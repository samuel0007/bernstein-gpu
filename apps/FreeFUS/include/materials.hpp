#pragma once
#include <vector>

#include "types.hpp"

namespace freefus {
namespace materials {
template <typename T> struct Material {
  T density;
  T sound_speed;
  int domain_id;
};

template <typename T> struct WaterData {
  static constexpr T density = 1000;
  static constexpr T sound_speed = 1500;
};

template <typename T>
std::vector<Material<T>> BP1Materials{
    {WaterData<T>::density, WaterData<T>::sound_speed, 1}};

template <typename T>
std::vector<Material<T>> getMaterialsData(MaterialCase material_case) {
  assert(material_case == MaterialCase::BP1); // TODO
  return BP1Materials<T>;
};
} // namespace materials

template <typename T>
auto create_materials_coefficients(std::shared_ptr<fem::FunctionSpace<T>> V_DG,
                                   const MeshData<T> mesh_data,
                                   MaterialCase material_case) {

  std::vector<materials::Material<T>> materials_data =
      materials::getMaterialsData<T>(material_case);
  auto c0 = std::make_shared<fem::Function<T>>(V_DG);
  auto rho0 = std::make_shared<fem::Function<T>>(V_DG);

  for (auto material : materials_data) {
    auto cells = mesh_data.cell_tags->find(material.domain_id);
    spdlog::info(
        "Material domain id {}, sounds speed {}, density {}, #cells {}",
        material.domain_id, material.sound_speed, material.density,
        cells.size());
    std::span<T> c0_ = c0->x()->mutable_array();
    std::for_each(cells.begin(), cells.end(),
                  [&](std::int32_t &i) { c0_[i] = material.sound_speed; });

    std::span<T> rho0_ = rho0->x()->mutable_array();
    std::for_each(cells.begin(), cells.end(),
                  [&](std::int32_t &i) { rho0_[i] = material.density; });
  }
  rho0->x()->scatter_fwd();
  c0->x()->scatter_fwd();

  return std::make_tuple(c0, rho0);
};

template <typename T> T get_source_sound_speed(MaterialCase material_case) {
  std::vector<materials::Material<T>> materials_data =
      materials::getMaterialsData<T>(material_case);
  for (auto material : materials_data) {
    if (material.domain_id == 1)
      return material.sound_speed;
  }
  assert(false && "No Material Source found");
  return 0.;
}
} // namespace freefus
