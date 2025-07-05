#pragma once
#include <vector>

#include "types.hpp"

namespace freefus {
namespace materials {
template <typename T> struct Material {
  T density;
  T sound_speed;
  T diffusivity;
  int domain_id;
};

template <typename T> struct WaterData {
  static constexpr T density = 1000;
  static constexpr T sound_speed = 1500;
  static constexpr T diffusivity = 0;
};

template <typename T> struct SkinData {
  static constexpr T density = 1090;
  static constexpr T sound_speed = 1610;
  static constexpr T diffusivity = 0.2;
};

template <typename T> struct BrainData {
  static constexpr T density = 1040;
  static constexpr T sound_speed = 1560;
  static constexpr T diffusivity = 0.3;
};

template <typename T> struct CorticalBoneData {
  static constexpr T density = 1850;
  static constexpr T sound_speed = 2800;
  static constexpr T diffusivity = 4;
};

template <typename T> struct TrabecularBoneData {
  static constexpr T density = 1700;
  static constexpr T sound_speed = 2300;
  static constexpr T diffusivity = 8;
};

// Material Cases. Domain "1" has to be the domain where the source lies.
template <typename T>
std::vector<Material<T>> BP1Materials{{WaterData<T>::density,
                                       WaterData<T>::sound_speed,
                                       WaterData<T>::diffusivity, 1}};

template <typename T>
std::vector<Material<T>> BP2Materials{
    {WaterData<T>::density, WaterData<T>::sound_speed,
     WaterData<T>::diffusivity, 1},
    {TrabecularBoneData<T>::density, TrabecularBoneData<T>::sound_speed,
     TrabecularBoneData<T>::diffusivity, 2}};

template <typename T>
std::vector<Material<T>> getMaterialsData(MaterialCase material_case) {
  switch (material_case) {
  case MaterialCase::BP1:
    return BP1Materials<T>;
  case MaterialCase::BP2:
    return BP2Materials<T>;
  }
  throw std::runtime_error("Invalid Material Case");
};
} // namespace materials

template <typename U>
auto create_materials_coefficients(std::shared_ptr<fem::FunctionSpace<U>> V_DG,
                                   const MeshData<U> mesh_data,
                                   MaterialCase material_case) {

  std::vector<materials::Material<U>> materials_data =
      materials::getMaterialsData<U>(material_case);
  auto c0 = std::make_shared<fem::Function<U>>(V_DG);
  auto rho0 = std::make_shared<fem::Function<U>>(V_DG);
  auto delta0 = std::make_shared<fem::Function<U>>(V_DG);

  for (auto material : materials_data) {
    auto cells = mesh_data.cell_tags->find(material.domain_id);
    // auto indices = mesh_data.cell_tags->indices();
    // auto values = mesh_data.cell_tags->values();
    // TODO: understand why the above doesn't work in parallel.
    // const int tdim = mesh_data.mesh->topology()->dim();
    // const int N = mesh_data.mesh->topology()->index_map(tdim)->size_local();
    // std::vector<int> cells(N);
    // std::iota(cells.begin(), cells.end(), 0);

    spdlog::info(
        "Material domain id {}, sounds speed {}, density {}, #cells {}",
        material.domain_id, material.sound_speed, material.density,
        cells.size());
    std::span<U> c0_ = c0->x()->mutable_array();
    std::for_each(cells.begin(), cells.end(),
                  [&](std::int32_t &i) { c0_[i] = material.sound_speed; });

    std::span<U> rho0_ = rho0->x()->mutable_array();
    std::for_each(cells.begin(), cells.end(),
                  [&](std::int32_t &i) { rho0_[i] = material.density; });

    std::span<U> delta0_ = delta0->x()->mutable_array();
    std::for_each(cells.begin(), cells.end(),
                  [&](std::int32_t &i) { delta0_[i] = material.diffusivity; });
  }
  rho0->x()->scatter_fwd();
  c0->x()->scatter_fwd();
  delta0->x()->scatter_fwd();

  return std::make_tuple(c0, rho0, delta0);
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
