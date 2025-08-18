#pragma once
#include <vector>

#include "types.hpp"

namespace freefus
{
  namespace materials
  {

    inline constexpr auto alpha_to_delta = []<typename T>(T alpha, T c,
                                                          T w) noexcept
    {
      return T(2) * (100 * alpha / 20 * log(10)) * c * c * c / (w * w);
    };

    template <typename T>
    struct Material
    {
      T density;
      T sound_speed;
      T alpha;
      int domain_id;
      T nonlinear_coefficient = 0;
    };

    template <typename T>
    struct WaterData
    {
      static constexpr T density = 1000;
      static constexpr T sound_speed = 1500;
      static constexpr T alpha = 0;
    };

    template <typename T>
    struct SkinData
    {
      static constexpr T density = 1090;
      static constexpr T sound_speed = 1610;
      static constexpr T alpha = 0.2;
    };

    template <typename T>
    struct BrainData
    {
      static constexpr T density = 1040;
      // static constexpr T density = 2300; for vis
      static constexpr T sound_speed = 1560;
      static constexpr T alpha = 0.3;
    };

    template <typename T>
    struct CorticalBoneData
    {
      static constexpr T density = 1850;
      // static constexpr T density = 1400; for vis
      static constexpr T sound_speed = 2800;
      static constexpr T alpha = 4;
    };

    template <typename T>
    struct TrabecularBoneData
    {
      static constexpr T density = 1700;
      static constexpr T sound_speed = 2300;
      static constexpr T alpha = 8;
    };

    template <typename T>
    struct NonlinearWaterData
    {
      static constexpr T density = 1000;
      static constexpr T sound_speed = 1480;
      static constexpr T alpha = 0.2;
      static constexpr T nonlinear_coefficient = 3.5;
    };

    template <typename T>
    struct NonlinearLiverData
    {
      static constexpr T density = 1060;
      static constexpr T sound_speed = 1590;
      static constexpr T alpha = 90;
      static constexpr T nonlinear_coefficient = 4.4;
    };

    // Material Cases. Domain "1" has to be the domain where the source lies.
    template <typename T>
    std::vector<Material<T>> BP1Materials{
        {WaterData<T>::density, WaterData<T>::sound_speed, WaterData<T>::alpha, 1}};

    template <typename T>
    std::vector<Material<T>> BP2Materials{
        {WaterData<T>::density, WaterData<T>::sound_speed, 1, 1}};

    template <typename T>
    std::vector<Material<T>> BP3Materials{
        {WaterData<T>::density, WaterData<T>::sound_speed, WaterData<T>::alpha, 1},
        {CorticalBoneData<T>::density, CorticalBoneData<T>::sound_speed,
         CorticalBoneData<T>::alpha, 2}};

    // Water: 1, Skin: 2, Cortical: 3, Trabecular: 4, Brain: 5
    template <typename T>
    std::vector<Material<T>> BP4Materials{
        {WaterData<T>::density, WaterData<T>::sound_speed, WaterData<T>::alpha, 1},
        {SkinData<T>::density, SkinData<T>::sound_speed, SkinData<T>::alpha, 2},
        {CorticalBoneData<T>::density, CorticalBoneData<T>::sound_speed,
         CorticalBoneData<T>::alpha, 3},
        {TrabecularBoneData<T>::density, TrabecularBoneData<T>::sound_speed,
         TrabecularBoneData<T>::alpha, 4},
        {BrainData<T>::density, BrainData<T>::sound_speed, BrainData<T>::alpha, 5}};

    // Water: 1, Liver: 2
    template <typename T>
    std::vector<Material<T>> H101Materials{
        {NonlinearWaterData<T>::density, NonlinearWaterData<T>::sound_speed,
         NonlinearWaterData<T>::alpha, 1,
         NonlinearWaterData<T>::nonlinear_coefficient},
        {NonlinearLiverData<T>::density, NonlinearLiverData<T>::sound_speed,
         NonlinearLiverData<T>::alpha, 2,
         NonlinearLiverData<T>::nonlinear_coefficient},
    };

    // Water: 1
    template <typename T>
    std::vector<Material<T>> NonlinearTestMaterials{
        {WaterData<T>::density, WaterData<T>::sound_speed,
         WaterData<T>::alpha, 1, 100.},
    };

    // Water: 1, Cortical Bone: 2, Brain: 3
    template <typename T>
    std::vector<Material<T>> SkullMaterials{
        {WaterData<T>::density, WaterData<T>::sound_speed, WaterData<T>::alpha, 1},
        {CorticalBoneData<T>::density, CorticalBoneData<T>::sound_speed,
         CorticalBoneData<T>::alpha, 2},
        {BrainData<T>::density, BrainData<T>::sound_speed, BrainData<T>::alpha, 3},
    };

    template <typename T>
    std::vector<Material<T>> getMaterialsData(MaterialCase material_case)
    {
      switch (material_case)
      {
      case MaterialCase::BP1:
        return BP1Materials<T>;
      case MaterialCase::BP2:
        return BP2Materials<T>;
      case MaterialCase::BP3:
        return BP3Materials<T>;
      case MaterialCase::BP4:
        return BP4Materials<T>;
      case MaterialCase::H101:
        return H101Materials<T>;
      case MaterialCase::Skull:
        return SkullMaterials<T>;
      case MaterialCase::NonlinearTest:
        return NonlinearTestMaterials<T>;
      }
      throw std::runtime_error("Invalid Material Case");
    };
  } // namespace materials

  template <typename U>
  auto create_materials_coefficients(std::shared_ptr<fem::FunctionSpace<U>> V_DG,
                                     const MeshData<U> mesh_data,
                                     MaterialCase material_case,
                                     const PhysicalParameters<U> &params)
  {

    std::vector<materials::Material<U>> materials_data =
        materials::getMaterialsData<U>(material_case);

    auto rho0 = std::make_shared<fem::Function<U>>(V_DG);
    auto c0 = std::make_shared<fem::Function<U>>(V_DG);
    auto delta0 = std::make_shared<fem::Function<U>>(V_DG);
    auto b0 = std::make_shared<fem::Function<U>>(V_DG);

    int cell_count = 0;
    for (const auto &material : materials_data)
    {
      auto cells = mesh_data.cell_tags->find(material.domain_id);
      cell_count += cells.size();
      // const int tdim = mesh_data.mesh->topology()->dim();
      // const int N = mesh_data.mesh->topology()->index_map(tdim)->size_local();
      // std::vector<int> cells(N);
      // std::iota(cells.begin(), cells.end(), 0);

      spdlog::info("Material domain id {}, sounds speed {}, density {}, "
                   "nonlinaer coeff: {}, #cells {}",
                   material.domain_id, material.sound_speed, material.density,
                   material.nonlinear_coefficient, cells.size());

      std::span<U> rho0_ = rho0->x()->mutable_array();
      std::for_each(cells.begin(), cells.end(),
                    [&](std::int32_t &i)
                    { rho0_[i] = material.density; });

      std::span<U> c0_ = c0->x()->mutable_array();
      std::for_each(cells.begin(), cells.end(),
                    [&](std::int32_t &i)
                    { c0_[i] = material.sound_speed; });

      std::span<U> delta0_ = delta0->x()->mutable_array();
      std::for_each(cells.begin(), cells.end(), [&](std::int32_t &i)
                    { delta0_[i] =
                          materials::alpha_to_delta(material.alpha, material.sound_speed,
                                                    params.source_angular_frequency); });

      std::span<U> b0_ = b0->x()->mutable_array();
      std::for_each(cells.begin(), cells.end(), [&](std::int32_t &i)
                    { b0_[i] = material.nonlinear_coefficient; });
    }
    const int tdim = V_DG->mesh()->topology()->dim();
    spdlog::info("Cells: local={}, ghost={}, total={}",
                 V_DG->mesh()->topology()->index_map(tdim)->size_local(),
                 V_DG->mesh()->topology()->index_map(tdim)->num_ghosts(),
                 V_DG->mesh()->topology()->index_map(tdim)->size_global());
    spdlog::info("Cell count: {}", cell_count);
    if (cell_count != V_DG->mesh()->topology()->index_map(tdim)->size_local())
    {
      throw std::runtime_error("invalid material setup. Mismatched number of "
                               "tagged cells and local cells.");
    };

    c0->x()->scatter_fwd();
    rho0->x()->scatter_fwd();
    delta0->x()->scatter_fwd();
    b0->x()->scatter_fwd();

    return std::make_tuple(c0, rho0, delta0, b0);
  };

  template <typename T>
  T get_source_sound_speed(MaterialCase material_case)
  {
    std::vector<materials::Material<T>> materials_data =
        materials::getMaterialsData<T>(material_case);
    for (auto material : materials_data)
    {
      if (material.domain_id == 1)
        return material.sound_speed;
    }
    assert(false && "No Material Source found");
    return 0.;
  }
} // namespace freefus
