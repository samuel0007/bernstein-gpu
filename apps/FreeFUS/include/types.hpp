#pragma once
#include <basix/finite-element.h>

enum class MaterialCase : int {
  BP1 = 1,
  BP2 = 2,
  BP3 = 3,
  BP4 = 4,
  BP5 = 5,
  BP6 = 6,
  BP7 = 7
};

template <typename T> struct UserConfig {
  std::string mesh_name;
  std::string mesh_filepath;
  std::string output_filepath;
  basix::element::lagrange_variant lvariant;
  MaterialCase material_case;
  T CFL;
  T source_frequency;
  T source_amplitude;
  T domain_length;
  int output_steps;
  spdlog::level::level_enum log_level;
  bool insitu;
  int insitu_output_steps;
};

template <typename T> struct PhysicalParameters {
  const T &source_frequency;
  const T &source_amplitude;
  const T &domain_length;

  const T period;

  PhysicalParameters(const UserConfig<T> &cfg)
      : source_frequency(cfg.source_frequency),
        source_amplitude(cfg.source_amplitude), domain_length(cfg.domain_length),
        period(static_cast<T>(1) / cfg.source_frequency) {}
};

template <typename T> struct MeshData {
  std::shared_ptr<mesh::Mesh<T>> mesh;
  std::shared_ptr<mesh::MeshTags<std::int32_t>> cell_tags;
  std::shared_ptr<mesh::MeshTags<std::int32_t>> facet_tags;

  int tdim() const noexcept { return mesh->topology()->dim(); }
  int gdim() const noexcept { return mesh->geometry()->dim(); }
};
