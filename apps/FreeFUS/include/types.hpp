#pragma once
#include "mass_baseline.hpp"
#include "stiffness_baseline.hpp"
#include "stiffness_sf.hpp"
#include <basix/finite-element.h>


template<auto> inline constexpr bool always_false_v = false;


enum class MaterialCase : int {
  BP1 = 1,
  BP2 = 2,
  BP3 = 3,
  BP4 = 4,
  // BP5 = 5,
  // BP6 = 6,
  // BP7 = 7,
  H101 = 8,
  Skull = 9,
};

enum class ModelType : int {
  LinearExplicit = 1,
  LinearImplicit = 2,
  LinearLossyImplicit = 3,
  NonLinearLossyImplicit = 4,
};

enum class TimesteppingType : int {
  ExplicitRK4 = 1,
  Newmark = 2,
  NonlinearNewmark = 3,
};

template <typename T> struct UserConfig {
  std::string mesh_name;
  std::string mesh_filepath;
  std::string mesh_dir;
  std::string output_filepath;

  basix::element::lagrange_variant lvariant;

  MaterialCase material_case;
  ModelType model_type;
  TimesteppingType timestepping_type;

  T CFL;
  T source_frequency;
  T source_amplitude;
  T domain_length;
  T domain_width;

  T window_length;

  int sample_harmonic;
  int sampling_periods;
  int sample_nx;
  int sample_nz;

  int output_steps;
  bool insitu;
  bool insitu_with_yaml;
  int insitu_output_steps;

  double cg_tol;
  int cg_max_steps;
  double nonlinear_tol;

  spdlog::level::level_enum log_level;
};

template <typename T> struct PhysicalParameters {
  const T &source_frequency;
  const T &source_amplitude;
  const T &domain_length;
  const T &window_length;

  const T period;
  const T source_angular_frequency;

  PhysicalParameters(const UserConfig<T> &cfg)
      : source_frequency(cfg.source_frequency),
        source_amplitude(cfg.source_amplitude),
        domain_length(cfg.domain_length), window_length(cfg.window_length),
        period(static_cast<T>(1) / cfg.source_frequency),
        source_angular_frequency(static_cast<T>(2) * M_PI *
                                 cfg.source_frequency) {}
};

template <typename U>
struct MeshData {
  std::shared_ptr<mesh::Mesh<U>> mesh;
  std::shared_ptr<mesh::MeshTags<std::int32_t>> cell_tags;
  std::shared_ptr<mesh::MeshTags<std::int32_t>> facet_tags;
};

template <typename T, typename U, int P, int Q, int D>
using MassAction = std::conditional_t<D == 2, acc::MatFreeMassBaseline<T, P, Q, U>,
                                      acc::MatFreeMassBaseline3D<T, P, Q, U>>;

template <typename T, typename U, int P, int Q, int D>
using StiffnessAction =
    std::conditional_t<D == 2, acc::MatFreeStiffness<T, P, Q, U>,
                       acc::MatFreeStiffness3D<T, P, Q, U>>;
                      //  acc::MatFreeStiffnessSF3D<T, P, Q, U>>;

template <typename T, typename U, int P, int Q, int D>
using ExteriorMassAction =
    std::conditional_t<D == 2, acc::MatFreeMassExteriorBaseline<T, P, Q, U>,
                       acc::MatFreeMassExteriorBaseline3D<T, P, Q, U>>;

// TODO in 2d
template <typename T, typename U, int P, int Q, int D>
using NonlinearMassAction =
    std::conditional_t<D == 2, acc::NonlinearMatFreeMassBaseline3D<T, P, Q, U>,
                       acc::NonlinearMatFreeMassBaseline3D<T, P, Q, U>>;
