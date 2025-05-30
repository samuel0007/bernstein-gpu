// Copyright (C) 2024 Adeeb Arif Kor
// SPDX-License-Identifier:    MIT

#pragma once

#include "quad_gpu.h"

#include <fstream>
#include <memory>
#include <string>

#include "petscksp.h"
#include "src/linalg.hpp"
#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/la/Vector.h>

#include "src/cg_gpu.hpp"
#include "src/mass_baseline.hpp"
#include "src/vector.hpp"

using namespace dolfinx;

namespace kernels {
// Copy data from a la::Vector in to a la::Vector out, including ghost entries.
// template <typename T> void copy(const la::Vector<T> &in, la::Vector<T> &out)
// {
//   std::span<const T> _in = in.array();
//   std::span<T> _out = out.mutable_array();
//   std::copy(_in.begin(), _in.end(), _out.begin());
// };

void copy_d(auto &in, auto &out) {
  thrust::copy(in.thrust_vector().begin(), in.thrust_vector().end(),
               out.thrust_vector().begin());
};
} // namespace kernels

/// Solver for the second order linear wave equation.
/// This solver uses GLL lattice and GLL quadrature such that it produces
/// a diagonal mass matrix.
/// @param [in] Mesh The mesh
/// @param [in] FacetTags The boundary facet tags
/// @param [in] speedOfSound A DG function defining the speed of sound within
/// the domain
/// @param [in] density A DG function defining the densities within the domain
/// @param [in] sourceFrequency The source frequency
/// @param [in] sourceAmplitude The source amplitude
/// @param [in] sourceSpeed The medium speed of sound that is in contact with
/// the source
template <typename T, int P, typename DeviceVector> class LinearSpectral {
  static constexpr int Q = P + 2; // TODO
public:
  LinearSpectral(std::shared_ptr<mesh::Mesh<T>> mesh,
                 std::shared_ptr<fem::FunctionSpace<T>> V,
                 std::shared_ptr<mesh::MeshTags<std::int32_t>> FacetTags,
                 std::shared_ptr<fem::Function<T>> speedOfSound,
                 std::shared_ptr<fem::Function<T>> density,
                 const T &sourceFrequency, const T &sourceAmplitude,
                 const T &sourceSpeed)
      : mesh(mesh) {
    // MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Physical parameters
    c0 = speedOfSound;
    rho0 = density;
    freq = sourceFrequency;
    w0 = 2 * M_PI * sourceFrequency;
    p0 = sourceAmplitude;
    s0 = sourceSpeed;
    period = 1.0 / sourceFrequency;
    window_length = 4.0;

    // Mesh data
    ft = FacetTags;

    // Define field functions
    index_map = V->dofmap()->index_map;
    bs = V->dofmap()->index_map_bs();

    u = std::make_shared<fem::Function<T>>(V);

    u_n = std::make_shared<fem::Function<T>>(V);
    v_n = std::make_shared<fem::Function<T>>(V);

    // Define source function
    g = std::make_shared<fem::Function<T>>(V);
    g_ = g->x()->mutable_array();

    // Define forms
    std::span<T> u_ = u->x()->mutable_array();
    std::fill(u_.begin(), u_.end(), 1.0);

    // Compute exterior facets
    std::vector<std::int32_t> ft_unique(ft->values().size());
    std::copy(ft->values().begin(), ft->values().end(), ft_unique.begin());
    std::sort(ft_unique.begin(), ft_unique.end());
    auto it = std::unique(ft_unique.begin(), ft_unique.end());
    ft_unique.erase(it, ft_unique.end());

    std::map<fem::IntegralType,
             std::vector<std::pair<std::int32_t, std::vector<std::int32_t>>>>
        fd;
    std::map<
        fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>
        fd_view;

    std::vector<std::int32_t> facet_domains;
    for (auto &tag : ft_unique) {
      facet_domains = fem::compute_integration_domains(
          fem::IntegralType::exterior_facet, *V->mesh()->topology_mutable(),
          ft->find(tag));
      fd[fem::IntegralType::exterior_facet].push_back({tag, facet_domains});
    }

    for (auto const &[key, val] : fd) {
      for (auto const &[tag, vec] : val) {
        fd_view[key].push_back({tag, std::span(vec.data(), vec.size())});
      }
    }

    a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_quad_gpu_a, {V}, {{"u", u}, {"c0", c0}, {"rho0", rho0}}, {},
        {}, {}));

    m = std::make_shared<la::Vector<T>>(index_map, bs);
    m_ = m->mutable_array();
    std::fill(m_.begin(), m_.end(), 0.0);
    fem::assemble_vector(m_, *a);

    // Define RHS form
    L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_quad_gpu_L, {V},
        {{"g", g}, {"u_n", u_n}, {"v_n", v_n}, {"c0", c0}, {"rho0", rho0}}, {},
        fd_view, {}, {}));

    b = std::make_shared<la::Vector<T>>(index_map, bs);
    b_ = b->mutable_array();
    b_d = std::make_shared<DeviceVector>(index_map, bs);
    u_n_d = std::make_shared<DeviceVector>(index_map, bs);
    v_n_d = std::make_shared<DeviceVector>(index_map, bs);
    m_d = std::make_shared<DeviceVector>(index_map, bs);
    m_d->copy_from_host(*m);
  }
  /// Set the initial values of u and v, i.e. u_0 and v_0
  void init() {
    u_n_d->set(0.);
    v_n_d->set(0.);
    u_n->x()->set(0.0);
    v_n->x()->set(0.0);
  }

  void f0(T &t, DeviceVector &u, DeviceVector &v, DeviceVector &result) {
    kernels::copy_d(v, result);
  }

  int f1(T &t, DeviceVector &u_d, DeviceVector &v_d, DeviceVector &result_d) {

    // Apply windowing
    if (t < period * window_length) {
      window = 0.5 * (1.0 - cos(freq * M_PI * t / window_length));
    } else {
      window = 1.0;
    }

    // Update boundary condition
    std::fill(g_.begin(), g_.end(),
              window * p0 * w0 / s0 * cos(w0 * t)); // homogenous domain

    // RHS Assembly is done on CPU, update form coefficients
    thrust::copy(u_d.thrust_vector().begin(), u_d.thrust_vector().end(),
                 u_n->x()->mutable_array().begin());
    thrust::copy(v_d.thrust_vector().begin(), v_d.thrust_vector().end(),
                 v_n->x()->mutable_array().begin());

    // Assemble RHS
    std::fill(b_.begin(), b_.end(), 0.0);
    fem::assemble_vector(b_, *L);
    b->scatter_rev(std::plus<T>());
    b_d->copy_from_host(*b);

    {
      // out = result->mutable_array();
      // _b = b->array();
      // _m = m->array();

      // // Element wise division
      // // out[i] = b[i]/m[i]
      // std::transform(_b.begin(), _b.end(), _m.begin(), out.begin(),
      //                [](const T &bi, const T &mi) { return bi / mi; });

      thrust::transform(thrust::device, b_d->mutable_array().begin(),
                        b_d->mutable_array().begin() + b_d->map()->size_local(),
                        m_d->mutable_array().begin(), result_d.mutable_array().begin(),
                        [] __host__ __device__( T bi, T mi) { return  bi / mi; });
    }

    return 0;

    // return cg_p->solve(gpu_action, result_d, *b_d, true);
  }

  /// Runge-Kutta 4th order solver
  /// @param[in] startTime initial time of the solver
  /// @param[in] finalTime final time of the solver
  /// @param[in] timeStep  time step size of the solver
  void rk4(const T &startTime, const T &finalTime, const T &timeStep,
           int output_frequency, std::shared_ptr<fem::Function<T>> u_out,
           dolfinx::io::VTXWriter<T> &f_out) {
    // Time-stepping parameters
    T t = startTime;
    T tf = finalTime;
    T dt = timeStep;
    int totalStep = (finalTime - startTime) / timeStep + 1;
    int outputStep = totalStep / output_frequency;

    int step = 0;

    // Time-stepping vectors
    std::shared_ptr<la::Vector<T>> u_, v_, un, vn, u0, v0, ku, kv;

    DeviceVector u__d(index_map, bs);
    DeviceVector v__d(index_map, bs);
    DeviceVector un_d(index_map, bs);
    DeviceVector vn_d(index_map, bs);
    DeviceVector u0_d(index_map, bs);
    DeviceVector v0_d(index_map, bs);
    DeviceVector ku_d(index_map, bs);
    DeviceVector kv_d(index_map, bs);

    kernels::copy_d(*u_n_d, u__d);
    kernels::copy_d(*v_n_d, v__d);

    kernels::copy_d(u__d, ku_d);
    kernels::copy_d(v__d, kv_d);

    // Runge-Kutta 4th order time-stepping data
    std::array<T, 4> a_runge = {0.0, 0.5, 0.5, 1.0};
    std::array<T, 4> b_runge = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
    std::array<T, 4> c_runge = {0.0, 0.5, 0.5, 1.0};

    // RK variables
    T tn;

    while (t < tf) {
      dt = std::min(dt, tf - t);

      kernels::copy_d(u__d, u0_d);
      kernels::copy_d(v__d, v0_d);

      // Runge-Kutta 4th order step
      for (int i = 0; i < 4; i++) {
        kernels::copy_d(u0_d, un_d);
        kernels::copy_d(v0_d, vn_d);

        acc::axpy(un_d, dt * a_runge[i], ku_d, un_d);
        acc::axpy(vn_d, dt * a_runge[i], kv_d, vn_d);

        // RK time evaluation
        tn = t + c_runge[i] * dt;

        f0(tn, un_d, vn_d, ku_d);
        int cg_its = f1(tn, un_d, vn_d, kv_d);
        // std::cout << "stage=" << i << " its=" << cg_its << std::endl;

        acc::axpy(u__d, dt * b_runge[i], ku_d, u__d);
        acc::axpy(v__d, dt * b_runge[i], kv_d, v__d);
      }

      // Update time
      t += dt;
      step += 1;

      if (step % 1 == 0) {
        if (mpi_rank == 0) {
          std::cout << "t: " << t << ",\t Steps: " << step << "/" << totalStep
                    << std::endl;
        }
      }
      if (step % outputStep == 0) {
        thrust::copy(u__d.thrust_vector().begin(), u__d.thrust_vector().end(),
                     u_n->x()->mutable_array().begin());
        u_out->interpolate(*u_n);
        f_out.write(t);
      }
    }

    // Prepare solution at final time
    thrust::copy(u__d.thrust_vector().begin(), u__d.thrust_vector().end(),
                 u_n->x()->mutable_array().begin());
    thrust::copy(v__d.thrust_vector().begin(), v__d.thrust_vector().end(),
                 v_n->x()->mutable_array().begin());

    // kernels::copy<T>(*u_, *u_n->x());
    // kernels::copy<T>(*v_, *v_n->x());
    u_n->x()->scatter_fwd();
    v_n->x()->scatter_fwd();
  }

  std::shared_ptr<fem::Function<T>> u_sol() const { return u_n; }

  std::int64_t number_of_dofs() const {
    return V->dofmap()->index_map->size_global();
  }

private:
  int mpi_rank, mpi_size; // MPI rank and size
  int bs;                 // block size
  T freq;                 // source frequency (Hz)
  T p0;                   // source amplitude (Pa)
  T w0;                   // angular frequency  (rad/s)
  T s0;                   // speed (m/s)
  T period, window_length, window;

  std::shared_ptr<mesh::Mesh<T>> mesh;
  std::shared_ptr<mesh::MeshTags<std::int32_t>> ft;
  std::shared_ptr<const common::IndexMap> index_map;
  std::shared_ptr<fem::FunctionSpace<T>> V;
  std::shared_ptr<fem::Function<T>> u, u_n, v_n, g, c0, rho0;
  std::shared_ptr<fem::Form<T>> a, L;
  std::shared_ptr<la::Vector<T>> m, b;

  std::shared_ptr<DeviceVector> b_d, u_n_d, v_n_d, m_d;

  std::function<void(const la::Vector<T> &, la::Vector<T> &)> action;
  std::map<std::pair<dolfinx::fem::IntegralType, int>,
           std::pair<std::vector<double>, int>>
      coeff;
  std::vector<T> constants;

  std::span<T> g_, m_, b_, out;
  std::span<const T> _m, _b;

  // acc::MatFreeMassBaseline3D<T, P, Q> gpu_action;
  // std::unique_ptr<dolfinx::acc::CGSolver<DeviceVector>> cg_p;
};

// Note:
// mutable array -> variable_name_
// array -> _variable_name