// Copyright (C) 2024 Adeeb Arif Kor
// SPDX-License-Identifier:    MIT

#pragma once

#include "planar_wave_triangles.h"

#include <fstream>
#include <memory>
#include <string>

#include "src/linalg.hpp"
#include <dolfinx.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/fem/petsc.h>
#include "petscksp.h" 


using namespace dolfinx;

namespace kernels {
// Copy data from a la::Vector in to a la::Vector out, including ghost entries.
template <typename T> void copy(const la::Vector<T> &in, la::Vector<T> &out) {
  std::span<const T> _in = in.array();
  std::span<T> _out = out.mutable_array();
  std::copy(_in.begin(), _in.end(), _out.begin());
}

/// Compute vector r = alpha*x + y
/// @param r Result
/// @param alpha
/// @param x
/// @param y
template <typename T>
void axpy(la::Vector<T> &r, T alpha, const la::Vector<T> &x,
          const la::Vector<T> &y) {
  std::transform(
      x.array().begin(), x.array().begin() + x.index_map()->size_local(),
      y.array().begin(), r.mutable_array().begin(),
      [&alpha](const T &vx, const T &vy) { return vx * alpha + vy; });
}

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
template <typename T, int P> class LinearSpectral {
public:
  LinearSpectral(basix::FiniteElement<T> element,
                 std::shared_ptr<mesh::Mesh<T>> Mesh,
                 std::shared_ptr<mesh::MeshTags<std::int32_t>> FacetTags,
                 std::shared_ptr<fem::Function<T>> speedOfSound,
                 std::shared_ptr<fem::Function<T>> density,
                 const T &sourceFrequency, const T &sourceAmplitude,
                 const T &sourceSpeed) {
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
    mesh = Mesh;
    ft = FacetTags;

    // Define function space
    V = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(mesh, element));

    // Define field functions
    index_map = V->dofmap()->index_map;
    bs = V->dofmap()->index_map_bs();

    u = std::make_shared<fem::Function<T>>(V);
    ui = std::make_shared<fem::Function<T>>(V);

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
      facet_domains = ffem::compute_integration_domains(
          fem::IntegralType::exterior_facet, *V->mesh()->topology_mutable(),
          ft->find(tag));
      fd[fem::IntegralType::exterior_facet].push_back({tag, facet_domains});
    }

    for (auto const &[key, val] : fd) {
      for (auto const &[tag, vec] : val) {
        fd_view[key].push_back({tag, std::span(vec.data(), vec.size())});
      }
    }

    // Define LHS form (bilinear)
    a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_planar_wave_triangles_a_M, {V, V},
        {{"c0", c0}, {"rho0", rho0}}, {}, {}, {}));

    auto A = la::petsc::Matrix(fem::petsc::create_matrix(*a), false);
    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                         *a, {});
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    MatNullSpace nsp;
    MatNullSpaceCreate(MPI_COMM_WORLD,
                      PETSC_TRUE,   // Constant Nullspace (Pure NBc problem, imposes 0 mean)
                      0,            
                      0,           
                      &nsp);
    MatSetNearNullSpace(A.mat(), nsp);
    // MatSetNullSpace(A.mat(), nsp);
    
    // la::petsc::options::set("ksp_type", "gmres");
    // la::petsc::options::set("ksp_type", "preonly");

    la::petsc::options::set("ksp_type", "cg");
    la::petsc::options::set("pc_type", "jacobi");
    // la::petsc::options::set("pc_type", "lu");

    lu.set_from_options();
    lu.set_operator(A.mat());

    m = std::make_shared<la::Vector<T>>(index_map, bs);
    m_ = m->mutable_array();
    std::fill(m_.begin(), m_.end(), 0.0);
    fem::assemble_vector(m_, *a);
    m->scatter_rev(std::plus<T>());

    // Define RHS form
    L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_planar_wave_triangles_L, {V},
        {{"g", g}, {"u_n", u_n}, {"v_n", v_n}, {"c0", c0}, {"rho0", rho0}}, {},
        fd_view, {}, {}));

    b = std::make_shared<la::Vector<T>>(index_map, bs);
    b_ = b->mutable_array();

    // matrix free
    a_linear = std::make_shared<fem::Form<T>>(fem::create_form<T>(
      *form_planar_wave_triangles_a, {V}, {{"u", ui}, {"c0", c0}, {"rho0", rho0}}, {},
      {}, {}));

    coeff = fem::allocate_coefficient_storage(*a_linear);
    constants = fem::pack_constants(*a_linear);

    action = [this](auto &x, auto &y) {
      y.set(0.0);

      // Update coefficient ui (just copy data from x to ui)
      std::ranges::copy(x.array(), this->ui->x()->mutable_array().begin());

      // Compute action of A on x
      fem::pack_coefficients(*a_linear, coeff);
      fem::assemble_vector(y.mutable_array(), *a_linear,
                           std::span<const T>(constants),
                           fem::make_coefficients_span(coeff));

      // // Accumulate ghost values
      // y.scatter_rev(std::plus<T>());

      // // Update ghost values
      // y.scatter_fwd();
    };
  }
  /// Set the initial values of u and v, i.e. u_0 and v_0
  void init() {
    u_n->x()->set(0.0);
    v_n->x()->set(0.0);
  }

  /// Evaluate du/dt = f0(t, u, v)
  /// @param[in] t Current time, i.e. tn
  /// @param[in] u Current u, i.e. un
  /// @param[in] v Current v, i.e. vn
  /// @param[out] result Result, i.e. dun/dtn
  void f0(T &t, std::shared_ptr<la::Vector<T>> u,
          std::shared_ptr<la::Vector<T>> v,
          std::shared_ptr<la::Vector<T>> result) {
    kernels::copy<T>(*v, *result);
  }

  /// Evaluate dv/dt = f1(t, u, v)
  /// @param[in] t Current time, i.e. tn
  /// @param[in] u Current u, i.e. un
  /// @param[in] v Current v, i.e. vn
  /// @param[out] result Result, i.e. dvn/dtn
  void f1(T &t, std::shared_ptr<la::Vector<T>> u,
          std::shared_ptr<la::Vector<T>> v,
          std::shared_ptr<la::Vector<T>> result) {

    // Apply windowing
    if (t < period * window_length) {
      window = 0.5 * (1.0 - cos(freq * M_PI * t / window_length));
    } else {
      window = 1.0;
    }

    // Update boundary condition
    std::fill(g_.begin(), g_.end(),
              window * p0 * w0 / s0 * cos(w0 * t)); // homogenous domain
    // std::fill(g_.begin(), g_.end(), 2.0 * window * p0 * w0 / s0 * cos(w0 *
    // t)); // heterogenous domain

    u->scatter_fwd();
    kernels::copy<T>(*u, *u_n->x());

    v->scatter_fwd();
    kernels::copy<T>(*v, *v_n->x());

    // Assemble RHS
    std::fill(b_.begin(), b_.end(), 0.0);
    fem::assemble_vector(b_, *L);
    b->scatter_rev(std::plus<T>());

    la::Vector<T> test_result(*result);

    // {
    //   la::petsc::Vector _u(la::petsc::create_vector_wrap(test_result), false);
    //   la::petsc::Vector _b(la::petsc::create_vector_wrap(*b), false);
    //   int its = lu.solve(_u.vec(), _b.vec());
    //   std::cout << "KSP CG its=" << its << "\n";
    //   auto ksp = lu.ksp();
    //   KSPConvergedReason reason;
    //   KSPGetConvergedReason(ksp, &reason);
    //   if (reason < 0)
    //     std::cerr << "KSP Failure: reason " << reason << "\n";
    // }

    int its = linalg::cg(*result, *b, action, 100, 1e-6);
    std::cout << "House CG its=" << its << std::endl;
  }

  /// Runge-Kutta 4th order solver
  /// @param[in] startTime initial time of the solver
  /// @param[in] finalTime final time of the solver
  /// @param[in] timeStep  time step size of the solver
  void rk4(const T &startTime, const T &finalTime, const T &timeStep) {

    // Time-stepping parameters
    T t = startTime;
    T tf = finalTime;
    T dt = timeStep;
    int totalStep = (finalTime - startTime) / timeStep + 1;
    int step = 0;

    // Time-stepping vectors
    std::shared_ptr<la::Vector<T>> u_, v_, un, vn, u0, v0, ku, kv;

    // Placeholder vectors at time step n
    u_ = std::make_shared<la::Vector<T>>(index_map, bs);
    v_ = std::make_shared<la::Vector<T>>(index_map, bs);

    kernels::copy<T>(*u_n->x(), *u_);
    kernels::copy<T>(*v_n->x(), *v_);

    // Placeholder vectors at intermediate time step n
    un = std::make_shared<la::Vector<T>>(index_map, bs);
    vn = std::make_shared<la::Vector<T>>(index_map, bs);

    // Placeholder vectors at start of time step
    u0 = std::make_shared<la::Vector<T>>(index_map, bs);
    v0 = std::make_shared<la::Vector<T>>(index_map, bs);

    // Placeholder at k intermediate time step
    ku = std::make_shared<la::Vector<T>>(index_map, bs);
    kv = std::make_shared<la::Vector<T>>(index_map, bs);

    kernels::copy<T>(*u_, *ku);
    kernels::copy<T>(*v_, *kv);

    // Runge-Kutta 4th order time-stepping data
    std::array<T, 4> a_runge = {0.0, 0.5, 0.5, 1.0};
    std::array<T, 4> b_runge = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
    std::array<T, 4> c_runge = {0.0, 0.5, 0.5, 1.0};

    // RK variables
    T tn;

    while (t < tf) {
      dt = std::min(dt, tf - t);

      // Store solution at start of time step
      kernels::copy<T>(*u_, *u0);
      kernels::copy<T>(*v_, *v0);

      // Runge-Kutta 4th order step
      for (int i = 0; i < 4; i++) {
        kernels::copy<T>(*u0, *un);
        kernels::copy<T>(*v0, *vn);

        kernels::axpy<T>(*un, dt * a_runge[i], *ku, *un);
        kernels::axpy<T>(*vn, dt * a_runge[i], *kv, *vn);

        // RK time evaluation
        tn = t + c_runge[i] * dt;

        // Compute RHS vector
        f0(tn, un, vn, ku);
        f1(tn, un, vn, kv);

        // Update solution
        kernels::axpy<T>(*u_, dt * b_runge[i], *ku, *u_);
        kernels::axpy<T>(*v_, dt * b_runge[i], *kv, *v_);
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
    }

    // Prepare solution at final time
    kernels::copy<T>(*u_, *u_n->x());
    kernels::copy<T>(*v_, *v_n->x());
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
  std::shared_ptr<fem::Function<T>> u, u_n, v_n, g, c0, rho0, ui;
  std::shared_ptr<fem::Form<T>> a, L, a_linear;
  std::shared_ptr<la::Vector<T>> m, b;

  std::function<void(const la::Vector<T> &, la::Vector<T> &)> action;
  std::map<std::pair<dolfinx::fem::IntegralType, int>,
           std::pair<std::vector<double>, int>>
      coeff;
  std::vector<T> constants;

  std::span<T> g_, m_, b_, out;
  std::span<const T> _m, _b;

  la::petsc::KrylovSolver lu{MPI_COMM_WORLD};
};

// Note:
// mutable array -> variable_name_
// array -> _variable_name