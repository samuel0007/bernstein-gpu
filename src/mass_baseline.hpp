#pragma once

#include "dof_ordering.hpp"
#include "geometry.hpp"
#include "kernels.hpp"
#include "quadrature.hpp"
#include "util.hpp"
#include "vector.hpp"
#include <memory>

namespace dolfinx::acc {

template <typename T, int P, int Q> class MatFreeMassBaseline {
public:
  using value_type = T;
  using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

  MatFreeMassBaseline(std::shared_ptr<mesh::Mesh<T>> mesh,
                      std::shared_ptr<fem::FunctionSpace<T>> V, T alpha)
      : mesh(mesh), V(V) {
    auto dofmap = V->dofmap();
    auto map = dofmap->index_map;
    int map_bs = dofmap->index_map_bs();
    la::Vector<T> alpha_vec(map, map_bs);
    alpha_vec.set(alpha);
    init(alpha_vec.array());
  }

  MatFreeMassBaseline(std::shared_ptr<mesh::Mesh<T>> mesh,
                      std::shared_ptr<fem::FunctionSpace<T>> V,
                      std::span<const T> alpha)
      : mesh(mesh), V(V) {
    init(alpha);
  }

  void init(std::span<const T> alpha) {
    // auto [lcells, bcells] = compute_boundary_cells(V);
    // spdlog::debug("#lcells = {}, #bcells = {}", lcells.size(),
    // bcells.size());

    auto element_p = this->V->element();
    // std::vector<int> dof_reordering = get_tp_ordering<P>(element_p);

    const std::size_t tdim = mesh->topology()->dim();
    const std::size_t gdim = mesh->geometry().dim();
    this->number_of_local_cells =
        mesh->topology()->index_map(tdim)->size_local() +
        mesh->topology()->index_map(tdim)->num_ghosts();
    // Transfer V dofmap to the GPU
    auto dofmap = V->dofmap();

    dofmap_d.resize(dofmap->map().size());
    thrust::copy(dofmap->map().data_handle(),
                 dofmap->map().data_handle() + dofmap->map().size(),
                 dofmap_d.begin());
    this->dofmap_d_span = std::span<const std::int32_t>(
        thrust::raw_pointer_cast(dofmap_d.data()), dofmap_d.size());

    std::cout << std::format("Sent dofmap to GPU (size = {} bytes)",
                             dofmap_d.size() * sizeof(std::int32_t))
              << std::endl;

    this->alpha_d.resize(alpha.size());
    thrust::copy(alpha.begin(), alpha.end(), this->alpha_d.begin());
    this->alpha_d_span = std::span<const T>(
        thrust::raw_pointer_cast(alpha_d.data()), alpha_d.size());

    std::cout << std::format("Sent alpha to GPU (size = {} bytes)",
                             alpha_d.size() * sizeof(T))
              << std::endl;

    // Construct quadrature points table
    auto [qpts, qwts] = basix::quadrature::make_quadrature<T>(
        basix::quadrature::type::Default, basix::cell::type::triangle,
        basix::polyset::type::standard, 2 * Q - 2);

    auto [phi_table, shape] =
        element_p->tabulate(qpts, {qpts.size() / 2, 2}, 0);

    std::cout << std::format("Table size = {}, qxn: {}x{}", phi_table.size(),
                             shape[1], shape[2])
              << std::endl;
    assert(shape[0] == 1 && shape[3] == 1);
    constexpr int nq = 6 * (P == 2) + 12 * (P == 3) + 16 * (P == 4) +
                       25 * (P == 5) + 33 * (P == 6) + 42 * (P == 7) +
                       55 * (P == 8) + 67 * (P == 9) + 79 * (P == 10) +
                       96 * (P == 11) + 112 * (P == 12);
    assert(nq == shape[1]);

    this->phi_d.resize(phi_table.size());
    thrust::copy(phi_table.begin(), phi_table.end(), phi_d.begin());
    this->phi_d_span = std::span<const T>(
        thrust::raw_pointer_cast(phi_d.data()), phi_d.size());

    // Precompute geometry data on cpu at collapsed quadrature points, and
    // copy it on the gpu
    std::vector<T> detJ_geom = compute_geometry(mesh, qpts, qwts);

    this->detJ_geom_d.resize(detJ_geom.size());
    thrust::copy(detJ_geom.begin(), detJ_geom.end(), detJ_geom_d.begin());
    this->detJ_geom_d_span = std::span<const T>(
        thrust::raw_pointer_cast(detJ_geom_d.data()), detJ_geom_d.size());

    std::cout << std::format("Send geometry to GPU (size = {} bytes)",
                             detJ_geom_d.size() * sizeof(T))
              << std::endl;
  }

  template <typename Vector> void operator()(const Vector &in, Vector &out) {
    out.set(T{0.0});

    const T *in_dofs = in.array().data();
    T *out_dofs = out.mutable_array().data();

    constexpr int N = P + 1;
    constexpr int nd =
        (N + 1) * N /
        2; // Number of dofs on triangle (is always smaller than nq)
    constexpr int nq = 6 * (P == 2) + 12 * (P == 3) + 16 * (P == 4) +
                       25 * (P == 5) + 33 * (P == 6) + 42 * (P == 7) +
                       55 * (P == 8) + 67 * (P == 9) + 79 * (P == 10) +
                       96 * (P == 11) + 112 * (P == 12);

    assert(dofmap_d_span.size() == this->number_of_local_cells * nd);
    assert(in.array().size() == out.mutable_array().size());
    assert(detJ_geom_d_span.size() == this->number_of_local_cells * Q * Q);

    // mass_operator<T, N, Q><<<grid_size, block_size>>>(
    //     in_dofs, out_dofs, this->alpha_d_span.data(),
    //     this->detJ_geom_d_span.data(), this->dofmap_d_span.data());
    // check_device_last_error();

    dim3 grid_size(this->number_of_local_cells);
    dim3 block_size(nq);
    mass_operator_baseline<T, nd, nq><<<grid_size, block_size>>>(
        in_dofs, out_dofs, this->alpha_d_span.data(),
        this->detJ_geom_d_span.data(), this->dofmap_d_span.data(),
        this->phi_d_span.data());
    check_device_last_error();
  }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    assert(false && "todo for jacobi preconditioning of cg");
  }

private:
  static constexpr int nq = Q * Q;
  std::size_t number_of_local_cells;

  std::shared_ptr<mesh::Mesh<T>> mesh;
  std::shared_ptr<fem::FunctionSpace<T>> V;

  thrust::device_vector<T> phi_d;
  std::span<const T> phi_d_span;

  thrust::device_vector<T> alpha_d;
  std::span<const T> alpha_d_span;

  thrust::device_vector<T> detJ_geom_d;
  std::span<const T> detJ_geom_d_span;

  thrust::device_vector<std::int32_t> dofmap_d;
  std::span<const std::int32_t> dofmap_d_span;
};

} // namespace dolfinx::acc