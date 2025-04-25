#pragma once

#include "dof_ordering.hpp"
#include "geometry.hpp"
#include "kernels.hpp"
#include "quadrature.hpp"
#include "util.hpp"
#include "vector.hpp"
#include <memory>

namespace dolfinx::acc {

template <typename T, int P, int Q> class MatFreeMass3D {
public:
  using value_type = T;
  using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

  MatFreeMass3D(std::shared_ptr<mesh::Mesh<T>> mesh,
                std::shared_ptr<fem::FunctionSpace<T>> V, T alpha)
      : mesh(mesh), V(V) {
    auto dofmap = V->dofmap();
    auto map = dofmap->index_map;
    int map_bs = dofmap->index_map_bs();
    la::Vector<T> alpha_vec(map, map_bs);
    alpha_vec.set(alpha);
    init(alpha_vec.array());
  }

  MatFreeMass3D(std::shared_ptr<mesh::Mesh<T>> mesh,
                std::shared_ptr<fem::FunctionSpace<T>> V,
                std::span<const T> alpha)
      : mesh(mesh), V(V) {
    init(alpha);
  }

  void init(std::span<const T> alpha) {
    // auto [lcells, bcells] = compute_boundary_cells(V);
    // spdlog::debug("#lcells = {}, #bcells = {}", lcells.size(),
    // bcells.size());

    // TODO
    // auto element_p = this->V->element();
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
    auto [rule0, rule1, rule2, ruleT] =
        create_quadrature_tetrahedron_duffy<T>(Q);

    auto &[qpts0, qwts0] = rule0;
    auto &[qpts1, qwts1] = rule1;
    auto &[qpts2, qwts2] = rule2;
    auto &[qptsT, _] = ruleT;

    // Precompute geometry data on cpu at collapsed quadrature points, and
    // copy it on the gpu
    std::cout << "start computing geometry..." << std::endl;
    std::vector<T> detJ_geom = compute_geometry(mesh, qptsT, 3);
    std::cout << "finished computing geometry." << std::endl;


    this->detJ_geom_d.resize(detJ_geom.size());
    thrust::copy(detJ_geom.begin(), detJ_geom.end(), detJ_geom_d.begin());
    this->detJ_geom_d_span = std::span<const T>(
        thrust::raw_pointer_cast(detJ_geom_d.data()), detJ_geom_d.size());

    std::cout << std::format("Send geometry to GPU (size = {} bytes)",
                             detJ_geom_d.size() * sizeof(T))
              << std::endl;

    // Copy quadrature weights as symbols to the gpu
    err_check(deviceMemcpyToSymbol((qwts0_d<T, Q>), qwts0.data(),
                                   qwts0.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((qwts1_d<T, Q>), qwts1.data(),
                                   qwts1.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((qwts2_d<T, Q>), qwts2.data(),
                                   qwts2.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((qpts0_d<T, Q>), qpts0.data(),
                                   qpts0.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((qpts1_d<T, Q>), qpts1.data(),
                                   qpts1.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((qpts2_d<T, Q>), qpts2.data(),
                                   qpts2.size() * sizeof(T)));
    // // Copy dofmap reordering to the gpu
    // err_check(deviceMemcpyToSymbol((dof_reordering_d<P + 1>),
    // dof_reordering.data(),
    //                                dof_reordering.size() * sizeof(int)));
  }

  template <typename Vector> void operator()(const Vector &in, Vector &out) {
    out.set(T{0.0});

    const T *in_dofs = in.array().data();
    T *out_dofs = out.mutable_array().data();

    dim3 grid_size(this->number_of_local_cells);
    dim3 block_size(Q, Q, Q);
    constexpr int N = P + 1;
    constexpr int K = N*(N+1)*(N+2) / 6; // Number of dofs on tet

    assert(dofmap_d_span.size() == this->number_of_local_cells * K);
    assert(in.array().size() == out.mutable_array().size());
    assert(detJ_geom_d_span.size() == this->number_of_local_cells * Q * Q * Q);

    mass_operator3D<T, N, Q><<<grid_size, block_size>>>(
        in_dofs, out_dofs, this->alpha_d_span.data(),
        this->detJ_geom_d_span.data(), this->dofmap_d_span.data());
    check_device_last_error();
  }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    assert(false && "todo for jacobi preconditioning of cg");
  }

private:
  // static constexpr int nq = Q * Q * Q;
  std::size_t number_of_local_cells;

  std::shared_ptr<mesh::Mesh<T>> mesh;
  std::shared_ptr<fem::FunctionSpace<T>> V;

  thrust::device_vector<T> alpha_d;
  std::span<const T> alpha_d_span;

  thrust::device_vector<T> detJ_geom_d;
  std::span<const T> detJ_geom_d_span;

  thrust::device_vector<std::int32_t> dofmap_d;
  std::span<const std::int32_t> dofmap_d_span;
};

} // namespace dolfinx::acc