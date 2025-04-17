#pragma once

#include "geometry.hpp"
#include "kernels.hpp"
#include "quadrature.hpp"
#include "util.hpp"
#include "vector.hpp"
#include <memory>

namespace dolfinx::acc {

template <typename T, int P, int Q> class MatFreeMass {
public:
  using value_type = T;
  using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

  MatFreeMass(std::shared_ptr<mesh::Mesh<T>> mesh,
              std::shared_ptr<fem::FunctionSpace<T>> V)
      : mesh(mesh), V(V) {

    // auto [lcells, bcells] = compute_boundary_cells(V);
    // spdlog::debug("#lcells = {}, #bcells = {}", lcells.size(),
    // bcells.size());

    const std::size_t tdim = mesh->topology()->dim();
    const std::size_t gdim = mesh->geometry().dim();
    this->number_of_local_cells =
        mesh->topology()->index_map(tdim)->size_local() +
        mesh->topology()->index_map(tdim)->num_ghosts();
    // Transfer V dofmap to the GPU
    auto dofmap = V->dofmap();

    dofmap_d.resize(dofmap->map().size());
    thrust::copy(dofmap->map().data_handle(), dofmap->map().data_handle()+dofmap->map().size(), dofmap_d.begin());
    this->dofmap_d_span = std::span<const std::int32_t>(
        thrust::raw_pointer_cast(dofmap_d.data()), dofmap_d.size());

    std::cout << std::format("Sent dofmap to GPU (size = {} bytes)",
                             dofmap_d.size() * sizeof(std::int32_t))
              << std::endl;

    // Construct quadrature points table
    auto [rule0, rule1, ruleT] = create_quadrature_triangle_duffy<T>(Q);

    auto &[qpts0, qwts0] = rule0;
    auto &[qpts1, qwts1] = rule1;
    auto &[qptsT, _] = ruleT;

    // Precompute geometry data on cpu at collapsed quadrature points, and
    // copy it on the gpu
    std::vector<T> detJ_geom = compute_geometry(mesh, qptsT);

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
    err_check(deviceMemcpyToSymbol((qpts0_d<T, Q>), qpts0.data(),
                                   qpts0.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((qpts1_d<T, Q>), qpts1.data(),
                                   qpts1.size() * sizeof(T)));
  }

  template <typename Vector> void operator()(const Vector &in, Vector &out) {
    out.set(T{0.0});
    
    const T *in_dofs = in.array().data();
    T *out_dofs = out.mutable_array().data();

    dim3 grid_size(this->number_of_local_cells);
    dim3 block_size(Q, Q);
    constexpr int N = P + 1;
    constexpr int K = (N + 1) * N / 2; // Number of dofs on triangle
    
    assert(dofmap_d_span.size() == this->number_of_local_cells * K);
    assert(in.array().size() == out.mutable_array().size());
    assert(detJ_geom_d_span.size() == this->number_of_local_cells * Q * Q);

    mass_operator<T, N, Q><<<grid_size, block_size>>>(
        in_dofs, out_dofs, this->detJ_geom_d_span.data(),
        this->dofmap_d_span.data());
    check_device_last_error();
  }

  template <typename Vector>
  void get_diag_inverse(Vector& diag_inv)
  {
    assert(false && "todo for jacobi preconditioning of cg");
  }

private:
  static constexpr int nq = Q * Q;
  std::size_t number_of_local_cells;

  std::shared_ptr<mesh::Mesh<T>> mesh;
  std::shared_ptr<fem::FunctionSpace<T>> V;

  thrust::device_vector<T> detJ_geom_d;
  std::span<const T> detJ_geom_d_span;

  thrust::device_vector<std::int32_t> dofmap_d;
  std::span<const std::int32_t> dofmap_d_span;
};

} // namespace dolfinx::acc