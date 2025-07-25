#pragma once

#include "geometry.hpp"
#include "geometry_gpu.hpp"
#include "kernels_fused.hpp"
#include "profiler.hpp"
#include "quadrature.hpp"
#include "util.hpp"
#include "vector.hpp"
#include <memory>

namespace dolfinx::acc {

template <typename T, int P, int Q, typename U = T> class MatFreeNewmark3D {
public:
  using value_type = T;
  using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

  MatFreeNewmark3D(std::shared_ptr<mesh::Mesh<U>> mesh,
                   std::shared_ptr<fem::FunctionSpace<U>> V, U alpha)
      : mesh(mesh), V(V) {
    auto dofmap = V->dofmap();
    auto map = dofmap->index_map;
    int map_bs = dofmap->index_map_bs();
    la::Vector<T> alpha_vec(map, map_bs);
    alpha_vec.set(alpha);
    init(alpha_vec.array());
  }

  MatFreeNewmark3D(std::shared_ptr<mesh::Mesh<U>> mesh,
                   std::shared_ptr<fem::FunctionSpace<U>> V,
                   std::span<const U> alpha)
      : mesh(mesh), V(V) {
    init(alpha);
  }

  void init(std::span<const U> alpha) {
    // auto [lcells, bcells] = compute_boundary_cells(V);
    // spdlog::debug("#lcells = {}, #bcells = {}", lcells.size(),
    // bcells.size());

    auto element_p = this->V->element();

    const std::size_t tdim = mesh->topology()->dim();
    const std::size_t gdim = mesh->geometry().dim();
    this->number_of_local_cells =
        mesh->topology()->index_map(tdim)->size_local() +
        mesh->topology()->index_map(tdim)->num_ghosts();

    auto dofmap = V->dofmap();
    dofmap_d_span = copy_to_device(
        dofmap->map().data_handle(),
        dofmap->map().data_handle() + dofmap->map().size(), dofmap_d, "dofmap");
    assert(dofmap_d_span.size() == this->number_of_local_cells * nd);

    alpha_d_span = copy_to_device(alpha.begin(), alpha.end(), alpha_d, "alpha");

    auto [qpts, qwts] = basix::quadrature::make_quadrature<U>(
        basix::quadrature::type::Default, basix::cell::type::tetrahedron,
        basix::polyset::type::standard, 2 * Q - 2);

    assert(nq == qpts.size() / tdim);

    auto [dphi, shape] = element_p->tabulate(qpts, {nq, tdim}, 1);

    dphi_d_span = copy_to_device(dphi.begin(), dphi.end(), dphi_d, "dphi");
    std::cout << std::format("Table size = {}, dxnqxnd: {}x{}x{}", dphi.size(),
                             shape[0], shape[1], shape[2])
              << std::endl;
    assert(shape[0] == tdim && shape[3] == 1);
    assert(nq == shape[1] && nd == shape[2]);

    geom_d_span = compute_stiffness_geometry_tetrahedron_GPU<T, U, nq>(
        mesh, qpts, qwts, geom_d);
    assert(geom_d_span.size() == this->number_of_local_cells * 6 * nq);
    assert(geom_d.size() == this->number_of_local_cells * 6 * nq);

    std::cout << "Precomputing geometry..." << std::endl;
    std::vector<U> detJ_geom = compute_geometry(mesh, qpts, qwts);

    this->detJ_geom_d_span = copy_to_device(detJ_geom.begin(), detJ_geom.end(),
                                            this->detJ_geom_d, "detJ_geom");
  }

  template <typename Vector>
  void operator()(Vector &in, Vector &out, U global_coefficient = 1.,
                  GpuStream stream = 0, int bs = nq, int cells_per_block = 1) {
    in.scatter_fwd();
    T dt = 0.1;

    const T *in_dofs = in.array().data();
    T *out_dofs = out.mutable_array().data();
    assert(in.array().size() == out.mutable_array().size());

    assert(nq >= nd);
    // dim3 grid_size(this->number_of_local_cells);
    // dim3 block_size(bs);
    // kernels::fused::fused_newmark<T, nd, nq>
    //     <<<grid_size, block_size, 0, stream>>>(
    //         out_dofs, in_dofs, this->alpha_d_span.data(),
    //         this->alpha_d_span.data(), this->alpha_d_span.data(),
    //         this->detJ_geom_d_span.data(), this->geom_d_span.data(),
    //         this->dofmap_d_span.data(), this->dphi_d_span.data(), dt,
    //         global_coefficient);

    dim3 block_size(bs, cells_per_block,
                    1); // e.g. blockDimX=64, cellsPerBlock=4
    dim3 grid_size((this->number_of_local_cells + block_size.y - 1) /
                   block_size.y);

    size_t shmem = sizeof(T) * (nd + 4 * nq) *
                   block_size.y; // per-cell shared * cellsPerBlock

    kernels::fused::fused_newmark<T, nd, nq>
        <<<grid_size, block_size, shmem, stream>>>(
            out_dofs, in_dofs, this->alpha_d_span.data(),
            this->alpha_d_span.data(), this->alpha_d_span.data(),
            this->detJ_geom_d_span.data(), this->geom_d_span.data(),
            this->dofmap_d_span.data(), this->dphi_d_span.data(), dt,
            global_coefficient, this->number_of_local_cells);

    // check_device_last_error();
  }

  template <typename Vector>
  void get_diag_inverse(Vector &diag_inv, U global_coefficient = 1.) {
    diag_inv.set(0.);
    this->get_diag(diag_inv, global_coefficient);
    thrust::transform(thrust::device, diag_inv.mutable_array().begin(),
                      diag_inv.mutable_array().begin() +
                          diag_inv.map()->size_local(),
                      diag_inv.mutable_array().begin(),
                      [] __host__ __device__(T yi) { return 1.0 / yi; });
  }

  template <typename Vector>
  void get_diag(Vector &diag, U global_coefficient = 1.) {
    T *out_dofs = diag.mutable_array().data();

    dim3 grid_size(this->number_of_local_cells);
    dim3 block_size(nd);
    kernels::stiffness::stiffness_operator3D_diagonal<T, nd, nq>
        <<<grid_size, block_size>>>(
            out_dofs, this->alpha_d_span.data(), this->geom_d_span.data(),
            this->dofmap_d_span.data(), this->dphi_d_span.data(),
            global_coefficient);
    check_device_last_error();
  }

private:
  static constexpr int N = P + 1;
  static constexpr int nd = N * (N + 1) * (N + 2) / 6;
  static constexpr int nq = 14 * (Q == 3) + 24 * (Q == 4) + 45 * (Q == 5) +
                            74 * (Q == 6) + 122 * (Q == 7) + 177 * (Q == 8) +
                            729 * (Q == 9) + 1000 * (Q == 10) +
                            1331 * (Q == 11);

  std::size_t number_of_local_cells;

  std::shared_ptr<mesh::Mesh<U>> mesh;
  std::shared_ptr<fem::FunctionSpace<U>> V;

  thrust::device_vector<T> dphi_d;
  std::span<const T> dphi_d_span;

  thrust::device_vector<T> alpha_d;
  std::span<const T> alpha_d_span;

  thrust::device_vector<T> geom_d;
  std::span<const T> geom_d_span;

  thrust::device_vector<T> detJ_geom_d;
  std::span<const T> detJ_geom_d_span;

  thrust::device_vector<T> buffer;

  thrust::device_vector<std::int32_t> dofmap_d;
  std::span<const std::int32_t> dofmap_d_span;
};

} // namespace dolfinx::acc
