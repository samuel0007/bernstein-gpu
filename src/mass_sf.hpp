#pragma once

#include "dof_ordering.hpp"
#include "geometry.hpp"
#include "kernels_mass.hpp"
#include "quadrature.hpp"
#include "util.hpp"
#include "vector.hpp"
#include <memory>

namespace dolfinx::acc {



template <typename T, int P, int Q> class MatFreeMassSF {
public:
  using value_type = T;
  using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

  MatFreeMassSF(std::shared_ptr<mesh::Mesh<T>> mesh,
                std::shared_ptr<fem::FunctionSpace<T>> V, T alpha)
      : mesh(mesh), V(V) {
    auto dofmap = V->dofmap();
    auto map = dofmap->index_map;
    int map_bs = dofmap->index_map_bs();
    la::Vector<T> alpha_vec(map, map_bs);
    alpha_vec.set(alpha);
    init(alpha_vec.array());
  }

  MatFreeMassSF(std::shared_ptr<mesh::Mesh<T>> mesh,
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
    std::vector<int> dof_reordering = get_tp_ordering2D<P>(element_p);

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
    auto [rule0, rule1, ruleT] = create_quadrature_triangle_duffy<T>(Q);

    auto &[qpts0, qwts0] = rule0;
    auto &[qpts1, qwts1] = rule1;
    auto &[qptsT, _] = ruleT;

    // Create 1D elements
    std::array<std::shared_ptr<basix::FiniteElement<T>>, P + 1>
    elems; // No default ctor
    std::array<std::vector<int>, P + 1> reordering_N;
    for (int p = 0; p < P + 1; ++p) {
      elems[p] =
      std::make_shared<basix::FiniteElement<T>>(basix::create_element<T>(
        basix::element::family::P, basix::cell::type::interval, p,
        basix::element::lagrange_variant::bernstein,
        basix::element::dpc_variant::unset, (p == 0)));
        reordering_N[p] = get_tp_ordering1D<T>(elems[p], p);
      }
      
      
      // As basix doesnt expose dof reodering for bernstein, we do it manually.
    auto [phi_1, shape_1] = elems[P]->tabulate(0, qpts1, {qpts1.size(), 1});
    std::cout << std::format("phi_1 size = {}, qxn: {}x{}", phi_1.size(),
                             shape_1[1], shape_1[2])
              << std::endl;
    assert(shape_1[1] == Q && shape_1[2] == N);
    phi_1 = permute_columns(phi_1, reordering_N[P], Q);

    std::vector<T> phi_0_N(
        N * Q * N,
        0.); // this could be more memory efficient (2x in 2D), but costly indexing?
    for (int p = 0; p < N; ++p) {
      auto [phi_0_p, shape_0_p] =
          elems[p]->tabulate(0, qpts0, {qpts0.size(), 1});
      phi_0_p = permute_columns(phi_0_p, reordering_N[p], Q);
      assert(shape_0_p[1] == Q && shape_0_p[2] == p + 1);
      for (int q = 0; q < shape_0_p[1]; ++q) {
        for (int i = 0; i < shape_0_p[2]; ++i) {
          phi_0_N[i + q * N + p * Q * N] = phi_0_p[i + q * shape_0_p[2]];
        }
      }
    }
    std::cout << std::format("phi_0_N size = {}, nxqxn: {}x{}x{}",
                             phi_0_N.size(), N, Q, N)
              << std::endl;

    this->phi_1_d.resize(phi_1.size());
    thrust::copy(phi_1.begin(), phi_1.end(), phi_1_d.begin());
    this->phi_1_d_span = std::span<const T>(
        thrust::raw_pointer_cast(phi_1_d.data()), phi_1_d.size());

    this->phi_0_N_d.resize(phi_0_N.size());
    thrust::copy(phi_0_N.begin(), phi_0_N.end(), phi_0_N_d.begin());
    this->phi_0_N_d_span = std::span<const T>(
        thrust::raw_pointer_cast(phi_0_N_d.data()), phi_0_N_d.size());

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

    // Copy dofmap reordering to the gpu
    err_check(deviceMemcpyToSymbol((dof_reordering_d<P + 1>),
                                   dof_reordering.data(),
                                   dof_reordering.size() * sizeof(int)));
    // Copy quadrature weights as symbols to the gpu
    err_check(deviceMemcpyToSymbol((qwts0_d<T, Q>), qwts0.data(),
                                   qwts0.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((qwts1_d<T, Q>), qwts1.data(),
                                   qwts1.size() * sizeof(T)));
  }

  template <typename Vector> void operator()(Vector &in, Vector &out) {
    in.scatter_fwd();

    const T *in_dofs = in.array().data();
    T *out_dofs = out.mutable_array().data();

    dim3 grid_size(this->number_of_local_cells);
    dim3 block_size(Q, Q);

    assert(dofmap_d_span.size() == this->number_of_local_cells * K);
    assert(in.array().size() == out.mutable_array().size());
    assert(detJ_geom_d_span.size() == this->number_of_local_cells * Q * Q);

    mass_operator_sf<T, N, Q><<<grid_size, block_size>>>(
        in_dofs, out_dofs, this->alpha_d_span.data(),
        this->detJ_geom_d_span.data(), this->dofmap_d_span.data(),
        phi_1_d_span.data(), phi_0_N_d_span.data());
    check_device_last_error();
  }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    assert(false && "todo for jacobi preconditioning of cg");
  }

private:
  static constexpr int N = P + 1;
  static constexpr int K = (N + 1) * N / 2; // Number of dofs on triangle
  std::size_t number_of_local_cells;

  std::shared_ptr<mesh::Mesh<T>> mesh;
  std::shared_ptr<fem::FunctionSpace<T>> V;

  thrust::device_vector<T> alpha_d;
  std::span<const T> alpha_d_span;

  thrust::device_vector<T> detJ_geom_d;
  std::span<const T> detJ_geom_d_span;

  thrust::device_vector<T> phi_1_d;
  std::span<const T> phi_1_d_span;

  thrust::device_vector<T> phi_0_N_d;
  std::span<const T> phi_0_N_d_span;

  thrust::device_vector<std::int32_t> dofmap_d;
  std::span<const std::int32_t> dofmap_d_span;
};

template <typename T, int P, int Q> class MatFreeMassSF3D {
public:
  using value_type = T;
  using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

  MatFreeMassSF3D(std::shared_ptr<mesh::Mesh<T>> mesh,
                  std::shared_ptr<fem::FunctionSpace<T>> V, T alpha)
      : mesh(mesh), V(V) {
    auto dofmap = V->dofmap();
    auto map = dofmap->index_map;
    int map_bs = dofmap->index_map_bs();
    la::Vector<T> alpha_vec(map, map_bs);
    alpha_vec.set(alpha);
    init(alpha_vec.array());
  }

  MatFreeMassSF3D(std::shared_ptr<mesh::Mesh<T>> mesh,
                  std::shared_ptr<fem::FunctionSpace<T>> V,
                  std::span<const T> alpha)
      : mesh(mesh), V(V) {
    init(alpha);
  }

  void init(std::span<const T> alpha) {
    // auto [lcells, bcells] = compute_boundary_cells(V);
    // spdlog::debug("#lcells = {}, #bcells = {}", lcells.size(),
    // bcells.size());

    // auto element_p = this->V->element();
    // std::vector<int> dof_reordering = get_tp_ordering2D<P>(element_p);
    // std::vector<int> dof_reordering = {0, 4, 1, 3, 2, 6, 5, 9, 8, 7};
    // std::vector<int> dof_reordering = {0, 1, 2, 3};
    std::vector<int> dof_reordering = lex_dof_ordering(
        basix::element::family::P, basix::cell::type::tetrahedron, P);
    // std::vector<int> dof_reordering = {0, 1, 2, 3};
    // std::vector<int> dof_reordering = {0, }
    // std::reverse(dof_reordering.begin(), dof_reordering.end());
    // for(int i = 0; i < dof_reordering.size(); ++i) {
    // dof_reordering[i] = i;
    // std::cout << "i=" << dof_reordering[i] << "\n";
    // }
    assert(dof_reordering.size() == K);

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

    // Create 1D elements
    std::array<std::shared_ptr<basix::FiniteElement<T>>, P + 1>
        elems; // No default ctor
    std::array<std::vector<int>, P + 1> reordering_N;

    for (int p = 0; p < P + 1; ++p) {
      elems[p] =
          std::make_shared<basix::FiniteElement<T>>(basix::create_element<T>(
              basix::element::family::P, basix::cell::type::interval, p,
              basix::element::lagrange_variant::bernstein,
              basix::element::dpc_variant::unset, (p == 0)));
      reordering_N[p] = get_tp_ordering1D<T>(elems[p], p);
    }

    auto [phi_2, shape_2] = elems[P]->tabulate(0, qpts2, {qpts2.size(), 1});
    phi_2 = permute_columns(phi_2, reordering_N[P], Q);

    std::cout << std::format("phi_2 size = {}, qxn: {}x{}", phi_2.size(),
                             shape_2[1], shape_2[2])
              << std::endl;
    assert(shape_2[1] == Q && shape_2[2] == N);

    std::vector<T> phi_1_N(
        N * Q * N,
        0.); // this could be more memory efficient (2x), but indexing?
    for (int p = 0; p < N; ++p) {
      auto [phi_1_p, shape_1_p] =
          elems[p]->tabulate(0, qpts1, {qpts1.size(), 1});
      phi_1_p = permute_columns(phi_1_p, reordering_N[p], Q);

      assert(shape_1_p[1] == Q && shape_1_p[2] == p + 1);
      for (int q = 0; q < shape_1_p[1]; ++q) {
        for (int i = 0; i < shape_1_p[2]; ++i) {
          phi_1_N[i + q * N + p * Q * N] = phi_1_p[i + q * shape_1_p[2]];
        }
      }
    }
    std::cout << std::format("phi_1_N size = {}, nxqxn: {}x{}x{}",
                             phi_1_N.size(), N, Q, N)
              << std::endl;

    std::vector<T> phi_0_N(
        N * Q * N,
        0.); // this could be more memory efficient (2x), but indexing?
    for (int p = 0; p < N; ++p) {
      auto [phi_0_p, shape_0_p] =
          elems[p]->tabulate(0, qpts0, {qpts0.size(), 1});
      phi_0_p = permute_columns(phi_0_p, reordering_N[p], Q);
      assert(shape_0_p[1] == Q && shape_0_p[2] == p + 1);
      for (int q = 0; q < shape_0_p[1]; ++q) {
        for (int i = 0; i < shape_0_p[2]; ++i) {
          phi_0_N[i + q * N + p * Q * N] = phi_0_p[i + q * shape_0_p[2]];
        }
      }
    }
    std::cout << std::format("phi_0_N size = {}, nxqxn: {}x{}x{}",
                             phi_0_N.size(), N, Q, N)
              << std::endl;

    this->phi_2_d.resize(phi_2.size());
    thrust::copy(phi_2.begin(), phi_2.end(), phi_2_d.begin());
    this->phi_2_d_span = std::span<const T>(
        thrust::raw_pointer_cast(phi_2_d.data()), phi_2_d.size());

    this->phi_1_N_d.resize(phi_1_N.size());
    thrust::copy(phi_1_N.begin(), phi_1_N.end(), phi_1_N_d.begin());
    this->phi_1_N_d_span = std::span<const T>(
        thrust::raw_pointer_cast(phi_1_N_d.data()), phi_1_N_d.size());

    this->phi_0_N_d.resize(phi_0_N.size());
    thrust::copy(phi_0_N.begin(), phi_0_N.end(), phi_0_N_d.begin());
    this->phi_0_N_d_span = std::span<const T>(
        thrust::raw_pointer_cast(phi_0_N_d.data()), phi_0_N_d.size());

    // Precompute geometry data on cpu at collapsed quadrature points, and
    // copy it on the gpu
    std::vector<T> detJ_geom = compute_geometry(mesh, qptsT, 3);

    this->detJ_geom_d.resize(detJ_geom.size());
    thrust::copy(detJ_geom.begin(), detJ_geom.end(), detJ_geom_d.begin());
    this->detJ_geom_d_span = std::span<const T>(
        thrust::raw_pointer_cast(detJ_geom_d.data()), detJ_geom_d.size());

    std::cout << std::format("Send geometry to GPU (size = {} bytes)",
                             detJ_geom_d.size() * sizeof(T))
              << std::endl;

    // Copy dofmap reordering to the gpu
    err_check(deviceMemcpyToSymbol((dof_reordering3d_d<N>),
                                   dof_reordering.data(),
                                   dof_reordering.size() * sizeof(int)));
    // Copy quadrature weights as symbols to the gpu
    err_check(deviceMemcpyToSymbol((qwts0_d<T, Q>), qwts0.data(),
                                   qwts0.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((qwts1_d<T, Q>), qwts1.data(),
                                   qwts1.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((qwts2_d<T, Q>), qwts2.data(),
                                   qwts2.size() * sizeof(T)));
  }

  template <typename Vector> void operator()(Vector &in, Vector &out) {
    in.scatter_fwd();

    const T *in_dofs = in.array().data();
    T *out_dofs = out.mutable_array().data();

    dim3 grid_size(this->number_of_local_cells);
    dim3 block_size(Q, Q, Q);

    assert(dofmap_d_span.size() == this->number_of_local_cells * K);
    assert(in.array().size() == out.mutable_array().size());
    assert(detJ_geom_d_span.size() == this->number_of_local_cells * Q * Q * Q);

    mass_operator3D_sf<T, N, Q><<<grid_size, block_size>>>(
        in_dofs, out_dofs, this->alpha_d_span.data(),
        this->detJ_geom_d_span.data(), this->dofmap_d_span.data(),
        phi_2_d_span.data(), phi_1_N_d_span.data(), phi_0_N_d_span.data());
    check_device_last_error();
  }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    assert(false && "todo for jacobi preconditioning of cg");
  }

private:
  static constexpr int N = P + 1;
  static constexpr int K = N * (N + 1) * (N + 2) / 6; // Number of dofs on tet
  std::size_t number_of_local_cells;

  std::shared_ptr<mesh::Mesh<T>> mesh;
  std::shared_ptr<fem::FunctionSpace<T>> V;

  thrust::device_vector<T> alpha_d;
  std::span<const T> alpha_d_span;

  thrust::device_vector<T> detJ_geom_d;
  std::span<const T> detJ_geom_d_span;

  thrust::device_vector<T> phi_2_d;
  std::span<const T> phi_2_d_span;

  thrust::device_vector<T> phi_1_N_d;
  std::span<const T> phi_1_N_d_span;

  thrust::device_vector<T> phi_0_N_d;
  std::span<const T> phi_0_N_d_span;

  thrust::device_vector<std::int32_t> dofmap_d;
  std::span<const std::int32_t> dofmap_d_span;
};

} // namespace dolfinx::acc