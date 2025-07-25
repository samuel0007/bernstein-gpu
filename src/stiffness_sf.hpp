#pragma once

#include "dof_ordering.hpp"
#include "geometry_gpu.hpp"
#include "kernels_stiffness.hpp"
#include "util.hpp"
#include "vector.hpp"

namespace dolfinx::acc {

/// @brief  Stiffness operator using bernstein tricks
/// @tparam T
/// @tparam U
/// @tparam P
/// @tparam Q
template <typename T, int P, int Q, typename U = T> class MatFreeStiffnessSF {
public:
  using value_type = T;
  using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

  MatFreeStiffnessSF(std::shared_ptr<mesh::Mesh<U>> mesh,
                     std::shared_ptr<fem::FunctionSpace<U>> V, U alpha)
      : mesh(mesh), V(V) {
    auto dofmap = V->dofmap();
    auto map = dofmap->index_map;
    int map_bs = dofmap->index_map_bs();
    la::Vector<T> alpha_vec(map, map_bs);
    alpha_vec.set(alpha);
    init(alpha_vec.array());
  }

  MatFreeStiffnessSF(std::shared_ptr<mesh::Mesh<U>> mesh,
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
    std::vector<int> dof_reordering = get_tp_ordering2D<P>(element_p);

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

    // Construct quadrature points table
    auto [rule0, rule1, ruleT] = create_quadrature_triangle_duffy<T>(Q);

    auto &[qpts0, qwts0] = rule0;
    auto &[qpts1, qwts1] = rule1;
    auto &[qptsT, _] = ruleT;

    // Create 1D elements
    std::array<std::shared_ptr<basix::FiniteElement<T>>, P>
        elems; // No default ctor
    std::array<std::vector<int>, P> reordering_N;
    for (int p = 0; p < P; ++p) {
      elems[p] =
          std::make_shared<basix::FiniteElement<T>>(basix::create_element<T>(
              basix::element::family::P, basix::cell::type::interval, p,
              basix::element::lagrange_variant::bernstein,
              basix::element::dpc_variant::unset, (p == 0)));
      reordering_N[p] = get_tp_ordering1D<T>(elems[p], p);
    }

    // As basix doesnt expose dof reodering for bernstein, we do it manually.
    auto [phi_1, shape_1] = elems[P - 1]->tabulate(0, qpts1, {qpts1.size(), 1});
    std::cout << std::format("phi_1 size = {}, Qx(N-1): {}x{}", phi_1.size(),
                             shape_1[1], shape_1[2])
              << std::endl;
    assert(shape_1[1] == Q && shape_1[2] == N - 1);
    phi_1 = permute_columns(phi_1, reordering_N[P - 1], Q);

    std::vector<T> phi_0_N((N - 1) * Q * (N - 1),
                           0.); // this could be more memory efficient (2x in
                                // 2D), but costly indexing?
    for (int p = 0; p < N - 1; ++p) {
      auto [phi_0_p, shape_0_p] =
          elems[p]->tabulate(0, qpts0, {qpts0.size(), 1});
      phi_0_p = permute_columns(phi_0_p, reordering_N[p], Q);
      assert(shape_0_p[1] == Q && shape_0_p[2] == p + 1);
      for (int q = 0; q < shape_0_p[1]; ++q) {
        for (int i = 0; i < shape_0_p[2]; ++i) {
          phi_0_N[i + q * (N - 1) + p * Q * (N - 1)] =
              phi_0_p[i + q * shape_0_p[2]];
        }
      }
    }
    std::cout << std::format("phi_0_N size = {}, (n-1)xqx(n-1): {}x{}x{}",
                             phi_0_N.size(), N - 1, Q, N - 1)
              << std::endl;

    // Copy only derivatives
    phi_1_d_span = copy_to_device(phi_1.begin(), phi_1.end(), phi_1_d, "phi_1");
    phi_0_N_d_span =
        copy_to_device(phi_0_N.begin(), phi_0_N.end(), phi_0_N_d, "phi_0_N");

    geom_d_span =
        compute_stiffness_geometry_triangle_GPU<T, U>(mesh, qptsT, geom_d);

    // Copy dofmap reordering to the gpu
    err_check(deviceMemcpyToSymbol(
        (kernels::stiffness::dof_reordering_d<P + 1>), dof_reordering.data(),
        dof_reordering.size() * sizeof(int)));
    // Copy quadrature weights as symbols to the gpu
    err_check(deviceMemcpyToSymbol((kernels::stiffness::qwts0_d<T, Q>),
                                   qwts0.data(), qwts0.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((kernels::stiffness::qwts1_d<T, Q>),
                                   qwts1.data(), qwts1.size() * sizeof(T)));
  }

  template <typename Vector>
  void operator()(Vector &in, Vector &out, T global_coefficient = 1.) {
    in.scatter_fwd();

    const T *in_dofs = in.array().data();
    T *out_dofs = out.mutable_array().data();
    assert(in.array().size() == out.mutable_array().size());
    // assert(dofmap_d_span.size() == this->number_of_local_cells * K);

    dim3 grid_size(this->number_of_local_cells);
    dim3 block_size(Q, Q);
    kernels::stiffness::stiffness_operator_sf<T, N, Q>
        <<<grid_size, block_size>>>(
            out_dofs, in_dofs, this->alpha_d_span.data(),
            this->geom_d_span.data(), this->dofmap_d_span.data(),
            this->phi_1_d_span.data(), this->phi_0_N_d_span.data(),
            global_coefficient);
    check_device_last_error();
  }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    assert(false && "todo for jacobi preconditioning of cg");
  }

private:
  static constexpr int N = P + 1;
  static constexpr int nd = (N + 1) * N / 2; // << nq
  static constexpr int nq = Q * Q;

  std::size_t number_of_local_cells;

  std::shared_ptr<mesh::Mesh<U>> mesh;
  std::shared_ptr<fem::FunctionSpace<U>> V;

  thrust::device_vector<T> alpha_d;
  std::span<const T> alpha_d_span;

  thrust::device_vector<T> geom_d;
  std::span<const T> geom_d_span;

  thrust::device_vector<std::int32_t> dofmap_d;
  std::span<const std::int32_t> dofmap_d_span;

  thrust::device_vector<T> phi_1_d;
  std::span<const T> phi_1_d_span;

  thrust::device_vector<T> phi_0_N_d;
  std::span<const T> phi_0_N_d_span;
};

template <typename T, int P, int Q, typename U = T> class MatFreeStiffnessSF3D {
public:
  using value_type = T;
  using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

  MatFreeStiffnessSF3D(std::shared_ptr<mesh::Mesh<U>> mesh,
                       std::shared_ptr<fem::FunctionSpace<U>> V, U alpha)
      : mesh(mesh), V(V) {
    auto dofmap = V->dofmap();
    auto map = dofmap->index_map;
    int map_bs = dofmap->index_map_bs();
    la::Vector<T> alpha_vec(map, map_bs);
    alpha_vec.set(alpha);
    init(alpha_vec.array());
  }

  MatFreeStiffnessSF3D(std::shared_ptr<mesh::Mesh<U>> mesh,
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

    dofmap_d_span = copy_to_device(
        dofmap->map().data_handle(),
        dofmap->map().data_handle() + dofmap->map().size(), dofmap_d, "dofmap");
    // assert(dofmap_d_span.size() == this->number_of_local_cells * nd);

    alpha_d_span = copy_to_device(alpha.begin(), alpha.end(), alpha_d, "alpha");

    // Construct quadrature points table
    auto [rule0, rule1, rule2, ruleT] =
        create_quadrature_tetrahedron_duffy<U>(Q);

    auto &[qpts0, qwts0] = rule0;
    auto &[qpts1, qwts1] = rule1;
    auto &[qpts2, qwts2] = rule2;
    auto &[qptsT, _] = ruleT;

    // Create 1D elements
    std::array<std::shared_ptr<basix::FiniteElement<U>>, P>
        elems; // No default ctor
    std::array<std::vector<int>, P> reordering_N;

    for (int p = 0; p < P; ++p) {
      elems[p] =
          std::make_shared<basix::FiniteElement<U>>(basix::create_element<U>(
              basix::element::family::P, basix::cell::type::interval, p,
              basix::element::lagrange_variant::bernstein,
              basix::element::dpc_variant::unset, (p == 0)));
      reordering_N[p] = get_tp_ordering1D<U>(elems[p], p);
    }

    auto [phi_2, shape_2] = elems[P - 1]->tabulate(0, qpts2, {qpts2.size(), 1});
    phi_2 = permute_columns(phi_2, reordering_N[P - 1], Q);

    std::cout << std::format("phi_2 size = {}, qxn: {}x{}", phi_2.size(),
                             shape_2[1], shape_2[2])
              << std::endl;
    assert(shape_2[1] == Q && shape_2[2] == N - 1);

    std::vector<U> phi_1_N(
        (N - 1) * Q * (N - 1),
        0.); // this could be more memory efficient (2x), but indexing?
    for (int p = 0; p < N - 1; ++p) {
      auto [phi_1_p, shape_1_p] =
          elems[p]->tabulate(0, qpts1, {qpts1.size(), 1});
      phi_1_p = permute_columns(phi_1_p, reordering_N[p], Q);

      assert(shape_1_p[1] == Q && shape_1_p[2] == p + 1);
      for (int q = 0; q < shape_1_p[1]; ++q) {
        for (int i = 0; i < shape_1_p[2]; ++i) {
          phi_1_N[i + q * (N - 1) + p * Q * (N - 1)] =
              phi_1_p[i + q * shape_1_p[2]];
        }
      }
    }
    std::cout << std::format("phi_1_N size = {}, nxqxn: {}x{}x{}",
                             phi_1_N.size(), N - 1, Q, N - 1)
              << std::endl;

    std::vector<U> phi_0_N(
        (N - 1) * Q * (N - 1),
        0.); // this could be more memory efficient (2x), but indexing?
    for (int p = 0; p < N - 1; ++p) {
      auto [phi_0_p, shape_0_p] =
          elems[p]->tabulate(0, qpts0, {qpts0.size(), 1});
      phi_0_p = permute_columns(phi_0_p, reordering_N[p], Q);
      assert(shape_0_p[1] == Q && shape_0_p[2] == p + 1);
      for (int q = 0; q < shape_0_p[1]; ++q) {
        for (int i = 0; i < shape_0_p[2]; ++i) {
          phi_0_N[i + q * (N - 1) + p * Q * (N - 1)] =
              phi_0_p[i + q * shape_0_p[2]];
        }
      }
    }
    std::cout << std::format("phi_0_N size = {}, nxqxn: {}x{}x{}",
                             phi_0_N.size(), N - 1, Q, N - 1)
              << std::endl;

    phi_2_d_span = copy_to_device(phi_2.begin(), phi_2.end(), phi_2_d, "phi_2");
    phi_1_N_d_span =
        copy_to_device(phi_1_N.begin(), phi_1_N.end(), phi_1_N_d, "phi_1_N");
    phi_0_N_d_span =
        copy_to_device(phi_0_N.begin(), phi_0_N.end(), phi_0_N_d, "phi_0_N");

    // Precompute geometry data on cpu at collapsed quadrature points, and
    // copy it on the gpu
    geom_d_span =
        compute_stiffness_geometry_tetrahedron_GPU<T, U>(mesh, qptsT, geom_d);

    // Copy dofmap reordering to the gpu
    err_check(deviceMemcpyToSymbol((kernels::stiffness::dof_reordering3d_d<N>),
                                   dof_reordering.data(),
                                   dof_reordering.size() * sizeof(int)));
    // Copy quadrature weights as symbols to the gpu
    err_check(deviceMemcpyToSymbol((kernels::stiffness::qwts0_d<T, Q>),
                                   qwts0.data(), qwts0.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((kernels::stiffness::qwts1_d<T, Q>),
                                   qwts1.data(), qwts1.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((kernels::stiffness::qwts2_d<T, Q>),
                                   qwts2.data(), qwts2.size() * sizeof(T)));

    auto [qpts, qwts] = basix::quadrature::make_quadrature<U>(
        basix::quadrature::type::Default, basix::cell::type::tetrahedron,
        basix::polyset::type::standard, 2 * Q - 2);

    // assert(nq == qpts.size() / tdim);

    auto [dphi, shape] = element_p->tabulate(qpts, {nq, tdim}, 1);

    // Copy only derivatives
    dphi_d_span = copy_to_device(dphi.begin() + dphi.size() / (tdim + 1),
                                 dphi.end(), dphi_d, "dphi");
  }

  template <typename Vector>
  void operator()(Vector &in, Vector &out, U global_coefficient = 1.) {
    in.scatter_fwd();

    const T *in_dofs = in.array().data();
    T *out_dofs = out.mutable_array().data();

    dim3 grid_size(this->number_of_local_cells);
    dim3 block_size(Q, Q, Q);

    assert(dofmap_d_span.size() == this->number_of_local_cells * K);
    assert(in.array().size() == out.mutable_array().size());

    kernels::stiffness::stiffness_operator3D_sf<T, N, Q>
        <<<grid_size, block_size>>>(
            out_dofs, in_dofs, this->alpha_d_span.data(),
            this->geom_d_span.data(), this->dofmap_d_span.data(),
            phi_2_d_span.data(), phi_1_N_d_span.data(), phi_0_N_d_span.data(),
            global_coefficient);
    check_device_last_error();
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
    dim3 block_size(K);
    kernels::stiffness::stiffness_operator3D_diagonal<T, K, nq>
        <<<grid_size, block_size>>>(
            out_dofs, this->alpha_d_span.data(), this->geom_d_span.data(),
            this->dofmap_d_span.data(), this->dphi_d_span.data(),
            global_coefficient);
    check_device_last_error();
  }

private:
  static constexpr int N = P + 1;
  static constexpr int K = N * (N + 1) * (N + 2) / 6; // Number of dofs on tet
  static constexpr int nq = 14 * (Q == 3) + 24 * (Q == 4) + 45 * (Q == 5) +
                            74 * (Q == 6) + 122 * (Q == 7) + 177 * (Q == 8) +
                            729 * (Q == 9) + 1000 * (Q == 10) +
                            1331 * (Q == 11);
  std::size_t number_of_local_cells;

  std::shared_ptr<mesh::Mesh<U>> mesh;
  std::shared_ptr<fem::FunctionSpace<U>> V;

  thrust::device_vector<T> alpha_d;
  std::span<const T> alpha_d_span;

  thrust::device_vector<T> geom_d;
  std::span<const T> geom_d_span;

  thrust::device_vector<T> phi_2_d;
  std::span<const T> phi_2_d_span;

  thrust::device_vector<T> phi_1_N_d;
  std::span<const T> phi_1_N_d_span;

  thrust::device_vector<T> phi_0_N_d;
  std::span<const T> phi_0_N_d_span;

  thrust::device_vector<std::int32_t> dofmap_d;
  std::span<const std::int32_t> dofmap_d_span;

  thrust::device_vector<T> dphi_d;
  std::span<const T> dphi_d_span;
};

} // namespace dolfinx::acc
