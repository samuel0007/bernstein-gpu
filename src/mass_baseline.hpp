#pragma once

#include "dof_ordering.hpp"
#include "geometry.hpp"
#include "kernels_mass.hpp"
#include "quadrature.hpp"
#include "util.hpp"
#include "vector.hpp"
#include <memory>

namespace dolfinx::acc {

template <typename T, int P, int Q, typename U = T> class MatFreeMassBaseline {
public:
  using value_type = T;
  using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

  MatFreeMassBaseline(std::shared_ptr<mesh::Mesh<U>> mesh,
                      std::shared_ptr<fem::FunctionSpace<U>> V, U alpha)
      : mesh(mesh), V(V) {
    auto dofmap = V->dofmap();
    auto map = dofmap->index_map;
    int map_bs = dofmap->index_map_bs();
    la::Vector<U> alpha_vec(map, map_bs);
    alpha_vec.set(alpha);
    init(alpha_vec.array());
  }

  MatFreeMassBaseline(std::shared_ptr<mesh::Mesh<U>> mesh,
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
    auto [qpts, qwts] = basix::quadrature::make_quadrature<U>(
        basix::quadrature::type::Default, basix::cell::type::triangle,
        basix::polyset::type::standard, 2 * Q - 2);

    auto [phi_table, shape] =
        element_p->tabulate(qpts, {qpts.size() / 2, 2}, 0);

    std::cout << std::format("Table size = {}, qxn: {}x{}", phi_table.size(),
                             shape[1], shape[2])
              << std::endl;
    assert(shape[0] == 1 && shape[3] == 1);

    assert(nq == shape[1]);

    this->phi_d_span =
        copy_to_device(phi_table.begin(), phi_table.end(), this->phi_d, "phi");

    std::vector<U> detJ_geom = compute_geometry(mesh, qpts, qwts);

    this->detJ_geom_d_span = copy_to_device(detJ_geom.begin(), detJ_geom.end(),
                                            this->detJ_geom_d, "detJ_geom");
  }

  template <typename Vector> void operator()(Vector &in, Vector &out) {
    in.scatter_fwd();

    const T *in_dofs = in.array().data();
    T *out_dofs = out.mutable_array().data();

    assert(dofmap_d_span.size() == this->number_of_local_cells * nd);
    assert(in.array().size() == out.mutable_array().size());
    assert(detJ_geom_d_span.size() == this->number_of_local_cells * nq);

    dim3 grid_size(this->number_of_local_cells);
    dim3 block_size(nq);
    mass_operator_baseline<T, nd, nq><<<grid_size, block_size>>>(
        in_dofs, out_dofs, this->alpha_d_span.data(),
        this->detJ_geom_d_span.data(), this->dofmap_d_span.data(),
        this->phi_d_span.data());
    check_device_last_error();
  }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    diag_inv.set(0.);
    T *out_dofs = diag_inv.mutable_array().data();

    dim3 grid_size(this->number_of_local_cells);
    dim3 block_size(nd);
    mass_diagonal<T, nd, nq><<<grid_size, block_size>>>(
        out_dofs, this->alpha_d_span.data(), this->detJ_geom_d_span.data(),
        this->dofmap_d_span.data(), this->phi_d_span.data());

    thrust::transform(thrust::device, diag_inv.array().begin(),
                      diag_inv.array().begin() + diag_inv.map()->size_local(),
                      diag_inv.mutable_array().begin(),
                      [] __host__ __device__(T yi) { return 1.0 / yi; });
    check_device_last_error();
  }

private:
  static constexpr int N = P + 1;
  static constexpr int nd = (N + 1) * N / 2; // << nq
  static constexpr int nq = 6 * (Q == 3) + 12 * (Q == 4) + 16 * (Q == 5) +
                            25 * (Q == 6) + 33 * (Q == 7) + 42 * (Q == 8) +
                            55 * (Q == 9) + 67 * (Q == 10) + 79 * (Q == 11) +
                            96 * (Q == 12) + 112 * (Q == 13);

  std::size_t number_of_local_cells;

  std::shared_ptr<mesh::Mesh<U>> mesh;
  std::shared_ptr<fem::FunctionSpace<U>> V;

  thrust::device_vector<T> phi_d;
  std::span<const T> phi_d_span;

  thrust::device_vector<T> alpha_d;
  std::span<const T> alpha_d_span;

  thrust::device_vector<T> detJ_geom_d;
  std::span<const T> detJ_geom_d_span;

  thrust::device_vector<std::int32_t> dofmap_d;
  std::span<const std::int32_t> dofmap_d_span;
};

template <typename T, int P, int Q, typename U = T>
class MatFreeMassBaseline3D {
public:
  using value_type = T;
  using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

  MatFreeMassBaseline3D(std::shared_ptr<mesh::Mesh<U>> mesh,
                        std::shared_ptr<fem::FunctionSpace<U>> V, U alpha)
      : mesh(mesh), V(V) {
    auto dofmap = V->dofmap();
    auto map = dofmap->index_map;
    int map_bs = dofmap->index_map_bs();
    la::Vector<T> alpha_vec(map, map_bs);
    alpha_vec.set(alpha);
    init(alpha_vec.array());
  }

  MatFreeMassBaseline3D(std::shared_ptr<mesh::Mesh<U>> mesh,
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
    // Transfer V dofmap to the GPU
    auto dofmap = V->dofmap();

    this->dofmap_d_span =
        copy_to_device(dofmap->map().data_handle(),
                       dofmap->map().data_handle() + dofmap->map().size(),
                       this->dofmap_d, "dofmap");
    this->alpha_d_span =
        copy_to_device(alpha.begin(), alpha.end(), this->alpha_d, "alpha");

    // Construct quadrature points table
    // auto [qpts, qwts] = basix::quadrature::make_quadrature<T>(
    //     basix::quadrature::type::gauss_jacobi,
    //     basix::cell::type::tetrahedron, basix::polyset::type::standard, 2 * Q
    //     - 2);
    auto [qpts, qwts] = basix::quadrature::make_quadrature<U>(
        basix::quadrature::type::Default, basix::cell::type::tetrahedron,
        basix::polyset::type::standard, 2 * Q - 2);

    auto [phi_table, shape] =
        element_p->tabulate(qpts, {qpts.size() / 3, 3}, 0);

    std::cout << std::format("Table size = {}, qxn: {}x{}", phi_table.size(),
                             shape[1], shape[2])
              << std::endl;
    assert(shape[0] == 1 && shape[3] == 1);
    assert(nq == shape[1]);

    this->phi_d_span =
        copy_to_device(phi_table.begin(), phi_table.end(), this->phi_d, "phi");
    std::cout << "Precomputing geometry..." << std::endl;
    std::vector<U> detJ_geom = compute_geometry(mesh, qpts, qwts);

    this->detJ_geom_d_span = copy_to_device(detJ_geom.begin(), detJ_geom.end(),
                                            this->detJ_geom_d, "detJ_geom");
  }

  template <typename Vector> void operator()(Vector &in, Vector &out) {
    in.scatter_fwd();

    const T *in_dofs = in.array().data();
    T *out_dofs = out.mutable_array().data();

    assert(dofmap_d_span.size() == this->number_of_local_cells * nd);
    assert(in.array().size() == out.mutable_array().size());
    assert(detJ_geom_d_span.size() == this->number_of_local_cells * nq);

    dim3 grid_size(this->number_of_local_cells);
    dim3 block_size(nq);
    mass_operator_baseline<T, nd, nq><<<grid_size, block_size>>>(
        in_dofs, out_dofs, this->alpha_d_span.data(),
        this->detJ_geom_d_span.data(), this->dofmap_d_span.data(),
        this->phi_d_span.data());
    check_device_last_error();
  }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    diag_inv.set(0.);
    this->get_diag(diag_inv);
    thrust::transform(thrust::device, diag_inv.array().begin(),
                  diag_inv.array().begin() + diag_inv.map()->size_local(),
                  diag_inv.mutable_array().begin(),
                  [] __host__ __device__(T yi) { return 1.0 / yi; });
  }

   template <typename Vector> void get_diag(Vector &diag) {
    T *out_dofs = diag.mutable_array().data();

    dim3 grid_size(this->number_of_local_cells);
    dim3 block_size(nd);
    mass_diagonal<T, nd, nq><<<grid_size, block_size>>>(
        out_dofs, this->alpha_d_span.data(), this->detJ_geom_d_span.data(),
        this->dofmap_d_span.data(), this->phi_d_span.data());
    check_device_last_error();
  }

private:
  static constexpr int N = P + 1;
  static constexpr int nd =
      N * (N + 1) * (N + 2) /
      6; // Number of dofs on tets (is always smaller than nq)
  static constexpr int nq = 14 * (Q == 3) + 24 * (Q == 4) + 45 * (Q == 5) +
                            74 * (Q == 6) + 122 * (Q == 7) + 177 * (Q == 8) +
                            729 * (Q == 9) + 1000 * (Q == 10) +
                            1331 * (Q == 11);
  std::size_t number_of_local_cells;

  std::shared_ptr<mesh::Mesh<U>> mesh;
  std::shared_ptr<fem::FunctionSpace<U>> V;

  thrust::device_vector<T> phi_d;
  std::span<const T> phi_d_span;

  thrust::device_vector<T> alpha_d;
  std::span<const T> alpha_d_span;

  thrust::device_vector<T> detJ_geom_d;
  std::span<const T> detJ_geom_d_span;

  thrust::device_vector<std::int32_t> dofmap_d;
  std::span<const std::int32_t> dofmap_d_span;
};

template <typename T, int P, int Q, typename U = T>
class MatFreeMassExteriorBaseline {
public:
  using value_type = T;
  using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

  MatFreeMassExteriorBaseline(std::shared_ptr<mesh::Mesh<U>> mesh,
                              std::shared_ptr<fem::FunctionSpace<U>> V,
                              std::span<const std::int32_t> cell_facet_data,
                              U alpha)
      : mesh(mesh), V(V) {
    auto dofmap = V->dofmap();
    auto map = dofmap->index_map;
    int map_bs = dofmap->index_map_bs();
    la::Vector<T> alpha_vec(map, map_bs);
    alpha_vec.set(alpha);
    init(cell_facet_data, alpha_vec.array());
  }

  MatFreeMassExteriorBaseline(std::shared_ptr<mesh::Mesh<U>> mesh,
                              std::shared_ptr<fem::FunctionSpace<U>> V,
                              std::span<const std::int32_t> cell_facet_data,
                              std::span<const U> alpha)
      : mesh(mesh), V(V) {
    init(cell_facet_data, alpha);
  }

  void init(std::span<const std::int32_t> cell_facet_data,
            std::span<const U> alpha) {
    // auto [lcells, bcells] = compute_boundary_cells(V);
    // spdlog::debug("#lcells = {}, #bcells = {}", lcells.size(),
    // bcells.size());
    auto element_p = this->V->element();

    const std::size_t tdim = mesh->topology()->dim();
    this->number_of_local_cells =
        mesh->topology()->index_map(tdim)->size_local() +
        mesh->topology()->index_map(tdim)->num_ghosts();
    this->number_of_local_facets = cell_facet_data.size() / 2;

    // TODO: one could be smart, and pack the necessary exterior data instead of
    // sending everything. Transfer V dofmap to the GPU Or just use the same
    // pointer as for the normal mass matrix? Like passing the dofmap as
    // argument
    auto dofmap = V->dofmap();

    dofmap_d_span = copy_to_device(
        dofmap->map().data_handle(),
        dofmap->map().data_handle() + dofmap->map().size(), dofmap_d, "dofmap");

    alpha_d_span = copy_to_device(alpha.begin(), alpha.end(), alpha_d, "alpha");

    cell_facet_d_span =
        copy_to_device(cell_facet_data.begin(), cell_facet_data.end(),
                       cell_facet_d, "cell facet data");

    // Construct quadrature points table on interval
    auto [qpts, qwts] = basix::quadrature::make_quadrature<U>(
        basix::quadrature::type::Default, basix::cell::type::interval,
        basix::polyset::type::standard, 2 * Q - 2);

    assert(qpts.size() == Q);
    // For each facet, we map it to the corresponding reference 2d point
    mesh::CellType cell_type = mesh->topology()->cell_type();
    this->n_faces = mesh::cell_num_entities(cell_type, tdim - 1);

    std::vector<U> facets_phi_table(n_faces * Q * K);
    for (int i = 0; i < n_faces; ++i) {
      // 1. Get mapped points on physical reference facet: tdim - 1 -> tdim
      std::vector<U> mapped_points = map_facet_points(qpts, i, cell_type);
      assert(mapped_points.size() / tdim == qpts.size() / (tdim - 1));

      // 2. Evaluate basis functions at mapped points
      auto [phi_table, shape] = element_p->tabulate(
          mapped_points, {mapped_points.size() / tdim, tdim}, 0);
      assert(shape[0] == 1 && shape[3] == 1);
      assert(shape[1] == Q && shape[2] == K);
      for (int q = 0; q < Q; ++q) {
        for (int k = 0; k < K; ++k) {
          facets_phi_table[i * Q * K + q * K + k] = phi_table[q * K + k];
          // std::cout << std::format("facets_phi[{}][{}][{}]={}", i, q, k,
          // phi_table[q * K + k]) << std::endl;
        }
      }
    }
    std::cout << std::format("Table size = {}, Fxqxk: {}x{}x{}",
                             facets_phi_table.size(), n_faces, Q, K)
              << std::endl;
    facets_phi_d_span =
        copy_to_device(facets_phi_table.begin(), facets_phi_table.end(),
                       facets_phi_d, "facets phi");

    // Precompute geometry data
    std::vector<U> detJ_geom =
        compute_geometry_facets(mesh, cell_facet_data, qpts, qwts);
    assert(detJ_geom.size() == this->number_of_local_cells * n_faces * nq);

    detJ_geom_d_span = copy_to_device(detJ_geom.begin(), detJ_geom.end(),
                                      detJ_geom_d, "detJ geometry");

    // Element Dof Layout, map from local facet idx -> local dof idx
    const fem::ElementDofLayout &dof_layout = dofmap->element_dof_layout();
    int n_face_dofs = dof_layout.num_entity_closure_dofs(tdim - 1);
    auto faces_dofs = make_faces_to_dofs_map(dof_layout, mesh, tdim);
    assert(faces_dofs.size() == n_face_dofs * n_faces);

    faces_dofs_d_span =
        copy_to_device(faces_dofs.begin(), faces_dofs.end(), faces_dofs_d,
                       "local face index to local closure dofs");
  }

  template <typename Vector>
  void operator()(Vector &in, Vector &out, U global_coefficient = 1.) {
    in.scatter_fwd();

    const T *in_dofs = in.array().data();
    T *out_dofs = out.mutable_array().data();

    assert(in.array().size() == out.mutable_array().size());

    if (this->number_of_local_facets != 0) {
      dim3 grid_size(this->number_of_local_facets);
      dim3 block_size(nd);
      facets_mass_operator_baseline<T, nd, nq><<<grid_size, block_size>>>(
          in_dofs, out_dofs, this->cell_facet_d_span.data(),
          this->detJ_geom_d_span.data(), this->alpha_d_span.data(),
          this->dofmap_d_span.data(), this->facets_phi_d_span.data(),
          this->faces_dofs_d_span.data(), this->n_faces, global_coefficient);
      check_device_last_error();
    }
  }

  template <typename Vector>
  void get_diag_inverse(Vector &diag_inv, U global_coefficient = 1.) {
    T *out_dofs = diag_inv.mutable_array().data();

    if (this->number_of_local_facets != 0) {
      dim3 grid_size(this->number_of_local_facets);
      dim3 block_size(nd);
      mass_exterior_diagonal<T, nd, nq><<<grid_size, block_size>>>(
          out_dofs, this->cell_facet_d_span.data(),
          this->detJ_geom_d_span.data(), this->alpha_d_span.data(),
          this->dofmap_d_span.data(), this->facets_phi_d_span.data(),
          this->faces_dofs_d_span.data(), this->n_faces, global_coefficient);
      thrust::transform(thrust::device, diag_inv.array().begin(),
                        diag_inv.array().begin() + diag_inv.map()->size_local(),
                        diag_inv.mutable_array().begin(),
                        [] __host__ __device__(T yi) { return 1.0 / yi; });
    }
  }

private:
  static constexpr int N = P + 1;
  static constexpr int K = (N + 1) * N / 2;
  static constexpr int nd = K;
  static constexpr int nq = Q;

  std::size_t number_of_local_facets;
  std::size_t number_of_local_cells;

  std::shared_ptr<mesh::Mesh<U>> mesh;
  std::shared_ptr<fem::FunctionSpace<U>> V;

  thrust::device_vector<T> facets_phi_d;
  std::span<const T> facets_phi_d_span;

  thrust::device_vector<T> alpha_d;
  std::span<const T> alpha_d_span;

  thrust::device_vector<T> detJ_geom_d;
  std::span<const T> detJ_geom_d_span;

  thrust::device_vector<int32_t> cell_facet_d;
  std::span<const int32_t> cell_facet_d_span;

  int n_faces; // Number of faces on the topological entity
  thrust::device_vector<int32_t> faces_dofs_d;
  std::span<const int32_t> faces_dofs_d_span;

  thrust::device_vector<std::int32_t> dofmap_d;
  std::span<const std::int32_t> dofmap_d_span;
};

template <typename T, int P, int Q, typename U = T>
class MatFreeMassExteriorBaseline3D {
public:
  using value_type = T;
  using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

  MatFreeMassExteriorBaseline3D(std::shared_ptr<mesh::Mesh<U>> mesh,
                                std::shared_ptr<fem::FunctionSpace<U>> V,
                                std::span<const std::int32_t> cell_facet_data,
                                U alpha)
      : mesh(mesh), V(V) {
    auto dofmap = V->dofmap();
    auto map = dofmap->index_map;
    int map_bs = dofmap->index_map_bs();
    la::Vector<T> alpha_vec(map, map_bs);
    alpha_vec.set(alpha);
    init(cell_facet_data, alpha_vec.array());
  }

  MatFreeMassExteriorBaseline3D(std::shared_ptr<mesh::Mesh<U>> mesh,
                                std::shared_ptr<fem::FunctionSpace<U>> V,
                                std::span<const std::int32_t> cell_facet_data,
                                std::span<const U> alpha)
      : mesh(mesh), V(V) {
    init(cell_facet_data, alpha);
  }

  void init(std::span<const std::int32_t> cell_facet_data,
            std::span<const U> alpha) {
    // auto [lcells, bcells] = compute_boundary_cells(V);
    // spdlog::debug("#lcells = {}, #bcells = {}", lcells.size(),
    // bcells.size());
    auto element_p = this->V->element();

    const std::size_t tdim = mesh->topology()->dim();
    this->number_of_local_cells =
        mesh->topology()->index_map(tdim)->size_local() +
        mesh->topology()->index_map(tdim)->num_ghosts();
    assert(cell_facet_data.size() % 2 == 0);
    this->number_of_local_facets = cell_facet_data.size() / 2;
    std::cout << "Exterior Mass local facets: " << this->number_of_local_facets
              << std::endl;

    // TODO: one could be smart, and pack the necessary exterior data instead of
    // sending everything. Transfer V dofmap to the GPU Or just use the same
    // pointer as for the normal mass matrix? Like passing the exterior dofmap
    // as argument
    auto dofmap = V->dofmap();

    dofmap_d_span = copy_to_device(
        dofmap->map().data_handle(),
        dofmap->map().data_handle() + dofmap->map().size(), dofmap_d, "dofmap");

    alpha_d_span = copy_to_device(alpha.begin(), alpha.end(), alpha_d, "alpha");

    cell_facet_d_span =
        copy_to_device(cell_facet_data.begin(), cell_facet_data.end(),
                       cell_facet_d, "cell facet data");

    // Construct quadrature points table on interval
    auto [qpts, qwts] = basix::quadrature::make_quadrature<U>(
        basix::quadrature::type::Default, basix::cell::type::triangle,
        basix::polyset::type::standard, 2 * Q - 2);

    assert(qpts.size() / (tdim - 1) == nq);
    // For each facet, we map it to the corresponding reference 2d point
    mesh::CellType cell_type = mesh->topology()->cell_type();
    this->n_faces = mesh::cell_num_entities(cell_type, tdim - 1);

    std::vector<U> facets_phi_table(n_faces * nq * nd);
    for (int i = 0; i < n_faces; ++i) {
      // 1. Get mapped points on physical reference facet: tdim - 1 -> tdim
      std::vector<U> mapped_points = map_facet_points(qpts, i, cell_type);
      assert(mapped_points.size() / tdim == qpts.size() / (tdim - 1));

      // 2. Evaluate basis functions at mapped points
      auto [phi_table, shape] = element_p->tabulate(
          mapped_points, {mapped_points.size() / tdim, tdim}, 0);
      assert(shape[0] == 1 && shape[3] == 1);
      assert(shape[1] == nq && shape[2] == nd);
      for (int q = 0; q < nq; ++q) {
        for (int k = 0; k < nd; ++k) {
          facets_phi_table[i * nq * nd + q * nd + k] = phi_table[q * nd + k];
          // std::cout << std::format("facets_phi[{}][{}][{}]={}", i, q, k,
          // phi_table[q * K + k]) << std::endl;
        }
      }
    }
    std::cout << std::format("Table size = {}, Fxnqxnd: {}x{}x{}",
                             facets_phi_table.size(), n_faces, nq, nd)
              << std::endl;
    facets_phi_d_span =
        copy_to_device(facets_phi_table.begin(), facets_phi_table.end(),
                       facets_phi_d, "facets phi");

    // Precompute geometry data
    std::vector<U> detJ_geom =
        compute_geometry_facets(mesh, cell_facet_data, qpts, qwts);
    assert(detJ_geom.size() == this->number_of_local_cells * n_faces * nq);

    detJ_geom_d_span = copy_to_device(detJ_geom.begin(), detJ_geom.end(),
                                      detJ_geom_d, "detJ geometry");

    // Element Dof Layout, map from local facet idx -> local dof idx
    // const fem::ElementDofLayout &dof_layout = dofmap->element_dof_layout();
    // int n_face_dofs = dof_layout.num_entity_closure_dofs(tdim - 1);
    // auto faces_dofs = make_faces_to_dofs_map(dof_layout, mesh, tdim);
    // assert(faces_dofs.size() == n_face_dofs * n_faces);

    // faces_dofs_d_span =
    //     copy_to_device(faces_dofs.begin(), faces_dofs.end(), faces_dofs_d,
    //                    "local face index to local closure dofs");
  }

  template <typename Vector>
  void operator()(Vector &in, Vector &out, U global_coefficient = 1.) {
    in.scatter_fwd();

    const T *in_dofs = in.array().data();
    T *out_dofs = out.mutable_array().data();

    assert(in.array().size() == out.mutable_array().size());
    if (this->number_of_local_facets != 0) {
      dim3 grid_size(this->number_of_local_facets);
      dim3 block_size(max(nq, nd));
      facets_mass_operator_baseline<T, nd, nq><<<grid_size, block_size>>>(
          in_dofs, out_dofs, this->cell_facet_d_span.data(),
          this->detJ_geom_d_span.data(), this->alpha_d_span.data(),
          this->dofmap_d_span.data(), this->facets_phi_d_span.data(),
          this->faces_dofs_d_span.data(), this->n_faces, global_coefficient);
      check_device_last_error();
    }
  }

  template <typename Vector>
  void get_diag(Vector &diag, U global_coefficient = 1.) {
    T *out_dofs = diag.mutable_array().data();

    if (this->number_of_local_facets != 0) {
      dim3 grid_size(this->number_of_local_facets);
      dim3 block_size(nd);
      mass_exterior_diagonal<T, nd, nq><<<grid_size, block_size>>>(
          out_dofs, this->cell_facet_d_span.data(),
          this->detJ_geom_d_span.data(), this->alpha_d_span.data(),
          this->dofmap_d_span.data(), this->facets_phi_d_span.data(),
          this->faces_dofs_d_span.data(), this->n_faces, global_coefficient);
    }
  }

  template <typename Vector>
  void get_diag_inverse(Vector &diag_inv, U global_coefficient = 1.) {
    this->get_diag(diag_inv, global_coefficient);
    if (this->number_of_local_facets != 0) {
      thrust::transform(thrust::device, diag_inv.array().begin(),
                        diag_inv.array().begin() + diag_inv.map()->size_local(),
                        diag_inv.mutable_array().begin(),
                        [] __host__ __device__(T yi) { return 1.0 / yi; });
    }
  }

private:
  static constexpr int N = P + 1;
  static constexpr int K = (N + 2) * (N + 1) * N / 6;
  static constexpr int nd = K;
  static constexpr int nq =
      6 * (Q == 3) + 12 * (Q == 4) + 16 * (Q == 5) + 25 * (Q == 6) +
      33 * (Q == 7) + 42 * (Q == 8) + 55 * (Q == 9) + 67 * (Q == 10) +
      79 * (Q == 11) + 96 * (Q == 12) + 112 * (Q == 13); // 2D rules (on facet)

  std::size_t number_of_local_facets;
  std::size_t number_of_local_cells;

  std::shared_ptr<mesh::Mesh<U>> mesh;
  std::shared_ptr<fem::FunctionSpace<U>> V;

  thrust::device_vector<T> facets_phi_d;
  std::span<const T> facets_phi_d_span;

  thrust::device_vector<T> alpha_d;
  std::span<const T> alpha_d_span;

  thrust::device_vector<T> detJ_geom_d;
  std::span<const T> detJ_geom_d_span;

  thrust::device_vector<int32_t> cell_facet_d;
  std::span<const int32_t> cell_facet_d_span;

  int n_faces; // Number of faces on the topological entity
  thrust::device_vector<int32_t> faces_dofs_d;
  std::span<const int32_t> faces_dofs_d_span;

  thrust::device_vector<std::int32_t> dofmap_d;
  std::span<const std::int32_t> dofmap_d_span;
};

} // namespace dolfinx::acc