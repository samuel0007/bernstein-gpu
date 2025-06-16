#pragma once

#include "kernels_stiffness.hpp"
#include "util.hpp"
#include "vector.hpp"
#include "geometry_gpu.hpp"

namespace dolfinx::acc
{

  template <typename T, int P, int Q>
  class MatFreeStiffness
  {
  public:
    using value_type = T;
    using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

    MatFreeStiffness(std::shared_ptr<mesh::Mesh<T>> mesh,
                     std::shared_ptr<fem::FunctionSpace<T>> V, T alpha)
        : mesh(mesh), V(V)
    {
      auto dofmap = V->dofmap();
      auto map = dofmap->index_map;
      int map_bs = dofmap->index_map_bs();
      la::Vector<T> alpha_vec(map, map_bs);
      alpha_vec.set(alpha);
      init(alpha_vec.array());
    }

    MatFreeStiffness(std::shared_ptr<mesh::Mesh<T>> mesh,
                     std::shared_ptr<fem::FunctionSpace<T>> V,
                     std::span<const T> alpha)
        : mesh(mesh), V(V)
    {
      init(alpha);
    }

    void init(std::span<const T> alpha)
    {
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

      auto [qpts, qwts] = basix::quadrature::make_quadrature<T>(
          basix::quadrature::type::Default, basix::cell::type::triangle,
          basix::polyset::type::standard, 2 * Q - 2);

      assert(nq == qpts.size() / tdim);

      auto [dphi, shape] = element_p->tabulate(qpts, {nq, tdim}, 1);

      // Copy only derivatives
      dphi_d_span = copy_to_device(dphi.begin() + dphi.size() / (tdim + 1), dphi.end(), dphi_d, "dphi");
      std::cout << std::format("Table size = {}, dxnqxnd: {}x{}x{}",
                               dphi.size() - dphi.size() / (tdim + 1), shape[0] - 1, shape[1], shape[2])
                << std::endl;
      assert(shape[0] == tdim + 1 && shape[3] == 1);
      assert(nq == shape[1] && nd == shape[2]);

      geom_d_span = compute_stiffness_geometry_triangle_GPU<T, nq>(mesh, qpts, qwts, geom_d);
      assert(geom_d_span.size() == this->number_of_local_cells * 3 * nq);
      assert(geom_d.size() == this->number_of_local_cells * 3 * nq);
    }

    template <typename Vector>
    void operator()(const Vector &in, Vector &out)
    {
      // out.set(T{0.0});

      const T *in_dofs = in.array().data();
      T *out_dofs = out.mutable_array().data();
      assert(in.array().size() == out.mutable_array().size());

      assert(nq >= nd);
      dim3 grid_size(this->number_of_local_cells);
      dim3 block_size(nq);
      kernels::stiffness::stiffness_operator<T, nd, nq><<<grid_size, block_size>>>(
          out_dofs, in_dofs, this->alpha_d_span.data(),
          this->geom_d_span.data(), this->dofmap_d_span.data(), this->dphi_d_span.data());
      check_device_last_error();
    }

    template <typename Vector>
    void get_diag_inverse(Vector &diag_inv)
    {
      assert(false && "todo for jacobi preconditioning of cg");
    }

  private:
    static constexpr int N = P + 1;
    static constexpr int nd = (N + 1) * N / 2; // << nq
    static constexpr int nq = 6 * (Q == 3) + 12 * (Q == 4) + 16 * (Q == 5) +
                              25 * (Q == 6) + 33 * (Q == 7) + 42 * (Q == 8) +
                              55 * (Q == 9) + 67 * (Q == 10) + 79 * (Q == 11) +
                              96 * (Q == 12) + 112 * (Q == 13);

    std::size_t number_of_local_cells;

    std::shared_ptr<mesh::Mesh<T>> mesh;
    std::shared_ptr<fem::FunctionSpace<T>> V;

    thrust::device_vector<T> dphi_d;
    std::span<const T> dphi_d_span;

    thrust::device_vector<T> alpha_d;
    std::span<const T> alpha_d_span;

    thrust::device_vector<T> geom_d;
    std::span<const T> geom_d_span;

    thrust::device_vector<std::int32_t> dofmap_d;
    std::span<const std::int32_t> dofmap_d_span;
  };

  template <typename T, int P, int Q>
  class MatFreeStiffness3D
  {
  public:
    using value_type = T;
    using quad_rule = std::pair<std::vector<T>, std::vector<T>>;

    MatFreeStiffness3D(std::shared_ptr<mesh::Mesh<T>> mesh,
                       std::shared_ptr<fem::FunctionSpace<T>> V, T alpha)
        : mesh(mesh), V(V)
    {
      auto dofmap = V->dofmap();
      auto map = dofmap->index_map;
      int map_bs = dofmap->index_map_bs();
      la::Vector<T> alpha_vec(map, map_bs);
      alpha_vec.set(alpha);
      init(alpha_vec.array());
    }

    MatFreeStiffness3D(std::shared_ptr<mesh::Mesh<T>> mesh,
                       std::shared_ptr<fem::FunctionSpace<T>> V,
                       std::span<const T> alpha)
        : mesh(mesh), V(V)
    {
      init(alpha);
    }

    void init(std::span<const T> alpha)
    {
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

      auto [qpts, qwts] = basix::quadrature::make_quadrature<T>(
          basix::quadrature::type::Default, basix::cell::type::tetrahedron,
          basix::polyset::type::standard, 2 * Q - 2);

      assert(nq == qpts.size() / tdim);

      auto [dphi, shape] = element_p->tabulate(qpts, {nq, tdim}, 1);

      // Copy only derivatives
      dphi_d_span = copy_to_device(dphi.begin() + dphi.size() / (tdim + 1), dphi.end(), dphi_d, "dphi");
      std::cout << std::format("Table size = {}, dxnqxnd: {}x{}x{}",
                               dphi.size() - dphi.size() / (tdim + 1), shape[0] - 1, shape[1], shape[2])
                << std::endl;
      assert(shape[0] == tdim + 1 && shape[3] == 1);
      assert(nq == shape[1] && nd == shape[2]);

      geom_d_span = compute_stiffness_geometry_tetrahedron_GPU<T, nq>(mesh, qpts, qwts, geom_d);
      assert(geom_d_span.size() == this->number_of_local_cells * 6 * nq);
      assert(geom_d.size() == this->number_of_local_cells * 6 * nq);
    }

    template <typename Vector>
    void operator()(const Vector &in, Vector &out)
    {
      out.set(T{0.0});

      const T *in_dofs = in.array().data();
      T *out_dofs = out.mutable_array().data();
      assert(in.array().size() == out.mutable_array().size());

      assert(nq >= nd);
      dim3 grid_size(this->number_of_local_cells);
      dim3 block_size(nq);
      kernels::stiffness::stiffness_operator3D<T, nd, nq><<<grid_size, block_size>>>(
          out_dofs, in_dofs, this->alpha_d_span.data(),
          this->geom_d_span.data(), this->dofmap_d_span.data(), this->dphi_d_span.data());
      check_device_last_error();
    }

    template <typename Vector>
    void get_diag_inverse(Vector &diag_inv)
    {
      assert(false && "todo for jacobi preconditioning of cg");
    }

  private:
    static constexpr int N = P + 1;
    static constexpr int nd = N * (N + 1) * (N + 2) / 6;
    static constexpr int nq = 14 * (Q == 3) + 24 * (Q == 4) + 45 * (Q == 5) +
                              74 * (Q == 6) + 122 * (Q == 7) + 177 * (Q == 8) +
                              729 * (Q == 9) + 1000 * (Q == 10) +
                              1331 * (Q == 11);

    std::size_t number_of_local_cells;

    std::shared_ptr<mesh::Mesh<T>> mesh;
    std::shared_ptr<fem::FunctionSpace<T>> V;

    thrust::device_vector<T> dphi_d;
    std::span<const T> dphi_d_span;

    thrust::device_vector<T> alpha_d;
    std::span<const T> alpha_d_span;

    thrust::device_vector<T> geom_d;
    std::span<const T> geom_d_span;

    thrust::device_vector<std::int32_t> dofmap_d;
    std::span<const std::int32_t> dofmap_d_span;
  };

} // namespace dolfinx::acc
