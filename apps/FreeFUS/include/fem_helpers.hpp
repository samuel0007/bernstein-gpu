#pragma once

#include <basix/finite-element.h>
#include <dolfinx.h>
#include <dolfinx/mesh/cell_types.h>
#include <memory>

namespace dolfinx::mesh {

// THis does a slice at a fixed y (x, z varying)
template <std::floating_point T = double>
Mesh<T>
create_slice(MPI_Comm comm, std::array<std::array<T, 3>, 2> p,
             std::array<std::int64_t, 2> n, CellType celltype,
             CellPartitionFunction partitioner,
             DiagonalType diagonal = DiagonalType::right,
             const CellReorderFunction &reorder_fn = graph::reorder_gps) {
  if (std::ranges::any_of(n, [](auto e) { return e < 1; }))
    throw std::runtime_error("At least one cell per dimension is required");

//   for (int32_t i = 0; i < 2; i++) {
//     if (p[0][i] >= p[2][i])
//       throw std::runtime_error("It must hold p[0] < p[1].");
//   }

  if (!partitioner and dolfinx::MPI::size(comm) > 1)
    partitioner = create_cell_partitioner();

  switch (celltype) {
//   case CellType::triangle:
//     return impl::build_tri<T>(comm, p, n, partitioner, diagonal, reorder_fn);
  case CellType::quadrilateral:
    return build_quad<T>(comm, p, n, partitioner, reorder_fn);
  default:
    throw std::runtime_error("Generate rectangle mesh. Wrong cell type");
  }
}

template <std::floating_point T>
Mesh<T> build_quad(MPI_Comm comm, const std::array<std::array<T, 3>, 2> p,
                   std::array<std::int64_t, 2> n,
                   const CellPartitionFunction &partitioner,
                   const CellReorderFunction &reorder_fn) {
  fem::CoordinateElement<T> element(CellType::quadrilateral, 1);
  if (dolfinx::MPI::rank(comm) == 0) {
    const auto [nx, ny] = n;
    const auto [a, r, c] = p[0];
    const auto [b, s, d] = p[1];
    assert(r == s);

    const T ab = (b - a) / static_cast<T>(nx);
    const T cd = (d - c) / static_cast<T>(ny);

    // Create vertices
    std::vector<T> x;
    x.reserve((nx + 1) * (ny + 1) * 3);
    for (std::int64_t ix = 0; ix <= nx; ix++) {
      T x0 = a + ab * static_cast<T>(ix);
      for (std::int64_t iy = 0; iy <= ny; iy++)
        x.insert(x.end(), {x0, r, c + cd * static_cast<T>(iy)});
    }

    // Create rectangles
    std::vector<std::int64_t> cells;
    cells.reserve(nx * ny * 4);
    for (std::int64_t ix = 0; ix < nx; ix++) {
      for (std::int64_t iy = 0; iy < ny; iy++) {
        std::int64_t i0 = ix * (ny + 1);
        cells.insert(cells.end(), {i0 + iy, i0 + iy + 1, i0 + iy + ny + 1,
                                   i0 + iy + ny + 2});
      }
    }

    return create_mesh(comm, MPI_COMM_SELF, cells, element, MPI_COMM_SELF, x,
                       {x.size() / 3, 3}, partitioner, reorder_fn);
  } else {
    return create_mesh(comm, MPI_COMM_NULL, {}, element, MPI_COMM_NULL,
                       std::vector<T>{}, {0, 3}, partitioner, reorder_fn);
  }
}
} // namespace dolfinx::mesh

namespace freefus {

template <typename T>
auto make_element_spaces(const std::shared_ptr<dolfinx::mesh::Mesh<T>> &mesh,
                         mesh::CellType cell_type,
                         basix::element::lagrange_variant lvariant,
                         int degree) {
  namespace be = basix::element;
  namespace bc = basix::cell;

  basix::cell::type b_cell_type = cell_type_to_basix_type(cell_type);

  auto el = basix::create_element<T>(be::family::P, b_cell_type, degree,
                                     lvariant, be::dpc_variant::unset, false);

  auto el_DG = basix::create_element<T>(be::family::P, b_cell_type, 0, lvariant,
                                        be::dpc_variant::unset, true);

  auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
      mesh, std::make_shared<const fem::FiniteElement<T>>(el)));

  auto V_DG = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
      mesh, std::make_shared<const fem::FiniteElement<T>>(el_DG)));

  spdlog::info("V Local dofs: {}, Global Dofs: {}",
               V->dofmap()->index_map->size_local(),
               V->dofmap()->index_map->size_global());
  spdlog::info("V_DG Local dofs: {}, Global Dofs: {}",
               V_DG->dofmap()->index_map->size_local(),
               V_DG->dofmap()->index_map->size_global());

  return std::make_tuple(el, el_DG, V, V_DG);
}

template <typename T>
auto make_output_spaces(const std::shared_ptr<mesh::Mesh<T>> &mesh_ptr,
                        mesh::CellType cell_type, int polynomial_degree) {

  basix::cell::type b_cell_type = cell_type_to_basix_type(cell_type);
  // Output space
  basix::FiniteElement lagrange_element =
      basix::create_element<T>(basix::element::family::P, b_cell_type,
                               polynomial_degree + 2, // "refine twice the mesh"
                               basix::element::lagrange_variant::gll_warped,
                               basix::element::dpc_variant::unset, false);

  auto V_out =
      std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
          mesh_ptr,
          std::make_shared<const fem::FiniteElement<T>>(lagrange_element)));

  auto u_out = std::make_shared<fem::Function<T>>(V_out);

  return std::make_tuple(V_out, u_out);
}

template <typename T>
auto make_sliced_output_spaces(std::shared_ptr<mesh::Mesh<T>> &mesh_ptr,
                               std::shared_ptr<fem::Function<T>> &u, T lx, T lz,
                               int nx, int nz) {

  //   basix::cell::type b_cell_type = cell_type_to_basix_type(cell_type);
  // Output space
  // CG1 mesh
  basix::FiniteElement lagrange_element = basix::create_element<T>(
      basix::element::family::P,
      cell_type_to_basix_type(mesh::CellType::quadrilateral), 1,
      basix::element::lagrange_variant::equispaced,
      basix::element::dpc_variant::unset, false);

  auto slice_mesh_ptr =
      std::make_shared<mesh::Mesh<T>>(mesh::create_slice<T>(
          mesh_ptr->comm(), {{{-lx / 2., 0.0, 0.0}, {lx / 2, 0.0, lz}}}, {nx - 1, nz - 1},
          mesh::CellType::quadrilateral,
          mesh::create_cell_partitioner(mesh::GhostMode::none)));

  auto V_out =
      std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
          slice_mesh_ptr,
          std::make_shared<const fem::FiniteElement<T>>(lagrange_element)));

  auto u_sliced_out = std::make_shared<fem::Function<T>>(V_out);

  assert(u_sliced_out->x()->array().size() == nx * nz);
  assert(slice_mesh_ptr->geometry().x().size() ==
         nx * nz * 3); // x: (num_points x 3)
  spdlog::info("Sliced output mesh size NX x NZ = {} x {}", nx, nz);

  auto cell_map =
      slice_mesh_ptr->topology()->index_map(slice_mesh_ptr->topology()->dim());
  assert(cell_map);
  std::vector<std::int32_t> cells(
      cell_map->size_local() + cell_map->num_ghosts(), 0);
  std::iota(cells.begin(), cells.end(), 0);
  geometry::PointOwnershipData<T> interpolation_point_ownership =
      fem::create_interpolation_data(
          u_sliced_out->function_space()->mesh()->geometry(),
          *u_sliced_out->function_space()->element(),
          *u->function_space()->mesh(), std::span(cells), 1e-8);
  // Yes, if you ask, this is pure lazyness. TODO: strong type.
  return std::make_tuple(V_out, u_sliced_out,
                         std::make_tuple(cells, interpolation_point_ownership));
}

template <typename T>
auto interpolate_to_slice(const std::shared_ptr<fem::Function<T>> &u,
                          const std::shared_ptr<fem::Function<T>> &u_sliced_out,
                          auto &interpolation_data) {
  auto [cells, interpolation_point_ownership] = interpolation_data;
  auto sliced_mesh_ptr = u_sliced_out->function_space()->mesh();
  u_sliced_out->interpolate(*u, cells, interpolation_point_ownership);
}

} // namespace freefus
