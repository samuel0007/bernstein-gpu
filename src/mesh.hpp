
#pragma once
#include <dolfinx/mesh/Mesh.h>

/// @brief Compute two lists of cell indices:
/// 1. cells which are "local", i.e. the dofs on
/// these cells are not shared with any other process.
/// 2. cells which share dofs with other processes.
///
template <typename T>
std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
compute_boundary_cells(std::shared_ptr<dolfinx::fem::FunctionSpace<T>> V) {
  auto mesh = V->mesh();
  auto topology = mesh->topology_mutable();
  int tdim = topology->dim();
  int fdim = tdim - 1;
  topology->create_connectivity(fdim, tdim);

  int ncells_local = topology->index_map(tdim)->size_local();
  int ncells_ghost = topology->index_map(tdim)->num_ghosts();
  int ndofs_local = V->dofmap()->index_map->size_local();

  std::vector<std::uint8_t> cell_mark(ncells_local + ncells_ghost, 0);
  for (int i = 0; i < ncells_local; ++i) {
    auto cell_dofs = V->dofmap()->cell_dofs(i);
    for (auto dof : cell_dofs)
      if (dof >= ndofs_local)
        cell_mark[i] = 1;
  }
  for (int i = ncells_local; i < ncells_local + ncells_ghost; ++i)
    cell_mark[i] = 1;

  std::vector<int> local_cells;
  std::vector<int> boundary_cells;
  for (int i = 0; i < cell_mark.size(); ++i) {
    if (cell_mark[i])
      boundary_cells.push_back(i);
    else
      local_cells.push_back(i);
  }

  spdlog::debug("lcells:{}, bcells:{}", local_cells.size(),
                boundary_cells.size());

  return {std::move(local_cells), std::move(boundary_cells)};
}

template <std::floating_point T>
mesh::Mesh<T> ghost_layer_mesh(
    dolfinx::mesh::Mesh<T> &mesh,
    dolfinx::fem::CoordinateElement<T> coord_element,
    std::shared_ptr<mesh::MeshTags<std::int32_t>> cell_tags,
    std::shared_ptr<mesh::MeshTags<std::int32_t>> facet_tags,
    const std::function<std::vector<std::int32_t>(
        const dolfinx::graph::AdjacencyList<std::int32_t> &)> &reorder_fn =
        graph::reorder_gps) {
  const unsigned long tdim = mesh.topology()->dim();
  const unsigned long gdim = mesh.geometry().dim();

  std::size_t ncells = mesh.topology()->index_map(tdim)->size_local();
  std::size_t num_vertices = mesh.topology()->index_map(0)->size_local();

  // Find which local vertices are ghosted elsewhere
  auto vertex_destinations =
      mesh.topology()->index_map(0)->index_to_dest_ranks();

  // Map from any local cells to processes where they should be ghosted
  std::map<int, std::vector<int>> cell_to_dests;
  auto c_to_v = mesh.topology()->connectivity(tdim, 0);

  std::vector<int> cdests;
  for (std::size_t c = 0; c < ncells; ++c) {
    cdests.clear();
    for (auto v : c_to_v->links(c)) {
      auto vdest = vertex_destinations.links(v);
      for (int dest : vdest)
        cdests.push_back(dest);
    }
    std::sort(cdests.begin(), cdests.end());
    cdests.erase(std::unique(cdests.begin(), cdests.end()), cdests.end());
    if (!cdests.empty())
      cell_to_dests[c] = cdests;
  }

  spdlog::info("cell_to_dests= {}, ncells = {}", cell_to_dests.size(), ncells);

  auto partitioner =
      [cell_to_dests,
       ncells](MPI_Comm comm, int nparts,
               const std::vector<dolfinx::mesh::CellType> &cell_types,
               const std::vector<std::span<const std::int64_t>> &cells) {
        int rank = dolfinx::MPI::rank(comm);
        std::vector<std::int32_t> dests;
        std::vector<int> offsets = {0};
        for (int c = 0; c < ncells; ++c) {
          dests.push_back(rank);
          if (auto it = cell_to_dests.find(c); it != cell_to_dests.end())
            dests.insert(dests.end(), it->second.begin(), it->second.end());

          // Ghost to other processes
          offsets.push_back(dests.size());
        }
        return dolfinx::graph::AdjacencyList<std::int32_t>(std::move(dests),
                                                           std::move(offsets));
      };

  std::array<std::size_t, 2> xshape = {num_vertices, gdim};
  std::span<T> x(mesh.geometry().x().data(), xshape[0] * xshape[1]);

  auto dofmap = mesh.geometry().dofmap();
  auto imap = mesh.geometry().index_map();
  std::vector<std::int32_t> permuted_dofmap;
  //   std::vector<int> perm = basix::tp_dof_ordering(
  //       basix::element::family::P,
  //       dolfinx::mesh::cell_type_to_basix_type(coord_element.cell_shape()),
  //       coord_element.degree(), coord_element.variant(),
  //       basix::element::dpc_variant::unset, false);
  for (std::size_t c = 0; c < dofmap.extent(0); ++c) {
    auto cell_dofs = std::submdspan(dofmap, c, std::full_extent);
    for (int i = 0; i < dofmap.extent(1); ++i)
      permuted_dofmap.push_back(cell_dofs(i));

    //   permuted_dofmap.push_back(cell_dofs(perm[i]));
  }
  std::vector<std::int64_t> permuted_dofmap_global(permuted_dofmap.size());
  imap->local_to_global(permuted_dofmap, permuted_dofmap_global);

  auto imap_cells = mesh.topology()->index_map(tdim);
  auto imap_facets = mesh.topology()->index_map(tdim - 1);

 

  auto new_mesh = dolfinx::mesh::create_mesh(
      mesh.comm(), mesh.comm(), std::span(permuted_dofmap_global),
      coord_element, mesh.comm(), x, xshape, partitioner, reorder_fn);


  // It's so wild that global indices have int64_t and local have int32_t
  std::vector<std::int64_t> original_cell_tags_global_indices(cell_tags->indices().size());
  imap_cells->local_to_global(cell_tags->indices(), original_cell_tags_global_indices);

  std::vector<std::int32_t> new_cell_tags_local_indices(cell_tags->indices().size());
  new_mesh.topology()->index_map(tdim)->global_to_local(original_cell_tags_global_indices, new_cell_tags_local_indices);
  

  spdlog::info("** NEW MESH num_ghosts_cells = {}",
               new_mesh.topology()->index_map(tdim)->num_ghosts());
  spdlog::info("** NEW MESH num_local_cells = {}",
               new_mesh.topology()->index_map(tdim)->size_local());

  return new_mesh;
}