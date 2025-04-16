
#pragma once

#include <dolfinx/mesh/Mesh.h>


/// @brief Compute two lists of cell indices:
/// 1. cells which are "local", i.e. the dofs on
/// these cells are not shared with any other process.
/// 2. cells which share dofs with other processes.
///
template <typename T>
std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
compute_boundary_cells(std::shared_ptr<dolfinx::fem::FunctionSpace<T>> V)
{
  auto mesh = V->mesh();
  auto topology = mesh->topology_mutable();
  int tdim = topology->dim();
  int fdim = tdim - 1;
  topology->create_connectivity(fdim, tdim);

  int ncells_local = topology->index_map(tdim)->size_local();
  int ncells_ghost = topology->index_map(tdim)->num_ghosts();
  int ndofs_local = V->dofmap()->index_map->size_local();

  std::vector<std::uint8_t> cell_mark(ncells_local + ncells_ghost, 0);
  for (int i = 0; i < ncells_local; ++i)
  {
    auto cell_dofs = V->dofmap()->cell_dofs(i);
    for (auto dof : cell_dofs)
      if (dof >= ndofs_local)
        cell_mark[i] = 1;
  }
  for (int i = ncells_local; i < ncells_local + ncells_ghost; ++i)
    cell_mark[i] = 1;

  std::vector<int> local_cells;
  std::vector<int> boundary_cells;
  for (int i = 0; i < cell_mark.size(); ++i)
  {
    if (cell_mark[i])
      boundary_cells.push_back(i);
    else
      local_cells.push_back(i);
  }

  spdlog::debug("lcells:{}, bcells:{}", local_cells.size(), boundary_cells.size());

  return {std::move(local_cells), std::move(boundary_cells)};
}
