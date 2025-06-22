#pragma once
#include <dolfinx/io/XDMFFile.h>
#include <mpi.h>

#include "types.hpp"

namespace freefus {

template <typename T>
MeshData<T> load_mesh(MPI_Comm comm, mesh::CellType cell_type,
                      const std::string &mesh_filepath) {
  auto coord_element = fem::CoordinateElement<T>(cell_type, 1);

  dolfinx::io::XDMFFile fmesh(comm, mesh_filepath, "r");
  auto mesh_ptr = std::make_shared<mesh::Mesh<T>>(
      fmesh.read_mesh(coord_element, mesh::GhostMode::none, "mesh"));

  const int tdim = mesh_ptr->topology()->dim();
  mesh_ptr->topology()->create_connectivity(tdim - 1, tdim);

  auto cell_tags = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(*mesh_ptr, "Cell tags", std::nullopt));
  auto facet_tags = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(*mesh_ptr, "Facet tags", std::nullopt));

  assert(!cell_tags->indices().empty() && "No cell tags found");
  assert(!facet_tags->indices().empty() && "No facet tags found");

  return MeshData<T>{mesh_ptr, cell_tags, facet_tags};
}

// TODO: move to mainlib source
template <typename T>
T compute_global_min_cell_size(
    const std::shared_ptr<dolfinx::mesh::Mesh<T>> &mesh_ptr) {
  int tdim = mesh_ptr->topology()->dim();
  const int N = mesh_ptr->topology()->index_map(tdim)->size_local();
  std::vector<int> cells(N);
  std::iota(cells.begin(), cells.end(), 0);

  std::vector<T> h_local = dolfinx::mesh::h(*mesh_ptr, cells, tdim);
  T min_local = *std::min_element(h_local.begin(), h_local.end());

  T min_global;
  MPI_Allreduce(&min_local, &min_global, 1, dolfinx::MPI::mpi_t<T>, MPI_MIN,
                mesh_ptr->comm());
  spdlog::info("Global mesh size: {}", min_global);
  return min_global;
}

} // namespace freefus
