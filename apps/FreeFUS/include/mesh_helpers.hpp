#pragma once
#include <dolfinx/io/XDMFFile.h>
#include <mpi.h>
#include "mesh.hpp"
#include "types.hpp"

namespace freefus {

template <typename U>
MeshData<U> load_mesh(MPI_Comm comm, mesh::CellType cell_type,
                      const std::string &mesh_filepath) {
  auto coord_element = fem::CoordinateElement<double>(cell_type, 1);

  dolfinx::io::XDMFFile fmesh(comm, mesh_filepath, "r");
  auto base_mesh_p = std::make_shared<mesh::Mesh<U>>(
    fmesh.read_mesh(coord_element, mesh::GhostMode::none, "mesh"));
  auto mesh_ptr = std::make_shared<mesh::Mesh<U>>(
      ghost_layer_mesh(*base_mesh_p, coord_element));
  // auto mesh_ptr = std::make_shared<mesh::Mesh<U>>(
  //   fmesh.read_mesh(coord_element, mesh::GhostMode::none, "mesh"));

  const int tdim = mesh_ptr->topology()->dim();
  mesh_ptr->topology()->create_connectivity(tdim - 1, tdim);

  auto cell_tags = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(*mesh_ptr, "Cell tags", std::nullopt));
  auto facet_tags = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(*mesh_ptr, "Facet tags", std::nullopt));

  // assert(!cell_tags->indices().empty() && "No cell tags found");
  assert(!facet_tags->indices().empty() && "No facet tags found");

  auto local_cells = mesh_ptr->topology()->index_map(tdim)->size_local();
  auto ghost_cells = mesh_ptr->topology()->index_map(tdim)->num_ghosts();
  spdlog::info("Cells: local={}, ghost={}, total={}",
              local_cells,
              ghost_cells,
              local_cells + ghost_cells);

  return MeshData<U>{mesh_ptr, cell_tags, facet_tags};
}

// TODO: move to mainlib source
template <typename U>
U compute_global_min_cell_size(
    const std::shared_ptr<dolfinx::mesh::Mesh<U>> &mesh_ptr) {
  int tdim = mesh_ptr->topology()->dim();
  const int N = mesh_ptr->topology()->index_map(tdim)->size_local();
  std::vector<int> cells(N);
  std::iota(cells.begin(), cells.end(), 0);

  std::vector<U> h_local = dolfinx::mesh::h(*mesh_ptr, cells, tdim);
  U min_local = *std::min_element(h_local.begin(), h_local.end());

  U min_global;
  MPI_Allreduce(&min_local, &min_global, 1, dolfinx::MPI::mpi_t<U>, MPI_MIN,
                mesh_ptr->comm());
  spdlog::info("Global min mesh size: {}", min_global);
  return min_global;
}

template <typename U>
U compute_global_minimum_sound_speed(MPI_Comm comm, auto material_coefficients) {
  auto &c0 = std::get<0>(material_coefficients);
  U min_local = *std::min_element(c0->x()->array().begin(), c0->x()->array().end());
  U max_local = *std::max_element(c0->x()->array().begin(), c0->x()->array().end());
  U min_global;
  MPI_Allreduce(&min_local, &min_global, 1, dolfinx::MPI::mpi_t<U>, MPI_MIN,
                comm);
  spdlog::info("Local min sound speed: {}", min_local);
  spdlog::info("Local max sound speed: {}", max_local);
  spdlog::info("Global min sound speed: {}", min_global);

  assert(min_global > 1e-16);
  return min_global;
}

} // namespace freefus
