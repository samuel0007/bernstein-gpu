#pragma once
#include "mesh.hpp"
#include "types.hpp"
#include <dolfinx/io/XDMFFile.h>
#include <mpi.h>

namespace freefus {

template <typename U>
MeshData<U> load_mesh(MPI_Comm comm, mesh::CellType cell_type,
                      const std::string &mesh_filepath) {
  auto coord_element = fem::CoordinateElement<double>(cell_type, 1);

  dolfinx::io::XDMFFile fmesh(comm, mesh_filepath, "r");
  auto base_mesh_p = std::make_shared<mesh::Mesh<U>>(
      fmesh.read_mesh(coord_element, mesh::GhostMode::none, "mesh"));
  // auto mesh_ptr = std::make_shared<mesh::Mesh<U>>(
  //     ghost_layer_mesh(*base_mesh_p, coord_element));
  // auto mesh_ptr = std::make_shared<mesh::Mesh<U>>(
  //   fmesh.read_mesh(coord_element, mesh::GhostMode::none, "mesh"));

  const int tdim = base_mesh_p->topology()->dim();
  base_mesh_p->topology()->create_connectivity(tdim - 1, tdim);

  auto cell_tags = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(*base_mesh_p, "Cell tags", std::nullopt));
  // auto facet_tags = std::make_shared<mesh::MeshTags<std::int32_t>>(
  //     fmesh.read_meshtags(*base_mesh_p, "Facet tags", std::nullopt));
  auto mesh_ptr = std::make_shared<mesh::Mesh<U>>(
      ghost_layer_mesh(*base_mesh_p, coord_element, cell_tags));
  // auto mesh_ptr = std::make_shared<mesh::Mesh<U>>(
  //     ghost_layer_mesh(*base_mesh_p, coord_element));
  mesh_ptr->topology()->create_connectivity(tdim - 1, tdim);
  // auto facet_tags = nullptr;
  // auto mesh_ptr = base_mesh_p;

  assert(!cell_tags->indices().empty() && "No cell tags found");
  // assert(!facet_tags->indices().empty() && "No facet tags found");

  auto local_cells = mesh_ptr->topology()->index_map(tdim)->size_local();
  auto ghost_cells = mesh_ptr->topology()->index_map(tdim)->num_ghosts();
  spdlog::info("Cells: local={}, ghost={}, total={}", local_cells, ghost_cells,
               local_cells + ghost_cells);

  // return MeshData<U>{mesh_ptr, cell_tags, facet_tags};
  return MeshData<U>{mesh_ptr, cell_tags, nullptr};
}

// TODO: move to mainlib source
template <typename U>
auto compute_global_cell_size(
    const std::shared_ptr<dolfinx::mesh::Mesh<U>> &mesh_ptr) {
  int tdim = mesh_ptr->topology()->dim();
  const int N = mesh_ptr->topology()->index_map(tdim)->size_local();
  std::vector<int> cells(N);
  std::iota(cells.begin(), cells.end(), 0);

  std::vector<U> h_local = dolfinx::mesh::h(*mesh_ptr, cells, tdim);
  U min_local = *std::min_element(h_local.begin(), h_local.end());
  U max_local = *std::max_element(h_local.begin(), h_local.end());
  U min_global;
  U max_global;
  MPI_Allreduce(&min_local, &min_global, 1, dolfinx::MPI::mpi_t<U>, MPI_MIN,
                mesh_ptr->comm());
  MPI_Allreduce(&max_local, &max_global, 1, dolfinx::MPI::mpi_t<U>, MPI_MAX,
                mesh_ptr->comm());
  spdlog::info("Global min mesh size: {}", min_global);
  spdlog::info("Global max mesh size: {}", max_global);

  return std::make_tuple(min_global, max_global);
}

template <typename U>
auto compute_global_sound_speed(MPI_Comm comm, auto material_coefficients) {
  auto &c0 = std::get<0>(material_coefficients);
  U min_local =
      *std::min_element(c0->x()->array().begin(), c0->x()->array().end());
  U max_local =
      *std::max_element(c0->x()->array().begin(), c0->x()->array().end());
  U min_global;
  U max_global;
  MPI_Allreduce(&min_local, &min_global, 1, dolfinx::MPI::mpi_t<U>, MPI_MIN,
                comm);
  MPI_Allreduce(&max_local, &max_global, 1, dolfinx::MPI::mpi_t<U>, MPI_MAX,
                comm);
  spdlog::info("Local min sound speed: {}", min_local);
  spdlog::info("Local max sound speed: {}", max_local);
  spdlog::info("Global min sound speed: {}", min_global);
  spdlog::info("Global max sound speed: {}", max_global);

  assert(min_global > 1e-16);
  return std::make_tuple(min_global, max_global);
}

template <typename U, int P>
void check_nyquist(U f0, U hmin, U hmax, U c0min, U c0max, U wave_cfl_dt) {
  // Highest spatial frequency mesh can support everywhere (Hz)
  const U f_mesh_max = c0min * P / (2.0 * hmax);

  // Highest temporal frequency timestep can support (Hz)
  const U f_dt_max = 1.0 / (2.0 * wave_cfl_dt);

  // Effective max frequency
  const U f_sim_max = std::min(f_mesh_max, f_dt_max);

  spdlog::info("Mesh-limited   f_max: {:#.6e} Hz", f_mesh_max);
  spdlog::info("Δt-limited     f_max: {:#.6e} Hz", f_dt_max);
  spdlog::info("Effective      f_max: {:#.6e} Hz", f_sim_max);

  if (f0 <= f_sim_max) {
    spdlog::info("OK   - f0 = {:#.6e} Hz is resolved.", f0);
  } else {
    spdlog::error("FAIL - f0 = {:#.6e} Hz exceeds resolved band.", f0);
  }
}

} // namespace freefus
