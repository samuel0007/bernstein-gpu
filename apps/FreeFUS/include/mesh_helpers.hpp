#pragma once
#include "mesh.hpp"
#include "types.hpp"
#include <dolfinx/io/XDMFFile.h>
#include <mpi.h>

namespace freefus
{

  template <typename U>
  MeshData<U> load_mesh(MPI_Comm comm, mesh::CellType cell_type,
                        const std::string &mesh_filepath)
  {
    auto coord_element = fem::CoordinateElement<double>(cell_type, 1);

    dolfinx::io::XDMFFile fmesh(comm, mesh_filepath, "r");
    auto base_mesh_p = std::make_shared<mesh::Mesh<U>>(
        fmesh.read_mesh(coord_element, mesh::GhostMode::none, "mesh"));

    const int tdim = base_mesh_p->topology()->dim();
    base_mesh_p->topology()->create_connectivity(tdim - 1, tdim);

    auto cell_tags = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(*base_mesh_p, "Cell tags", std::nullopt));
    auto facet_tags = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(*base_mesh_p, "Facet tags", std::nullopt));
    auto mesh_ptr = std::make_shared<mesh::Mesh<U>>(
        ghost_layer_mesh(*base_mesh_p, coord_element, cell_tags));
    mesh_ptr->topology()->create_connectivity(tdim - 1, tdim);

    assert(!cell_tags->indices().empty() && "No cell tags found");
    assert(!facet_tags->indices().empty() && "No facet tags found");

    auto local_cells = mesh_ptr->topology()->index_map(tdim)->size_local();
    auto ghost_cells = mesh_ptr->topology()->index_map(tdim)->num_ghosts();
    spdlog::info("Cells: local={}, ghost={}, total={}", local_cells, ghost_cells,
                 local_cells + ghost_cells);

    return MeshData<U>{mesh_ptr, cell_tags, facet_tags};
    // return MeshData<U>{mesh_ptr, cell_tags, nullptr};
  }

  // TODO: move to mainlib source
  template <typename U>
  auto compute_global_cell_size(
      const std::shared_ptr<dolfinx::mesh::Mesh<U>> &mesh_ptr)
  {
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
  auto compute_global_sound_speed(MPI_Comm comm, auto material_coefficients)
  {
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

  // Reports PPW, PPP (evolution), PPP (sampling; should be integer), and Nyquist checks.
  // No alignment checks between dt and sampling_dt.
  template <typename U, int P>
  void check_nyquist(U f0, U hmin, U hmax, U c0min, U c0max, U dt, U sampling_dt)
  {
    const U period0 = U(1) / f0;

    // Nyquist limits
    const U f_mesh_max = c0min * P / (U(2) * hmax);
    const U f_dt_max = U(1) / (U(2) * dt);            // evolution
    const U f_samp_max = U(1) / (U(2) * sampling_dt); // sampling
    const U f_sim_max = std::min(f_mesh_max, f_dt_max);

    // PPW (space) and PPP (time)
    const U lambda0_min = c0min / f0;
    const U ppw = (lambda0_min / hmin) * P;
    const U ppp_evo = period0 / dt;
    const U ppp_samp = period0 / sampling_dt;

    // Integer check for sampling PPP
    const long long ppp_samp_int = static_cast<long long>(std::llround(ppp_samp));
    const U ppp_samp_err = std::abs(ppp_samp - U(ppp_samp_int));

    spdlog::info("max_dt={}, sampling_dt={}", dt, sampling_dt);

    spdlog::info("Mesh-limited   f_max: {:#.6e} Hz", f_mesh_max);
    spdlog::info("Δt-evolution   f_max: {:#.6e} Hz", f_dt_max);
    spdlog::info("Δt-sampling    f_max: {:#.6e} Hz", f_samp_max);
    spdlog::info("Effective      f_max: {:#.6e} Hz", f_sim_max);

    spdlog::info("PPW @ f0: {:.2f}", ppw);
    spdlog::info("PPP evolution @ f0: {:.2f}", ppp_evo);
    spdlog::info("PPP sampling  @ f0: {:.2f} (nearest int = {} ; |err| = {:.3e})",
                 ppp_samp, ppp_samp_int, ppp_samp_err);

    if (ppp_samp_err < U(1e-6))
    {
      spdlog::info("COHERENCE OK   - sampling PPP is integer.");
    }
    else
    {
      spdlog::warn("COHERENCE WARN - sampling PPP not integer. Consider sampling_dt = period0 / {}.", ppp_samp_int);
    }

    if (f0 <= f_sim_max)
    {
      spdlog::info("ALIASING OK    - f0 resolved by evolution.");
    }
    else
    {
      spdlog::error("ALIASING FAIL  - f0 exceeds evolution band.");
    }
    if (f0 <= f_samp_max)
    {
      spdlog::info("SAMPLING OK    - f0 Nyquist-safe for sampling_dt.");
    }
    else
    {
      spdlog::error("SAMPLING FAIL  - f0 exceeds sampling Nyquist.");
    }
  }

  template <typename T>
  T compute_evolution_time(T traversal_time, T sampling_dt)
  {
    const std::int64_t n_steps = static_cast<std::int64_t>(std::ceil(traversal_time / sampling_dt));
    return static_cast<T>(n_steps) * sampling_dt;
  }

} // namespace freefus
