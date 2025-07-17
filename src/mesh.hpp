
#pragma once
#include <dolfinx/mesh/Mesh.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <dolfinx/io/utils.h>


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

template <std::floating_point T, typename TagT>
mesh::Mesh<T> ghost_layer_mesh(
    dolfinx::mesh::Mesh<T> &mesh,
    dolfinx::fem::CoordinateElement<T> coord_element,
    std::shared_ptr<mesh::MeshTags<TagT>>& cell_tags,
    // std::shared_ptr<mesh::MeshTags<TagT>> facet_tags,
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
  // auto imap_facets = mesh.topology()->index_map(tdim - 1);

  auto new_mesh = dolfinx::mesh::create_mesh(
      mesh.comm(), mesh.comm(), std::span(permuted_dofmap_global),
      coord_element, mesh.comm(), x, xshape, partitioner, reorder_fn);


  
  auto new_vertex_to_original_global_vertex_map = new_mesh.geometry().input_global_indices();
  auto new_dofmap = new_mesh.geometry().dofmap();

  // spdlog::info("New mesh #cells={}, #map size={}", new_dofmap.extent(0), new_vertex_to_original_global_vertex_map.size());
  
  // auto original_local_cell_tag_values = cell_tags->values();
  // const int N_cell_tags = original_local_cell_tag_indices.size();
  // assert(N_cell_tags == dofmap.extent(0)); // Every cell should have a tag
  // std::vector<std::int64_t> original_global_cell_tag_indices(N_cell_tags);
  // original_imap_cells->local_to_global(original_local_cell_tag_indices, original_global_cell_tag_indices);

  // auto original_cell_tag_values = cell_tags->values();
  // assert(original_cell_tag_values.size() == dofmap.extent(0));
  std::mdspan<std::int64_t, std::dextents<std::size_t, 2>> dofmap_global_span(permuted_dofmap_global.data(), dofmap.extent(0), dofmap.extent(1));
  
  // std::pair<std::vector<std::int32_t>, std::vector<TagT>> entities_values = io::distribute_entity_data<TagT>(
  //       *new_mesh.topology(), new_mesh.geometry().input_global_indices(),
  //       new_mesh.geometry().index_map()->size_global(),
  //       new_mesh.geometry().cmap().create_dof_layout(), new_mesh.geometry().dofmap(),
  //       tdim, dofmap_global_span, original_cell_tag_values);

  // std::cout << "size: " << entities_values.first.size() << " " << entities_values.second.size() << std::endl;
  
  // std::vector<std::int32_t> local_indices(
  //   cell_tags->indices().begin(), cell_tags->indices().end());

  // cell_tags = std::make_shared<mesh::MeshTags<TagT>>(
  //     cell_tags->topology(),
  //     cell_tags->dim(),
  //     local_indices, // All local cells (ghost cells have no data)
  //     entities_values.second
  // );

  // const graph::AdjacencyList<std::int32_t> entities_adj
  //     = graph::regular_adjacency_list(std::move(entities_values.first),
  //                                     dofmap.extent(1));
  // mesh::MeshTags meshtags = mesh::create_meshtags(
  //     mesh.topology(), tdim, entities_adj,
  //     std::span<const TagT>(entities_values.second));

  // std::cout << "number of resulting meshtags: " << meshtags.values().size() << std::endl;

  // cell_tags = std::make_shared<mesh::MeshTags<TagT>>(meshtags);
  
  // (local) Map between original dofmap to a dofmap with mapped vertices
  // Note that the ordering of the cells is not the same as in the new_dofmap, but they each cell as the same indices.
  spdlog::info("-- Building cells to cells map... --");

  // make a key from any number of sorted indices
  auto make_key = [&](const std::vector<std::int32_t>& A) {
    std::ostringstream oss;
    for (size_t i = 0; i < A.size(); ++i) {
      oss << ' ';
      oss << A[i];
    }
    return oss.str();
  };


  std::unordered_map<std::string, std::int32_t> sig2cell;
  sig2cell.reserve(new_dofmap.extent(0));
  std::vector<std::int32_t> sig(new_dofmap.extent(1));

  spdlog::info("Creating cells hashmap");
  for (std::size_t nc = 0; nc < new_dofmap.extent(0); ++nc) {
    auto vd = std::submdspan(new_dofmap, nc, std::full_extent);
    for (int i = 0; i < new_dofmap.extent(1); ++i)
      sig[i] = new_vertex_to_original_global_vertex_map[vd(i)];
    std::sort(sig.begin(), sig.end());
    sig2cell[make_key(sig)] = nc;
  }

  spdlog::info("Matching original to new...");
  std::vector<std::int32_t> orig_cell_to_new_cell(dofmap.extent(0), std::int32_t(-1));

  for (std::size_t c = 0; c < dofmap_global_span.extent(0); ++c) {
    auto vd = std::submdspan(dofmap_global_span, c, std::full_extent);
    std::vector<std::int32_t> sig(dofmap_global_span.extent(1));
    for (size_t i = 0; i < dofmap_global_span.extent(1); ++i)
      sig[i] = vd(i);

    std::sort(sig.begin(), sig.end());
    auto it = sig2cell.find(make_key(sig));
    if (!(it == sig2cell.end())) // We only care about local cells
      orig_cell_to_new_cell[c] = it->second;
      // throw std::runtime_error("No matching cell for original cell " + std::to_string(c));
  }
  spdlog::info("-- Done! --");
  
  // Map tags
  auto original_imap_cells = mesh.topology()->index_map(tdim);
  auto new_imap_cells = new_mesh.topology()->index_map(tdim);

  auto original_local_cell_tag_indices = cell_tags->indices();
  auto original_cell_tag_values = cell_tags->values();
  const int N_cell_tags = original_local_cell_tag_indices.size();
  std::vector<std::int64_t> original_global_cell_tag_indices(N_cell_tags);
  original_imap_cells->local_to_global(original_local_cell_tag_indices, original_global_cell_tag_indices);


  std::vector<std::int32_t> new_local_cell_tag_indices(N_cell_tags);
  std::vector<TagT> new_cell_tag_values(N_cell_tags);

  for (int i = 0; i < N_cell_tags; ++i) {
    std::int32_t orig_idx = original_local_cell_tag_indices[i];
    TagT orig_value = original_cell_tag_values[i];
    std::int64_t new_idx  = orig_cell_to_new_cell[orig_idx];
    if(new_idx == std::int64_t(-1))
      throw std::runtime_error("Missing tag");
    new_local_cell_tag_indices[i] = new_idx;
    new_cell_tag_values[i] = orig_value;
  }

  // new_imap_cells->global_to_local(new_global_cell_tag_indices, new_local_cell_tag_indices);


  // std::sort(new_cell_tags.begin(), new_cell_tags.end(),
  //         [](auto &a, auto &b){ return a.first < b.first; });

  // Unpack


  // for (size_t i = 0; i < N_cell_tags; ++i) {
  //   new_cell_tag_indices[i] = new_cell_tags[i].first;
  //   new_cell_tag_values[i] = std::move(new_cell_tags[i].second);
  // }

  cell_tags = std::make_shared<mesh::MeshTags<TagT>>(
      cell_tags->topology(),
      cell_tags->dim(),
      new_local_cell_tag_indices,
      new_cell_tag_values
  );
  
  spdlog::info("** NEW MESH num_ghosts_cells = {}",
               new_mesh.topology()->index_map(tdim)->num_ghosts());
  spdlog::info("** NEW MESH num_local_cells = {}",
               new_mesh.topology()->index_map(tdim)->size_local());
  return new_mesh;
}