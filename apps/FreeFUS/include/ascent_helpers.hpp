#pragma once

#include <ascent.hpp>
#include <conduit_blueprint.hpp>
#include <dolfinx.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ascent_h {

using namespace ascent;
using namespace conduit;
using namespace dolfinx;

static const std::unordered_map<mesh::CellType, std::string>
    dolfinx_celltype_to_blueprint = {{mesh::CellType::point, "point"},
                                     {mesh::CellType::interval, "line"},
                                     {mesh::CellType::triangle, "tri"},
                                     {mesh::CellType::quadrilateral, "quad"},
                                     {mesh::CellType::tetrahedron, "tet"},
                                     {mesh::CellType::hexahedron, "hex"},
                                     {mesh::CellType::prism, "prism"},
                                     {mesh::CellType::pyramid, "pyramid"}};

/// @brief Mesh the reference triangle assuming a lagrangian space structure
/// @param P
/// @return dof connectivity
std::vector<int> MeshRefTriangle(const int P) {
  switch (P) {
  case 1:
    return {0, 1, 2};
  case 2:
    return {0, 5, 4, 5, 1, 3, 3, 4, 5, 4, 3, 2};
  case 3:
    return {0, 7, 5, 7, 8, 9, 8, 1, 3, 9, 5, 7, 3, 9,
            8, 5, 9, 6, 9, 3, 4, 4, 6, 9, 6, 4, 2};
  case 4:
    return {3,  11, 1,  4,  5,  14, 8,  5,  2, 4,  13, 3, 6,  12, 7,  7,
            12, 14, 9,  12, 0,  12, 6,  0,  5, 8,  7,  5, 7,  14, 11, 13,
            10, 13, 11, 3,  12, 13, 14, 13, 4, 14, 13, 9, 10, 13, 12, 9};
  case 5:
    return {17, 3,  4,  3,  14, 1,  6,  20, 5,  10, 6,  2,  5,  19, 4,
            14, 17, 13, 17, 14, 3,  13, 16, 12, 17, 16, 13, 18, 9,  8,
            9,  18, 20, 10, 9,  20, 10, 20, 6,  15, 11, 12, 16, 15, 12,
            15, 7,  0,  11, 15, 0,  7,  15, 8,  15, 18, 8,  15, 19, 18,
            19, 15, 16, 19, 16, 17, 19, 17, 4,  18, 19, 20, 20, 19, 5};
  default:
    throw std::invalid_argument("MeshRefTriangle: unsupported P = " +
                                std::to_string(P));
  }
}


std::vector<int> MeshRefTetrahedron(const int P) {
  switch (P) {
  case 1:
    return {0, 1, 2, 3};
  case 2:
    return {8, 6, 4, 2, 7, 5, 4, 3, 9, 5, 6, 1, 9, 8, 4, 0,
            9, 5, 6, 4, 9, 8, 6, 4, 9, 7, 5, 4, 9, 7, 4, 0};
  case 3:
    return {13, 9,  4,  2, 11, 5,  7,  3,  15, 6,  8,  1,  18, 16, 6,  7,
            19, 16, 9,  8, 17, 16, 5,  4,  19, 15, 14, 6,  19, 16, 6,  8,
            19, 15, 6,  8, 19, 18, 16, 6,  19, 18, 16, 14, 19, 18, 14, 6,
            17, 13, 12, 9, 17, 16, 9,  4,  17, 13, 9,  4,  17, 19, 16, 9,
            17, 19, 12, 9, 17, 16, 5,  7,  17, 11, 5,  7,  17, 18, 11, 7,
            17, 18, 16, 7, 17, 10, 18, 11, 17, 14, 12, 0,  17, 18, 16, 14,
            17, 18, 14, 0, 17, 19, 16, 14, 17, 19, 14, 12, 17, 10, 18, 0};
  case 4:
    return {18, 12, 4,  2,  21, 7,  10, 1,  32, 22, 10, 11, 23, 5,  26, 4,  23,
            33, 12, 11, 15, 6,  9,  3,  29, 22, 7,  8,  24, 23, 34, 22, 24, 27,
            6,  5,  24, 30, 9,  8,  23, 26, 17, 4,  23, 18, 12, 17, 23, 18, 12,
            4,  23, 18, 17, 4,  23, 33, 12, 17, 23, 33, 26, 17, 29, 21, 7,  20,
            29, 21, 10, 20, 29, 22, 7,  10, 29, 21, 7,  10, 29, 32, 22, 20, 29,
            32, 22, 10, 29, 32, 10, 20, 31, 32, 33, 11, 31, 32, 22, 11, 31, 23,
            33, 11, 31, 23, 34, 33, 31, 23, 34, 22, 31, 23, 22, 11, 31, 32, 22,
            20, 31, 29, 22, 20, 31, 28, 34, 22, 31, 19, 28, 20, 31, 19, 28, 34,
            31, 19, 32, 20, 31, 29, 28, 22, 31, 29, 28, 20, 16, 19, 28, 0,  16,
            19, 28, 34, 16, 28, 25, 34, 16, 13, 28, 25, 16, 13, 28, 0,  16, 31,
            19, 34, 16, 33, 26, 17, 16, 25, 34, 26, 16, 23, 34, 26, 16, 23, 33,
            26, 16, 23, 34, 33, 16, 31, 34, 33, 24, 29, 22, 8,  24, 28, 34, 22,
            24, 29, 28, 22, 24, 29, 28, 8,  24, 30, 28, 8,  24, 23, 5,  26, 24,
            25, 5,  26, 24, 25, 34, 26, 24, 23, 34, 26, 24, 27, 25, 5,  24, 15,
            9,  14, 24, 15, 6,  14, 24, 15, 6,  9,  24, 27, 6,  14, 24, 30, 9,
            14, 24, 28, 25, 34, 24, 27, 25, 14, 24, 13, 28, 25, 24, 13, 25, 14,
            24, 13, 28, 14, 24, 30, 28, 14};
  default:
    throw std::invalid_argument("MeshRefTets: unsupported P = " +
                                std::to_string(P));
  }
}

template <typename T>
void MeshToBlueprintMesh(std::shared_ptr<const fem::FunctionSpace<T>> V, conduit::Node &out) {
  // Topology: get connectivity array
  auto topology =  V->mesh()->topology();
  const int tdim = topology->dim();
  std::vector<int> conn = topology->connectivity(tdim, 0)->array();
  std::cout << "local cells in insitu output: " << conn.size() / 4 << std::endl;

  // Geometry: get coordinates
  std::span<const T> coords = V->mesh()->geometry().x();
  const int n_coords = coords.size() / 3;
  std::vector<T> X(n_coords), Y(n_coords), Z(n_coords);
  for (int i = 0; i < n_coords; ++i) {
    X[i] = coords[3 * i];
    Y[i] = coords[3 * i + 1];
    Z[i] = coords[3 * i + 2];
  }

  // Fill Conduit node for Blueprint mesh
  out["coordsets/coords/type"] = "explicit";
  out["coordsets/coords/values/x"].set(X.data(), n_coords);
  out["coordsets/coords/values/y"].set(Y.data(), n_coords);
  out["coordsets/coords/values/z"].set(Z.data(), n_coords);

  out["topologies/mesh/type"] = "unstructured";
  out["topologies/mesh/coordset"] = "coords";
  auto it = dolfinx_celltype_to_blueprint.find(topology->cell_type());
  if (it == dolfinx_celltype_to_blueprint.end())
    throw std::runtime_error(
        "Unknown cell type in dolfinx_celltype_to_blueprint mapping");
  out["topologies/mesh/elements/shape"] = it->second;
  out["topologies/mesh/elements/connectivity"].set(conn.data(), conn.size());

  Node verify_info;
  if(!blueprint::mesh::verify(out, verify_info))
  {
      std::cout << "Mesh Verify failed!" << std::endl;
      std::cout << verify_info.to_yaml() << std::endl;
  } else {
      std::cout << "Mesh verify success!" << std::endl;
  }
}

template <typename T>
void MeshToBlueprintMesh(std::shared_ptr<fem::FunctionSpace<T>> V, const int P,
                         conduit::Node &out) {
  // Shape: (num_dofs, 3)
  std::vector<T> coords = V->tabulate_dof_coordinates(false);
  const int n_coords = coords.size() / 3;
  std::vector<T> X(n_coords), Y(n_coords), Z(n_coords);
  for (int i = 0; i < n_coords; ++i) {
    X[i] = coords[3 * i];
    Y[i] = coords[3 * i + 1];
    Z[i] = coords[3 * i + 2];
  }

  // Fill Conduit node for Blueprint mesh
  out["coordsets/coords/type"] = "explicit";
  out["coordsets/coords/values/x"].set(X.data(), n_coords);
  out["coordsets/coords/values/y"].set(Y.data(), n_coords);
  out["coordsets/coords/values/z"].set(Z.data(), n_coords);

  out["topologies/mesh/type"] = "unstructured";
  out["topologies/mesh/coordset"] = "coords";

  std::shared_ptr<const mesh::Topology> topology = V->mesh()->topology();
  auto it = dolfinx_celltype_to_blueprint.find(topology->cell_type());
  if (it == dolfinx_celltype_to_blueprint.end())
    throw std::runtime_error(
        "Unknown cell type in dolfinx_celltype_to_blueprint mapping");
  out["topologies/mesh/elements/shape"] = it->second;

  // Connectivity
  const int tdim = topology->dim();
  const int num_local_cells = topology->index_map(tdim)->size_local();

  // Ref  connectivity
  std::vector<int> local_connectivity;
  if (tdim == 2) {
    local_connectivity = MeshRefTriangle(P);
  } else {
    local_connectivity = MeshRefTetrahedron(P);
  }
  const int N = local_connectivity.size();
  std::vector<int> global_connectivity(N * num_local_cells);

  std::shared_ptr<const fem::DofMap> dofmap = V->dofmap();

  for (int i = 0; i < num_local_cells; ++i) {
    std::span<const std::int32_t> global_dofs = dofmap->cell_dofs(i);
    for (int k = 0; k < N; ++k) {
      global_connectivity[i * N + k] = global_dofs[local_connectivity[k]];
    }
  }

  out["topologies/mesh/elements/connectivity"].set(global_connectivity.data(),
                                                   global_connectivity.size());

  Node verify_info;
  if(!blueprint::mesh::verify(out, verify_info))
  {
      std::cout << "Mesh Verify failed!" << std::endl;
      std::cout << verify_info.to_yaml() << std::endl;
  } else {
      std::cout << "Mesh verify success!" << std::endl;
  }
}

template <typename T>
void DG0FunctionToBlueprintField(std::shared_ptr<fem::Function<T>> f,
                                 conduit::Node &out,
                                 const std::string &field_name) {
  std::span<T> values = f->x()->mutable_array();
  out["fields"][field_name]["association"] = "element"; // DG
  out["fields"][field_name]["topology"] = "mesh";
  out["fields"][field_name]["values"].set_external(values.data(),
                                                   values.size());
}

template <typename T>
void CG1FunctionToBlueprintField(std::shared_ptr<fem::Function<T>> f,
                                 conduit::Node &out,
                                 const std::string &field_name) {
  std::span<T> values = f->x()->mutable_array();
  out["fields"][field_name]["association"] = "vertex"; // CG1
  out["fields"][field_name]["topology"] = "mesh";
  out["fields"][field_name]["values"].set_external(values.data(),
                                                   values.size());
}


template <typename T>
void FunctionToBlueprintField(std::shared_ptr<fem::Function<T>> f,
                              conduit::Node &out,
                              const std::string &field_name) {
  std::span<T> values = f->x()->mutable_array();
  out["fields"][field_name]["association"] = "vertex";
  out["fields"][field_name]["topology"] = "mesh";
  out["fields"][field_name]["values"].set_external(values.data(),
                                                   values.size());
  //   out["fields"][field_name]["values"].set(values.data(), values.size());
}

} // namespace ascent_h
