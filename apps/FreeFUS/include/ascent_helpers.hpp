#pragma once

#include <ascent.hpp>
#include <conduit_blueprint.hpp>
#include <dolfinx.h>
#include <vector>
#include <stdexcept>
#include <unordered_map>


using namespace ascent;
using namespace conduit;
using namespace dolfinx;

static const std::unordered_map<mesh::CellType, std::string> dolfinx_celltype_to_blueprint = {
    {mesh::CellType::point, "point"},
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
    switch(P) {
        case 1: return {0, 1, 2};
        case 2: return {0, 5, 4, 5, 1, 3, 3, 4, 5, 4, 3, 2};
        case 3: return {0, 7, 5, 7, 8, 9, 8, 1, 3, 9, 5, 7, 3, 9, 8, 5, 9, 6, 9, 3, 4, 4, 6, 9, 6, 4, 2};
        case 4: return {3, 11, 1, 4, 5, 14, 8, 5, 2, 4, 13, 3, 6, 12, 7, 7, 12, 14, 9, 12, 0, 12, 6, 0, 5, 8, 7, 5, 7, 14, 11, 13, 10, 13, 11, 3, 12, 13, 14, 13, 4, 14, 13, 9, 10, 13, 12, 9};
        case 5: return {17, 3, 4, 3, 14, 1, 6, 20, 5, 10, 6, 2, 5, 19, 4, 14, 17, 13, 17, 14, 3, 13, 16, 12, 17, 16, 13, 18, 9, 8, 9, 18, 20, 10, 9, 20, 10, 20, 6, 15, 11, 12, 16, 15, 12, 15, 7, 0, 11, 15, 0, 7, 15, 8, 15, 18, 8, 15, 19, 18, 19, 15, 16, 19, 16, 17, 19, 17, 4, 18, 19, 20, 20, 19, 5};
        default:
            throw std::invalid_argument("MeshRefTriangle: unsupported P = " + std::to_string(P));
    }
}

template <typename T>
void MeshToBlueprintMesh(std::shared_ptr<fem::FunctionSpace<T>> V, const int P, conduit::Node &out)
{
    // Shape: (num_dofs, 3)
    std::vector<T> coords = V->tabulate_dof_coordinates(false);
    const int n_coords = coords.size() / 3;
    std::vector<T> X(n_coords), Y(n_coords), Z(n_coords);
    for (int i = 0; i < n_coords; ++i)
    {
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
        throw std::runtime_error("Unknown cell type in dolfinx_celltype_to_blueprint mapping");
    out["topologies/mesh/elements/shape"] = it->second;


    // Connectivity
    const int tdim = topology->dim();
    const int num_local_cells = topology->index_map(tdim)->size_local();

    // Ref triangle connectivity
    std::vector<int> local_connectivity = MeshRefTriangle(P);
    const int P2 = P * P;
    assert(local_connectivity.size() == P2 * 3);
    // For triangles: one has a triangular mesh of P^2 triangles for each triangle
    std::vector<int> global_connectivity(P2 * 3 * num_local_cells);

    std::shared_ptr<const fem::DofMap> dofmap = V->dofmap();

    for(int i = 0; i < num_local_cells; ++i) {
        std::span<const std::int32_t> global_dofs = dofmap->cell_dofs(i);
        for(int k = 0; k < P2 * 3; ++k) {
            global_connectivity[i * P2 * 3 + k] = global_dofs[local_connectivity[k]];
        }
    }
    
    out["topologies/mesh/elements/connectivity"].set(global_connectivity.data(), global_connectivity.size());
}

template <typename T>
void DG0FunctionToBlueprintField(std::shared_ptr<fem::Function<T>> f,
                                 conduit::Node &out,
                                 const std::string &field_name)
{
    std::span<T> values = f->x()->mutable_array();
    out["fields"][field_name]["association"] = "element"; // DG
    out["fields"][field_name]["topology"] = "mesh";
    out["fields"][field_name]["values"].set_external(values.data(), values.size());
}

template <typename T>
void FunctionToBlueprintField(std::shared_ptr<fem::Function<T>> f,
                                 conduit::Node &out,
                                 const std::string &field_name)
{
    std::span<T> values = f->x()->mutable_array();
    out["fields"][field_name]["association"] = "vertex"; // CG1
    out["fields"][field_name]["topology"] = "mesh";
    out["fields"][field_name]["values"].set_external(values.data(), values.size());
}