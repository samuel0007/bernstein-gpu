#pragma once

#include <vector>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/common/math.h>


using namespace dolfinx;
/// Compute the determinant of the Jacobian ([cell][point]): nc x nq
/// @param[in] mesh The mesh object (which contains the coordinate map)
/// @param[in] points The quadrature points to compute Jacobian of the map
template <typename T>
std::vector<T> compute_geometry(std::shared_ptr<dolfinx::mesh::Mesh<T>> mesh,
                                std::vector<T> points, int dim = 2)
{
    // Number of quadrature points
    std::size_t nq = points.size() / dim; // x and y

    // Get geometry data
    const fem::CoordinateElement<T> &cmap = mesh->geometry().cmap();
    auto x_dofmap = mesh->geometry().dofmap();
    const std::size_t num_dofs_g = cmap.dim();
    std::span<const T> x_g = mesh->geometry().x();

    // Get dimensions
    const std::size_t tdim = mesh->topology()->dim();
    const std::size_t gdim = mesh->geometry().dim();
    const std::size_t nc = mesh->topology()->index_map(tdim)->size_local() + mesh->topology()->index_map(tdim)->num_ghosts();

    // Tabulate basis functions at quadrature points
    std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, nq);
    std::vector<T> phi_b(std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
    std::mdspan<const T, std::dextents<std::size_t, 4>> phi(phi_b.data(), phi_shape);
    cmap.tabulate(1, points, {nq, gdim}, phi_b);

    // Create working arrays
    std::vector<T> coord_dofs_b(num_dofs_g * gdim);
    std::mdspan<T, std::dextents<std::size_t, 2>> coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);

    std::vector<T> J_b(tdim * gdim);
    std::mdspan<T, std::dextents<std::size_t, 2>> J(J_b.data(), tdim, gdim);
    std::vector<T> detJ_b(nc * nq);
    std::mdspan<T, std::dextents<std::size_t, 2>> detJ(detJ_b.data(), nc, nq);
    std::vector<T> det_scratch(2 * tdim * gdim);

    for (std::size_t c = 0; c < nc; ++c)
    {
        // Get cell geometry (coordinates dofs)
        for (std::size_t i = 0; i < x_dofmap.extent(1); ++i)
        {
            for (std::size_t j = 0; j < gdim; ++j)
            {
                coord_dofs(i, j) = x_g[3 * x_dofmap(c, i) + j];
            }
        }

        // Compute the scaled Jacobian determinant
        for (std::size_t q = 0; q < nq; ++q)
        {
            std::fill(J_b.begin(), J_b.end(), 0.0);

            // Get the derivatives at each quadrature points
            auto dphi = std::submdspan(phi, std::pair(1, tdim + 1), q, std::full_extent, 0);

            // Compute Jacobian matrix
            auto _J = std::submdspan(J, std::full_extent, std::full_extent);
            cmap.compute_jacobian(dphi, coord_dofs, _J);

            // Compute the determinant of the Jacobian
            detJ(c, q) = cmap.compute_jacobian_determinant(_J, det_scratch);
        }
    }

    return detJ_b;
}

/// Compute the determinant of the Jacobian ([cell][point]): nc x nq
/// @param[in] mesh The mesh object (which contains the coordinate map)
/// @param[in] points The quadrature points to compute Jacobian of the map
template <typename T>
std::vector<T> compute_geometry(std::shared_ptr<dolfinx::mesh::Mesh<T>> mesh,
                                std::vector<T> points, std::vector<T> weights)
{
    // Number of quadrature points
    std::size_t nq = weights.size();

    // Get geometry data
    const fem::CoordinateElement<T> &cmap = mesh->geometry().cmap();
    auto x_dofmap = mesh->geometry().dofmap();
    const std::size_t num_dofs_g = cmap.dim();
    std::span<const T> x_g = mesh->geometry().x();

    // Get dimensions
    const std::size_t tdim = mesh->topology()->dim();
    const std::size_t gdim = mesh->geometry().dim();
    const std::size_t nc = mesh->topology()->index_map(tdim)->size_local() + mesh->topology()->index_map(tdim)->num_ghosts();

    // Tabulate basis functions at quadrature points
    std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, nq);
    std::vector<T> phi_b(std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
    std::mdspan<const T, std::dextents<std::size_t, 4>> phi(phi_b.data(), phi_shape);
    cmap.tabulate(1, points, {nq, gdim}, phi_b);

    // Create working arrays
    std::vector<T> coord_dofs_b(num_dofs_g * gdim);
    std::mdspan<T, std::dextents<std::size_t, 2>> coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);

    std::vector<T> J_b(tdim * gdim);
    std::mdspan<T, std::dextents<std::size_t, 2>> J(J_b.data(), gdim, tdim);
    std::vector<T> detJ_b(nc * nq);
    std::mdspan<T, std::dextents<std::size_t, 2>> detJ(detJ_b.data(), nc, nq);
    std::vector<T> det_scratch(2 * tdim * gdim);

    for (std::size_t c = 0; c < nc; ++c)
    {
        // Get cell geometry (coordinates dofs)
        for (std::size_t i = 0; i < x_dofmap.extent(1); ++i)
        {
            for (std::size_t j = 0; j < gdim; ++j)
            {
                coord_dofs(i, j) = x_g[3 * x_dofmap(c, i) + j];
            }
        }

        // Compute the scaled Jacobian determinant
        for (std::size_t q = 0; q < nq; ++q)
        {
            std::fill(J_b.begin(), J_b.end(), 0.0);

            // Get the derivatives at each quadrature points
            auto dphi = std::submdspan(phi, std::pair(1, tdim + 1), q, std::full_extent, 0);

            // Compute Jacobian matrix
            auto _J = std::submdspan(J, std::full_extent, std::full_extent);
            cmap.compute_jacobian(dphi, coord_dofs, _J);

            // Compute the determinant of the Jacobian
            detJ(c, q) = cmap.compute_jacobian_determinant(_J, det_scratch);

            detJ(c, q) = std::fabs(detJ(c, q)) * weights[q];
        }
    }

    return detJ_b;
}

/// Compute the determinant of the Jacobian of the facets ([cell][face][point]): nc x n_topological_faces x nq
/// @param[in] mesh The mesh object
/// @param[in] cell_facet_data nf x (cell, local_facet)
/// @param[in] points The quadrature points to compute Jacobian of the map
template <typename T>
std::vector<T> compute_geometry_facets(std::shared_ptr<dolfinx::mesh::Mesh<T>> mesh,
                                       std::span<const int32_t> cell_facet_data,
                                       std::vector<T> points, std::vector<T> weights)
{
    // Get dimensions
    const std::size_t tdim = mesh->topology()->dim();
    const std::size_t gdim = mesh->geometry().dim();

    // Number of quadrature points
    std::size_t nq = weights.size();

    // Number of cells local to this processor
    const std::size_t nc = mesh->topology()->index_map(tdim)->size_local() + mesh->topology()->index_map(tdim)->num_ghosts();

    // Number of facets
    std::size_t nf = cell_facet_data.size() / 2;

    // Get geometry data
    const fem::CoordinateElement<T> &cmap = mesh->geometry().cmap();
    auto x_dofmap = mesh->geometry().dofmap();
    const std::size_t num_dofs_g = cmap.dim();
    std::span<const T> x_g = mesh->geometry().x();

    // Tabulate basis functions at quadrature points
    std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, nq);
    std::vector<T> phi_b(std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
    std::mdspan<const T, std::dextents<std::size_t, 4>> phi(phi_b.data(), phi_shape);
    cmap.tabulate(1, points, {nq, gdim}, phi_b);

    // Create working arrays
    fem::ElementDofLayout element_dof_layout = cmap.create_dof_layout();
    mesh::CellType cell_type = mesh->topology()->cell_type();
    int n_faces_topological = mesh::cell_num_entities(cell_type, tdim - 1);
    basix::cell::type b_cell_type = mesh::cell_type_to_basix_type(cell_type);

    std::vector<T> coord_dofs_b(num_dofs_g * gdim);
    std::mdspan<T, std::dextents<std::size_t, 2>> coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);

    std::vector<T> J_b(gdim * tdim);
    std::mdspan<T, std::dextents<std::size_t, 2>> J(J_b.data(), gdim, tdim);
    std::vector<T> J_facet_b(gdim * (tdim-1));
    std::mdspan<T, std::dextents<std::size_t, 2>> J_facet(J_facet_b.data(), gdim, tdim-1);

    std::vector<T> detJ_b(nc * n_faces_topological * nq);
    std::fill(detJ_b.begin(), detJ_b.end(), 0);
    std::mdspan<T, std::dextents<std::size_t, 3>> detJ(detJ_b.data(), nc, n_faces_topological, nq);
    std::vector<T> det_scratch(2 * tdim * gdim);

    auto [facets_jacobian_b, facets_jacobian_shape] = basix::cell::facet_jacobians<T>(b_cell_type);
    assert(facets_jacobian_shape[0] == n_faces_topological);
    assert(facets_jacobian_shape[1] == tdim && facets_jacobian_shape[2] == tdim - 1);
    std::mdspan<T, std::dextents<std::size_t, 3>> facets_jacobian(facets_jacobian_b.data(), n_faces_topological, tdim, tdim - 1);

    for (std::size_t f = 0; f < nf; ++f)
    {
        int32_t c = cell_facet_data[2 * f];
        int32_t local_facet = cell_facet_data[2 * f + 1];

        // std::vector<int> facet_dofs = element_dof_layout.entity_closure_dofs(tdim - 1, local_facet);
        auto reference_facet_jacobian = std::submdspan(facets_jacobian, local_facet, std::full_extent, std::full_extent);

        // Get cell geometry (coordinates dofs)
        // for (std::size_t i = 0; i < ; facet_dofs.size()++i)
        for (std::size_t i = 0; i < x_dofmap.extent(1); ++i)
        {
            for (std::size_t j = 0; j < gdim; ++j)
            {
                coord_dofs(i, j) = x_g[3 * x_dofmap(c, i) + j];
            }
        }

        // Compute the scaled Gram determinant for exterior facets
        for (std::size_t q = 0; q < nq; ++q)
        {
            std::fill(J_b.begin(), J_b.end(), 0.0);
            std::fill(J_facet_b.begin(), J_facet_b.end(), 0.0);

            // Get the derivatives at each quadrature points
            auto dphi = std::submdspan(phi, std::pair(1, tdim+1), q, std::full_extent, 0);

            // Compute Jacobian matrix
            cmap.compute_jacobian(dphi, coord_dofs, J);

            // J (cell jacobian) gdim x tdim
            // reference_facet jacobian, tdim x (tdim-1)
            // J_facet -> gdim x (tdim - 1)
            math::dot(J, reference_facet_jacobian, J_facet, false);

            // Compute the determinant of the gram matrix
            detJ(c, local_facet, q) = cmap.compute_jacobian_determinant(J_facet, det_scratch);

            detJ(c, local_facet, q) = std::fabs(detJ(c, local_facet, q)) * weights[q];
        }
    }

    return detJ_b;
}

/// @brief Creates a map from local face index to closure of dofs living on that face ([face][dof])
/// @tparam T 
/// @param dof_layout  
/// @param mesh 
/// @param tdim 
/// @return faces_to_dofs vector, size n_faces * n_faces_dofs
template <typename T>
std::vector<int32_t> make_faces_to_dofs_map(const fem::ElementDofLayout &dof_layout,
                                          const std::shared_ptr<mesh::Mesh<T>> &mesh,
                                          int tdim)
{
    mesh::CellType cell_type = mesh->topology()->cell_type();
    int n_faces = mesh::cell_num_entities(cell_type, tdim - 1);
    int n_face_dofs = dof_layout.num_entity_closure_dofs(tdim - 1);

    std::vector<int32_t> f_to_v(n_faces * n_face_dofs);
    for (int face = 0; face < n_faces; ++face)
    {
        auto closure = dof_layout.entity_closure_dofs(tdim - 1, face);
        for (int i = 0; i < n_face_dofs; ++i)
            f_to_v[face * n_face_dofs + i] = closure[i];
    }
    return f_to_v;
}


// Map points from a reference facet to a physical facet of a given cell type.
template <typename T>
std::vector<T> map_facet_points(
    std::vector<T> const& points,  // flattened (nq × nd)
    int facet,                     // local facet index
    mesh::CellType cell_type)
{
    basix::cell::type b_cell_type = mesh::cell_type_to_basix_type(cell_type);
    int tdim = basix::cell::topological_dimension(b_cell_type);

    auto [vertices, vshape] = basix::cell::geometry<T>(b_cell_type);
    int nverts = vshape[0];
    int gdim   = vshape[1];

    // Get facet→vertex connectivity
    auto topo = basix::cell::sub_entity_connectivity(b_cell_type);
    const std::vector<int>& facet_verts = topo[tdim - 1][facet][0];  // list of vertex indices

    std::vector<std::vector<T>> F(nverts, std::vector<T>(gdim));
    for(int i = 0; i < facet_verts.size(); ++i) {
        for(int j = 0; j < gdim; ++j) {
            F[i][j] = vertices[gdim*facet_verts[i] + j];
        }
    }

    int nd = tdim - 1;  // dim of facet
    int nq = points.size() / nd;  // number of quadrature points

    // result: (nq × gdim)
    std::vector<T> result;
    result.reserve(nq * gdim);

    for (int q = 0; q < nq; ++q)
    {
        // pointer to the qp-th reference point
        T const* p = &points[q * nd];

        // affine map: X = F[0] + Σ_k (F[k+1] - F[0]) * p[k]
        for (int d = 0; d < gdim; ++d)
        {
            T x = F[0][d];
            for (int k = 0; k < nd; ++k)
                x += (F[k+1][d] - F[0][d]) * p[k];
            result.push_back(x);
        }
    }
    return result;
}

