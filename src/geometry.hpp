#pragma once

#include <vector>
#include <dolfinx/mesh/Mesh.h>

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
            for (std::size_t j = 0; j < gdim; ++j) {
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
    std::mdspan<T, std::dextents<std::size_t, 2>> J(J_b.data(), tdim, gdim);
    std::vector<T> detJ_b(nc * nq);
    std::mdspan<T, std::dextents<std::size_t, 2>> detJ(detJ_b.data(), nc, nq);
    std::vector<T> det_scratch(2 * tdim * gdim);

    for (std::size_t c = 0; c < nc; ++c)
    {
        // Get cell geometry (coordinates dofs)
        for (std::size_t i = 0; i < x_dofmap.extent(1); ++i)
        {
            for (std::size_t j = 0; j < gdim; ++j) {
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
