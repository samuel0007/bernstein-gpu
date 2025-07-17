#pragma once

#include "kernels_stiffness.hpp"
#include "util.hpp"
#include "vector.hpp"
#include <dolfinx/mesh/Mesh.h>
#include <vector>

/// Compute the (symmetric) geometry factor for the stiffness operator
/// ([cell][G][point]) This runs on the GPU.
/// @param[in] mesh The mesh object
/// @param[in] points The quadrature points to compute Jacobian of the map
template <typename T, typename U, int nq>
std::span<T> compute_stiffness_geometry_triangle_GPU(
    std::shared_ptr<dolfinx::mesh::Mesh<U>> mesh, std::vector<U> points,
    std::vector<U> weights, thrust::device_vector<T> &G_entity) {
  const std::size_t tdim = mesh->topology()->dim();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t nc = mesh->topology()->index_map(tdim)->size_local() +
                         mesh->topology()->index_map(tdim)->num_ghosts();
  assert(weights.size() == nq);
  G_entity.resize(nc * 3 * nq);

  const fem::CoordinateElement<U> &cmap = mesh->geometry().cmap();
  auto xdofmap = mesh->geometry().dofmap();

  // Geometry dofmap
  thrust::device_vector<std::int32_t> xdofmap_d;
  auto xdofmap_d_span = copy_to_device(xdofmap.data_handle(),
                                       xdofmap.data_handle() + xdofmap.size(),
                                       xdofmap_d, "geometry dofmap");

  // Geometry coordinates
  thrust::device_vector<T> xgeom_d;
  auto xgeom_d_span =
      copy_to_device(mesh->geometry().x().begin(), mesh->geometry().x().end(),
                     xgeom_d, "geometry coordinates");

  // Evalute dphi at quadrature points
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, nq);
  std::vector<U> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmap.tabulate(1, points, {nq, gdim}, phi_b);

  thrust::device_vector<T> dphi_d;
  auto dphi_d_span = copy_to_device(phi_b.begin() + phi_b.size() / (tdim + 1),
                                    phi_b.end(), dphi_d, "dphi geom");

  thrust::device_vector<T> weights_d;
  auto weights_d_span =
      copy_to_device(weights.begin(), weights.end(), weights_d, "weights");

  kernels::stiffness::geometry_computation_tri<T, nq><<<nc, nq>>>(
      thrust::raw_pointer_cast(G_entity.data()), xgeom_d_span.data(),
      xdofmap_d_span.data(), dphi_d_span.data(), weights_d_span.data());

  return std::span<T>(thrust::raw_pointer_cast(G_entity.data()),
                      G_entity.size());
}

/// Compute the (symmetric) geometry factor for the stiffness operator
/// ([cell][G][point]) This runs on the GPU.
/// @param[in] mesh The mesh object
/// @param[in] points The quadrature points to compute Jacobian of the map
template <typename T, typename U>
std::span<T> compute_stiffness_geometry_triangle_GPU(
    std::shared_ptr<dolfinx::mesh::Mesh<U>> mesh, std::vector<U> points,
    thrust::device_vector<T> &G_entity) {
  const std::size_t tdim = mesh->topology()->dim();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t nc = mesh->topology()->index_map(tdim)->size_local() +
                         mesh->topology()->index_map(tdim)->num_ghosts();
  const std::size_t nq = points.size() / tdim;
  G_entity.resize(nc * 3 * nq);

  const fem::CoordinateElement<U> &cmap = mesh->geometry().cmap();
  auto xdofmap = mesh->geometry().dofmap();

  // Geometry dofmap
  thrust::device_vector<std::int32_t> xdofmap_d;
  auto xdofmap_d_span = copy_to_device(xdofmap.data_handle(),
                                       xdofmap.data_handle() + xdofmap.size(),
                                       xdofmap_d, "geometry dofmap");

  // Geometry coordinates
  thrust::device_vector<T> xgeom_d;
  auto xgeom_d_span =
      copy_to_device(mesh->geometry().x().begin(), mesh->geometry().x().end(),
                     xgeom_d, "geometry coordinates");

  // Evalute dphi at quadrature points
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, nq);
  std::vector<U> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmap.tabulate(1, points, {nq, gdim}, phi_b);

  thrust::device_vector<T> dphi_d;
  auto dphi_d_span = copy_to_device(phi_b.begin() + phi_b.size() / (tdim + 1),
                                    phi_b.end(), dphi_d, "dphi geom");

  kernels::stiffness::geometry_computation_tri<T><<<nc, nq>>>(
      thrust::raw_pointer_cast(G_entity.data()), xgeom_d_span.data(),
      xdofmap_d_span.data(), dphi_d_span.data(), nq);

  return std::span<T>(thrust::raw_pointer_cast(G_entity.data()),
                      G_entity.size());
}

/// Compute the (symmetric) geometry factor for the stiffness operator
/// ([cell][G][point]) This runs on the GPU.
/// @param[in] mesh The mesh object
/// @param[in] points The quadrature points to compute Jacobian of the map
template <typename T, typename U, int nq>
std::span<T> compute_stiffness_geometry_tetrahedron_GPU(
    std::shared_ptr<dolfinx::mesh::Mesh<U>> mesh, std::vector<U> points,
    std::vector<U> weights, thrust::device_vector<T> &G_entity) {
  const std::size_t tdim = mesh->topology()->dim();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t nc = mesh->topology()->index_map(tdim)->size_local() +
                         mesh->topology()->index_map(tdim)->num_ghosts();
  assert(weights.size() == nq);
  G_entity.resize(nc * 6 * nq);

  const fem::CoordinateElement<U> &cmap = mesh->geometry().cmap();
  auto xdofmap = mesh->geometry().dofmap();

  // Geometry dofmap
  thrust::device_vector<std::int32_t> xdofmap_d;
  auto xdofmap_d_span = copy_to_device(xdofmap.data_handle(),
                                       xdofmap.data_handle() + xdofmap.size(),
                                       xdofmap_d, "geometry dofmap");

  // Geometry coordinates
  thrust::device_vector<T> xgeom_d;
  auto xgeom_d_span =
      copy_to_device(mesh->geometry().x().begin(), mesh->geometry().x().end(),
                     xgeom_d, "geometry coordinates");

  // Evalute dphi at quadrature points
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, nq);
  std::vector<U> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmap.tabulate(1, points, {nq, gdim}, phi_b);

  thrust::device_vector<T> dphi_d;
  auto dphi_d_span = copy_to_device(phi_b.begin() + phi_b.size() / (tdim + 1),
                                    phi_b.end(), dphi_d, "dphi geom");

  thrust::device_vector<T> weights_d;
  auto weights_d_span =
      copy_to_device(weights.begin(), weights.end(), weights_d, "weights");

  kernels::stiffness::geometry_computation_tet<T, nq><<<nc, nq>>>(
      thrust::raw_pointer_cast(G_entity.data()), xgeom_d_span.data(),
      xdofmap_d_span.data(), dphi_d_span.data(), weights_d_span.data());

  return std::span<T>(thrust::raw_pointer_cast(G_entity.data()),
                      G_entity.size());
}

/// Compute the (symmetric) geometry factor for the stiffness operator
/// ([cell][G][point]) This runs on the GPU.
/// @param[in] mesh The mesh object
/// @param[in] points The quadrature points to compute Jacobian of the map
template <typename T, typename U>
std::span<T> compute_stiffness_geometry_tetrahedron_GPU(
    std::shared_ptr<dolfinx::mesh::Mesh<U>> mesh, std::vector<U> points,
    thrust::device_vector<T> &G_entity) {
  const std::size_t tdim = mesh->topology()->dim();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t nc = mesh->topology()->index_map(tdim)->size_local() +
                         mesh->topology()->index_map(tdim)->num_ghosts();
  const std::size_t nq = points.size() / tdim;
  G_entity.resize(nc * 6 * nq);

  const fem::CoordinateElement<U> &cmap = mesh->geometry().cmap();
  auto xdofmap = mesh->geometry().dofmap();

  // Geometry dofmap
  thrust::device_vector<std::int32_t> xdofmap_d;
  auto xdofmap_d_span = copy_to_device(xdofmap.data_handle(),
                                       xdofmap.data_handle() + xdofmap.size(),
                                       xdofmap_d, "geometry dofmap");

  // Geometry coordinates
  thrust::device_vector<T> xgeom_d;
  auto xgeom_d_span =
      copy_to_device(mesh->geometry().x().begin(), mesh->geometry().x().end(),
                     xgeom_d, "geometry coordinates");

  // Evalute dphi at quadrature points
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, nq);
  std::vector<U> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmap.tabulate(1, points, {nq, gdim}, phi_b);

  thrust::device_vector<T> dphi_d;
  auto dphi_d_span = copy_to_device(phi_b.begin() + phi_b.size() / (tdim + 1),
                                    phi_b.end(), dphi_d, "dphi geom");

  kernels::stiffness::geometry_computation_tet<T><<<nc, nq>>>(
      thrust::raw_pointer_cast(G_entity.data()), xgeom_d_span.data(),
      xdofmap_d_span.data(), dphi_d_span.data(), nq);

  return std::span<T>(thrust::raw_pointer_cast(G_entity.data()),
                      G_entity.size());
}