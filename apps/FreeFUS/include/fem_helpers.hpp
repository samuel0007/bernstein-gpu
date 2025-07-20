#pragma once

#include <basix/finite-element.h>
#include <dolfinx.h>
#include <dolfinx/mesh/cell_types.h>
#include <memory>

namespace freefus {

template <typename T>
auto make_element_spaces(const std::shared_ptr<dolfinx::mesh::Mesh<T>> &mesh,
                         mesh::CellType cell_type,
                         basix::element::lagrange_variant lvariant,
                         int degree) {
  namespace be = basix::element;
  namespace bc = basix::cell;

  basix::cell::type b_cell_type = cell_type_to_basix_type(cell_type);

  auto el = basix::create_element<T>(be::family::P, b_cell_type, degree,
                                     lvariant, be::dpc_variant::unset, false);

  auto el_DG = basix::create_element<T>(be::family::P, b_cell_type, 0, lvariant,
                                        be::dpc_variant::unset, true);

  auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
      mesh, std::make_shared<const fem::FiniteElement<T>>(el)));

  auto V_DG = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
      mesh, std::make_shared<const fem::FiniteElement<T>>(el_DG)));

  spdlog::info("V Local dofs: {}, Global Dofs: {}", V->dofmap()->index_map->size_local(), V->dofmap()->index_map->size_global());
  spdlog::info("V_DG Local dofs: {}, Global Dofs: {}", V_DG->dofmap()->index_map->size_local(), V_DG->dofmap()->index_map->size_global());

  return std::make_tuple(el, el_DG, V, V_DG);
}

template <typename T>
auto make_output_spaces(const std::shared_ptr<mesh::Mesh<T>> &mesh_ptr,
                        mesh::CellType cell_type, int polynomial_degree) {

  basix::cell::type b_cell_type = cell_type_to_basix_type(cell_type);
  // Output space
  basix::FiniteElement lagrange_element = basix::create_element<T>(
      basix::element::family::P, b_cell_type, polynomial_degree + 2, // "refine twice the mesh"
      basix::element::lagrange_variant::gll_warped,
      basix::element::dpc_variant::unset, false);

  auto V_out =
      std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
          mesh_ptr,
          std::make_shared<const fem::FiniteElement<T>>(lagrange_element)));

  auto u_out = std::make_shared<fem::Function<T>>(V_out);

  return std::make_tuple(V_out, u_out);
}
} // namespace freefus
