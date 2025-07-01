#pragma once

#include "types.hpp"

namespace freefus {

/// @brief Linear Wave Explicit model
// a_linearwave = inner(u/rho0/c0/c0, v) * dx
// L_linearwave =
//     - inner(1/rho0*grad(u_n), grad(v)) * dx
//     + inner(1/rho0*g, v) * ds(1)
//     - inner(1/rho0/c0*v_n, v) * ds(2)
template <typename T, typename U, int P, int Q, int D> class LinearExplicit {
  using Func_ptr = std::shared_ptr<fem::Function<U>>;
  using MassAction = MassAction<T, U, P, Q, D>;
  using StiffnessAction = StiffnessAction<T, U, P, Q, D>;
  using ExteriorMassAction = ExteriorMassAction<T, U, P, Q, D>;

public:
  LinearExplicit(auto spaces, std::shared_ptr<mesh::Mesh<U>> mesh,
                 Func_ptr rho0, Func_ptr c0,
                 std::vector<std::vector<std::int32_t>> facet_domains) {
    auto [el, el_DG, V, V_DG] = spaces;


    std::span<U> c0_ = c0->x()->mutable_array();
    std::span<U> rho0_ = rho0->x()->mutable_array();
    const int ncells = c0_.size();

    std::vector<U> alpha_mass(ncells);
    std::vector<U> alpha_stiffness(ncells);
    std::vector<U> alpha_exterior1(ncells);
    std::vector<U> alpha_exterior2(ncells);

    for (std::size_t i = 0; i < ncells; ++i) {
      alpha_mass[i] = 1. / (rho0_[i] * c0_[i] * c0_[i]);
      alpha_stiffness[i] = -1. / rho0_[i];
      alpha_exterior1[i] = 1. / rho0_[i];
      alpha_exterior2[i] = -1. / (rho0_[i] * c0_[i]);
    }
    mass_action_ptr =
        std::make_unique<MassAction>(mesh, V, alpha_mass);
    stiffness_action_ptr = std::make_unique<StiffnessAction>(
        mesh, V, alpha_stiffness);

    exterior_mass_action1_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[0], alpha_exterior1);
    exterior_mass_action2_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[1], alpha_exterior2);
  };

  template <typename Vector> void operator()(Vector &in, Vector &out) {
    out.set(0.);
    (*mass_action_ptr)(in, out);
  };

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    mass_action_ptr->get_diag_inverse(diag_inv);
  }

  template <typename Vector>
  void rhs(Vector &u_n, Vector &v_n, Vector &g, Vector &out) {
    out.set(0.);
    (*stiffness_action_ptr)(u_n, out);
    (*exterior_mass_action1_ptr)(g, out);
    (*exterior_mass_action2_ptr)(v_n, out);
  }

private:
  std::unique_ptr<MassAction> mass_action_ptr;
  std::unique_ptr<StiffnessAction> stiffness_action_ptr;
  std::unique_ptr<ExteriorMassAction> exterior_mass_action1_ptr;
  std::unique_ptr<ExteriorMassAction> exterior_mass_action2_ptr;
};

template <typename T, typename U, int P, int Q, int D>
auto create_model(const auto &spaces, const auto &material_coefficients,
                  const MeshData<U> &mesh_data,
                  const PhysicalParameters<U> &params, ModelType model_type) {

  std::vector<std::vector<std::int32_t>> facet_domains;
  // std::vector<int> ft_unique = {1, 2};
  // for (int i = 0; i < ft_unique.size(); ++i) {
  //   int tag = ft_unique[i];
  //   std::vector<std::int32_t> facet_domain = fem::compute_integration_domains(
  //       fem::IntegralType::exterior_facet,
  //       *(mesh_data.mesh->topology_mutable()), mesh_data.facet_tags->find(tag));
  //   std::cout << std::format("Domain {}: {}\n", tag, facet_domain.size() / 2);
  //   facet_domains.push_back(facet_domain);
  // }
    
  // TODO
   std::vector<std::int32_t> facets1 = mesh::locate_entities_boundary(
        *mesh_data.mesh, 2,
    [](auto x)
    {
      std::vector<std::int8_t> marker(x.extent(1), false);
      for (std::size_t p = 0; p < x.extent(1); ++p)
      {
        auto x0 = x(2, p);
        if (x0 > -0.002)
          marker[p] = true;
      }
      return marker;
    });
  
   std::vector<std::int32_t> facets2 = mesh::locate_entities_boundary(
        *mesh_data.mesh, 2,
    [](auto x)
    {
      std::vector<std::int8_t> marker(x.extent(1), false);
      for (std::size_t p = 0; p < x.extent(1); ++p)
      {
        auto x0 = x(2, p);
        if (x0 <= -0.002)
          marker[p] = true;
      }
      return marker;
    });

  std::vector<std::int32_t> facet_domain = fem::compute_integration_domains(
        fem::IntegralType::exterior_facet,
        *(mesh_data.mesh->topology_mutable()), facets1);
    std::cout << std::format("Domain {}: {}\n", 1, facet_domain.size() / 2);
    facet_domains.push_back(facet_domain);

  facet_domain = fem::compute_integration_domains(
        fem::IntegralType::exterior_facet,
        *(mesh_data.mesh->topology_mutable()), facets2);
    std::cout << std::format("Domain {}: {}\n", 2, facet_domain.size() / 2);
    facet_domains.push_back(facet_domain);

  auto [rho0, c0] = material_coefficients;

  if (model_type == ModelType::LinearExplicit) {
    return std::make_unique<LinearExplicit<T, U, P, Q, D>>(spaces, mesh_data.mesh, rho0, c0,
                                      facet_domains);
  }
  throw std::runtime_error("Unsupport model type");
};

} // namespace freefus
