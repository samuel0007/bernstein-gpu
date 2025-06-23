#pragma once

#include "types.hpp"

namespace freefus {

/// @brief Linear Wave Explicit model
// a_linearwave = inner(u/rho0/c0/c0, v) * dx
// L_linearwave =
//     - inner(1/rho0*grad(u_n), grad(v)) * dx
//     + inner(1/rho0*g, v) * ds(1)
//     - inner(1/rho0/c0*v_n, v) * ds(2)
template <typename T, int P, int Q, int D> class LinearExplicit {
  using Func_ptr = std::shared_ptr<fem::Function<T>>;
  using MassAction = MassAction<T, P, Q, D>;
  using StiffnessAction = StiffnessAction<T, P, Q, D>;
  using ExteriorMassAction = ExteriorMassAction<T, P, Q, D>;

public:
  LinearExplicit(auto spaces, std::shared_ptr<mesh::Mesh<T>> mesh,
                 Func_ptr rho0, Func_ptr c0,
                 std::vector<std::vector<std::int32_t>> facet_domains) {
    auto [el, el_DG, V, V_DG] = spaces;

    Func_ptr alpha_mass = std::make_shared<fem::Function<T>>(V_DG);
    Func_ptr alpha_stiffness = std::make_shared<fem::Function<T>>(V_DG);
    Func_ptr alpha_exterior1 = std::make_shared<fem::Function<T>>(V_DG);
    Func_ptr alpha_exterior2 = std::make_shared<fem::Function<T>>(V_DG);

    std::span<T> c0_ = c0->x()->mutable_array();
    std::span<T> rho0_ = rho0->x()->mutable_array();

    std::span<T> alpha_mass_ = alpha_mass->x()->mutable_array();
    std::span<T> alpha_stiffness_ = alpha_stiffness->x()->mutable_array();
    std::span<T> alpha_exterior1_ = alpha_exterior1->x()->mutable_array();
    std::span<T> alpha_exterior2_ = alpha_exterior2->x()->mutable_array();

    const int ncells = alpha_mass_.size();

    for (std::size_t i = 0; i < ncells; ++i) {
      alpha_mass_[i] = 1. / (rho0_[i] * c0_[i] * c0_[i]);
      alpha_stiffness_[i] = -1. / rho0_[i];
      alpha_exterior1_[i] = 1. / rho0_[i];
      alpha_exterior2_[i] = -1. / (rho0_[i] * c0_[i]);
    }
    mass_action_ptr =
        std::make_unique<MassAction>(mesh, V, alpha_mass->x()->array());
    stiffness_action_ptr = std::make_unique<StiffnessAction>(
        mesh, V, alpha_stiffness->x()->array());

    exterior_mass_action1_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[0], alpha_exterior1->x()->array());
    exterior_mass_action2_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[1], alpha_exterior2->x()->array());
  };

  template <typename Vector> void operator()(const Vector &in, Vector &out) {
    out.set(0.);
    (*mass_action_ptr)(in, out);
  };

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    mass_action_ptr->get_diag_inverse(diag_inv);
  }

  template <typename Vector>
  void rhs(const Vector &u_n, const Vector &v_n, const Vector &g, Vector &out) {
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

template <typename T, int P, int Q, int D>
auto create_model(const auto &spaces, const auto &material_coefficients,
                  const MeshData<T> &mesh_data,
                  const PhysicalParameters<T> &params, ModelType model_type) {

  std::vector<std::vector<std::int32_t>> facet_domains;
  std::vector<int> ft_unique = {1, 2};
  for (int i = 0; i < ft_unique.size(); ++i) {
    int tag = ft_unique[i];
    std::vector<std::int32_t> facet_domain = fem::compute_integration_domains(
        fem::IntegralType::exterior_facet,
        *(mesh_data.mesh->topology_mutable()), mesh_data.facet_tags->find(tag));
    std::cout << std::format("Domain {}: {}\n", tag, facet_domain.size() / 2);
    facet_domains.push_back(facet_domain);
  }

  auto [rho0, c0] = material_coefficients;

  if (model_type == ModelType::LinearExplicit) {
    return std::make_unique<LinearExplicit<T, P, Q, D>>(spaces, mesh_data.mesh, rho0, c0,
                                      facet_domains);
  }
  throw std::runtime_error("Unsupport model type");
};

} // namespace freefus
