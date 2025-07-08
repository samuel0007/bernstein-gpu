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
    mass_action_ptr = std::make_unique<MassAction>(mesh, V, alpha_mass);
    stiffness_action_ptr =
        std::make_unique<StiffnessAction>(mesh, V, alpha_stiffness);

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
class LinearImplicit {
  using Func_ptr = std::shared_ptr<fem::Function<U>>;
  using MassAction = MassAction<T, U, P, Q, D>;
  using StiffnessAction = StiffnessAction<T, U, P, Q, D>;
  using ExteriorMassAction = ExteriorMassAction<T, U, P, Q, D>;

public:
  /// @brief TODO this is not very dry. We could split "operator provider" and
  /// simply a Model which combines these operators (and calls somt like
  /// operator_provider.init())
  /// @param spaces
  /// @param mesh
  /// @param rho0
  /// @param c0
  /// @param facet_domains
  LinearImplicit(auto spaces, std::shared_ptr<mesh::Mesh<U>> mesh,
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
      alpha_stiffness[i] = 1. / rho0_[i];
      alpha_exterior1[i] = 1. / rho0_[i];
      alpha_exterior2[i] = 1. / (rho0_[i] * c0_[i]);
    }
    mass_action_ptr = std::make_unique<MassAction>(mesh, V, alpha_mass);
    stiffness_action_ptr =
        std::make_unique<StiffnessAction>(mesh, V, alpha_stiffness);

    exterior_mass_action1_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[0], alpha_exterior1);
    exterior_mass_action2_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[1], alpha_exterior2);
  };

  // LHS: (M + C*gamma*dt + K*beta*dt2) @ in
  template <typename Vector> void operator()(Vector &in, Vector &out) {
    out.set(0.);
    (*stiffness_action_ptr)(in, out, beta *m_dt *m_dt);
    (*exterior_mass_action2_ptr)(in, out, gamma *m_dt);
    (*mass_action_ptr)(in, out);
  };

  // Note: I really dislike this interface.
  void set_dt(U dt) { m_dt = dt; }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    diag_inv.set(0.);
    mass_action_ptr->get_diag(diag_inv);
    exterior_mass_action2_ptr->get_diag(diag_inv, gamma * m_dt);
    stiffness_action_ptr->get_diag(diag_inv, beta * m_dt * m_dt);
    acc::inverse(diag_inv);
  }

  template <typename Vector>
  void rhs(Vector &u, Vector &ud, Vector &g, Vector &out) {
    out.set(0.);
    (*stiffness_action_ptr)(u, out, -1);
    (*exterior_mass_action2_ptr)(ud, out, -1);
    (*exterior_mass_action1_ptr)(g, out);
  }

private:
  static constexpr double beta = 0.25;
  static constexpr double gamma = 0.5;
  std::unique_ptr<MassAction> mass_action_ptr;
  std::unique_ptr<StiffnessAction> stiffness_action_ptr;
  std::unique_ptr<ExteriorMassAction> exterior_mass_action1_ptr;
  std::unique_ptr<ExteriorMassAction> exterior_mass_action2_ptr;
  double m_dt;
};

template <typename T, typename U, int P, int Q, int D>
class LinearLossyImplicit {
  using Func_ptr = std::shared_ptr<fem::Function<U>>;
  using MassAction = MassAction<T, U, P, Q, D>;
  using StiffnessAction = StiffnessAction<T, U, P, Q, D>;
  using ExteriorMassAction = ExteriorMassAction<T, U, P, Q, D>;

public:
  /// @brief TODO this is not very dry. We could split "operator provider" and
  /// simply a Model which combines these operators (and calls somt like
  /// operator_provider.init())
  /// @param spaces
  /// @param mesh
  /// @param rho0
  /// @param c0
  /// @param facet_domains
  LinearLossyImplicit(auto spaces, std::shared_ptr<mesh::Mesh<U>> mesh,
                      Func_ptr rho0, Func_ptr c0, Func_ptr delta0,
                      std::vector<std::vector<std::int32_t>> facet_domains) {
    auto [el, el_DG, V, V_DG] = spaces;

    std::span<U> c0_ = c0->x()->mutable_array();
    std::span<U> rho0_ = rho0->x()->mutable_array();
    std::span<U> delta0_ = delta0->x()->mutable_array();
    const int ncells = c0_.size();

    std::vector<U> alpha_M(ncells);
    std::vector<U> alpha_Mgamma(ncells);
    std::vector<U> alpha_C(ncells);
    std::vector<U> alpha_Cgamma(ncells);
    std::vector<U> alpha_K(ncells);
    std::vector<U> alpha_F(ncells);
    std::vector<U> alpha_Fd(ncells);

    for (std::size_t i = 0; i < ncells; ++i) {
      alpha_M[i] = 1. / (rho0_[i] * c0_[i] * c0_[i]);
      alpha_Mgamma[i] = delta0_[i] / (rho0_[i] * c0_[i] * c0_[i] * c0_[i]);

      alpha_C[i] = delta0_[i] / (rho0_[i] * c0_[i] * c0_[i]);
      alpha_Cgamma[i] = 1. / (rho0_[i] * c0_[i]);

      alpha_K[i] = 1. / rho0_[i];

      alpha_F[i] = 1. / rho0_[i];
      alpha_Fd[i] = delta0_[i] / (rho0_[i] * c0_[i] * c0_[i]);
    }

    M_action_ptr = std::make_unique<MassAction>(mesh, V, alpha_M);
    Mgamma_action_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[1], alpha_Mgamma);
    C_action_ptr = std::make_unique<StiffnessAction>(mesh, V, alpha_C);
    Cgamma_action_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[1], alpha_Cgamma);
    K_action_ptr = std::make_unique<StiffnessAction>(mesh, V, alpha_K);
    F_action_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[0], alpha_F);
    Fd_action_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[0], alpha_Fd);
  };

  // LHS: (M+M_Γ) + (C+C_Γ)*gamma*dt + K*beta*dt2) @ in
  template <typename Vector> void operator()(Vector &in, Vector &out) {
    out.set(0.);
    (*M_action_ptr)(in, out);
    (*Mgamma_action_ptr)(in, out);
    (*C_action_ptr)(in, out, gamma *m_dt);
    (*Cgamma_action_ptr)(in, out, gamma *m_dt);
    (*K_action_ptr)(in, out, beta *m_dt *m_dt);
  };

  // Note: I really dislike this interface.
  void set_dt(U dt) { m_dt = dt; }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    diag_inv.set(0.);
    M_action_ptr->get_diag(diag_inv);
    Mgamma_action_ptr->get_diag(diag_inv);
    C_action_ptr->get_diag(diag_inv, gamma * m_dt);
    Cgamma_action_ptr->get_diag(diag_inv, gamma * m_dt);
    K_action_ptr->get_diag(diag_inv, beta * m_dt * m_dt);
    acc::inverse(diag_inv);
  }

  // RHS: F - (C+C_Γ) @ ud - K @ u
  template <typename Vector>
  void rhs(Vector &u, Vector &ud, Vector &g, Vector &gd, Vector &out) {
    out.set(0.);
    (*F_action_ptr)(g, out);
    (*Fd_action_ptr)(gd, out);
    (*Cgamma_action_ptr)(ud, out, -1);
    (*C_action_ptr)(ud, out, -1);
    (*K_action_ptr)(u, out, -1);
  }

private:
  static constexpr double beta = 0.25;
  static constexpr double gamma = 0.5;
  std::unique_ptr<MassAction> M_action_ptr;
  std::unique_ptr<ExteriorMassAction> Mgamma_action_ptr;
  std::unique_ptr<StiffnessAction> C_action_ptr;
  std::unique_ptr<ExteriorMassAction> Cgamma_action_ptr;
  std::unique_ptr<StiffnessAction> K_action_ptr;
  std::unique_ptr<ExteriorMassAction> F_action_ptr;
  std::unique_ptr<ExteriorMassAction> Fd_action_ptr;
  double m_dt;
};

template <typename T, typename U, int P, int Q, int D>
class NonLinearLossyImplicit {
  using Func_ptr = std::shared_ptr<fem::Function<U>>;
  using MassAction = MassAction<T, U, P, Q, D>;
  using NonlinearMassAction = NonlinearMassAction<T, U, P, Q, D>;
  using StiffnessAction = StiffnessAction<T, U, P, Q, D>;
  using ExteriorMassAction = ExteriorMassAction<T, U, P, Q, D>;


public:
  /// @brief TODO this is not very dry. We could split "operator provider" and
  /// simply a Model which combines these operators (and calls somt like
  /// operator_provider.init())
  /// @param spaces
  /// @param mesh
  /// @param rho0
  /// @param c0
  /// @param facet_domains
  NonLinearLossyImplicit(auto spaces, std::shared_ptr<mesh::Mesh<U>> mesh,
                         Func_ptr rho0, Func_ptr c0, Func_ptr delta0,
                         Func_ptr b0,
                         std::vector<std::vector<std::int32_t>> facet_domains) {
    auto [el, el_DG, V, V_DG] = spaces;

    std::span<U> c0_ = c0->x()->mutable_array();
    std::span<U> rho0_ = rho0->x()->mutable_array();
    std::span<U> delta0_ = delta0->x()->mutable_array();
    std::span<U> b0_ = b0->x()->mutable_array();
    const int ncells = c0_.size();

    // alpha_Mp.resize(ncells);
    std::vector<U> alpha_Mp(ncells);
    std::vector<U> alpha_M(ncells);
    std::vector<U> alpha_Mgamma(ncells);
    std::vector<U> alpha_C(ncells);
    std::vector<U> alpha_Cgamma(ncells);
    std::vector<U> alpha_K(ncells);
    std::vector<U> alpha_F(ncells);
    std::vector<U> alpha_Fd(ncells);

    for (std::size_t i = 0; i < ncells; ++i) {
      const U c02 = c0_[i] * c0_[i];
      alpha_Mp[i] = -2. * b0_[i] / (rho0_[i] * rho0_[i] * c02 * c02);
      alpha_M[i] = 1. / (rho0_[i] * c02);
      alpha_Mgamma[i] = delta0_[i] / (rho0_[i] * c02 * c0_[i]);

      alpha_C[i] = delta0_[i] / (rho0_[i] * c02);
      alpha_Cgamma[i] = 1. / (rho0_[i] * c0_[i]);

      alpha_K[i] = 1. / rho0_[i];

      alpha_F[i] = 1. / rho0_[i];
      alpha_Fd[i] = delta0_[i] / (rho0_[i] * c02);
    }

    Mp_action_ptr = std::make_unique<NonlinearMassAction>(mesh, V, alpha_Mp);
    Mpdot_action_ptr = std::make_unique<NonlinearMassAction>(mesh, V, alpha_Mp);
    Mpdotdot_action_ptr = std::make_unique<MassAction>(mesh, V, alpha_Mp);

    M_action_ptr = std::make_unique<MassAction>(mesh, V, alpha_M);
    Mgamma_action_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[1], alpha_Mgamma);
    C_action_ptr = std::make_unique<StiffnessAction>(mesh, V, alpha_C);
    Cgamma_action_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[1], alpha_Cgamma);
    K_action_ptr = std::make_unique<StiffnessAction>(mesh, V, alpha_K);
    F_action_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[0], alpha_F);
    Fd_action_ptr = std::make_unique<ExteriorMassAction>(
        mesh, V, facet_domains[0], alpha_Fd);
  };

  template <typename Vector>
  void residual(Vector &u, Vector &ud, Vector &udd, Vector &g, Vector &gd,
                Vector &out) {
    out.set(0.);
    // - LHS
    // (M(u) + M + Mgamma) @ udd
    (*Mp_action_ptr)(udd, out, -1);
    (*M_action_ptr)(udd, out, -1);
    (*Mgamma_action_ptr)(udd, out, -1);
    // (M(ud) + C + Cgamma) @ ud
    (*Mpdot_action_ptr)(ud, out, -1);
    (*Cgamma_action_ptr)(ud, out, -1);
    (*C_action_ptr)(ud, out, -1);
    // K @ u
    (*K_action_ptr)(u, out, -1);
    // +RHS
    (*F_action_ptr)(g, out);
    (*Fd_action_ptr)(gd, out);
  };

  // - Jacobian:  - ((M+M_Γ) + (C+C_Γ)*gamma*dt + K*beta*dt2) @ in
  template <typename Vector> void operator()(Vector &in, Vector &out) {
    out.set(0.);
    // Linear Jacobian
    (*M_action_ptr)(in, out);
    (*Mgamma_action_ptr)(in, out);
    (*C_action_ptr)(in, out, gamma *m_dt);
    (*Cgamma_action_ptr)(in, out, gamma *m_dt);
    (*K_action_ptr)(in, out, beta *m_dt *m_dt);
    // Nonlinear part
    (*Mp_action_ptr)(in, out);
    (*Mpdot_action_ptr)(in, out, 2 *gamma *m_dt);
    (*Mpdotdot_action_ptr)(in, out, 2 *(beta *m_dt *m_dt + gamma*gamma *m_dt*m_dt));
  };

  // Note: I really dislike this interface.
  void set_dt(U dt) { m_dt = dt; }

  // Get diagonal inverse of jacobian
  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    diag_inv.set(0.);
    M_action_ptr->get_diag(diag_inv);
    Mgamma_action_ptr->get_diag(diag_inv);
    C_action_ptr->get_diag(diag_inv, gamma * m_dt);
    Cgamma_action_ptr->get_diag(diag_inv, gamma * m_dt);
    K_action_ptr->get_diag(diag_inv, beta * m_dt * m_dt);
   
    Mp_action_ptr->get_diag(diag_inv);
    Mpdot_action_ptr->get_diag(diag_inv, 2 *gamma *m_dt);
    Mpdotdot_action_ptr->get_diag(diag_inv, 2 *(beta *m_dt *m_dt + gamma*gamma *m_dt*m_dt));
    acc::inverse(diag_inv);
  }

  template <typename Vector>
  void init_coefficients(Vector &u, Vector &ud, Vector &udd) {
    update_coefficients(u, ud, udd);
  }

  template <typename Vector>
  void update_coefficients(Vector &u, Vector &ud, Vector &udd) {
    Mp_action_ptr->set_coefficient(u);
    Mpdot_action_ptr->set_coefficient(ud);
    // Mpdotdot_action_ptr->set_coefficient(udd);
  }

private:
  static constexpr U beta = 0.25;
  static constexpr U gamma = 0.5;

  std::unique_ptr<NonlinearMassAction> Mp_action_ptr;
  std::unique_ptr<NonlinearMassAction> Mpdot_action_ptr;
  std::unique_ptr<MassAction> Mpdotdot_action_ptr;

  // std::unique_ptr<MassAction> Mp_jacobian_action_ptr;
  std::unique_ptr<MassAction> M_action_ptr;
  std::unique_ptr<ExteriorMassAction> Mgamma_action_ptr;
  std::unique_ptr<StiffnessAction> C_action_ptr;
  std::unique_ptr<ExteriorMassAction> Cgamma_action_ptr;
  std::unique_ptr<StiffnessAction> K_action_ptr;
  std::unique_ptr<ExteriorMassAction> F_action_ptr;
  std::unique_ptr<ExteriorMassAction> Fd_action_ptr;

  // I'm not a fan of this interface
  // std::vector<U> alpha_Mp;
  // thrust::device_vector<T> alpha_static_coeff_Mp;

  double m_dt;
};

template <ModelType MT, typename T, typename U, int P, int Q, int D>
auto create_model(const auto &spaces, const auto &material_coefficients,
                  const MeshData<U> &mesh_data,
                  const PhysicalParameters<U> &params, ModelType model_type) {

  std::vector<std::vector<std::int32_t>> facet_domains;
  std::vector<int> ft_unique = {1, 2};
  const std::vector<std::int32_t> bfacets = mesh::exterior_facet_indices(*(mesh_data.mesh->topology()));
  for (int i = 0; i < ft_unique.size(); ++i) {
    int tag = ft_unique[i];
    std::vector<std::int32_t> facets_tags;
    if(tag == 2) {
      facets_tags = bfacets;
    } else {
      facets_tags = mesh_data.facet_tags->find(tag);
    }
    std::vector<std::int32_t> facet_domain =
    fem::compute_integration_domains(
        fem::IntegralType::exterior_facet,
        *(mesh_data.mesh->topology_mutable()),
        facets_tags);
    std::cout << std::format("Domain {}: {}\n", tag, facet_domain.size() /
    2); facet_domains.push_back(facet_domain);
  }

  // TODO
  // std::vector<std::int32_t> facets1 =
  //     mesh::locate_entities_boundary(*mesh_data.mesh, 2, [](auto x) {
  //       std::vector<std::int8_t> marker(x.extent(1), false);
  //       for (std::size_t p = 0; p < x.extent(1); ++p) {
  //         auto x0 = x(2, p);
  //         if (x0 > -0.002)
  //           marker[p] = true;
  //       }
  //       return marker;
  //     });

  // std::vector<std::int32_t> facets2 =
  //     mesh::locate_entities_boundary(*mesh_data.mesh, 2, [](auto x) {
  //       std::vector<std::int8_t> marker(x.extent(1), false);
  //       for (std::size_t p = 0; p < x.extent(1); ++p) {
  //         auto x0 = x(2, p);
  //         if (x0 <= -0.002)
  //           marker[p] = true;
  //       }
  //       return marker;
  //     });

  // std::vector<std::int32_t> facet_domain = fem::compute_integration_domains(
  //     fem::IntegralType::exterior_facet, *(mesh_data.mesh->topology_mutable()),
  //     facets1);
  // std::cout << std::format("Domain {}: {}\n", 1, facet_domain.size() / 2);
  // facet_domains.push_back(facet_domain);

  // facet_domain = fem::compute_integration_domains(
      // fem::IntegralType::exterior_facet, *(mesh_data.mesh->topology_mutable()),
      // facets2);
  // std::cout << std::format("Domain {}: {}\n", 2, facet_domain.size() / 2);
  // facet_domains.push_back(facet_domain);

  auto [c0, rho0, delta0, b0] = material_coefficients;

  if constexpr (MT == ModelType::LinearExplicit) {
    return std::make_unique<LinearExplicit<T, U, P, Q, D>>(
        spaces, mesh_data.mesh, rho0, c0, facet_domains);
  } else if constexpr (MT == ModelType::LinearImplicit) {
    return std::make_unique<LinearImplicit<T, U, P, Q, D>>(
        spaces, mesh_data.mesh, rho0, c0, facet_domains);
  } else if constexpr (MT == ModelType::LinearLossyImplicit) {
    return std::make_unique<LinearLossyImplicit<T, U, P, Q, D>>(
        spaces, mesh_data.mesh, rho0, c0, delta0, facet_domains);
  } else if constexpr (MT == ModelType::NonLinearLossyImplicit) {
    return std::make_unique<NonLinearLossyImplicit<T, U, P, Q, D>>(
        spaces, mesh_data.mesh, rho0, c0, delta0, b0, facet_domains);
  } else {
    static_assert(always_false_v<MT>, "Unsupported timestepping type");
  }
};

} // namespace freefus
