#pragma once

#include <array>
#include <chrono>

#include "types.hpp"
#include "vector.hpp"

namespace freefus {

template <typename U, typename Vector> class ExplicitRK4 {
public:
  ExplicitRK4(std::shared_ptr<fem::FunctionSpace<U>> V,
              const PhysicalParameters<U> &params, U source_sound_speed)
      : params(params), source_sound_speed(source_sound_speed) {

    auto index_map = V->dofmap()->index_map;
    auto bs = V->dofmap()->index_map_bs();
    assert(bs == 1 && "not implemented");

    g = std::make_unique<Vector>(index_map, bs);   // Source vector
    RHS = std::make_unique<Vector>(index_map, bs); // RHS vector

    u = std::make_unique<Vector>(index_map, bs);
    v = std::make_unique<Vector>(index_map, bs);
    u->set(0);
    v->set(0);
    ku = std::make_unique<Vector>(index_map, bs);
    kv = std::make_unique<Vector>(index_map, bs);
    u0 = std::make_unique<Vector>(index_map, bs);
    v0 = std::make_unique<Vector>(index_map, bs);
    ui = std::make_unique<Vector>(index_map, bs);
    vi = std::make_unique<Vector>(index_map, bs);
    ui->set(0);
    vi->set(0);
  };

  /// Evolve a solution under a model using a solver for a timestep dt
  void evolve(auto &model, auto &solver, U t, U dt) {
    copy_d_to_d(*u, *u0);
    copy_d_to_d(*v, *v0);

    for (int i = 0; i < 4; i++) {
      copy_d_to_d(*u0, *ui);
      copy_d_to_d(*v0, *vi);

      acc::axpy(*ui, dt * a[i], *ku, *ui);
      acc::axpy(*vi, dt * a[i], *kv, *vi);

      // RK time evaluation
      U tn = t + c[i] * dt;

      f0(tn, *ui, *vi, *ku);
      int solver_its = f1(tn, *ui, *vi, *kv, model, solver);
      spdlog::info("stage={} its={}", i, solver_its);

      acc::axpy(*u, dt * b[i], *ku, *u);
      acc::axpy(*v, dt * b[i], *kv, *v);
    }
  }

  void get_solution(auto &solution) {
    thrust::copy(u->thrust_vector().begin(), u->thrust_vector().end(),
                 solution->x()->mutable_array().begin());
  }

private:
  static constexpr std::array<U, 4> a = {0.0, 0.5, 0.5, 1.0};
  static constexpr std::array<U, 4> b = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0,
                                         1.0 / 6.0};
  static constexpr std::array<U, 4> c = {0.0, 0.5, 0.5, 1.0};

  PhysicalParameters<U> params;
  U source_sound_speed;

  std::unique_ptr<Vector> g;
  std::unique_ptr<Vector> RHS;

  std::unique_ptr<Vector> ku;
  std::unique_ptr<Vector> kv;
  std::unique_ptr<Vector> u0;
  std::unique_ptr<Vector> v0;
  std::unique_ptr<Vector> ui;
  std::unique_ptr<Vector> vi;

  std::unique_ptr<Vector> u;
  std::unique_ptr<Vector> v;

  void f0(U t, Vector &u, Vector &v, Vector &result) { copy_d_to_d(v, result); }

  int f1(U t, Vector &u, Vector &v, Vector &result, auto &model, auto &solver) {
    // Apply windowing
    U window;
    if (t < params.period * params.window_length) {
      window = 0.5 * (1.0 - cos(params.source_frequency * M_PI * t /
                                params.window_length));
    } else {
      window = 1.0;
    }

    // Update boundary condition
    const U homogeneous_source =
        window * params.source_amplitude * params.source_angular_frequency /
        source_sound_speed * cos(params.source_angular_frequency * t);
    g->set(homogeneous_source);

    model->rhs(u, v, *g, *RHS);

    return solver->solve(*model, result, *RHS, true);
  }
};

template <typename U, typename Vector, double beta = 0.25, double gamma = 0.5>
class Newmark {
public:
  Newmark(std::shared_ptr<fem::FunctionSpace<U>> V,
          const PhysicalParameters<U> &params, U source_sound_speed)
      : params(params), source_sound_speed(source_sound_speed) {

    auto index_map = V->dofmap()->index_map;
    auto bs = V->dofmap()->index_map_bs();
    assert(bs == 1 && "not implemented");

    g = std::make_unique<Vector>(index_map, bs);   // Source vector
    RHS = std::make_unique<Vector>(index_map, bs); // RHS vector

    u = std::make_unique<Vector>(index_map, bs);
    ud = std::make_unique<Vector>(index_map, bs);
    udd = std::make_unique<Vector>(index_map, bs);
    u->set(0);
    ud->set(0);
    udd->set(0);
  };

  /// Evolve a solution under a model using a solver for a timestep dt
  void evolve(auto &model, auto &solver, U t, U dt) {
    update_source(t + dt); // f(t + dt)
    const U dt2 = dt * dt;
    // Predictor
    // 1. Displacement: u = u + ud * dt + udd * (0.5 - beta) * dt2;
    acc::axpy(*u, dt, *ud, *u);
    acc::axpy(*u, (0.5 - beta) * dt2, *udd, *u);
    // 2. Velocity: ud = ud + udd * (1 - gamma) * dt;
    acc::axpy(*ud, (1 - gamma) * dt, *udd, *ud);

    // Solve LSE
    model->set_dt(dt);
    model->rhs(*u, *ud, *g, *RHS);
    int solver_its = solver->solve(*model, *udd, *RHS, true);
    spdlog::info("solver its={}", solver_its);

    // Correctors
    // 1. Displacement:  u = u + udd * beta * dt2
    acc::axpy(*u, beta * dt2, *udd, *u);
    // 2. Velocity: ud = ud + udd * gamma * dt
    acc::axpy(*ud, gamma * dt, *udd, *ud);
  }

  void get_solution(auto &solution) {
    thrust::copy(u->thrust_vector().begin(), u->thrust_vector().end(),
                 solution->x()->mutable_array().begin());
  }

private:
  PhysicalParameters<U> params;
  U source_sound_speed;

  std::unique_ptr<Vector> g;
  std::unique_ptr<Vector> RHS;

  std::unique_ptr<Vector> u;
  std::unique_ptr<Vector> ud;
  std::unique_ptr<Vector> udd;

  void update_source(U t) {
    // Apply windowing
    U window;
    if (t < params.period * params.window_length) {
      window = 0.5 * (1.0 - cos(params.source_frequency * M_PI * t /
                                params.window_length));
    } else {
      window = 1.0;
    }

    // Update boundary condition
    const U homogeneous_source =
        window * params.source_amplitude * params.source_angular_frequency /
        source_sound_speed * cos(params.source_angular_frequency * t);
    g->set(homogeneous_source);
  }
};

template <TimesteppingType TS, typename U, typename Vector>
auto create_timestepper(std::shared_ptr<fem::FunctionSpace<U>> V,
                        PhysicalParameters<U> const &params,
                        UserConfig<U> const &config) {
  U c0 = get_source_sound_speed<U>(config.material_case);

  if constexpr (TS == TimesteppingType::ExplicitRK4) {
    return std::make_unique<ExplicitRK4<U, Vector>>(V, params, c0);
  } else if constexpr (TS == TimesteppingType::Newmark) {
    return std::make_unique<Newmark<U, Vector>>(V, params, c0);
  } else {
    static_assert(always_false_v<TS>, "Unsupported timestepping type");
  }
}

template <typename U, int P>
U compute_dt(std::shared_ptr<fem::Function<U>> solution, U h, U sound_speed,
             U period, U CFL) {
  // this could implement better heuristics (for nonlinear case) dependent on
  // the solution value. Note that this would require a global reduction.
  assert(false && "TODO");
  return 0.;
}

template <typename U, int P> U compute_dt(U h, U sound_speed, U period, U CFL) {
  std::cout << "DT" << P << " " << h << " " << sound_speed << " " << CFL << " "
            << period << std::endl;
  U dt = CFL * h / (sound_speed * P * P);
  const int steps_per_period = period / dt + 1;
  return period / steps_per_period;
}

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

template <class U>
void log_progress(int steps, U dt, U current_time, U final_time,
                  const TimePoint &start) {
  U fraction = double(current_time) / double(final_time);
  auto now = Clock::now();
  U elapsed = std::chrono::duration<U>(now - start).count();
  U eta = (fraction > 0.0) ? elapsed * (1.0 - fraction) / fraction : 0.0;

  spdlog::info("Step {:4d} — dt = {:10.6e}s — t = {:10.6e}/{:7.4f} ({:5.1f}%) "
               "— elapsed {:6.2f}s — ETA {:6.2f}s",
               steps, double(dt), double(current_time), double(final_time),
               fraction * 100.0, elapsed, eta);
}

} // namespace freefus