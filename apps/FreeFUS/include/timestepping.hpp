#pragma once

#include <array>
#include <chrono>

#include "types.hpp"
#include "vector.hpp"

namespace freefus {

template <typename T, typename Vector> class ExplicitRK4 {
public:
  ExplicitRK4(std::shared_ptr<fem::FunctionSpace<T>> V,
              const PhysicalParameters<T> &params, T source_sound_speed)
      : params(params), source_sound_speed(source_sound_speed) {

    auto index_map = V->dofmap()->index_map;
    auto bs = V->dofmap()->index_map_bs();
    assert(bs == 1 && "not implemented");

    g = std::make_unique<Vector>(index_map, bs);   // Source vector
    RHS = std::make_unique<Vector>(index_map, bs); // RHS vector

    u = std::make_unique<Vector>(index_map, bs);
    v = std::make_unique<Vector>(index_map, bs);
    ku = std::make_unique<Vector>(index_map, bs);
    kv = std::make_unique<Vector>(index_map, bs);
    u0 = std::make_unique<Vector>(index_map, bs);
    v0 = std::make_unique<Vector>(index_map, bs);
    ui = std::make_unique<Vector>(index_map, bs);
    vi = std::make_unique<Vector>(index_map, bs);
  };

  /// Evolve a solution under a model using a solver for a timestep dt
  void evolve(auto& model, auto& solver, T t, T dt) {
    copy_d_to_d(*u, *u0);
    copy_d_to_d(*v, *v0);

    for (int i = 0; i < 4; i++) {
      copy_d_to_d(*u0, *ui);
      copy_d_to_d(*v0, *vi);

      acc::axpy(*ui, dt * a[i], *ku, *ui);
      acc::axpy(*vi, dt * a[i], *kv, *vi);

      // RK time evaluation
      T tn = t + c[i] * dt;

      f0(tn, *ui, *vi, *ku);
      int solver_its = f1(tn, *ui, *vi, *kv, model, solver);
      spdlog::info("stage={} its={}", i, solver_its);

      acc::axpy(*u, dt * b[i], *ku, *u);
      acc::axpy(*v, dt * b[i], *kv, *v);
    }
  }

private:
  T source_sound_speed;
  static constexpr std::array<T, 4> a = {0.0, 0.5, 0.5, 1.0};
  static constexpr std::array<T, 4> b = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0,
                                         1.0 / 6.0};
  static constexpr std::array<T, 4> c = {0.0, 0.5, 0.5, 1.0};

  PhysicalParameters<T> params;
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

  void f0(T t, Vector &u, Vector &v, Vector &result) {
    copy_d_to_d(v, result);
  }

  int f1(T t, Vector &u, Vector &v, Vector &result,
         auto& model, auto& solver) {
    // Apply windowing
    T window;
    if (t < params.period * params.window_length) {
      window = 0.5 * (1.0 - cos(params.source_frequency * M_PI * t /
                                params.window_length));
    } else {
      window = 1.0;
    }

    // Update boundary condition
    const T homogeneous_source =
        window * params.source_amplitude * params.source_angular_frequency /
        source_sound_speed * cos(params.source_angular_frequency * t);
    g->set(homogeneous_source);

    model->rhs(u, v, *g, *RHS);

    return solver->solve(*model, result, *RHS, true);
  }
};

template <typename T, typename Vector>
auto create_timestepper(std::shared_ptr<fem::FunctionSpace<T>> V,
                        const PhysicalParameters<T> &params,
                        const UserConfig<T> &config) {
  T source_sound_speed = get_source_sound_speed<T>(config.material_case);
  return std::make_unique<ExplicitRK4<T, Vector>>(V, params,
                                                  source_sound_speed);
};

template <typename T, int P>
T compute_dt(std::shared_ptr<fem::Function<T>> solution, T h, T sound_speed,
             T period, T CFL) {
  // this could implement better heuristics (for nonlinear case) dependent on
  // the solution value. Note that this would require a global reduction.
  assert(false && "TODO");
  return 0.;
}

template <typename T, int P> T compute_dt(T h, T sound_speed, T period, T CFL) {
  T dt = CFL * h / (sound_speed * P * P);
  const int steps_per_period = period / dt + 1;
  return period / steps_per_period;
}

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

template <class T>
void log_progress(int steps, T dt, T current_time, T final_time,
                  const TimePoint &start) {
  double fraction = double(current_time) / double(final_time);
  auto now = Clock::now();
  double elapsed = std::chrono::duration<double>(now - start).count();
  double eta = (fraction > 0.0) ? elapsed * (1.0 - fraction) / fraction : 0.0;

  spdlog::info("Step {:4d} — dt = {:7.4f}s — t = {:7.4f}/{:7.4f} ({:5.1f}%) "
               "— elapsed {:6.2f}s — ETA {:6.2f}s",
               steps, double(dt), double(current_time), double(final_time),
               fraction * 100.0, elapsed, eta);
}

} // namespace freefus