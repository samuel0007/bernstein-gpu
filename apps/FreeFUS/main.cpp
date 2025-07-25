// #define PROF_ACTIVATE  
#include "profiler.hpp"

#include <dolfinx.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <mpi.h>
#include <spdlog/spdlog.h>

#include "util.hpp"
#include "vector.hpp"

#include "ascent_helpers.hpp"
#include "cli.hpp"
#include "fem_helpers.hpp"
#include "insitu.hpp"
#include "materials.hpp"
#include "mesh_helpers.hpp"
#include "model.hpp"
#include "solver.hpp"
#include "timestepping.hpp"
#include "types.hpp"


using namespace dolfinx;
using T = SCALAR_TYPE;
using U = double;

template <typename T, typename U, typename Vector, int D>
void solver(MPI_Comm comm, const UserConfig<U> &config,
            const PhysicalParameters<U> params) {
  PROF_CPU_SCOPE("solver", 1);
  PROF_CPU_START("setup", 1);
  static_assert(D >= 2 && D <= 3, "Unsupported dimension");

  // Global Parameters
  static constexpr int P =
      POLYNOMIAL_DEGREE;          // Polynomial Degree of approximation
  static constexpr int Q = P + 2; // Quadrature order
  static constexpr mesh::CellType cell_type =
      (D == 2) ? mesh::CellType::triangle : mesh::CellType::tetrahedron;
  
  MeshData mesh_data =
      freefus::load_mesh<U>(comm, cell_type, config.mesh_filepath);

  auto spaces = freefus::make_element_spaces(mesh_data.mesh, cell_type,
                                             config.lvariant, P);

  auto [el, el_DG, V, V_DG] = spaces;

  auto solution = std::make_shared<fem::Function<U>>(V);

  auto material_coefficients = freefus::create_materials_coefficients<U>(
      V_DG, mesh_data, config.material_case, params);

  auto [V_out, u_out] =
      freefus::make_output_spaces(mesh_data.mesh, cell_type, P);

  auto [V_sliced_out, u_sliced_out, interpolation_data] =
      freefus::make_sliced_output_spaces(
          mesh_data.mesh, solution, config.domain_width, config.domain_length,
          config.sample_nx, config.sample_nz);

  io::VTXWriter<U> fwriter(mesh_data.mesh->comm(), config.output_filepath,
                           {u_sliced_out}, "bp5");

  ascent::Ascent ascent_runner;
  conduit::Node conduit_mesh;
  conduit::Node ascent_actions;

  if (config.insitu) {
    freefus::setup_insitu(V_out, P, u_out, ascent_runner, conduit_mesh,
                          ascent_actions, config);

    // Initial material output
    freefus::insitu_output_DG(material_coefficients, ascent_runner);
  }

  auto model = freefus::create_model<ModelType::LinearLossyImplicit, T, U, P, Q, D>(
      spaces, material_coefficients, mesh_data, params, config.model_type);
  auto solver = freefus::create_solver<T, U>(V, config);

  auto timestepper =
      freefus::create_timestepper<TimesteppingType::Newmark, U, Vector>(
          V, params, config);
  PROF_CPU_START("MESH_GLOBAL_REDUCTION", 3);
  auto [h_min, h_max] = freefus::compute_global_cell_size(mesh_data.mesh);
  auto [sound_speed_min, sound_speed_max] = freefus::compute_global_sound_speed<T>(
      mesh_data.mesh->comm(), material_coefficients);
  PROF_CPU_STOP("MESH_GLOBAL_REDUCTION");

  T current_time = 0.;
  T final_time =
      params.domain_length / sound_speed_min + 4.0 / params.source_frequency;

  int steps = 1;
  auto start = freefus::Clock::now();

  T max_dt = freefus::compute_dt<T, P>(h_min, sound_speed_min, params.period,
                                       config.CFL);

  freefus::check_nyquist<U, P>(params.source_frequency, h_min, h_max, sound_speed_min, sound_speed_max, max_dt);

  T harmonic_dt = 1. / (2. * params.source_frequency * config.sample_harmonic);
  T sampling_dt = std::min(max_dt, harmonic_dt);
  // this actually should capture all the simulated harmonics, if the simulation respect the nyquist limit
  // T sampling_dt = max_dt; 
  T sampling_T =
      final_time +
      (params.period * config.sampling_periods);
  spdlog::info("max harmonic={}, max_dt={}, sampling_dt={}", config.sample_harmonic, max_dt,
               sampling_dt);

  timestepper->init(model, solver);

  PROF_CPU_STOP("setup");
  PROF_REPORT();

  PROF_CPU_START("timestepping", 1);

  // This runs the simulation until the end of the domain
  while (current_time < final_time) {
    const std::string current_timestep_name = std::string("timestep_") + std::to_string(steps);
    PROF_CPU_START(current_timestep_name, 2);
    // This might be needed for nonlinear case
    // T max_dt = freefus::compute_dt<T, P>(solution, h_min, sound_speed_min,
    //                                      params.period, config.CFL);
    T dt = std::min(max_dt, final_time - current_time);
    timestepper->evolve(model, solver, current_time, dt);
    current_time += dt;

    if (!(steps % config.output_steps)) {
      PROF_CPU_SCOPE("FILE_OUTPUT", 3);
      timestepper->get_solution(solution);
      freefus::interpolate_to_slice(solution, u_sliced_out, interpolation_data);

      // u_out->interpolate(*solution);
      fwriter.write(current_time);
      spdlog::info("File output: wrote solution at time {} (step {})",
                   current_time, steps);
    }

    if (config.insitu && !(steps % config.insitu_output_steps)) {
      PROF_CPU_SCOPE("INSITU_OUTPUT", 3);
      timestepper->get_solution(solution);
      u_out->interpolate(*solution);
      freefus::publish_insitu(u_out, ascent_runner, conduit_mesh,
                              ascent_actions);
      spdlog::info(
          "In-situ output: executed Ascent actions at time {} (step {})",
          current_time, steps);
    }

    ++steps;
    if (!(steps % 10)) {
      freefus::log_progress(steps, dt, current_time, final_time, start);
    }
    PROF_CPU_STOP(current_timestep_name);
    PROF_REPORT_ABOVE(2);
    PROF_RESET_LVL(2);
  }

  PROF_CPU_STOP("timestepping");
  PROF_CPU_START("sampling", 1);

  // At the end of the simulation, to sample up to the nth harmonic for a single
  // period, we need a timestep of 1 / (2 * source_frequency * n) > delta t for
  // one period
  if (config.sample_harmonic > 0) {
    int sampling_steps = 0;
    while (current_time < sampling_T) {
      T dt = std::min(sampling_dt, sampling_T - current_time);
      timestepper->evolve(model, solver, current_time, dt);
      current_time += dt;
      timestepper->get_solution(solution);
      // u_out->interpolate(*solution);
      freefus::interpolate_to_slice(solution, u_sliced_out, interpolation_data);

      fwriter.write(current_time);
      spdlog::info("File output: wrote solution at time {} (step {}), sampling "
                   "final time={}",
                   current_time, steps, sampling_T);
      ++steps;
      ++sampling_steps;
    }
    spdlog::info("Number of sampling steps={}, max harmonic={}, samping_dt={}",
                 sampling_steps, config.sample_harmonic, sampling_dt);
  }
  PROF_CPU_STOP("sampling");
  // PROF_CPU_START("finalize", 1);


  // timestepper->get_solution(solution);
  // u_out->interpolate(*solution);
  // fwriter.write(current_time);
  // spdlog::info("File output: wrote solution at time {} (step {})", current_time,
  //              steps);

  // if (config.insitu) {
  //   timestepper->get_solution(solution);
  //   u_out->interpolate(*solution);
  //   freefus::publish_insitu(u_out, ascent_runner, conduit_mesh, ascent_actions);
  //   spdlog::info("In-situ output: executed Ascent actions at time {} (step {})",
  //                current_time, steps);
  //   ascent_runner.close();
  // }

  PROF_REPORT();
}

int main(int argc, char *argv[]) {
  PROF_CPU_SCOPE("main", 0);
  auto vm = parse_cli_config<U>(argc, argv);
  if (vm.count("help")) {
    std::cout << "...";
    return 0;
  }

  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const UserConfig<U> config = make_user_config<U>(vm);

    spdlog::info(asciiLogo());
    display_user_config(config);
    spdlog::info("\n{}", device_information());

    PhysicalParameters params(config);

    solver<T, U, acc::DeviceVector<T>, 3>(comm, config, params);
  }
  MPI_Finalize();

  return 0;
}