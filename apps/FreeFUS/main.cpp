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
            const PhysicalParameters<U> params)
{
  PROF_CPU_SCOPE("solver", 1);
  PROF_CPU_START("setup", 1);
  static_assert(D >= 2 && D <= 3, "Unsupported dimension");

  // Global Parameters
  static constexpr int P =
      POLYNOMIAL_DEGREE;      // Polynomial Degree of approximation
  static constexpr int Q = P; // Quadrature order
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

  io::VTXWriter<U> fslicedwriter(mesh_data.mesh->comm(), config.output_filepath + "-sliced.bp",
                                 {u_sliced_out}, "bp5");
  io::VTXWriter<U> fwriter(mesh_data.mesh->comm(), config.output_filepath + ".bp",
                           {u_out}, "bp5");

  ascent::Ascent ascent_runner;
  conduit::Node conduit_mesh;
  conduit::Node ascent_actions;

  if (config.insitu)
  {
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
  // Worst case traversal time, with added periods for buffering of the initial windowing
  T traversal_time = params.domain_length / sound_speed_min + config.buffer_periods * params.period;

  T evo_dt = freefus::compute_dt<T, P>(h_min, sound_speed_min, params.period, config.CFL);
  T sampling_dt = params.period / (2. * T(config.sample_harmonic));

  T evolution_time = freefus::compute_evolution_time(traversal_time, sampling_dt);
  int evolution_steps = static_cast<int>(std::ceil(evolution_time / evo_dt));

  T sampling_time = params.period * config.sampling_periods;
  int sampling_steps = static_cast<int>(std::ceil(sampling_time / sampling_dt));
  T final_time = evolution_time + sampling_time;
  int total_steps = evolution_steps + sampling_steps;

  freefus::check_nyquist<U, P>(params.source_frequency, h_min, h_max, sound_speed_min, sound_speed_max, evo_dt, sampling_dt);

  timestepper->init(model, solver);

  PROF_CPU_STOP("setup");
  PROF_REPORT();

  PROF_CPU_START("timestepping", 1);

  auto start = freefus::Clock::now();

  int steps;
  // This runs the simulation until the end of the domain (for traversal_time)
  for (steps = 1; (steps <= evolution_steps) && (steps <= config.max_steps); ++steps)
  {
    const std::string current_timestep_name = std::string("timestep_") + std::to_string(steps);
    PROF_CPU_START(current_timestep_name, 2);
    // This might be needed for nonlinear case
    // T evo_dt = freefus::compute_dt<T, P>(solution, h_min, sound_speed_min,
    //                                      params.period, config.CFL);
    T dt = std::min(evo_dt, traversal_time - current_time);
    timestepper->evolve(model, solver, current_time, dt);
    current_time += dt;

    if (!(steps % config.output_steps))
    {
      PROF_CPU_SCOPE("FILE_OUTPUT", 3);
      timestepper->get_solution(solution);
      freefus::interpolate_to_slice(solution, u_sliced_out, interpolation_data);
      fslicedwriter.write(current_time);

      u_out->interpolate(*solution);
      fwriter.write(current_time);
      spdlog::info("File output: wrote solution at time {} (step {})",
                   current_time, steps);
    }

    if (config.insitu && !(steps % config.insitu_output_steps))
    {
      PROF_CPU_SCOPE("INSITU_OUTPUT", 3);
      timestepper->get_solution(solution);
      u_out->interpolate(*solution);
      freefus::publish_insitu(u_out, ascent_runner, conduit_mesh,
                              ascent_actions);
      spdlog::info(
          "In-situ output: executed Ascent actions at time {} (step {})",
          current_time, steps);
    }

    if (!(steps % 10))
    {
      freefus::log_progress(steps, evolution_steps, dt, current_time, evolution_time, start);
    }
    PROF_CPU_STOP(current_timestep_name);
    PROF_REPORT_ABOVE(2);
    PROF_RESET_LVL(2);
  }
  assert(steps == evolution_steps);

  PROF_CPU_STOP("timestepping");
  PROF_CPU_START("sampling", 1);

  // At the end of the simulation, to sample up to the nth harmonic for a single
  // period, we need a timestep of 1 / (2 * source_frequency * n) > delta t for
  // one period
  auto sampling_start = freefus::Clock::now();

  if (config.sample_harmonic > 0)
  {
    for (int sampling_step = 0; sampling_step < sampling_steps; ++sampling_step)
    {
      // T dt = std::min(sampling_dt, sampling_T - current_time);
      T dt = sampling_dt; // Sampling has to be uniform
      timestepper->evolve(model, solver, current_time, dt);
      current_time += dt;
      timestepper->get_solution(solution);

      freefus::interpolate_to_slice(solution, u_sliced_out, interpolation_data);
      fslicedwriter.write(current_time);

      // u_out->interpolate(*solution);
      // fwriter.write(current_time);

      freefus::log_progress(sampling_step, sampling_steps, dt, current_time, final_time, sampling_start);
      ++steps;
    }
    spdlog::info("Number of sampling steps={}, max harmonic={}, samping_dt={}",
                 sampling_steps, config.sample_harmonic, sampling_dt);
  }
  PROF_CPU_STOP("sampling");

  auto stop = freefus::Clock::now();
  U sampling_elapsed = std::chrono::duration<U>(stop - sampling_start).count();
  U total_elapsed = std::chrono::duration<U>(stop - start).count();
  U evolution_elapsed = total_elapsed - sampling_elapsed;
  spdlog::info("Evolution:  {:6.2f}s, Sampling:  {:6.2f}s, Total runtime: {:6.2f}s",
               evolution_elapsed, sampling_elapsed, total_elapsed);

  PROF_REPORT();
}

int main(int argc, char *argv[])
{
  PROF_CPU_SCOPE("main", 0);
  auto vm = parse_cli_config<U>(argc, argv);
  if (vm.count("help"))
  {
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