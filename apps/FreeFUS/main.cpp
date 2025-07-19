#include <mpi.h>
#include <spdlog/spdlog.h>
#include <dolfinx.h>
#include <dolfinx/io/ADIOS2Writers.h>

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

  // io::VTXWriter<U> fwriter(mesh_data.mesh->comm(), config.output_filepath,
  //                          {u_out}, "bp5");

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

  auto timestepper = freefus::create_timestepper<TimesteppingType::Newmark, U, Vector>(V, params, config);

  T h_min = freefus::compute_global_min_cell_size(mesh_data.mesh);
  T sound_speed_min = freefus::compute_global_minimum_sound_speed<T>(
      mesh_data.mesh->comm(), material_coefficients);

  T current_time = 0.;
  T final_time =
      params.domain_length / sound_speed_min + 4.0 / params.source_frequency;

  int steps = 0;
  auto start = freefus::Clock::now();

  T max_dt = freefus::compute_dt<T, P>(h_min, sound_speed_min, params.period,
                                       config.CFL);

  timestepper->init(model, solver);

  while (current_time < final_time) {
    // This might be needed for nonlinear case
    // T max_dt = freefus::compute_dt<T, P>(solution, h_min, sound_speed_min,
    //                                      params.period, config.CFL);
    T dt = std::min(max_dt, final_time - current_time);
    timestepper->evolve(model, solver, current_time, dt);

    // if (!(steps % config.output_steps)) {
    //   timestepper->get_solution(solution);
    //   u_out->interpolate(*solution);
    //   fwriter.write(current_time);
    //   spdlog::info("File output: wrote solution at time {} (step {})",
    //                current_time, steps);
    // }

    if (config.insitu && !(steps % config.insitu_output_steps)) {
      timestepper->get_solution(solution);
      u_out->interpolate(*solution);
      freefus::publish_insitu(u_out, ascent_runner, conduit_mesh,
                              ascent_actions);
      spdlog::info(
          "In-situ output: executed Ascent actions at time {} (step {})",
          current_time, steps);
    }

    ++steps;
    current_time += dt;
    // break;
    if (!(steps % 10)) {
      freefus::log_progress(steps, dt, current_time, final_time, start);
    }
  }

  timestepper->get_solution(solution);
  u_out->interpolate(*solution);
  // fwriter.write(current_time);
  spdlog::info("File output: wrote solution at time {} (step {})",
                current_time, steps);

  if (config.insitu) {
    timestepper->get_solution(solution);
    u_out->interpolate(*solution);
    freefus::publish_insitu(u_out, ascent_runner, conduit_mesh,
                            ascent_actions);
    spdlog::info(
        "In-situ output: executed Ascent actions at time {} (step {})",
        current_time, steps);
    ascent_runner.close();
  }

}

int main(int argc, char *argv[]) {
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