#include "util.hpp"
#include "vector.hpp"
#include <mpi.h>
#include <spdlog/spdlog.h>

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

template <typename T, typename Vector, int D>
void solver(MPI_Comm comm, const UserConfig<T> &config,
            const PhysicalParameters<T> params) {
  static_assert(D >= 2 && D <= 3, "Unsupported dimension");

  // Global Parameters
  static constexpr int P =
      POLYNOMIAL_DEGREE;          // Polynomial Degree of approximation
  static constexpr int Q = P + 2; // Quadrature order
  static constexpr mesh::CellType cell_type =
      (D == 2) ? mesh::CellType::triangle : mesh::CellType::tetrahedron;

  MeshData mesh_data =
      freefus::load_mesh<T>(comm, cell_type, config.mesh_filepath);

  auto spaces = freefus::make_element_spaces(mesh_data.mesh, cell_type,
                                             config.lvariant, P);

  auto [el, el_DG, V, V_DG] = spaces;

  auto solution = std::make_shared<fem::Function<T>>(V);

  auto material_coefficients = freefus::create_materials_coefficients<T>(
      V_DG, mesh_data, config.material_case);

  auto [V_out, u_out] =
      freefus::make_output_spaces(mesh_data.mesh, cell_type, P);

  io::VTXWriter<T> fwriter(mesh_data.mesh->comm(), config.output_filepath,
                           {u_out}, "bp5");

  Ascent ascent_runner;
  Node conduit_mesh;
  Node ascent_actions;
  if (config.insitu)
    freefus::setup_insitu(V, P, solution, ascent_runner, conduit_mesh,
                          ascent_actions);

  auto model = freefus::create_model<T, P, Q, D>(
      spaces, material_coefficients, mesh_data, params, config.model_type);
  auto solver = freefus::create_solver(V, config);

  auto timestepper = freefus::create_timestepper<T, Vector>(V, params, config);

  T h_min = freefus::compute_global_min_cell_size(mesh_data.mesh);
  T sound_speed_min = freefus::compute_global_minimum_sound_speed<T>(
      mesh_data.mesh->comm(), material_coefficients);

  T current_time = 0.;
  T final_time =
      params.domain_length / sound_speed_min + 4.0 / params.source_frequency;

  int steps = 0;
  auto start = freefus::Clock::now();

  T max_dt = freefus::compute_dt<T, P>(h_min, sound_speed_min,
                                       params.period, config.CFL);
  while (current_time < final_time) {
    // This might be needed for nonlinear case
    // T max_dt = freefus::compute_dt<T, P>(solution, h_min, sound_speed_min,
    //                                      params.period, config.CFL);
    T dt = min(max_dt, final_time - current_time);
    timestepper->evolve(model, solver, current_time, dt);

    ++steps;
    current_time += dt;
    freefus::log_progress(steps, dt, current_time, final_time, start);
  }

  // for timestep in timesteps:
  //   compute_dt();
  //   evolve();
  //   output_file();
  //   output_insitu();

  ascent_runner.execute(ascent_actions);
  ascent_runner.close();
}

int main(int argc, char *argv[]) {
  auto vm = parse_cli_config<T>(argc, argv);
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

    const UserConfig<T> config = make_user_config<T>(vm);

    spdlog::info(asciiLogo());
    display_user_config(config);
    spdlog::info("\n{}", device_information());

    PhysicalParameters params(config);

    solver<T, acc::DeviceVector<T>, 2>(comm, config, params);
  }
  MPI_Finalize();

  return 0;
}