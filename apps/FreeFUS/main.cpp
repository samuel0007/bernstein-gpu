#include <mpi.h>
#include <spdlog/spdlog.h>

#include "ascent_helpers.hpp"
#include "cli.hpp"
#include "fem_helpers.hpp"
#include "materials.hpp"
#include "mesh_helpers.hpp"
#include "types.hpp"
#include "util.hpp"
#include "vector.hpp"
#include "insitu.hpp"

using namespace dolfinx;
using T = SCALAR_TYPE;

template <typename T, int D>
void solver(MPI_Comm comm, const UserConfig<T> &config,
            const PhysicalParameters<T> params)
{
  static_assert(D >= 2 && D <= 3, "Unsupported dimension");

  // Global Parameters
  static constexpr int P =
      POLYNOMIAL_DEGREE;          // Polynomial Degree of approximation
  static constexpr int Q = P + 2; // Quadrature order
  static constexpr mesh::CellType cell_type =
      (D == 2) ? mesh::CellType::triangle : mesh::CellType::tetrahedron;

  MeshData mesh_data =
      freefus::load_mesh<T>(comm, cell_type, config.mesh_filepath);

  T h_min = freefus::compute_global_min_cell_size(mesh_data.mesh);
  auto [el, el_DG, V, V_DG] = freefus::make_element_spaces(
      mesh_data.mesh, cell_type, config.lvariant, P);

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
    setup_insitu(V, P, solution, ascent_runner, conduit_mesh, ascent_actions);

  ascent_runner.execute(ascent_actions);

  ascent_runner.close();
}

int main(int argc, char *argv[])
{
  auto vm = get_cli_config<T>(argc, argv);
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

    const UserConfig<T> config = make_user_config<T>(vm);

    spdlog::info(asciiLogo());
    display_user_config(config);
    spdlog::info("\n{}", device_information());

    PhysicalParameters params(config);

    solver<T, 2>(comm, config, params);
  }
  MPI_Finalize();

  return 0;
}