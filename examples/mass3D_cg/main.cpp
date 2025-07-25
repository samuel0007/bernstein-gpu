// # Mass matrix operation

#include "mass3D_cg.h"
#include <algorithm>
#include <basix/finite-element.h>
#include <cmath>
#include <concepts>
#include <dolfinx.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>
#include <format>
#include <memory>
#include <petscsystypes.h>

#include <boost/program_options.hpp>

#include "src/cg_gpu.hpp"
#include "src/geometry.hpp"
#include "src/linalg.hpp"
#include "src/mass.hpp"
#include "src/mass_baseline.hpp"
#include "src/mass_sf.hpp"
#include "src/mesh.hpp"
#include "src/quadrature.hpp"
#include "src/util.hpp"
#include "src/vector.hpp"

using namespace dolfinx;
namespace po = boost::program_options;

using T = SCALAR_TYPE;

#if USE_HIP
using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;
#elif USE_CUDA
using DeviceVector = dolfinx::acc::Vector<T, acc::Device::CUDA>;
#else
static_assert(false)
#endif

po::variables_map get_cli_config(int argc, char *argv[])
{
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()("help,h", "print usage message")
      ("matrix_comparison", po::bool_switch()->default_value(false), "Compare result to CPU matrix CG");
  // clang-format on

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run(),
            vm);
  po::notify(vm);

  return vm;
}

template <std::floating_point T>
void solver(MPI_Comm comm, po::variables_map vm)
{
  constexpr int polynomial_degree = POLYNOMIAL_DEGREE;
  constexpr int quadrature_points = polynomial_degree + 2;

  const bool matrix_comparison = vm["matrix_comparison"].as<bool>();


  // ----------- 1. Problem Setup -----------
  // Create mesh and function space
  // auto mesh = std::make_shared<mesh::Mesh<T>>(mesh::create_rectangle<T>(
  //     comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {10, 10}, mesh::CellType::triangle,
  //     mesh::create_cell_partitioner(mesh::GhostMode::none)));

  // Read mesh and mesh tags
  auto coord_element = fem::CoordinateElement<T>(mesh::CellType::tetrahedron, 1);
  io::XDMFFile fmesh(MPI_COMM_WORLD, std::string(DATA_DIR) + "/mesh.xdmf", "r");
  auto mesh = std::make_shared<mesh::Mesh<T>>(fmesh.read_mesh(
      coord_element, mesh::GhostMode::none, "mesh"));
  mesh->topology()->create_connectivity(1, 2);

  auto element = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::tetrahedron, polynomial_degree,
      basix::element::lagrange_variant::bernstein,
      basix::element::dpc_variant::unset, false);
  auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
      mesh, std::make_shared<const fem::FiniteElement<T>>(element), {}));

  auto f = std::make_shared<fem::Function<T>>(V);
  auto ui = std::make_shared<fem::Function<T>>(V);
  auto M = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_mass3D_cg_M, {V}, {{"ui", ui}}, {{}}, {}, {}));
  auto L = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_mass3D_cg_L, {V}, {{"f", f}}, {}, {}, {}));

  f->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> out;
        for (std::size_t p = 0; p < x.extent(1); ++p)
        {
          out.push_back(sin(x(0, p)) * cos(x(1, p)));
        }
        return {out, {out.size()}};
      });
  auto topology = V->mesh()->topology_mutable();
  auto dofmap = V->dofmap();
  auto map = dofmap->index_map;
  int map_bs = dofmap->index_map_bs();

  const std::size_t tdim = topology->dim();
  std::size_t ncells = topology->index_map(tdim)->size_global();
  std::size_t ndofs_global = dofmap->index_map->size_global();
  std::size_t ndofs_local = dofmap->index_map->size_local();

  std::cout << "ncells=" << ncells << "\n";
  std::cout << "ndofs_global=" << ndofs_global << "\n";
  std::cout << "ndofs_local=" << ndofs_local << "\n";

  // Assemble RHS vector
  la::Vector<T> b(map, map_bs);
  fem::assemble_vector(b.mutable_array(), *L);

  // ----------- 3. GPU Matrix Free setup -----------
  DeviceVector b_d(map, map_bs);
  b_d.copy_from_host(b);
  // acc::MatFreeMass<T, polynomial_degree, quadrature_points> gpu_action(mesh, V,
  //                                                                      1.0);
  acc::MatFreeMassBaseline3D<T, polynomial_degree, quadrature_points> gpu_action(mesh, V,
                                                                               1.0);
  // acc::MatFreeMassSF<T, polynomial_degree, quadrature_points> gpu_action(mesh, V,
  //                                                                      1.0);

  // ----------- 4. CG -----------
  int max_iters = 2000;
  double rtol = 1e-7;

  // GPU
  dolfinx::acc::CGSolver<DeviceVector> cg(map, map_bs);
  cg.set_max_iterations(max_iters);
  cg.set_tolerance(rtol);

  DeviceVector x_d(map, map_bs);
  x_d.set(0.);
  int pcg_gpu_its = cg.solve(gpu_action, x_d, b_d, true);
 
  x_d.set(0.);
  int cg_gpu_its = cg.solve(gpu_action, x_d, b_d, false);

  if (dolfinx::MPI::rank(comm) == 0)
  {
    std::cout << "Number of GPU PCG iterations " << pcg_gpu_its << std::endl;
    std::cout << "Number of GPU CG iterations " << cg_gpu_its << std::endl;
  }

  if (matrix_comparison)
  {
    // ----------- 2. CPU Matrix Free setup -----------
    auto coeff = fem::allocate_coefficient_storage(*M);
    std::vector<T> constants = fem::pack_constants(*M);

    // Create function for computing the action of A on x (y = Ax)
    auto cpu_action = [&M, &ui, &coeff, &constants](auto &x, auto &y)
    {
      y.set(0.0);

      // Update coefficient ui (just copy data from x to ui)
      std::ranges::copy(x.array(), ui->x()->mutable_array().begin());

      // Compute action of A on x
      fem::pack_coefficients(*M, coeff);
      fem::assemble_vector(y.mutable_array(), *M, std::span<const T>(constants),
                           fem::make_coefficients_span(coeff));
    };

    // CPU
    la::Vector<T> x(map, map_bs);
    int cpu_its = linalg::cg(x, b, cpu_action, max_iters, rtol);

    la::Vector<T> x_h(map, map_bs);
    thrust::copy(x_d.thrust_vector().begin(), x_d.thrust_vector().end(),
                 x_h.mutable_array().begin());

    // for(int i = 0; i < ndofs_local; ++i) {
    //   std::cout << "y_d[" << i << "]=" << y_h.array()[i] << std::endl;
    // }
    if (dolfinx::MPI::rank(comm) == 0)
    {
      std::cout << "Number of CPU CG iterations " << cpu_its << std::endl;
    }

    double eps = rtol * 10;
    bool check = true;
    for (int i = 0; i < ndofs_local; ++i)
    {
      if (std::abs(x.array()[i] - x_h.array()[i]) > eps)
      {
        // std::cout << x.array()[i] << " " << x_h.array()[i] << std::endl;
        check = false;
        // break;
      }
    }
    std::cout << (check ? "PASSED" : "FAILED") << std::endl;
  }
}

/// Main program
int main(int argc, char *argv[])
{
  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (rank == 0)
    {
      std::cout << device_information();
    }
    auto vm = get_cli_config(argc, argv);


    solver<T>(MPI_COMM_WORLD, vm);
  }
  MPI_Finalize();
  return 0;
}
