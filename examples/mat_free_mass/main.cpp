// # Mass matrix operation

#include "mat_free_mass.h"
#include <algorithm>
#include <basix/finite-element.h>
#include <cmath>
#include <concepts>
#include <dolfinx.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <memory>
#include <petscsystypes.h>

#include "src/geometry.hpp"
#include "src/mass.hpp"
#include "src/mass_baseline.hpp"
#include "src/mesh.hpp"
#include "src/quadrature.hpp"
#include "src/util.hpp"
#include "src/vector.hpp"

#include <boost/program_options.hpp>

using namespace dolfinx;
namespace po = boost::program_options;

using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

#if USE_HIP
using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;
#elif USE_CUDA
using DeviceVector = dolfinx::acc::Vector<T, acc::Device::CUDA>;
#else
static_assert(false)
#endif

#ifndef POLYNOMIAL_DEGREE
#define POLYNOMIAL_DEGREE 10
#endif

namespace linalg {

template <typename T> void copy(const la::Vector<T> &in, la::Vector<T> &out) {
  std::span<const T> _in = in.array();
  std::span<T> _out = out.mutable_array();
  std::copy(_in.begin(), _in.end(), _out.begin());
}

/// @brief Compute vector r = alpha * x + y.
/// @param[out] r
/// @param[in] alpha
/// @param[in] x
/// @param[in] y
void axpy(auto &&r, auto alpha, auto &&x, auto &&y) {
  std::ranges::transform(x.array(), y.array(), r.mutable_array().begin(),
                         [alpha](auto x, auto y) { return alpha * x + y; });
}
} // namespace linalg

po::variables_map get_cli_config(int argc, char *argv[]) {
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()("help,h", "print usage message")
      ("nelements", po::value<int>()->default_value(10), "Number of elements (1D)")
      ("nreps", po::value<int>()->default_value(10), "number of repetitions")
      ("matrix_comparison", po::bool_switch()->default_value(false), "Compare result to CPU matrix operator");
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

template <typename T, std::floating_point U>
void solver(MPI_Comm comm, po::variables_map vm) {
  constexpr int polynomial_degree = POLYNOMIAL_DEGREE;
  // TODO: verify if expression integrates exactly. Probably? Comes from Basix.
  // constexpr int quadrature_points = (polynomial_degree + 2) / 2;
  constexpr int quadrature_points = polynomial_degree + 1;

  const int nelements = vm["nelements"].as<int>();
  const int nreps = vm["nreps"].as<int>();
  const bool matrix_comparison = vm["matrix_comparison"].as<bool>();

  // ----------- 1. Problem Setup -----------
  // Create mesh and function space
  auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_rectangle<U>(
      comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {nelements, nelements},
      mesh::CellType::triangle,
      mesh::create_cell_partitioner(mesh::GhostMode::none)));
  auto element = basix::create_element<U>(
      basix::element::family::P, basix::cell::type::triangle, polynomial_degree,
      basix::element::lagrange_variant::bernstein,
      basix::element::dpc_variant::unset, false);

  auto element_DG = basix::create_element<U>(
      basix::element::family::P, basix::cell::type::triangle, 0,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, true);

  auto V = std::make_shared<fem::FunctionSpace<U>>(
      fem::create_functionspace(mesh, element, {}));

  auto V_DG = std::make_shared<fem::FunctionSpace<U>>(
      fem::create_functionspace(mesh, element_DG, {}));

  auto alpha = std::make_shared<fem::Function<T>>(V_DG);
  alpha->x()->set(0.5);

  // Action of the bilinear form "a" on a function ui
  auto ui = std::make_shared<fem::Function<T, U>>(V);
  auto M = std::make_shared<fem::Form<T, U>>(
      fem::create_form<T>(*form_mat_free_mass_M, {V},
                          {{"ui", ui}, {"alpha", alpha}}, {{}}, {}, {}));

  const std::size_t tdim = mesh->topology()->dim();
  std::size_t ncells = mesh->topology()->index_map(tdim)->size_global();
  std::size_t ndofs_global = V->dofmap()->index_map->size_global();
  std::size_t ndofs_local = V->dofmap()->index_map->size_local();

  // if (rank == 0)
  {
    std::string fp_type = "float";
    if (std::is_same_v<T, float>)
      fp_type += "32";
    else if (std::is_same_v<T, double>)
      fp_type += "64";

    constexpr int N = polynomial_degree + 1;
    constexpr int K = (N + 1) * N / 2; // Number of dofs on triangle

    std::cout << device_information();
    std::cout << "-----------------------------------\n";
    std::cout << "Polynomial degree: " << polynomial_degree << "\n";
    std::cout << "Number of quadrature points (1D): " << quadrature_points
              << "\n";
    std::cout << "Number of dofs per element: " << K << "\n";
    // std::cout << "Number of ranks : " << size << "\n";
    std::cout << "Number of cells-global: " << ncells << "\n";
    std::cout << "Number of dofs-global: " << ndofs_global << "\n";
    std::cout << "Number of dofs-local: " << ndofs_local << "\n";
    // std::cout << "Number of cells-rank : " << ncells / size << "\n";
    // std::cout << "Number of dofs-rank : " << ndofs_global / size << "\n";
    std::cout << "Number of repetitions: " << nreps << "\n";
    std::cout << "Scalar Type: " << fp_type << "\n";
    std::cout << "-----------------------------------\n";
    std::cout << std::flush;
  }

  // ----------- 2. GPU Matrix Free setup -----------
  acc::MatFreeMass<U, polynomial_degree, quadrature_points> gpu_action(
      mesh, V, alpha->x()->array());
  acc::MatFreeMassBaseline<U, polynomial_degree, quadrature_points>
      gpu_action_baseline(mesh, V, alpha->x()->array());

  // ----------- 3. Matrix Free apply -----------

  auto map = V->dofmap()->index_map;
  int map_bs = V->dofmap()->index_map_bs();

  // CPU
  la::Vector<T> x(map, map_bs);

  for (double i = 0; i < ndofs_local; ++i) {
    // x.mutable_array()[i] = sin(i / ndofs_local);
    x.mutable_array()[i] = 1.;
  }

  // GPU
  DeviceVector x_d(map, map_bs);
  DeviceVector y_d(map, map_bs);
  y_d.set(U{0.0});
  x_d.copy_from_host(x);
  std::cout << "norm(x_d)=" << acc::norm(x_d) << "\n";

  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nreps; ++i) {
      gpu_action_baseline(x_d, y_d);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;
    std::cout << "Baseline Mat-free Matvec time: " << duration.count()
              << std::endl;
    std::cout << "Baseline Mat-free action Gdofs/s: "
              << ndofs_global * nreps / (1e9 * duration.count()) << std::endl;
  }
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nreps; ++i) {
      gpu_action(x_d, y_d);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;
    std::cout << "Mat-free Matvec time: " << duration.count() << std::endl;
    std::cout << "Mat-free action Gdofs/s: "
              << ndofs_global * nreps / (1e9 * duration.count()) << std::endl;
  }

  std::cout << "norm(y_d)=" << acc::norm(y_d) << "\n";

  if (matrix_comparison) {
    la::Vector<T> y_h(map, map_bs);
    thrust::copy(y_d.thrust_vector().begin(), y_d.thrust_vector().end(),
                 y_h.mutable_array().begin());

    // ----------- 2. CPU Matrix Free setup -----------
    auto coeff = fem::allocate_coefficient_storage(*M);
    std::vector<T> constants = fem::pack_constants(*M);

    // Create function for computing the action of A on x (y = Ax)
    auto cpu_action = [&M, &ui, &coeff, &constants](auto &x, auto &y) {
      y.set(0.0);

      // Update coefficient ui (just copy data from x to ui)
      std::ranges::copy(x.array(), ui->x()->mutable_array().begin());

      // Compute action of A on x
      fem::pack_coefficients(*M, coeff);
      fem::assemble_vector(y.mutable_array(), *M, std::span<const T>(constants),
                           fem::make_coefficients_span(coeff));

      // // Accumulate ghost values
      // y.scatter_rev(std::plus<T>());

      // // Update ghost values
      // y.scatter_fwd();
    };

    la::Vector<T> y(map, map_bs);
    y.set(0.);
    std::cout << "norm(x)=" << la::norm(x) << "\n";
    cpu_action(x, y);
    std::cout << "norm(y)=" << la::norm(y) << "\n";
    double eps = 1e-6;
    bool check = true;
    for (int i = 0; i < ndofs_local; ++i) {
      if (std::abs(y.array()[i] - y_h.array()[i]) > eps) {
        check = false;
        break;
      }
    }
    std::cout << "S:" << (check ? "PASSED" : "FAILED") << std::endl;
  }
}

/// Main program
int main(int argc, char *argv[]) {
  using T = PetscScalar;
  using U = typename dolfinx::scalar_value_type_t<T>;
  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    auto vm = get_cli_config(argc, argv);

    solver<T, U>(MPI_COMM_WORLD, vm);
  }
  MPI_Finalize();
  return 0;
}
