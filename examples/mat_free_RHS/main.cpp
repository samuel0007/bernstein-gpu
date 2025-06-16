// # Exterior Mass matrix operation
#include "mat_free_RHS.h"
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

#include "src/geometry.hpp"
#include "src/mass_baseline.hpp"
#include "src/mesh.hpp"
#include "src/quadrature.hpp"
#include "src/stiffness_baseline.hpp"
#include "src/util.hpp"
#include "src/vector.hpp"

#include <boost/program_options.hpp>

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

#ifndef POLYNOMIAL_DEGREE
#define POLYNOMIAL_DEGREE 10
#endif

po::variables_map get_cli_config(int argc, char *argv[]) {
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()("help,h", "print usage message")
      ("nreps", po::value<int>()->default_value(1), "number of repetitions")
      ("matrix_comparison", po::bool_switch()->default_value(true), "Compare result to CPU matrix operator");
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
void solver(MPI_Comm comm, po::variables_map vm) {
  constexpr int polynomial_degree = POLYNOMIAL_DEGREE;
  constexpr int quadrature_points = polynomial_degree + 2;

  const int nreps = vm["nreps"].as<int>();
  const bool matrix_comparison = vm["matrix_comparison"].as<bool>();

  // ----------- 1. Problem Setup -----------
  // Read mesh and mesh tags
  auto coord_element =
      fem::CoordinateElement<T>(mesh::CellType::tetrahedron, 1);
  io::XDMFFile fmesh(MPI_COMM_WORLD, std::string(DATA_DIR) + "/mesh.xdmf", "r");
  auto mesh = std::make_shared<mesh::Mesh<T>>(
      fmesh.read_mesh(coord_element, mesh::GhostMode::none, "mesh"));
  mesh->topology()->create_connectivity(2, 3);
  auto mt_cell = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(*mesh, "Cell tags", std::nullopt));
  auto mt_facet = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(*mesh, "Facet tags", std::nullopt));

  auto element = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::tetrahedron,
      polynomial_degree, basix::element::lagrange_variant::equispaced,
      basix::element::dpc_variant::unset, false);

  auto element_DG = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::tetrahedron, 0,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, true);

  auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
      mesh, std::make_shared<const fem::FiniteElement<T>>(element)));
  auto V_DG = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
      mesh, std::make_shared<const fem::FiniteElement<T>>(element_DG)));

  auto c0 = std::make_shared<fem::Function<T>>(V_DG);
  auto rho0 = std::make_shared<fem::Function<T>>(V_DG);
  auto alpha1 = std::make_shared<fem::Function<T>>(V_DG);
  auto alpha2 = std::make_shared<fem::Function<T>>(V_DG);
  auto alpha3 = std::make_shared<fem::Function<T>>(V_DG);
  auto cells_1 = mt_cell->find(1);

  std::span<T> c0_ = c0->x()->mutable_array();
  std::for_each(cells_1.begin(), cells_1.end(),
                [&](std::int32_t &i) { c0_[i] = 0.015; });
  std::span<T> rho0_ = rho0->x()->mutable_array();
  std::for_each(cells_1.begin(), cells_1.end(),
                [&](std::int32_t &i) { rho0_[i] = 0.01; });

  std::span<T> alpha1_ = alpha1->x()->mutable_array();
  std::for_each(cells_1.begin(), cells_1.end(),
                [&](std::int32_t &i) { alpha1_[i] = -1. / rho0_[i]; });

  std::span<T> alpha2_ = alpha2->x()->mutable_array();
  std::for_each(cells_1.begin(), cells_1.end(),
                [&](std::int32_t &i) { alpha2_[i] = 1. / (rho0_[i]); });

  std::span<T> alpha3_ = alpha3->x()->mutable_array();
  std::for_each(cells_1.begin(), cells_1.end(), [&](std::int32_t &i) {
    alpha3_[i] = -1. / (rho0_[i] * c0_[i]);
  });

  std::cout << "cells_1 size:" << cells_1.size() << std::endl;
  std::cout << "alpha1 size:" << alpha1->x()->mutable_array().size()
            << std::endl;

  std::shared_ptr<mesh::Topology> topology = mesh->topology();
  const std::size_t tdim = topology->dim();
  std::size_t ncells = topology->index_map(tdim)->size_global();
  std::size_t ndofs_global = V->dofmap()->index_map->size_global();
  std::size_t ndofs_local = V->dofmap()->index_map->size_local();

  auto ui = std::make_shared<fem::Function<T>>(V);

  {
    std::string fp_type = "float";
    if (std::is_same_v<T, float>)
      fp_type += "32";
    else if (std::is_same_v<T, double>)
      fp_type += "64";

    constexpr int N = polynomial_degree + 1;
    constexpr int K = (N + 2) * (N + 1) * N / 6; // Number of dofs on triangle

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

  std::vector<std::pair<std::int32_t, std::vector<std::int32_t>>> facet_domains;
  std::map<fem::IntegralType,
           std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>
      facet_domains_data;

  // const std::vector bfacets = mesh::exterior_facet_indices(*topology);

  std::vector<int> ft_unique = {1, 2};
  for (int i = 0; i < ft_unique.size(); ++i) {
    int tag = ft_unique[i];
    std::vector<std::int32_t> facet_domain = fem::compute_integration_domains(
        fem::IntegralType::exterior_facet, *V->mesh()->topology_mutable(),
        mt_facet->find(tag));
    // bfacets);
    std::cout << std::format("Domain {}: {}\n", tag, facet_domain.size() / 2);

    facet_domains.push_back({tag, facet_domain});
    facet_domains_data[fem::IntegralType::exterior_facet].push_back(
        {tag, facet_domains[i].second});
  }

  // ----------- 2. GPU Matrix Free setup -----------
  acc::MatFreeStiffness3D<T, polynomial_degree, quadrature_points>
      gpu_action_stiffness(mesh, V, alpha1->x()->array());
  acc::MatFreeMassExteriorBaseline3D<T, polynomial_degree, quadrature_points>
      gpu_action_exterior1(mesh, V, facet_domains[0].second,
                           alpha2->x()->array());
  acc::MatFreeMassExteriorBaseline3D<T, polynomial_degree, quadrature_points>
      gpu_action_exterior2(mesh, V, facet_domains[1].second,
                           alpha3->x()->array());

  // ----------- 3. Matrix Free apply -----------

  auto map = V->dofmap()->index_map;
  int map_bs = V->dofmap()->index_map_bs();
  la::Vector<T> &x = *(ui->x());

  for (double i = 0; i < ndofs_local; ++i) {
    x.mutable_array()[i] = sin(i / ndofs_local);
    // x.mutable_array()[i] = 1.;
  }

  // GPU
  DeviceVector x_d(map, map_bs);
  DeviceVector y_d(map, map_bs);
  y_d.set(T{0.0});
  x_d.copy_from_host(x);
  std::cout << "norm(x_d)=" << acc::norm(x_d) << "\n";

  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nreps; ++i) {
      y_d.set(0.);
      gpu_action_stiffness(x_d, y_d);
      gpu_action_exterior1(x_d, y_d);
      gpu_action_exterior2(x_d, y_d);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "norm(y_d)=" << acc::norm(y_d) << "\n";
    std::chrono::duration<double> duration = stop - start;
    std::cout << "Baseline Mat-free Matvec time: " << duration.count()
              << std::endl;
    std::cout << "Baseline Mat-free action Gdofs/s: "
              << ndofs_global * nreps / (1e9 * duration.count()) << std::endl;

  }
  // {
  //     auto start = std::chrono::high_resolution_clock::now();
  //     for (int i = 0; i < nreps; ++i)
  //     {
  //         gpu_action_sf(x_d, y_d);
  //     }
  //     auto stop = std::chrono::high_resolution_clock::now();
  //     std::chrono::duration<double> duration = stop - start;
  //     std::cout << "SF Mat-free Matvec time: " << duration.count()
  //               << std::endl;
  //     std::cout << "SF Mat-free action Gdofs/s: "
  //               << ndofs_global * nreps / (1e9 * duration.count()) <<
  //               std::endl;
  // }
  // {
  //     auto start = std::chrono::high_resolution_clock::now();
  //     for (int i = 0; i < nreps; ++i)
  //     {
  //         gpu_action(x_d, y_d);
  //     }
  //     auto stop = std::chrono::high_resolution_clock::now();
  //     std::chrono::duration<double> duration = stop - start;
  //     std::cout << "SF OTF Mat-free Matvec time: " << duration.count() <<
  //     std::endl; std::cout << "SF OTF Mat-free action Gdofs/s: "
  //               << ndofs_global * nreps / (1e9 * duration.count()) <<
  //               std::endl;
  // }

  // std::cout << "norm(y_d)=" << acc::norm(y_d) << "\n";

  if (matrix_comparison && std::is_same_v<T, PetscScalar>) {
    // Action of the bilinear form "a" on a function ui
    auto M = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_mat_free_RHS_L, {V}, {{"u_n", ui}, {"rho0", rho0}, {"c0", c0}},
        {}, facet_domains_data, {}, {}));

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
    double eps = 1e-8;
    bool check = true;
    for (int i = 0; i < ndofs_local; ++i) {
      if (std::abs(y.array()[i] - y_h.array()[i]) > eps) {
        std::cout << std::format("y={}, y_h={}\n", y.array()[i],
                                 y_h.array()[i]);
        check = false;
        break;
      }
    }
    std::cout << "S:" << (check ? "PASSED" : "FAILED") << std::endl;
  }
}

/// Main program
int main(int argc, char *argv[]) {
  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    auto vm = get_cli_config(argc, argv);

    solver<T>(MPI_COMM_WORLD, vm);
  }
  MPI_Finalize();
  return 0;
}
