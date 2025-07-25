// # Mass matrix operation
#define PROF_ACTIVATE
#include <spdlog/spdlog.h>

#include "profiler.hpp"

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
#include "src/newmark.hpp"
#include "src/quadrature.hpp"
#include "src/stiffness_baseline.hpp"
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

template <typename T, int P, int Q> class FusedJacobianStream {
  using FusedNewmarkAction = acc::MatFreeNewmark3D<T, P, Q>;
  using ExteriorMassAction = acc::MatFreeMassExteriorBaseline3D<T, P, Q>;

public:
  FusedJacobianStream(auto mesh, auto V, auto alpha, auto facet_domain) {
    Newmark_action_ptr = std::make_unique<FusedNewmarkAction>(mesh, V, alpha);
    Mgamma_action_ptr =
        std::make_unique<ExteriorMassAction>(mesh, V, facet_domain, alpha);
    Cgamma_action_ptr =
        std::make_unique<ExteriorMassAction>(mesh, V, facet_domain, alpha);

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
  }

  // LHS: (M+M_Γ) + (C+C_Γ)*gamma*dt + K*beta*dt2) @ in
  template <typename Vector> void operator()(Vector &in, Vector &out) {
    // PROF_GPU_SCOPE("JACOBIAN", 7, 0);
    out.set(0.);
    PROF_GPU_SCOPE("Newmark", 3, stream1);
    PROF_GPU_SCOPE("Mgamma", 3, stream2);
    PROF_GPU_SCOPE("Cgamma", 3, stream3);
    (*Newmark_action_ptr)(in, out, 1, stream1);
    (*Mgamma_action_ptr)(in, out, 1, stream2);
    (*Cgamma_action_ptr)(in, out, gamma *m_dt, stream3);
  };

  // void set_dt(U dt) { m_dt = dt; }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    // PROF_GPU_SCOPE("PC", 7, 0);
    // diag_inv.set(0.);
    // M_action_ptr->get_diag(diag_inv);
    // Mgamma_action_ptr->get_diag(diag_inv);
    // C_action_ptr->get_diag(diag_inv, gamma * m_dt);
    // Cgamma_action_ptr->get_diag(diag_inv, gamma * m_dt);
    // K_action_ptr->get_diag(diag_inv, beta * m_dt * m_dt);
    acc::inverse(diag_inv);
  }

private:
  T m_dt = 0.1;
  static constexpr T beta = 0.25;
  static constexpr T gamma = 0.5;
  std::unique_ptr<FusedNewmarkAction> Newmark_action_ptr;
  std::unique_ptr<ExteriorMassAction> Mgamma_action_ptr;
  std::unique_ptr<ExteriorMassAction> Cgamma_action_ptr;
  GpuStream stream1, stream2, stream3;
};

template <typename T, int P, int Q> class FusedJacobian {
  using FusedNewmarkAction = acc::MatFreeNewmark3D<T, P, Q>;
  using ExteriorMassAction = acc::MatFreeMassExteriorBaseline3D<T, P, Q>;

public:
  FusedJacobian(auto mesh, auto V, auto alpha, auto facet_domain) {
    Newmark_action_ptr = std::make_unique<FusedNewmarkAction>(mesh, V, alpha);
    Mgamma_action_ptr =
        std::make_unique<ExteriorMassAction>(mesh, V, facet_domain, alpha);
    Cgamma_action_ptr =
        std::make_unique<ExteriorMassAction>(mesh, V, facet_domain, alpha);
  }

  // LHS: (M+M_Γ) + (C+C_Γ)*gamma*dt + K*beta*dt2) @ in
  template <typename Vector> void operator()(Vector &in, Vector &out) {
    // PROF_GPU_SCOPE("JACOBIAN", 7, 0);
    out.set(0.);
    PROF_GPU_START("Newmark", 3, 0);
    (*Newmark_action_ptr)(in, out);
    PROF_GPU_STOP("Newmark");
    PROF_GPU_START("Mgamma", 3, 0);
    (*Mgamma_action_ptr)(in, out);
    PROF_GPU_STOP("Mgamma");
    PROF_GPU_START("Cgamma", 3, 0);
    (*Cgamma_action_ptr)(in, out, gamma *m_dt);
    PROF_GPU_STOP("Cgamma");
  };

  // void set_dt(U dt) { m_dt = dt; }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    // PROF_GPU_SCOPE("PC", 7, 0);
    // diag_inv.set(0.);
    // M_action_ptr->get_diag(diag_inv);
    // Mgamma_action_ptr->get_diag(diag_inv);
    // C_action_ptr->get_diag(diag_inv, gamma * m_dt);
    // Cgamma_action_ptr->get_diag(diag_inv, gamma * m_dt);
    // K_action_ptr->get_diag(diag_inv, beta * m_dt * m_dt);
    acc::inverse(diag_inv);
  }

private:
  T m_dt = 0.1;
  static constexpr T beta = 0.25;
  static constexpr T gamma = 0.5;
  std::unique_ptr<FusedNewmarkAction> Newmark_action_ptr;
  std::unique_ptr<ExteriorMassAction> Mgamma_action_ptr;
  std::unique_ptr<ExteriorMassAction> Cgamma_action_ptr;
};

template <typename T, int P, int Q> class Jacobian {
  using MassAction = acc::MatFreeMassBaseline3D<T, P, Q>;
  using StiffnessAction = acc::MatFreeStiffness3D<T, P, Q>;
  using ExteriorMassAction = acc::MatFreeMassExteriorBaseline3D<T, P, Q>;

public:
  Jacobian(auto mesh, auto V, auto alpha, auto facet_domain) {
    M_action_ptr = std::make_unique<MassAction>(mesh, V, alpha);
    Mgamma_action_ptr =
        std::make_unique<ExteriorMassAction>(mesh, V, facet_domain, alpha);
    C_action_ptr = std::make_unique<StiffnessAction>(mesh, V, alpha);
    Cgamma_action_ptr =
        std::make_unique<ExteriorMassAction>(mesh, V, facet_domain, alpha);
    K_action_ptr = std::make_unique<StiffnessAction>(mesh, V, alpha);
  }

  // LHS: (M+M_Γ) + (C+C_Γ)*gamma*dt + K*beta*dt2) @ in
  // Important note: the timer add roughly 1ms to execution time.
  template <typename Vector> void operator()(Vector &in, Vector &out) {
    // PROF_GPU_SCOPE("JACOBIAN", 7, 0);
    out.set(0.);
    PROF_GPU_START("M", 3, 0);
    (*M_action_ptr)(in, out);
    PROF_GPU_STOP("M");
    PROF_GPU_START("Mgamma", 3, 0);
    (*Mgamma_action_ptr)(in, out);
    PROF_GPU_STOP("Mgamma");
    PROF_GPU_START("C+K", 3, 0);
    (*C_action_ptr)(in, out, gamma *m_dt + beta * m_dt * m_dt);
    PROF_GPU_STOP("C+K");
    PROF_GPU_START("Cgamma", 3, 0);
    (*Cgamma_action_ptr)(in, out, gamma *m_dt);
    PROF_GPU_STOP("Cgamma");
    // PROF_GPU_START("K", 3, 0);
    // (*K_action_ptr)(in, out, beta *m_dt *m_dt);
    // PROF_GPU_STOP("K");
  };

  // void set_dt(U dt) { m_dt = dt; }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    // PROF_GPU_SCOPE("PC", 7, 0);
    diag_inv.set(0.);
    M_action_ptr->get_diag(diag_inv);
    Mgamma_action_ptr->get_diag(diag_inv);
    C_action_ptr->get_diag(diag_inv, gamma * m_dt);
    Cgamma_action_ptr->get_diag(diag_inv, gamma * m_dt);
    K_action_ptr->get_diag(diag_inv, beta * m_dt * m_dt);
    acc::inverse(diag_inv);
  }

private:
  T m_dt = 0.1;
  static constexpr T beta = 0.25;
  static constexpr T gamma = 0.5;
  std::unique_ptr<MassAction> M_action_ptr;
  std::unique_ptr<ExteriorMassAction> Mgamma_action_ptr;
  std::unique_ptr<StiffnessAction> C_action_ptr;
  std::unique_ptr<ExteriorMassAction> Cgamma_action_ptr;
  std::unique_ptr<StiffnessAction> K_action_ptr;
};

template <typename T, int P, int Q> class JacobianStream {
  using MassAction = acc::MatFreeMassBaseline3D<T, P, Q>;
  using StiffnessAction = acc::MatFreeStiffness3D<T, P, Q>;
  using ExteriorMassAction = acc::MatFreeMassExteriorBaseline3D<T, P, Q>;

public:
  JacobianStream(auto mesh, auto V, auto alpha, auto facet_domain) {
    M_action_ptr = std::make_unique<MassAction>(mesh, V, alpha);
    Mgamma_action_ptr =
        std::make_unique<ExteriorMassAction>(mesh, V, facet_domain, alpha);
    C_action_ptr = std::make_unique<StiffnessAction>(mesh, V, alpha);
    Cgamma_action_ptr =
        std::make_unique<ExteriorMassAction>(mesh, V, facet_domain, alpha);
    K_action_ptr = std::make_unique<StiffnessAction>(mesh, V, alpha);

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    // cudaStreamCreate(&stream5);
  }

  // LHS: (M+M_Γ) + (C+C_Γ)*gamma*dt + K*beta*dt2) @ in
  template <typename Vector> void operator()(Vector &in, Vector &out) {
    // PROF_GPU_SCOPE("JACOBIAN", 7, 0);
    out.set(0.);
    PROF_GPU_SCOPE("M", 3, stream1);
    PROF_GPU_SCOPE("Mgamma", 3, stream2);
    PROF_GPU_SCOPE("C+K", 3, stream3);
    PROF_GPU_SCOPE("Cgamma", 3, stream4);
    // PROF_GPU_SCOPE("K", 3, stream5);
    (*M_action_ptr)(in, out, 1, stream1);
    (*Mgamma_action_ptr)(in, out, 1, stream2);
    (*C_action_ptr)(in, out, gamma *m_dt + beta * m_dt * m_dt, stream3);
    (*Cgamma_action_ptr)(in, out, gamma *m_dt, stream4);
    // (*K_action_ptr)(in, out, beta *m_dt *m_dt, stream5);
  };

  // void set_dt(U dt) { m_dt = dt; }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    // PROF_GPU_SCOPE("PC", 7, 0);
    diag_inv.set(0.);
    M_action_ptr->get_diag(diag_inv);
    Mgamma_action_ptr->get_diag(diag_inv);
    C_action_ptr->get_diag(diag_inv, gamma * m_dt);
    Cgamma_action_ptr->get_diag(diag_inv, gamma * m_dt);
    K_action_ptr->get_diag(diag_inv, beta * m_dt * m_dt);
    acc::inverse(diag_inv);
  }

private:
  T m_dt = 0.1;
  static constexpr T beta = 0.25;
  static constexpr T gamma = 0.5;
  std::unique_ptr<MassAction> M_action_ptr;
  std::unique_ptr<ExteriorMassAction> Mgamma_action_ptr;
  std::unique_ptr<StiffnessAction> C_action_ptr;
  std::unique_ptr<ExteriorMassAction> Cgamma_action_ptr;
  std::unique_ptr<StiffnessAction> K_action_ptr;

  GpuStream stream1, stream2, stream3, stream4, stream5;
};

po::variables_map get_cli_config(int argc, char *argv[]) {
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()("help,h", "print usage message")
      ("matrix_comparison", po::bool_switch()->default_value(false), "Compare result to CPU matrix CG")
      ("N", po::value<int>()->default_value(10), "N")
      ("jacobian-type", po::value<int>()->default_value(1), "J");
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
  //   constexpr int polynomial_degree = POLYNOMIAL_DEGREE;
  constexpr int quadrature_points = polynomial_degree + 2;

  const int N = vm["N"].as<int>();
  const int jacobian_type = vm["jacobian-type"].as<int>();

  //   std::cout << "polynomial degree: " << polynomial_degree << std::endl;

  // ----------- 1. Problem Setup -----------
  // Create mesh and function space
  auto mesh = std::make_shared<mesh::Mesh<T>>(mesh::create_box<T>(
      comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {N, N, N},
      mesh::CellType::tetrahedron,
      mesh::create_cell_partitioner(mesh::GhostMode::none)));
  auto element = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::tetrahedron,
      polynomial_degree, basix::element::lagrange_variant::bernstein,
      basix::element::dpc_variant::unset, false);
  auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
      mesh, std::make_shared<const fem::FiniteElement<T>>(element), {}));
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

  // std::map<fem::IntegralType,
  //          std::vector<std::pair<std::int32_t, std::span<const
  //          std::int32_t>>>>
  //     facet_domains_data;
  // facet_domains_data[fem::IntegralType::exterior_facet].push_back(
  //     {1, facet_domain});
  topology->create_connectivity(2, 3);
  const std::vector bfacets = mesh::exterior_facet_indices(*topology);
  std::vector<std::int32_t> facet_domain =
      fem::compute_integration_domains(fem::IntegralType::exterior_facet,
                                       *V->mesh()->topology_mutable(), bfacets);
  // ----------- 3. GPU Matrix Free setup -----------
  auto element_p = this->V->element();

  const std::size_t tdim = mesh->topology()->dim();
  const std::size_t gdim = mesh->geometry().dim();
  this->number_of_local_cells =
      mesh->topology()->index_map(tdim)->size_local() +
      mesh->topology()->index_map(tdim)->num_ghosts();
  // Transfer V dofmap to the GPU

  thrust::device_vector<T> phi_d;
  std::span<const T> phi_d_span;

  thrust::device_vector<T> alpha_d;
  std::span<const T> alpha_d_span;

  thrust::device_vector<T> detJ_geom_d;
  std::span<const T> detJ_geom_d_span;

  thrust::device_vector<std::int32_t> dofmap_d;
  std::span<const std::int32_t> dofmap_d_span;

  auto dofmap = V->dofmap();

  dofmap_d_span = copy_to_device(
      dofmap->map().data_handle(),
      dofmap->map().data_handle() + dofmap->map().size(), dofmap_d, "dofmap");
  alpha_d_span = copy_to_device(alpha.begin(), alpha.end(), alpha_d, "alpha");

  // Construct quadrature points table
  // auto [qpts, qwts] = basix::quadrature::make_quadrature<T>(
  //     basix::quadrature::type::gauss_jacobi,
  //     basix::cell::type::tetrahedron, basix::polyset::type::standard, 2 * Q
  //     - 2);
  auto [qpts, qwts] = basix::quadrature::make_quadrature<U>(
      basix::quadrature::type::Default, basix::cell::type::tetrahedron,
      basix::polyset::type::standard, 2 * Q - 2);

  auto [phi_table, shape] = element_p->tabulate(qpts, {qpts.size() / 3, 3}, 0);

  std::cout << std::format("Table size = {}, qxn: {}x{}", phi_table.size(),
                           shape[1], shape[2])
            << std::endl;
  assert(shape[0] == 1 && shape[3] == 1);
  assert(nq == shape[1]);

  phi_d_span = copy_to_device(phi_table.begin(), phi_table.end(), phi_d, "phi");
  std::cout << "Precomputing geometry..." << std::endl;
  std::vector<U> detJ_geom = compute_geometry(mesh, qpts, qwts);

  detJ_geom_d_span = copy_to_device(detJ_geom.begin(), detJ_geom.end(),
                                    detJ_geom_d, "detJ_geom");

  Jacobian<T, polynomial_degree, quadrature_points> j_action(mesh, V, 1.0,
                                                             facet_domain);
  JacobianStream<T, polynomial_degree, quadrature_points> js_action(
      mesh, V, 1.0, facet_domain);
  FusedJacobian<T, polynomial_degree, quadrature_points> fj_action(
      mesh, V, 1.0, facet_domain);
  FusedJacobianStream<T, polynomial_degree, quadrature_points> fjs_action(
      mesh, V, 1.0, facet_domain);

  DeviceVector b_d(map, map_bs);
  b_d.set(1.);

  // ----------- 4. CG -----------
  int max_iters = 1;
  double rtol = 1e-7;

  // GPU
  dolfinx::acc::CGSolver<DeviceVector> cg(map, map_bs);
  cg.set_max_iterations(max_iters);
  cg.set_tolerance(rtol);

  DeviceVector x_d(map, map_bs);
  x_d.set(0.);
  int pcg_gpu_its;
  switch (jacobian_type) {
  case 1:
    std::cout << "BASELINE JACOBIAN ACTION" << std::endl;
    pcg_gpu_its = cg.solve(j_action, x_d, b_d, false);
    PROF_REPORT();
    PROF_RESET_ALL();
    break;
  case 2:
    std::cout << "STREAM JACOBIAN ACTION" << std::endl;
    pcg_gpu_its = cg.solve(js_action, x_d, b_d, false);
    PROF_REPORT();
    PROF_RESET_ALL();
    break;
  case 3:
    std::cout << "FUSED JACOBIAN ACTION" << std::endl;
    pcg_gpu_its = cg.solve(fj_action, x_d, b_d, false);
    PROF_REPORT();
    PROF_RESET_ALL();
    break;
  case 4:
    std::cout << "STREAM+FUSED JACOBIAN ACTION" << std::endl;
    pcg_gpu_its = cg.solve(fjs_action, x_d, b_d, false);
    PROF_REPORT();
    PROF_RESET_ALL();
    break;
  }

  // x_d.set(0.);
  // int cg_gpu_its = cg.solve(gpu_action, x_d, b_d, false);

  if (dolfinx::MPI::rank(comm) == 0) {
    std::cout << "Number of GPU PCG iterations " << pcg_gpu_its << std::endl;
    // std::cout << "Number of GPU CG iterations " << cg_gpu_its << std::endl;
  }

  // if (matrix_comparison) {
  //   // ----------- 2. CPU Matrix Free setup -----------
  //   auto coeff = fem::allocate_coefficient_storage(*M);
  //   std::vector<T> constants = fem::pack_constants(*M);

  //   // Create function for computing the action of A on x (y = Ax)
  //   auto cpu_action = [&M, &ui, &coeff, &constants](auto &x, auto &y) {
  //     y.set(0.0);

  //     // Update coefficient ui (just copy data from x to ui)
  //     std::ranges::copy(x.array(), ui->x()->mutable_array().begin());

  //     // Compute action of A on x
  //     fem::pack_coefficients(*M, coeff);
  //     fem::assemble_vector(y.mutable_array(), *M, std::span<const
  //     T>(constants),
  //                          fem::make_coefficients_span(coeff));
  //   };

  //   // CPU
  //   la::Vector<T> x(map, map_bs);
  //   int cpu_its = linalg::cg(x, b, cpu_action, max_iters, rtol);

  //   la::Vector<T> x_h(map, map_bs);
  //   thrust::copy(x_d.thrust_vector().begin(), x_d.thrust_vector().end(),
  //                x_h.mutable_array().begin());

  //   // for(int i = 0; i < ndofs_local; ++i) {
  //   //   std::cout << "y_d[" << i << "]=" << y_h.array()[i] << std::endl;
  //   // }
  //   if (dolfinx::MPI::rank(comm) == 0) {
  //     std::cout << "Number of CPU CG iterations " << cpu_its << std::endl;
  //   }

  //   double eps = rtol * 10;
  //   bool check = true;
  //   for (int i = 0; i < ndofs_local; ++i) {
  //     if (std::abs(x.array()[i] - x_h.array()[i]) > eps) {
  //       // std::cout << x.array()[i] << " " << x_h.array()[i] << std::endl;
  //       check = false;
  //       // break;
  //     }
  //   }
  //   std::cout << (check ? "PASSED" : "FAILED") << std::endl;
  // }
}

/// Main program
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (rank == 0) {
      std::cout << device_information();
    }
    auto vm = get_cli_config(argc, argv);

    solver<T>(MPI_COMM_WORLD, vm);
  }
  MPI_Finalize();
  return 0;
}
