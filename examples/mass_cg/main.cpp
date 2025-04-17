// # Mass matrix operation

#include "mass_cg.h"
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
#include "src/mesh.hpp"
#include "src/quadrature.hpp"
#include "src/util.hpp"
#include "src/vector.hpp"
#include "src/cg.hpp"


using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

#if USE_HIP
using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;
#elif USE_CUDA
using DeviceVector = dolfinx::acc::Vector<T, acc::Device::CUDA>;
#else
static_assert(false)
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

/// @brief Solve problem A.x = b using the conjugate gradient (CG)
/// method.
///
/// @param[in, out] x Solution vector, may be set to an initial guess
/// hence no zeroed.
/// @param[in] b Right-hand side vector.
/// @param[in] action Function that computes the action of the linear
/// operator on a vector.
/// @param[in] kmax Maximum number of iterations
/// @param[in] rtol Relative tolerances for convergence
/// @return Number of CG iterations.
/// @pre The ghost values of `x` and `b` must be updated before this
/// function is called.
int cg(auto &x, auto &b, auto action, int kmax = 50, double rtol = 1e-8) {
  using T = typename std::decay_t<decltype(x)>::value_type;

  // Create working vectors
  la::Vector r(b), y(b);

  // Compute initial residual r0 = b - Ax0
  action(x, y);
  axpy(r, T(-1), y, b);

  // Create p work vector
  la::Vector p(r);

  // Iterations of CG
  auto rnorm0 = la::squared_norm(r);
  auto rtol2 = rtol * rtol;
  auto rnorm = rnorm0;
  int k = 0;
  while (k < kmax) {
    ++k;

    // Compute y = A p
    action(p, y);

    // Compute alpha = r.r/p.y
    T alpha = rnorm / la::inner_product(p, y);

    // Update x (x <- x + alpha*p)
    axpy(x, alpha, p, x);

    // Update r (r <- r - alpha*y)
    axpy(r, -alpha, y, r);

    // Update residual norm
    auto rnorm_new = la::squared_norm(r);
    T beta = rnorm_new / rnorm;
    rnorm = rnorm_new;

    if (rnorm / rnorm0 < rtol2)
      break;

    // Update p (p <- beta * p + r)
    axpy(p, beta, p, r);
  }

  return k;
}
} // namespace linalg

template <typename T, std::floating_point U> void solver(MPI_Comm comm) {
  constexpr int polynomial_degree = 2;
  // TODO: verify if expression integrates exactly. Probably? Comes from Basix.
  // constexpr int quadrature_points = (polynomial_degree + 2) / 2;
  constexpr int quadrature_points = 3;

  // ----------- 1. Problem Setup -----------
  // Create mesh and function space
  auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_rectangle<U>(
      comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {10, 10}, mesh::CellType::triangle,
      mesh::create_cell_partitioner(mesh::GhostMode::none)));
  auto element = basix::create_element<U>(
      basix::element::family::P, basix::cell::type::triangle, polynomial_degree,
      basix::element::lagrange_variant::bernstein,
      basix::element::dpc_variant::unset, false);
  auto V = std::make_shared<fem::FunctionSpace<U>>(
      fem::create_functionspace(mesh, element, {}));

  auto f = std::make_shared<fem::Function<T>>(V);
  auto ui = std::make_shared<fem::Function<T, U>>(V);
  auto M = std::make_shared<fem::Form<T, U>>(fem::create_form<T>(
      *form_mass_cg_M, {V}, {{"ui", ui}}, {{}}, {}, {}));
  auto L = std::make_shared<fem::Form<T, U>>(
      fem::create_form<T>(*form_mass_cg_L, {V}, {{"f", f}}, {}, {}, {}));

  f->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>> {
        std::vector<T> out;
        for (std::size_t p = 0; p < x.extent(1); ++p) {
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
  };

  // ----------- 3. GPU Matrix Free setup -----------
  DeviceVector b_d(map, map_bs);
  b_d.copy_from_host(b);
  la::Vector<T> alpha(map, map_bs);
  alpha.set(1.0);
  acc::MatFreeMass<U, polynomial_degree, quadrature_points> gpu_action(mesh, V, alpha.array());

  // ----------- 4. CG -----------
  int max_iters = 200;
  double rtol = 1e-6;

  // CPU
  la::Vector<T> x(map, map_bs);
  int cpu_its = linalg::cg(x, b, cpu_action, max_iters, rtol);

  // GPU
  DeviceVector x_d(map, map_bs);

  dolfinx::acc::CGSolver<DeviceVector> cg(map, map_bs);
  cg.set_max_iterations(max_iters);
  cg.set_tolerance(rtol);

  int gpu_its = cg.solve(gpu_action, x_d, b_d, false);

  la::Vector<T> x_h(map, map_bs);
  thrust::copy(x_d.thrust_vector().begin(), x_d.thrust_vector().end(),
               x_h.mutable_array().begin());

  if (dolfinx::MPI::rank(comm) == 0) {
    std::cout << "Number of CPU CG iterations " << cpu_its << std::endl;
    std::cout << "Number of GPU CG iterations " << gpu_its << std::endl;
  }
  // for(int i = 0; i < ndofs_local; ++i) {
  //   std::cout << "y_d[" << i << "]=" << y_h.array()[i] << std::endl;
  // }

  double eps = rtol;
  bool check = true;
  for (int i = 0; i < ndofs_local; ++i) {
    if (std::abs(x.array()[i] - x_h.array()[i]) > eps) {
      std::cout << x.array()[i] << " " << x_h.array()[i] << std::endl;
      check = false;
      // break;
    }
  }
  std::cout << (check ? "PASSED" : "FAILED") << std::endl;
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
    if (rank == 0) {
      std::cout << device_information();
    }

    solver<T, U>(MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return 0;
}
