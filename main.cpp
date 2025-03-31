// # Mass matrix operation

#include "mass.h"
#include <algorithm>
#include <basix/finite-element.h>
#include <cmath>
#include <concepts>
#include <dolfinx.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <memory>
#include <petscsystypes.h>

#include "src/util.hpp"
#include "src/vector.hpp"
#include "src/mass.hpp"



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

namespace linalg
{

template <typename T>
void copy(const la::Vector<T>& in, la::Vector<T>& out) {
  std::span<const T> _in = in.array();
  std::span<T> _out = out.mutable_array();
  std::copy(_in.begin(), _in.end(), _out.begin());
}

/// @brief Compute vector r = alpha * x + y.
/// @param[out] r
/// @param[in] alpha
/// @param[in] x
/// @param[in] y
void axpy(auto&& r, auto alpha, auto&& x, auto&& y)
{
  std::ranges::transform(x.array(), y.array(), r.mutable_array().begin(),
                         [alpha](auto x, auto y) { return alpha * x + y; });
}
} // namespace linalg

template <typename T, std::floating_point U>
void solver(MPI_Comm comm)
{
  int order = 2;
  int nq = 3;

  // ----------- 1. Problem Setup -----------
  // Create mesh and function space
  auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_rectangle<U>(
      comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {10, 10}, mesh::CellType::triangle,
      mesh::create_cell_partitioner(mesh::GhostMode::none)));
  auto element = basix::create_element<U>(
      basix::element::family::P, basix::cell::type::triangle, 2,
      basix::element::lagrange_variant::bernstein,
      basix::element::dpc_variant::unset, false);
  auto V = std::make_shared<fem::FunctionSpace<U>>(
      fem::create_functionspace(mesh, element, {}));
  // Prepare and set Constants for the bilinear form
  auto f = std::make_shared<fem::Constant<T>>(-6.0);

  // Define variational forms
  auto L = std::make_shared<fem::Form<T, U>>(
      fem::create_form<T>(*form_mass_L, {V}, {}, {{"f", f}}, {}, {}));

  // Action of the bilinear form "a" on a function ui
  auto ui = std::make_shared<fem::Function<T, U>>(V);
  auto M = std::make_shared<fem::Form<T, U>>(
      fem::create_form<T>(*form_mass_M, {V}, {{"ui", ui}}, {{}}, {}, {}));

  // -------- 1.2 Boundary Conditions --------
  auto u_D = std::make_shared<fem::Function<T, U>>(V);
  u_D->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> f;
        for (std::size_t p = 0; p < x.extent(1); ++p)
          f.push_back(1 + x(0, p) * x(0, p) + 2 * x(1, p) * x(1, p));
        return {f, {f.size()}};
      });
  mesh->topology_mutable()->create_connectivity(1, 2);
  const std::vector<std::int32_t> facets
      = mesh::exterior_facet_indices(*mesh->topology());
  std::vector<std::int32_t> bdofs = fem::locate_dofs_topological(
      *V->mesh()->topology_mutable(), *V->dofmap(), 1, facets);
  auto bc = std::make_shared<const fem::DirichletBC<T>>(u_D, bdofs);
  
  // Assemble RHS vector
  la::Vector<T> b(V->dofmap()->index_map, V->dofmap()->index_map_bs());
  fem::assemble_vector(b.mutable_array(), *L);

  b.scatter_rev(std::plus<T>());
  b.scatter_fwd();

  // ----------- 2. CPU Matrix Free setup -----------
  // Pack coefficients and constants
  auto coeff = fem::allocate_coefficient_storage(*M);
  std::vector<T> constants = fem::pack_constants(*M);

  // Create function for computing the action of A on x (y = Ax)
  auto cpu_action = [&M, &ui, &bc, &coeff, &constants](auto& x, auto& y)
  {
    y.set(0.0);

    // Update coefficient ui (just copy data from x to ui)
    std::ranges::copy(x.array(), ui->x()->mutable_array().begin());

    // Compute action of A on x
    fem::pack_coefficients(*M, coeff);
    fem::assemble_vector(y.mutable_array(), *M, std::span<const T>(constants),
                         fem::make_coefficients_span(coeff));

    // Accumulate ghost values
    y.scatter_rev(std::plus<T>());

    // Update ghost values
    y.scatter_fwd();
  };

  // ----------- 3. GPU Matrix Free setup -----------
  // Transfer V dofmap to the GPU
  auto dofmap = V->dofmap();
  thrust::device_vector<std::int32_t> dofmap_d(
  dofmap->map().data_handle(), dofmap->map().data_handle() + dofmap->map().size());
  std::span<const std::int32_t> dofmap_d_span(thrust::raw_pointer_cast(dofmap_d.data()),
                                              dofmap_d.size());
  std::cout << std::format("Send dofmap to GPU (size = {} bytes)", dofmap_d.size() * sizeof(std::int32_t)) << std::endl;
  
  auto map = V->dofmap()->index_map;
  int map_bs = V->dofmap()->index_map_bs();
  
  acc::MatFreeMass<T> gpu_action(order, nq);

  // ----------- 4. Matrix Free apply -----------
  
  // CPU
  // auto u = std::make_shared<fem::Function<T>>(V);
  // auto x = *u->x();
  
  la::Vector<T> x(map, map_bs);
  la::Vector<T> y(map, map_bs);
  y.set(0.);
  linalg::copy(b, x);

  std::cout << "norm(x)=" << la::norm(x) << "\n";
  cpu_action(x, y);
  std::cout << "norm(y)=" << la::norm(y) << "\n";


  // GPU
  DeviceVector x_d(map, map_bs);
  DeviceVector y_d(map, map_bs);
  y_d.set(T{0.0});
  x_d.copy_from_host(b);
  std::cout << "norm(x_d)=" << acc::norm(x_d) << "\n";
  // GPU action
  gpu_action(x_d, y_d);
  std::cout << "norm(y_d)=" << acc::norm(y_d) << "\n";


  // Compute L2 error (squared) of the solution vector e = (u - u_d, u
  // - u_d)*dx
  // auto E = std::make_shared<fem::Form<T>>(fem::create_form<T, U>(
      // *form_mass_E, {}, {{"uexact", u_D}, {"usol", u}}, {}, {}, {}, mesh));
  // T error = fem::assemble_scalar(*E);
  // if (dolfinx::MPI::rank(comm) == 0)
  // {
  //   // std::cout << "Number of CG iterations " << num_it << std::endl;
  //   std::cout << "Finite element error (L2 norm (squared)) " << std::abs(error)
  //             << std::endl;
  // }
}


/// Main program
int main(int argc, char* argv[])
{
  using T = PetscScalar;
  using U = typename dolfinx::scalar_value_type_t<T>;
  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (rank == 0)
    {
      std::cout << device_information();
    }

    solver<T, U>(MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return 0;
}
