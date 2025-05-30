// Copyright(C) 2023-2024 Igor A. Baratta, Chris N. Richardson
// SPDX-License-Identifier:    MIT

#pragma once

#include "util.hpp"
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <basix/quadrature.h>
#include <iomanip>
#include <thrust/device_vector.h>
/// @brief Computes weighted geometry tensor G from the coordinates and
/// quadrature weights.
/// @param [in] xgeom Geometry points [*, 3]
/// @param [out] G_entity geometry data [n_entities, nq, 6]
/// @param [in] geometry_dofmap Location of coordinates for each cell in
/// xgeom [*, ncdofs]
/// @param [in] _dphi Basis derivative tabulation for cell at quadrature
/// points [3, nq, ncdofs]
/// @param [in] weights Quadrature weights [nq]
/// @param [in] entities list of cells to compute for [n_entities]
/// @param [in] n_entities total number of cells to compute for
/// @tparam T scalar type
/// @tparam D degree
template <typename T, int D>
__global__ void geometry_computation(const T *xgeom, T *G_entity,
                                     const std::int32_t *geometry_dofmap,
                                     const T *_dphi, const T *weights,
                                     const int *entities, int n_entities) {
  // One block per cell
  int c = blockIdx.x;

  // Limit to cells in list
  if (c >= n_entities)
    return;

  // Cell index
  int cell = entities[c];

  // Number of quadrature points (must match arrays in weights and dphi)
  constexpr int nq =
      (D == 2) * 4 + (D == 3) * 14 + (D == 4) * 24 + (D == 5) * 45;

  // Number of coordinate dofs
  constexpr int ncdofs = 4;

  // Geometric dimension
  constexpr int gdim = 3;

  extern __shared__ T shared_mem[];

  // coord_dofs has shape [ncdofs, gdim]
  T *_coord_dofs = shared_mem;

  // First collect geometry into shared memory
  int iq = threadIdx.x;

  if (blockDim.x < 12 and iq == 0) {
    for (int k = 0; k < 12; ++k) {
      int i = k / gdim;
      int j = k % gdim;
      _coord_dofs[k] = xgeom[3 * geometry_dofmap[cell * ncdofs + i] + j];
    }
  } else if (iq < ncdofs * gdim) {
    int i = iq / gdim;
    int j = iq % gdim;
    _coord_dofs[iq] = xgeom[3 * geometry_dofmap[cell * ncdofs + i] + j];
  }

  __syncthreads();

  // One quadrature point per thread

  // Jacobian
  T J[3][3];
  auto coord_dofs = [&_coord_dofs](int i, int j) -> T & {
    return _coord_dofs[i * gdim + j];
  };

  // For each quadrature point / thread
  {
    // dphi has shape [gdim, ncdofs, nq]
    auto dphi = [&_dphi, iq](int i, int j) -> const T {
      return _dphi[(i * nq + iq) * ncdofs + j];
    };
    for (std::size_t i = 0; i < gdim; i++) {
      for (std::size_t j = 0; j < gdim; j++) {
        J[i][j] = 0.0;
        for (std::size_t k = 0; k < ncdofs; k++)
          J[i][j] += coord_dofs(k, i) * dphi(j, k);
      }
    }

    // Components of K = J^-1 (detJ)
    T K[3][3] = {{J[1][1] * J[2][2] - J[1][2] * J[2][1],
                  -J[0][1] * J[2][2] + J[0][2] * J[2][1],
                  J[0][1] * J[1][2] - J[0][2] * J[1][1]},
                 {-J[1][0] * J[2][2] + J[1][2] * J[2][0],
                  J[0][0] * J[2][2] - J[0][2] * J[2][0],
                  -J[0][0] * J[1][2] + J[0][2] * J[1][0]},
                 {J[1][0] * J[2][1] - J[1][1] * J[2][0],
                  -J[0][0] * J[2][1] + J[0][1] * J[2][0],
                  J[0][0] * J[1][1] - J[0][1] * J[1][0]}};

    T detJ = J[0][0] * K[0][0] - J[0][1] * K[1][0] + J[0][2] * K[2][0];

    int offset = (c * nq * 6 + iq);
    const T w = weights[iq];
    G_entity[offset] =
        (K[0][0] * K[0][0] + K[0][1] * K[0][1] + K[0][2] * K[0][2]) * w / detJ;
    G_entity[offset + nq] =
        (K[1][0] * K[0][0] + K[1][1] * K[0][1] + K[1][2] * K[0][2]) * w / detJ;
    G_entity[offset + 2 * nq] =
        (K[2][0] * K[0][0] + K[2][1] * K[0][1] + K[2][2] * K[0][2]) * w / detJ;
    G_entity[offset + 3 * nq] =
        (K[1][0] * K[1][0] + K[1][1] * K[1][1] + K[1][2] * K[1][2]) * w / detJ;
    G_entity[offset + 4 * nq] =
        (K[2][0] * K[1][0] + K[2][1] * K[1][1] + K[2][2] * K[1][2]) * w / detJ;
    G_entity[offset + 5 * nq] =
        (K[2][0] * K[2][0] + K[2][1] * K[2][1] + K[2][2] * K[2][2]) * w / detJ;
  }
}

/// Compute 3d index from 1d indices
template <int nq> __device__ __forceinline__ int ijk(int i, int j, int k) {
  // Precompute (nq * (nq + 1)) since it's used multiple times

  // NB pad=0 may be better for AMD, pad=1 for NVIDIA
  constexpr int pad = 0;
  constexpr int nq2 = nq * (nq + pad);
  return i * nq2 + j * (nq + pad) + k;
}

/// Compute b = A * u where A is the stiffness operator for a set of
/// entities (cells or facets) in a mesh.
///
/// The stiffness operator is defined as:
///
///     A = ∑_i ∫_Ω C ∇ϕ_i ∇ϕ_j dx
///
/// where C is a constant, ϕ_i and ϕ_j are the basis functions of the
/// finite element space, and ∇ϕ_i is the gradient of the basis
/// function. The integral is computed over the domain Ω of the entity
/// using sum factorization. The basis functions are defined on the
/// reference element and are transformed to the physical element using
/// the geometry operator G. G is a 3x3 matrix per quadrature point per
/// entity.
///
/// @tparam T Data type of the input and output arrays
/// @tparam P Polynomial degree of the basis functions
/// @tparam Q Number of quadrature points in 1D
/// @param u Input vector of size (ndofs,)
/// @param entity_constants Array of size (n_entities,) with the
/// constant C for each entity
/// @param b Output vector of size (ndofs,)
/// @param G_entity Array of size (n_entities, nq, 6) with the geometry
/// operator G for each entity
/// @param entity_dofmap Array of size (n_entities, ndofs) with the
/// dofmap for each entity
/// @param phi0_in Array of size (nq, ndofs) with the interpolation basis
/// functions in 1D. u1_i = phi0_(ij) u_j, where u are the dofs
/// associated with the element (degree P), and u1 are the dofs for the
/// finite elment (degree >= P) that u is interpolated into.
/// @param dphi1_in Array of size (nq, nq) with the 1D basis function
/// derivatives. FIXME: layout is (point_idx, dphi_i)?
/// @param entities List of entities to compute on
/// @param n_entities Number of entries in `entities`
/// @param bc_marker Array of size (ndofs,) with the boundary condition
/// marker
///
/// @note The kernel is launched with grid of blocks, where each
/// block is responsible for computing the stiffness operator for several cells.
/// The block size is ncells*ndofs.
template <typename T, int P>
__global__ void
stiffness_operator(const T *__restrict__ u,
                   const T *__restrict__ entity_constants, T *__restrict__ b,
                   const T *__restrict__ G_entity,
                   const std::int32_t *__restrict__ entity_dofmap,
                   const std::int16_t *__restrict__ dphi_lookup,
                   const T *__restrict__ dphi_value,
                   const int *__restrict__ entities, int n_entities) {
  constexpr int ncells = 32 * (P == 2) + 16 * (P == 3) + 10 * (P == 4);
  constexpr int nd = 10 * (P == 2) + 20 * (P == 3) + 35 * (P == 4);
  constexpr int nq = 4 * (P == 2) + 14 * (P == 3) + 24 * (P == 4);

  // block_id is the cell (or facet) index
  const int block_id = blockIdx.x;

  // thread_id is either the dof or quadrature point index
  const int thread_id = threadIdx.x;

  __shared__ T scratch1[nd * ncells];
  __shared__ T scratch2[nd * ncells];
  __shared__ T scratch3[nd * ncells];

  // Copy dphi to shared memory
  __shared__ T dphi1[3 * nq * nd];

  constexpr int nw = (3 * nq) / ncells + 1;
  for (int i = 0; i < nw; ++i) {
    int id = i * ncells * nd + thread_id;
    if (id < 3 * nd * nq)
      dphi1[id] = dphi_value[id];
  }

  // Get cell and thread "within cell" index
  int tx = thread_id / ncells;
  int cell_local = thread_id - tx * ncells;
  int cell = block_id * ncells + cell_local;

  if (cell >= n_entities)
    return;

  // Get dof value that this thread is responsible for, and
  // place in shared memory.
  int dof = -1;

  const int entity_index = entities[cell];
  scratch1[thread_id] = 0.0;
  {
    dof = entity_dofmap[entity_index * nd + tx];
    // if (bc_marker[dof])
    // {
    //   b[dof] = u[dof];
    //   dof = -1;
    // }
    // else
    scratch1[thread_id] = u[dof];
  }

  __syncthreads(); // Make sure all threads have written to shared memory

  auto dphix = [&nd](auto j, auto k) { return dphi1[j * nd + k]; };
  auto dphiy = [&nd, &nq](auto j, auto k) {
    return dphi1[nd * nq + j * nd + k];
  };
  auto dphiz = [&nd, &nq](auto j, auto k) {
    return dphi1[2 * nd * nq + j * nd + k];
  };
  // auto dphiy = [&nd, &dofperm_y, &qpperm_y](auto j, auto k)
  //   { return dphi1[qpperm_y[j] * nd + dofperm_y[k]]; };
  //   auto dphiz = [&nd](auto j, auto k) { return dphi1[qpperm_z[j] * nd +
  //   dofperm_z[k]]; };

  // Compute val{x,y,z} at quadrature points

  T fw0 = 0, fw1 = 0, fw2 = 0;

  if (tx < nq) {
    T val_x = 0.0;
    T val_y = 0.0;
    T val_z = 0.0;
    for (int idof = 0; idof < nd; ++idof) {
      val_x += dphix(tx, idof) * scratch1[idof * ncells + cell_local];
      val_y += dphiy(tx, idof) * scratch1[idof * ncells + cell_local];
      val_z += dphiz(tx, idof) * scratch1[idof * ncells + cell_local];
    }

    // Apply geometric transformation to data at quadrature point
    const int gid = cell * nq * 6 + tx;
    const T G0 = G_entity[gid + nq * 0];
    const T G1 = G_entity[gid + nq * 1];
    const T G2 = G_entity[gid + nq * 2];
    const T G3 = G_entity[gid + nq * 3];
    const T G4 = G_entity[gid + nq * 4];
    const T G5 = G_entity[gid + nq * 5];

    const T coeff = entity_constants[cell];

    fw0 = coeff * (G0 * val_x + G1 * val_y + G2 * val_z);
    fw1 = coeff * (G1 * val_x + G3 * val_y + G4 * val_z);
    fw2 = coeff * (G2 * val_x + G4 * val_y + G5 * val_z);
  }

  __syncthreads();

  // Store values at quadrature points
  scratch1[thread_id] = fw0;
  scratch2[thread_id] = fw1;
  scratch3[thread_id] = fw2;
  __syncthreads();

  T grad_x = 0.0;
  T grad_y = 0.0;
  T grad_z = 0.0;

  for (int iq = 0; iq < nq; ++iq) {
    grad_x += dphix(iq, tx) * scratch1[iq * ncells + cell_local];
    grad_y += dphiy(iq, tx) * scratch2[iq * ncells + cell_local];
    grad_z += dphiz(iq, tx) * scratch3[iq * ncells + cell_local];
  }

  // Sum contributions
  T yd = grad_x + grad_y + grad_z;

  // Write back to global memory
  if (dof != -1)
    atomicAdd(&b[dof], yd);
}

template <typename T, int P>
__global__ void
mat_diagonal_tet(const T *entity_constants, T *b, const T *G_entity,
                 const std::int32_t *entity_dofmap, const T *dphi,
                 const int *entities, int n_entities) {
  constexpr int ncells = 32 * (P == 2) + 16 * (P == 3) + 10 * (P == 4);
  constexpr int nd = 10 * (P == 2) + 20 * (P == 3) + 35 * (P == 4);
  constexpr int nq = 4 * (P == 2) + 14 * (P == 3) + 24 * (P == 4);

  // block_id is the cell (or facet) index
  const int block_id = blockIdx.x;

  // Get cell and thread "within cell" index
  int thread_id = threadIdx.x;
  int tx = thread_id / ncells;
  int cell_local = thread_id - tx * ncells;
  int cell = block_id * ncells + cell_local;

  if (cell >= n_entities)
    return;

  // DG-0 Coefficient
  T coeff = entity_constants[cell];

  auto dphi_ = [&](auto r, auto j, auto k) {
    return dphi[r * nd * nq + j * nd + k];
  };
  auto G_ = [&](int r, int ix) { return G_entity[nq * (cell * 6 + r) + ix]; };

  T val = 0.0;
  for (int iq = 0; iq < nq; ++iq) {
    val += G_(0, iq) * dphi_(0, iq, tx) * dphi_(0, iq, tx);
    val += G_(1, iq) * dphi_(0, iq, tx) * dphi_(1, iq, tx);
    val += G_(2, iq) * dphi_(0, iq, tx) * dphi_(2, iq, tx);
    val += G_(1, iq) * dphi_(1, iq, tx) * dphi_(0, iq, tx);
    val += G_(3, iq) * dphi_(1, iq, tx) * dphi_(1, iq, tx);
    val += G_(4, iq) * dphi_(1, iq, tx) * dphi_(2, iq, tx);
    val += G_(2, iq) * dphi_(2, iq, tx) * dphi_(0, iq, tx);
    val += G_(4, iq) * dphi_(2, iq, tx) * dphi_(1, iq, tx);
    val += G_(5, iq) * dphi_(2, iq, tx) * dphi_(2, iq, tx);
  }

  const int entity_index = entities[cell];

  int dof = entity_dofmap[entity_index * nd + tx];
  // if (bc_marker[dof])
  // b[dof] = T(1.0);
  // else
  atomicAdd(&b[dof], coeff * val);
}

namespace dolfinx::acc {

const std::map<int, int> Qdegree_map = {{1, 1}, {2, 3},  {3, 4},  {4, 6},
                                        {5, 8}, {6, 10}, {7, 12}, {8, 14}};

template <typename T> class MatFreeLaplacianTet {
public:
  using value_type = T;

  MatFreeLaplacianTet(int degree, std::span<const T> coefficients,
                      std::span<const std::int32_t> dofmap,
                      std::span<const T> xgeom,
                      std::span<const std::int32_t> geometry_dofmap,
                      std::span<const T> dphi_geometry,
                      std::span<const T> G_weights, std::span<const T> G_points,
                      const std::vector<int> &lcells,
                      const std::vector<int> &bcells,
                      std::size_t batch_size = 0)
      : degree(degree), cell_constants(coefficients), cell_dofmap(dofmap),
        xgeom(xgeom), geometry_dofmap(geometry_dofmap),
        dphi_geometry(dphi_geometry), G_weights(G_weights),
        batch_size(batch_size) {
    auto element = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::tetrahedron, degree,
        basix::element::lagrange_variant::gll_warped,
        basix::element::dpc_variant::unset, false);

    // Tabulate tet at quadrature points
    auto [table, shape] =
        element.tabulate(1, G_points, {G_points.size() / 3, 3});

    spdlog::info("Table size = {} {}x{}x{}x{}", table.size(), shape[0],
                 shape[1], shape[2], shape[3]);
    // Compress table to device

    std::vector<T> table_values(table.begin() + table.size() / 4, table.end());
    // std::sort(table_values.begin(), table_values.end());
    // table_values.erase(std::unique(table_values.begin(), table_values.end(),
    //                                [&eps](T a, T b) { return std::abs(a - b)
    //                                < eps; }),
    //                    table_values.end());
    spdlog::info("Unique table entries: {}", table_values.size());
    dphi_value_d.resize(table_values.size());
    thrust::copy(table_values.begin(), table_values.end(),
                 dphi_value_d.begin());

    // Find where the entries are
    T eps = 1e-6;
    std::vector<std::int16_t> table_lookup;
    for (int i = table.size() / 4; i < table.size(); ++i) {
      T v = table[i];
      auto it = std::find_if(table_values.begin(), table_values.end(),
                             [&v, &eps](T w) { return std::abs(v - w) < eps; });
      assert(it != table_values.end());
      assert(it - table_values.begin() < 32768);
      table_lookup.push_back(it - table_values.begin());
    }
    dphi_lookup_d.resize(table_lookup.size());
    thrust::copy(table_lookup.begin(), table_lookup.end(),
                 dphi_lookup_d.begin());

    // Copy lists of local and boundary cells to device
    lcells_device.resize(lcells.size());
    thrust::copy(lcells.begin(), lcells.end(), lcells_device.begin());
    bcells_device.resize(bcells.size());
    thrust::copy(bcells.begin(), bcells.end(), bcells_device.begin());

    // If we're not batching the geometry, precompute it
    if (batch_size == 0) {
      // FIXME Store cells and local/ghost offsets instead to avoid this?
      spdlog::info("Precomputing geometry");
      thrust::device_vector<std::int32_t> cells_d(lcells_device.size() +
                                                  bcells_device.size());
      thrust::copy(lcells_device.begin(), lcells_device.end(), cells_d.begin());
      thrust::copy(bcells_device.begin(), bcells_device.end(),
                   cells_d.begin() + lcells_device.size());
      std::span<std::int32_t> cell_list_d(
          thrust::raw_pointer_cast(cells_d.data()), cells_d.size());

      compute_geometry(degree, cell_list_d);
      device_synchronize();
    }
  }

  // Compute weighted geometry data on GPU
  template <int D = 2>
  void compute_geometry(int degree, std::span<int> cell_list_d) {
    if constexpr (D < 10) {
      if (degree > D)
        compute_geometry<D + 1>(degree, cell_list_d);
      else {
        assert(degree == D);
        G_entity.resize(G_weights.size() * cell_list_d.size() * 6);
        dim3 block_size(G_weights.size());
        dim3 grid_size(cell_list_d.size());

        spdlog::info("xgeom size {}", xgeom.size());
        spdlog::info("G_entity size {}", G_entity.size());
        spdlog::info("geometry_dofmap size {}", geometry_dofmap.size());
        spdlog::info("dphi_geometry size {}", dphi_geometry.size());
        spdlog::info("G_weights size {}", G_weights.size());
        spdlog::info("cell_list_d size {}", cell_list_d.size());
        spdlog::info("Calling geometry_computation [{} {}]", D, degree);

        std::size_t shm_size = 12 * sizeof(T); // coordinate size (4x3)
        geometry_computation<T, D><<<grid_size, block_size, shm_size, 0>>>(
            xgeom.data(), thrust::raw_pointer_cast(G_entity.data()),
            geometry_dofmap.data(), dphi_geometry.data(), G_weights.data(),
            cell_list_d.data(), cell_list_d.size());
      }
    } else
      throw std::runtime_error("Unsupported degree [geometry]");
  }

  template <int P, typename Vector>
  void impl_operator(Vector &in, Vector &out) {
    spdlog::debug("impl_operator operator start");

    in.scatter_fwd_begin();

    // Set ndofs and ncells for degree
    constexpr int ncells = 32 * (P == 2) + 16 * (P == 3) + 10 * (P == 4);
    constexpr int ndofs = 10 * (P == 2) + 20 * (P == 3) + 35 * (P == 4);
    constexpr int nq = 4 * (P == 2) + 14 * (P == 3) + 24 * (P == 4);

    if (!lcells_device.empty()) {
      std::size_t i = 0;
      std::size_t i_batch_size =
          (batch_size == 0) ? lcells_device.size() : batch_size;
      while (i < lcells_device.size()) {
        std::size_t i_next = std::min(lcells_device.size(), i + i_batch_size);

        std::span<int> cell_list_d(
            thrust::raw_pointer_cast(lcells_device.data()) + i, i_next - i);
        i = i_next;

        if (batch_size > 0) {
          spdlog::debug("Calling compute_geometry on local cells [{}]",
                        cell_list_d.size());
          compute_geometry(P, cell_list_d);
          device_synchronize();
        }

        dim3 block_size(ndofs * ncells);
        dim3 grid_size(cell_list_d.size() / ncells + 1);

        spdlog::debug("Calling stiffness_operator on local cells [{}]",
                      cell_list_d.size());
        T *x = in.mutable_array().data();
        T *y = out.mutable_array().data();

        stiffness_operator<T, P><<<grid_size, block_size>>>(
            x, cell_constants.data(), y,
            thrust::raw_pointer_cast(G_entity.data()), cell_dofmap.data(),
            thrust::raw_pointer_cast(dphi_lookup_d.data()),
            thrust::raw_pointer_cast(dphi_value_d.data()), cell_list_d.data(),
            cell_list_d.size());

        check_device_last_error();
      }
    }

    spdlog::debug("impl_operator done lcells");

    spdlog::debug("cell_constants size {}", cell_constants.size());
    spdlog::debug("in size {}", in.array().size());
    spdlog::debug("out size {}", out.array().size());
    spdlog::debug("G_entity size {}", G_entity.size());
    spdlog::debug("cell_dofmap size {}", cell_dofmap.size());
    spdlog::debug("dphi1_value size {}", dphi_value_d.size());
    spdlog::debug("dphi1_lookup size {}", dphi_lookup_d.size());
    // spdlog::debug("bc_marker size {}", bc_marker.size());

    in.scatter_fwd_end();

    spdlog::debug("impl_operator after scatter");

    if (!bcells_device.empty()) {
      spdlog::debug("impl_operator doing bcells. bcells size = {}",
                    bcells_device.size());
      std::span<int> cell_list_d(thrust::raw_pointer_cast(bcells_device.data()),
                                 bcells_device.size());

      T *geometry_ptr = thrust::raw_pointer_cast(G_entity.data());
      if (batch_size > 0) {
        compute_geometry(P, cell_list_d);
        device_synchronize();
      } else
        geometry_ptr += 6 * nq * lcells_device.size();

      dim3 block_size(ndofs * ncells);
      dim3 grid_size(cell_list_d.size() / ncells + 1);

      T *x = in.mutable_array().data();
      T *y = out.mutable_array().data();

      stiffness_operator<T, P><<<grid_size, block_size>>>(
          x, cell_constants.data(), y, geometry_ptr, cell_dofmap.data(),
          thrust::raw_pointer_cast(dphi_lookup_d.data()),
          thrust::raw_pointer_cast(dphi_value_d.data()), cell_list_d.data(),
          cell_list_d.size());

      check_device_last_error();
    }

    device_synchronize();

    spdlog::debug("impl_operator done bcells");
  }

  /// Compute matrix diagonal entries
  template <int P, typename Vector> void compute_mat_diag_inv(Vector &out) {
    constexpr int ncells =
        80 * (P == 1) + 32 * (P == 2) + 16 * (P == 3) + 10 * (P == 4);
    constexpr int ndofs =
        4 * (P == 1) + 10 * (P == 2) + 20 * (P == 3) + 35 * (P == 4);
    constexpr int nq =
        1 * (P == 1) + 4 * (P == 2) + 14 * (P == 3) + 24 * (P == 4);

    if (!lcells_device.empty()) {
      spdlog::debug("mat_diagonal doing lcells. lcells size = {}",
                    lcells_device.size());
      std::span<int> cell_list_d(thrust::raw_pointer_cast(lcells_device.data()),
                                 lcells_device.size());
      compute_geometry(P, cell_list_d);
      device_synchronize();

      out.set(T{0.0});
      T *y = out.mutable_array().data();

      dim3 block_size(ndofs * ncells);
      dim3 grid_size(cell_list_d.size() / ncells + 1);
      spdlog::debug("Calling mat_diagonal");
      mat_diagonal_tet<T, P><<<grid_size, block_size, 0>>>(
          cell_constants.data(), y, thrust::raw_pointer_cast(G_entity.data()),
          cell_dofmap.data(), thrust::raw_pointer_cast(dphi_value_d.data()),
          cell_list_d.data(), cell_list_d.size());
      check_device_last_error();
    }

    if (!bcells_device.empty()) {
      spdlog::debug("mat_diagonal doing bcells. bcells size = {}",
                    bcells_device.size());
      std::span<int> cell_list_d(thrust::raw_pointer_cast(bcells_device.data()),
                                 bcells_device.size());
      compute_geometry(P, cell_list_d);
      device_synchronize();

      T *y = out.mutable_array().data();

      dim3 block_size(ndofs * ncells);
      dim3 grid_size(cell_list_d.size() / ncells + 1);
      mat_diagonal_tet<T, P><<<grid_size, block_size, 0>>>(
          cell_constants.data(), y, thrust::raw_pointer_cast(G_entity.data()),
          cell_dofmap.data(), thrust::raw_pointer_cast(dphi_value_d.data()),
          cell_list_d.data(), cell_list_d.size());
      check_device_last_error();
    }

    // Invert
    thrust::transform(thrust::device, out.array().begin(),
                      out.array().begin() + out.map()->size_local(),
                      out.mutable_array().begin(),
                      [] __host__ __device__(T yi) { return 1.0 / yi; });
  }

  template <typename Vector> void operator()(Vector &in, Vector &out) {
    spdlog::debug("Mat free operator start");
    out.set(T{0.0});

    if (degree == 2)
      impl_operator<2>(in, out);
    else if (degree == 3)
      impl_operator<3>(in, out);
    else if (degree == 4)
      impl_operator<4>(in, out);
    else
      throw std::runtime_error("Unsupported degree [mat-free operator]");

    spdlog::debug("Mat free operator end");
  }

  template <typename Vector> void get_diag_inverse(Vector &diag_inv) {
    spdlog::debug("Mat diagonal operator start");

    if (degree == 2)
      compute_mat_diag_inv<2>(diag_inv);
    else if (degree == 3)
      compute_mat_diag_inv<3>(diag_inv);
    else if (degree == 4)
      compute_mat_diag_inv<4>(diag_inv);
    else
      throw std::runtime_error("Unsupported degree [mat diag]");

    spdlog::debug("Mat diagonal operator end");
  }

private:
  int degree;

  // Reference to on-device storage for constants, dofmap etc.
  std::span<const T> cell_constants;
  std::span<const std::int32_t> cell_dofmap;

  // Reference to on-device storage of geometry data
  std::span<const T> xgeom;
  std::span<const std::int32_t> geometry_dofmap;
  std::span<const T> dphi_geometry;
  std::span<const T> G_weights;
  std::span<const std::int8_t> bc_marker;

  // On device storage for geometry data (computed for each batch of cells)
  thrust::device_vector<T> G_entity;

  // On device storage for dphi
  thrust::device_vector<T> dphi_value_d;

  // On device storage for phi
  thrust::device_vector<std::int16_t> dphi_lookup_d;

  // Interpolation is the identity
  bool is_identity;

  // Lists of cells which are local (lcells) and boundary (bcells)
  thrust::device_vector<int> lcells_device, bcells_device;

  // On device storage for the inverse diagonal, needed for Jacobi
  // preconditioner (to remove in future)
  thrust::device_vector<T> _diag_inv;

  // Batch size for geometry computation (set to 0 for no batching)
  std::size_t batch_size;
};

} // namespace dolfinx::acc
