#pragma once

namespace kernels::stiffness {

template <typename T, int nq>
__global__ void geometry_computation_tri(T *G_entity, const T *xgeom,
                                         const std::int32_t *geometry_dofmap,
                                         const T *_dphi, const T *weights) {
  // Cell index
  int cell = blockIdx.x;

  constexpr int ncdofs = 3;
  constexpr int gdim = 2;

  __shared__ T _coord_dofs[ncdofs * gdim];

  int iq = threadIdx.x;

  // Gather geometry into shared memory
  if (blockDim.x < 6 && iq == 0) {
    for (int k = 0; k < 6; ++k) {
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

  // Jacobian
  T J[2][2];
  auto coord_dofs = [&](int i, int j) -> T & {
    return _coord_dofs[i * gdim + j];
  };

  // For each quadrature point / thread
  {
    // dphi has shape [gdim, nq, ncdofs]
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

    // K = J^-1 * det(J)
    T K[2][2] = {{J[1][1], -J[0][1]}, {-J[1][0], J[0][0]}};
    T detJ = fabs(J[0][0] * J[1][1] - J[0][1] * J[1][0]);

    int offset = (cell * nq * 3 + iq);
    const T w = weights[iq];

    G_entity[offset] = (K[0][0] * K[0][0] + K[0][1] * K[0][1]) * w / detJ;
    G_entity[offset + nq] = (K[1][0] * K[0][0] + K[1][1] * K[0][1]) * w / detJ;
    G_entity[offset + 2 * nq] =
        (K[1][0] * K[1][0] + K[1][1] * K[1][1]) * w / detJ;
  }
}

/// @brief Computes weighted geometry tensor G from the coordinates and
/// quadrature weights.
/// @param [in] xgeom Geometry points [*, 3]
/// @param [out] G_entity geometry data [n_entities, nq, 6]
/// @param [in] geometry_dofmap Location of coordinates for each cell in
/// xgeom [*, ncdofs]
/// @param [in] _dphi Basis derivative tabulation for cell at quadrature
/// points [3, nq, ncdofs]
/// @param [in] weights Quadrature weights [nq]
/// @tparam T scalar type
/// @tparam D degree
template <typename T, int nq>
__global__ void geometry_computation_tet(T *G_entity, const T *xgeom,
                                         const std::int32_t *geometry_dofmap,
                                         const T *_dphi, const T *weights) {
  int cell = blockIdx.x;
  constexpr int ncdofs = 4;
  constexpr int gdim = 3;

  __shared__ T _coord_dofs[ncdofs * gdim];

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

  // Jacobian
  T J[3][3];
  auto coord_dofs = [&](int i, int j) -> T & {
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

    T detJ = fabs(J[0][0] * K[0][0] + J[0][1] * K[1][0] + J[0][2] * K[2][0]);

    int offset = (cell * nq * 6 + iq);
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
/// @tparam T Data type of the input and output arrays
/// @tparam P Polynomial degree of the basis functions
/// @tparam Q Number of quadrature points in 1D
/// @param u Input vector of size (ndofs,)
/// @param entity_constants Array of size (ncells,) with the
/// constant C for each entity
/// @param b Output vector of size (ndofs,)
/// @param G_entity Array of size (ncells, nq, 6) with the geometry
/// operator G for each entity
/// @param dofmap Array of size (ncells, ndofs) with the
/// dofmap for each entity
/// @param phi0_in Array of size (nq, ndofs) with the interpolation basis
/// functions in 1D. u1_i = phi0_(ij) u_j, where u are the dofs
/// associated with the element (degree P), and u1 are the dofs for the
/// finite elment (degree >= P) that u is interpolated into.

///
/// @note The kernel is launched with grid of blocks, where each
/// block is responsible for computing the stiffness operator for several cells.
/// The block size is ncells*ndofs.
template <typename T, int nd, int nq>
__global__ void stiffness_operator(T *__restrict__ out_dofs,
                                   const T *__restrict__ in_dofs,
                                   const T *__restrict__ alpha_cells,
                                   const T *__restrict__ G_cells,
                                   const std::int32_t *__restrict__ dofmap,
                                   const T *__restrict__ dphi) {
  const int cell_idx = blockIdx.x;
  const int tx = threadIdx.x;
  int gdof = -1;
  const T alpha = alpha_cells[cell_idx];

  __shared__ T scratch1[nq];
  __shared__ T scratch2[nq];

  scratch1[tx] = 0.0;

  if (tx < nd) {
    gdof = dofmap[tx + cell_idx * nd];
    scratch1[tx] = in_dofs[gdof];
  }
  __syncthreads();

  auto dphix = [&](auto j, auto k) { return dphi[j * nd + k]; };
  auto dphiy = [&](auto j, auto k) { return dphi[nd * nq + j * nd + k]; };

  // Compute val{x,y} at quadrature points
  T fw0 = 0, fw1 = 0;
  if (tx < nq) {
    T val_x = 0.0;
    T val_y = 0.0;
    for (int idof = 0; idof < nd; ++idof) {
      val_x += dphix(tx, idof) * scratch1[idof];
      val_y += dphiy(tx, idof) * scratch1[idof];
    }

    // Apply geometric transformation to data at quadrature point
    const int gid = tx + cell_idx * nq * 3;
    const T G0 = G_cells[gid + nq * 0];
    const T G1 = G_cells[gid + nq * 1];
    const T G2 = G_cells[gid + nq * 2];

    fw0 = alpha * (G0 * val_x + G1 * val_y);
    fw1 = alpha * (G1 * val_x + G2 * val_y);

    // Store values at quadrature points
    scratch1[tx] = fw0;
    scratch2[tx] = fw1;
  }

  __syncthreads();

  if (tx < nd) {
    T grad_x = 0.0;
    T grad_y = 0.0;
    for (int iq = 0; iq < nq; ++iq) {
      grad_x += dphix(iq, tx) * scratch1[iq];
      grad_y += dphiy(iq, tx) * scratch2[iq];
    }

    // Sum contributions
    T yd = grad_x + grad_y;
    // Write back to global memory
    atomicAdd(&out_dofs[gdof], yd);
  }
}

template <typename T, int nd, int nq>
__global__ void stiffness_operator3D(T *__restrict__ out_dofs,
                                   const T *__restrict__ in_dofs,
                                   const T *__restrict__ alpha_cells,
                                   const T *__restrict__ G_cells,
                                   const std::int32_t *__restrict__ dofmap,
                                   const T *__restrict__ dphi) {
  const int cell_idx = blockIdx.x;
  const int tx = threadIdx.x;
  int gdof = -1;
  const T alpha = alpha_cells[cell_idx];

  __shared__ T scratch1[nq];
  __shared__ T scratch2[nq];
  __shared__ T scratch3[nq];

  scratch1[tx] = 0.0;

  if (tx < nd) {
    gdof = dofmap[tx + cell_idx * nd];
    scratch1[tx] = in_dofs[gdof];
  }
  __syncthreads();

  auto dphix = [&](auto j, auto k) { return dphi[j * nd + k]; };
  auto dphiy = [&](auto j, auto k) { return dphi[nd * nq + j * nd + k]; };
  auto dphiz = [&](auto j, auto k) { return dphi[2 * nd * nq + j * nd + k]; };

  // Compute val{x,y} at quadrature points
  T fw0 = 0, fw1 = 0, fw2 = 0;
  if (tx < nq) {
    T val_x = 0.0;
    T val_y = 0.0;
    T val_z = 0.0;
    for (int idof = 0; idof < nd; ++idof) {
      val_x += dphix(tx, idof) * scratch1[idof];
      val_y += dphiy(tx, idof) * scratch1[idof];
      val_z += dphiz(tx, idof) * scratch1[idof];
    }

    // Apply geometric transformation to data at quadrature point
    const int gid = tx + cell_idx * nq * 6;
    const T G0 = G_cells[gid + nq * 0];
    const T G1 = G_cells[gid + nq * 1];
    const T G2 = G_cells[gid + nq * 2];
    const T G3 = G_cells[gid + nq * 3];
    const T G4 = G_cells[gid + nq * 4];
    const T G5 = G_cells[gid + nq * 5];

    fw0 = alpha * (G0 * val_x + G1 * val_y + G2 * val_z);
    fw1 = alpha * (G1 * val_x + G3 * val_y + G4 * val_z);
    fw2 = alpha * (G2 * val_x + G4 * val_y + G5 * val_z);

    // Store values at quadrature points
    scratch1[tx] = fw0;
    scratch2[tx] = fw1;
    scratch3[tx] = fw2;
  }

  __syncthreads();

  if (tx < nd) {
    T grad_x = 0.0;
    T grad_y = 0.0;
    T grad_z = 0.0;
    for (int iq = 0; iq < nq; ++iq) {
      grad_x += dphix(iq, tx) * scratch1[iq];
      grad_y += dphiy(iq, tx) * scratch2[iq];
      grad_z += dphiz(iq, tx) * scratch3[iq];
    }

    // Sum contributions
    T yd = grad_x + grad_y + grad_z;
    // Write back to global memory
    atomicAdd(&out_dofs[gdof], yd);
  }
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
stiffness_operator3D_(const T *__restrict__ u,
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

} // namespace kernels::stiffness