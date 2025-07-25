#pragma once

namespace kernels::stiffness {

template <typename T, int Q> __constant__ T qwts0_d[Q];
template <typename T, int Q> __constant__ T qpts0_d[Q];

template <typename T, int Q> __constant__ T qwts1_d[Q];
template <typename T, int Q> __constant__ T qpts1_d[Q];

template <typename T, int Q> __constant__ T qwts2_d[Q];
template <typename T, int Q> __constant__ T qpts2_d[Q];

template <int N> __constant__ int dof_reordering_d[(N + 1) * N / 2];
template <int N> __constant__ int dof_reordering3d_d[N * (N + 1) * (N + 2) / 6];

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

/// @brief Computes *non-weighted* geometry tensor G from the coordinates and
/// quadrature weights.
/// @param [in] xgeom Geometry points [*, 3]
/// @param [out] G_entity geometry data [n_entities, nq, 6]
/// @param [in] geometry_dofmap Location of coordinates for each cell in
/// xgeom [*, ncdofs]
/// @param [in] _dphi Basis derivative tabulation for cell at quadrature
/// points [3, nq, ncdofs]
/// @param [in] weights Quadrature weights [nq]
/// @tparam T scalar type
template <typename T>
__global__ void geometry_computation_tri(T *G_entity, const T *xgeom,
                                         const std::int32_t *geometry_dofmap,
                                         const T *_dphi, int nq) {
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
    auto dphi = [&_dphi, iq, nq](int i, int j) -> const T {
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

    G_entity[offset] = (K[0][0] * K[0][0] + K[0][1] * K[0][1]) / detJ;
    G_entity[offset + nq] = (K[1][0] * K[0][0] + K[1][1] * K[0][1]) / detJ;
    G_entity[offset + 2 * nq] = (K[1][0] * K[1][0] + K[1][1] * K[1][1]) / detJ;
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

/// @brief Computes *non-weighted* geometry tensor G from the coordinates and
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
template <typename T>
__global__ void geometry_computation_tet(T *G_entity, const T *xgeom,
                                         const std::int32_t *geometry_dofmap,
                                         const T *_dphi, int nq) {
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
    auto dphi = [&_dphi, iq, nq](int i, int j) -> const T {
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
    G_entity[offset] =
        (K[0][0] * K[0][0] + K[0][1] * K[0][1] + K[0][2] * K[0][2]) / detJ;
    G_entity[offset + nq] =
        (K[1][0] * K[0][0] + K[1][1] * K[0][1] + K[1][2] * K[0][2]) / detJ;
    G_entity[offset + 2 * nq] =
        (K[2][0] * K[0][0] + K[2][1] * K[0][1] + K[2][2] * K[0][2]) / detJ;
    G_entity[offset + 3 * nq] =
        (K[1][0] * K[1][0] + K[1][1] * K[1][1] + K[1][2] * K[1][2]) / detJ;
    G_entity[offset + 4 * nq] =
        (K[2][0] * K[1][0] + K[2][1] * K[1][1] + K[2][2] * K[1][2]) / detJ;
    G_entity[offset + 5 * nq] =
        (K[2][0] * K[2][0] + K[2][1] * K[2][1] + K[2][2] * K[2][2]) / detJ;
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
__global__ void
stiffness_operator(T *__restrict__ out_dofs, const T *__restrict__ in_dofs,
                   const T *__restrict__ alpha_cells,
                   const T *__restrict__ G_cells,
                   const std::int32_t *__restrict__ dofmap,
                   const T *__restrict__ dphi, T global_coefficient) {
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
  }
  __syncthreads();

  // Store values at quadrature points
  scratch1[tx] = fw0;
  scratch2[tx] = fw1;

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
    atomicAdd(&out_dofs[gdof], global_coefficient * yd);
  }
}

template <typename T, int N, int Q>
__launch_bounds__(Q *Q) __global__ void stiffness_operator_sf(
    T *__restrict__ out_dofs, const T *__restrict__ in_dofs,
    const T *__restrict__ alpha_cells, const T *__restrict__ G_cells,
    const std::int32_t *__restrict__ dofmap, const T *__restrict__ phi_1,
    const T *__restrict__ phi_0_N, T global_coefficient) {
  auto triangle_ij = [](auto i, auto j) {
    return (i + j + 1) * (i + j) / 2 + j;
  }; // Maps 2d grid to triangle

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int cell_idx = blockIdx.x;
  constexpr int K = (N + 1) * N / 2; // Number of dofs on triangle

  const int l_dof_idx =
      dof_reordering_d<N>[triangle_ij(ty, tx)]; // Only valid for tx < N - ty
  int g_dof_idx = -1;

  const T alpha = alpha_cells[cell_idx];

  __shared__ T c0[N][N];

  __shared__ T c1x[N - 1][Q];
  __shared__ T c1y[N - 1][Q];

  __shared__ T c2x[Q][Q];
  __shared__ T c2y[Q][Q];

  if (ty < N - 1) {
    c1x[ty][tx] = 0.;
    c1y[ty][tx] = 0.;
  }

  if (tx < N && ty < N) {
    c0[ty][tx] = 0.;
  }
  if (tx < N - ty && ty < N) {
    g_dof_idx = dofmap[l_dof_idx + cell_idx * K];
    c0[ty][tx] = in_dofs[g_dof_idx];
  }
  __syncthreads();

  // 1. Evaluate du/dx(x_q) and du/dy(x_q) (dofs -> c2x, c2y)
  // 1.1 c0[N][N] -> c1[N][Q]
  // ty := alpha1, tx := i2
  if (ty < N - 1) {
    for (int alpha2 = 0; alpha2 < N - 1 - ty; ++alpha2) {
      // B_alpha2^{p - alpha1}(t0)
      T phi_0 = phi_0_N[(N - 2 - ty) * Q * (N - 1) + tx * (N - 1) + alpha2];
      c1x[ty][tx] += phi_0 * (c0[ty + 1][alpha2] - c0[ty][alpha2]);
      c1y[ty][tx] += phi_0 * (c0[ty][alpha2 + 1] - c0[ty][alpha2]);
    }
    // printf("c1[%d, %d]=%f\n", ty, tx, c1[ty][tx]);
  }

  __syncthreads();

  T qvalx = 0.;
  T qvaly = 0.;

  // i1:= ty, i2 := tx
  for (int alpha1 = 0; alpha1 < N - 1; ++alpha1) {
    // Load B_alpha1^N(t1)
    T phi_1_ = phi_1[ty * N + alpha1];
    qvalx += phi_1_ * c1x[alpha1][tx];
    qvaly += phi_1_ * c1y[alpha1][tx];
  }

  // 2. Apply geometric transformation to data at quadrature point
  constexpr int nq = Q * Q;
  const int gid = ty + tx * Q + cell_idx * nq * 3;
  const T G0 = G_cells[gid + nq * 0];
  const T G1 = G_cells[gid + nq * 1];
  const T G2 = G_cells[gid + nq * 2];

  c2x[ty][tx] = alpha * (G0 * qvalx + G1 * qvaly);
  c2y[ty][tx] = alpha * (G1 * qvalx + G2 * qvaly);
  __syncthreads();

  T(&f0x)
  [Q][Q] = c2x;
  T(&f0y)
  [Q][Q] = c2y;
  T(&f1x)
  [N - 1][Q] = c1x;
  T(&f1y)
  [N - 1][Q] = c1y;

  __shared__ T f2x[N - 1][N - 1];
  __shared__ T f2y[N - 1][N - 1];

  if (ty < N - 1 && tx < N - 1) {
    f2x[ty][tx] = 0.;
    f2y[ty][tx] = 0.;
  }
  if (ty < N - 1) {
    f1x[ty][tx] = 0.;
    f1y[ty][tx] = 0.;
  }
  __syncthreads();

  // 3.1 f0x[Q][Q] -> f1x[N - 1][Q]
  // and f0y[Q][Q] -> f1y[N - 1][Q]
  // ty := alpha1, tx := i2
  {
    if (ty < N - 1) {
      for (int i1 = 0; i1 < Q; ++i1) {
        T w = qwts1_d<T, Q>[i1];
        T phi_1_ = phi_1[i1 * (N - 1) + ty];
        f1x[ty][tx] += w * phi_1_ * f0x[i1][tx];
        f1y[ty][tx] += w * phi_1_ * f0y[i1][tx];
      }
      // printf("f1[%d, %d]=%f\n", ty, tx, f1[ty][tx]);
    }
  }
  __syncthreads();

  // 3.2 f1x[N - 1][Q] -> f2x[N - 1][N - 1]
  // and f1y[N - 1][Q] -> f2y[N - 1][N - 1]
  // ty := alpha1, tx := alpha2
  {
    if (ty < N - 1 && tx < N - 1) {
      for (int i2 = 0; i2 < Q; ++i2) {
        T w = qwts0_d<T, Q>[i2];
        T phi_0 = phi_0_N[(N - 1 - ty) * Q * N + i2 * N + tx];
        f2x[ty][tx] += w * phi_0 * f1x[ty][i2];
        f2y[ty][tx] += w * phi_0 * f1y[ty][i2];
      }
    }
  }
  __syncthreads();

  T yd = 0.;
  if (ty < N - 1 && tx < N - 1)
    yd -= (f2x[ty][tx] + f2y[ty][tx]);
  if (ty > 1 && ty < N && tx < N - 1)
    yd += f2x[ty - 1][tx];
  if (ty < N - 1 && tx > 1 && tx < N)
    yd += f2y[ty][tx - 1];

  if (tx < N - ty && ty < N) {
    // Write back to global memory
    atomicAdd(&out_dofs[g_dof_idx], N * N * global_coefficient * yd);
  }
}

template <typename T, int nd, int nq>
__global__ void
stiffness_operator3D(T *__restrict__ out_dofs, const T *__restrict__ in_dofs,
                     const T *__restrict__ alpha_cells,
                     const T *__restrict__ G_cells,
                     const std::int32_t *__restrict__ dofmap,
                     const T *__restrict__ dphi, T global_coefficient) {
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
  }

  __syncthreads();

  // Store values at quadrature points
  scratch1[tx] = fw0;
  scratch2[tx] = fw1;
  scratch3[tx] = fw2;

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
    atomicAdd(&out_dofs[gdof], global_coefficient * yd);
  }
}

template <typename T, int nd, int nq>
__global__ void stiffness_operator3D_opti(T *__restrict__ out_dofs,
                                     const T *__restrict__ in_dofs,
                                     const T *__restrict__ alpha_cells,
                                     const T *__restrict__ G_cells,
                                     const std::int32_t *__restrict__ dofmap,
                                     const T *__restrict__ dphi,
                                     T global_coefficient, int ncells) {
  const int tx = threadIdx.x;
  const int stride = blockDim.x;
  const int c_local = threadIdx.y;
  const int cells_blk = blockDim.y;
  const int cell_idx = blockIdx.x * cells_blk + c_local;
  const bool active = (cell_idx < ncells);

  // --- shared memory layout (dynamic) ---
  // per cell: nd (dofs) + 3*nq (scratch1/2/3)
  extern __shared__ T sh[];
  const int per_cell = nd + 3 * nq;
  T *in_local_dofs = sh + c_local * per_cell; // [0 .. nd)
  T *scratch1 = in_local_dofs + nd;           // [0 .. nq)
  T *scratch2 = scratch1 + nq;                // [0 .. nq)
  T *scratch3 = scratch2 + nq;                // [0 .. nq)

  // ---- load DOFs (stride over nd) ----
  if (active) {
    for (int i = tx; i < nd; i += stride) {
      const int gdof = dofmap[cell_idx * nd + i];
      in_local_dofs[i] = in_dofs[gdof];
    }
  }
  __syncthreads();

  auto dphix = [&](int j, int k) { return dphi[j * nd + k]; };
  auto dphiy = [&](int j, int k) { return dphi[nd * nq + j * nd + k]; };
  auto dphiz = [&](int j, int k) { return dphi[2 * nd * nq + j * nd + k]; };

  // ---- phase 1: compute at quadrature points (stride over nq) ----
  if (active) {
    const T alpha = alpha_cells[cell_idx];
    for (int iq = tx; iq < nq; iq += stride) {
      T val_x = 0.0, val_y = 0.0, val_z = 0.0;
      for (int idof = 0; idof < nd; ++idof) {
        const T u = in_local_dofs[idof];
        val_x += dphix(iq, idof) * u;
        val_y += dphiy(iq, idof) * u;
        val_z += dphiz(iq, idof) * u;
      }

      const int gid = iq + cell_idx * nq * 6;
      const T G0 = G_cells[gid + nq * 0];
      const T G1 = G_cells[gid + nq * 1];
      const T G2 = G_cells[gid + nq * 2];
      const T G3 = G_cells[gid + nq * 3];
      const T G4 = G_cells[gid + nq * 4];
      const T G5 = G_cells[gid + nq * 5];

      scratch1[iq] = alpha * (G0 * val_x + G1 * val_y + G2 * val_z);
      scratch2[iq] = alpha * (G1 * val_x + G3 * val_y + G4 * val_z);
      scratch3[iq] = alpha * (G2 * val_x + G4 * val_y + G5 * val_z);
    }
  }
  __syncthreads();

  // ---- phase 2: accumulate gradients back to DOFs (stride over nd) ----
  if (active) {
    for (int id = tx; id < nd; id += stride) {
      T grad_x = 0.0, grad_y = 0.0, grad_z = 0.0;
      for (int iq = 0; iq < nq; ++iq) {
        grad_x += dphix(iq, id) * scratch1[iq];
        grad_y += dphiy(iq, id) * scratch2[iq];
        grad_z += dphiz(iq, id) * scratch3[iq];
      }
      const T yd = grad_x + grad_y + grad_z;
      const int gdof = dofmap[cell_idx * nd + id];
      atomicAdd(&out_dofs[gdof], global_coefficient * yd);
    }
  }
}

template <typename T, int nd, int nq>
__global__ void stiffness_operator3D_optimem(
    T *__restrict__ out_dofs, const T *__restrict__ in_dofs,
    const T *__restrict__ alpha_cells, const T *__restrict__ G_cells,
    const std::int32_t *__restrict__ dofmap, const T *__restrict__ dphi,
    T global_coefficient, int ncells)
{
  const int tx        = threadIdx.x;
  const int stride    = blockDim.x;
  const int c_local   = threadIdx.y;
  const int cells_blk = blockDim.y;
  const int cell_idx  = blockIdx.x * cells_blk + c_local;
  const bool active   = (cell_idx < ncells);

  // --- shared memory ---
  // Per cell: nd dofs + 3*nq scratch
  // Plus: 3 * nd * nq for dphi copy (x,y,z)
  extern __shared__ T sh[];
  const int per_cell = nd + 3 * nq;
  T *in_local_dofs = sh + c_local * per_cell;
  T *scratch1      = in_local_dofs + nd;
  T *scratch2      = scratch1 + nq;
  T *scratch3      = scratch2 + nq;

  // copy dphi to shared
  T *dphix_sh = sh + per_cell * cells_blk;
  T *dphiy_sh = dphix_sh + nd * nq;
  T *dphiz_sh = dphiy_sh + nd * nq;
  for (int i = tx + threadIdx.y * stride; i < nd * nq; i += stride * blockDim.y) {
    dphix_sh[i] = dphi[i];
    dphiy_sh[i] = dphi[nd * nq + i];
    dphiz_sh[i] = dphi[2 * nd * nq + i];
  }
  __syncthreads();

  if (active) {
    // load dofs
    for (int i = tx; i < nd; i += stride) {
      const int gdof = dofmap[cell_idx * nd + i];
      in_local_dofs[i] = in_dofs[gdof];
    }
  }
  __syncthreads();

  auto dphix = [&](int j, int k) { return dphix_sh[j * nd + k]; };
  auto dphiy = [&](int j, int k) { return dphiy_sh[j * nd + k]; };
  auto dphiz = [&](int j, int k) { return dphiz_sh[j * nd + k]; };

  // phase 1
  if (active) {
    const T alpha = alpha_cells[cell_idx];
    for (int iq = tx; iq < nq; iq += stride) {
      T val_x = 0.0, val_y = 0.0, val_z = 0.0;
      for (int idof = 0; idof < nd; ++idof) {
        const T u = in_local_dofs[idof];
        val_x += dphix(iq, idof) * u;
        val_y += dphiy(iq, idof) * u;
        val_z += dphiz(iq, idof) * u;
      }

      const int gid = iq + cell_idx * nq * 6;
      const T G0 = G_cells[gid + nq * 0];
      const T G1 = G_cells[gid + nq * 1];
      const T G2 = G_cells[gid + nq * 2];
      const T G3 = G_cells[gid + nq * 3];
      const T G4 = G_cells[gid + nq * 4];
      const T G5 = G_cells[gid + nq * 5];

      scratch1[iq] = alpha * (G0 * val_x + G1 * val_y + G2 * val_z);
      scratch2[iq] = alpha * (G1 * val_x + G3 * val_y + G4 * val_z);
      scratch3[iq] = alpha * (G2 * val_x + G4 * val_y + G5 * val_z);
    }
  }
  __syncthreads();

  // phase 2
  if (active) {
    for (int id = tx; id < nd; id += stride) {
      T grad_x = 0.0, grad_y = 0.0, grad_z = 0.0;
      for (int iq = 0; iq < nq; ++iq) {
        grad_x += dphix(iq, id) * scratch1[iq];
        grad_y += dphiy(iq, id) * scratch2[iq];
        grad_z += dphiz(iq, id) * scratch3[iq];
      }
      const T yd = grad_x + grad_y + grad_z;
      const int gdof = dofmap[cell_idx * nd + id];
      atomicAdd(&out_dofs[gdof], global_coefficient * yd);
    }
  }
}


template <typename T, int N, int Q>
__launch_bounds__(Q *Q *Q) __global__ void stiffness_operator3D_sf(
    T *__restrict__ out_dofs, const T *__restrict__ in_dofs,
    const T *__restrict__ alpha_cells, const T *__restrict__ G_cells,
    const std::int32_t *__restrict__ dofmap, const T *__restrict__ phi_2,
    const T *__restrict__ phi_1_N, const T *__restrict__ phi_0_N,
    T global_coefficient) {
  auto tet_ijk = [](int i, int j, int k) {
    int w = i + j + k;
    int s = j + k;
    return (w + 2) * (w + 1) * w / 6 + (s + 1) * s / 2 + k;
  }; // Maps 3d grid to tet using twice the cantor pairing function

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tz = threadIdx.z;

  const int cell_idx = blockIdx.x;
  constexpr int K = N * (N + 1) * (N + 2) / 6; // Number of dofs on tet

  const int l_dof_idx = dof_reordering3d_d<N>[tet_ijk(
      tz, ty, tx)]; // Only valid for ty < N - tz, tx < N - ty - tz
  int g_dof_idx = -1;

  const T alpha = alpha_cells[cell_idx];

  // Load tables in shared memory
  __shared__ T phi_0_N_s[(N - 1)][Q][(N - 1)];
  __shared__ T phi_1_N_s[(N - 1)][Q][(N - 1)];
  __shared__ T phi_2_s[Q][(N - 1)];

  if (tz < N - 1 && tx < N - 1) {
    phi_0_N_s[tz][ty][tx] = phi_0_N[tz * Q * (N - 1) + ty * (N - 1) + tx];
    phi_1_N_s[tz][ty][tx] = phi_1_N[tz * Q * (N - 1) + ty * (N - 1) + tx];
  }
  if (tz == 0 && tx < N - 1) {
    phi_2_s[ty][tx] = phi_2[ty * (N - 1) + tx];
  }

  __shared__ T c0[N][N][N];

  __shared__ T c1x[N - 1][N - 1][Q];
  __shared__ T c1y[N - 1][N - 1][Q];
  __shared__ T c1z[N - 1][N - 1][Q];

  __shared__ T c2x[N - 1][Q][Q];
  __shared__ T c2y[N - 1][Q][Q];
  __shared__ T c2z[N - 1][Q][Q];

  __shared__ T c3x[Q][Q][Q];
  __shared__ T c3y[Q][Q][Q];
  __shared__ T c3z[Q][Q][Q];

  if (tz < N && ty < N && tx < N) {
    c0[tz][ty][tx] = 0.;
  }
  if (ty < N - 1 && tz < N - 1) {
    c1x[tz][ty][tx] = 0.;
    c1y[tz][ty][tx] = 0.;
    c1z[tz][ty][tx] = 0.;
  }
  if (tz < N - 1) {
    c2x[tz][ty][tx] = 0.;
    c2y[tz][ty][tx] = 0.;
    c2z[tz][ty][tx] = 0.;
  }

  c3x[tz][ty][tx] = 0.;
  c3y[tz][ty][tx] = 0.;
  c3z[tz][ty][tx] = 0.;

  if (tx < N - ty - tz && ty < N - tz && tz < N) {
    g_dof_idx = dofmap[l_dof_idx + cell_idx * K];
    c0[tz][ty][tx] = in_dofs[g_dof_idx];
  }
  __syncthreads();

  // 1. Evaluate du/dx(x_q) and du/dy(x_q) (dofs -> c2x, c2y)
  // 1.1 c0[N][N][N] -> c1x[N-1][N-1][Q], c1y[N-1][N-1][Q], c1z[N-1][N-1][Q]
  // ty := alpha1, tx := i2
  if (ty < N - 1 - tz && tz < N - 1) {
    for (int alpha3 = 0; alpha3 < N - 1 - ty; ++alpha3) {
      // B_alpha3^{p - alpha1}(t0)
      T phi_0 = phi_0_N_s[(N - 2 - tz - ty)][tx][alpha3];
      c1x[tz][ty][tx] += phi_0 * (c0[tz + 1][ty][alpha3] - c0[tz][ty][alpha3]);
      c1y[tz][ty][tx] += phi_0 * (c0[tz][ty + 1][alpha3] - c0[tz][ty][alpha3]);
      c1z[tz][ty][tx] += phi_0 * (c0[tz][ty][alpha3 + 1] - c0[tz][ty][alpha3]);
    }
    // printf("c1[%d, %d]=%f\n", ty, tx, c1[ty][tx]);
  }
  __syncthreads();

  if (ty < N - 1) {
    for (int alpha2 = 0; alpha2 < N - 1 - tz; ++alpha2) {
      // B_alpha2^{p - alpha1}(t0)
      T phi_1 = phi_1_N_s[(N - 2 - tz)][ty][alpha2];
      c2x[tz][ty][tx] += phi_1 * c1x[tz][alpha2][tx];
      c2y[tz][ty][tx] += phi_1 * c1y[tz][alpha2][tx];
      c2z[tz][ty][tx] += phi_1 * c1z[tz][alpha2][tx];
    }
    // printf("c1[%d, %d]=%f\n", ty, tx, c1[ty][tx]);
  }

  __syncthreads();

  T qvalx = 0.;
  T qvaly = 0.;
  T qvalz = 0.;

  // i1:= ty, i2 := tx
  for (int alpha1 = 0; alpha1 < N - 1; ++alpha1) {
    // Load B_alpha1^N(t1)
    T phi_2 = phi_2_s[tz][alpha1];
    qvalx += phi_2 * c2x[alpha1][ty][tx];
    qvaly += phi_2 * c2y[alpha1][ty][tx];
    qvalz += phi_2 * c2z[alpha1][ty][tx];
  }

  // 2. Apply geometric transformation to data at quadrature point
  constexpr int nq = Q * Q * Q;
  const int gid = tz + ty * Q + tx * Q * Q + cell_idx * nq * 3;

  const T G0 = G_cells[gid + nq * 0];
  const T G1 = G_cells[gid + nq * 1];
  const T G2 = G_cells[gid + nq * 2];
  const T G3 = G_cells[gid + nq * 3];
  const T G4 = G_cells[gid + nq * 4];
  const T G5 = G_cells[gid + nq * 5];

  c3x[tz][ty][tx] = alpha * (G0 * qvalx + G1 * qvaly + G2 * qvalz);
  c3y[tz][ty][tx] = alpha * (G1 * qvalx + G3 * qvaly + G4 * qvalz);
  c3z[tz][ty][tx] = alpha * (G2 * qvalx + G4 * qvaly + G5 * qvalz);

  __syncthreads();

  T(&f0x)
  [Q][Q][Q] = c3x;
  T(&f0y)
  [Q][Q][Q] = c3y;
  T(&f0z)
  [Q][Q][Q] = c3z;

  T(&f1x)
  [N - 1][Q][Q] = c2x;
  T(&f1y)
  [N - 1][Q][Q] = c2y;
  T(&f1z)
  [N - 1][Q][Q] = c2z;

  T(&f2x)
  [N - 1][N - 1][Q] = c1x;
  T(&f2y)
  [N - 1][N - 1][Q] = c1y;
  T(&f2z)
  [N - 1][N - 1][Q] = c1z;

  __shared__ T f3x[N - 1][N - 1][N - 1];
  __shared__ T f3y[N - 1][N - 1][N - 1];
  __shared__ T f3z[N - 1][N - 1][N - 1];

  if (tz < N - 1 && ty < N - 1 && tx < N - 1) {
    f3x[tz][ty][tx] = 0.;
    f3y[tz][ty][tx] = 0.;
    f3z[tz][ty][tx] = 0.;
  }
  if (tz < N - 1 && ty < N - 1) {
    f2x[tz][ty][tx] = 0.;
    f2y[tz][ty][tx] = 0.;
    f2z[tz][ty][tx] = 0.;
  }
  if (tz < N - 1) {
    f1x[tz][ty][tx] = 0.;
    f1y[tz][ty][tx] = 0.;
    f1z[tz][ty][tx] = 0.;
  }
  __syncthreads();

  // 3.1 f0x[Q][Q] -> f1x[N - 1][Q]
  // and f0y[Q][Q] -> f1y[N - 1][Q]
  // ty := alpha1, tx := i2
  {
    if (tz < N - 1) {
      for (int i1 = 0; i1 < Q; ++i1) {
        T w = qwts2_d<T, Q>[i1];
        T phi_2 = phi_2_s[i1][tz];
        f1x[tz][ty][tx] += w * phi_2 * f0x[i1][ty][tx];
        f1y[tz][ty][tx] += w * phi_2 * f0y[i1][ty][tx];
        f1z[tz][ty][tx] += w * phi_2 * f0z[i1][ty][tx];
      }
      // printf("f1[%d, %d]=%f\n", ty, tx, f1[ty][tx]);
    }
  }
  __syncthreads();

  // ty := alpha1, tx := i2
  {
    if (tz < N - 1 && ty < N - 1) {
      for (int i2 = 0; i2 < Q; ++i2) {
        T w = qwts1_d<T, Q>[i2];
        T phi_1 = phi_1_N_s[(N - 2 - tz)][i2][ty];
        f2x[tz][ty][tx] += w * phi_1 * f1x[tz][i2][tx];
        f2y[tz][ty][tx] += w * phi_1 * f1y[tz][i2][tx];
        f2z[tz][ty][tx] += w * phi_1 * f1z[tz][i2][tx];
      }
      // printf("f1[%d, %d]=%f\n", ty, tx, f1[ty][tx]);
    }
  }

  // 3.2 f1x[N - 1][Q] -> f2x[N - 1][N - 1]
  // and f1y[N - 1][Q] -> f2y[N - 1][N - 1]
  // ty := alpha1, tx := alpha2
  {
    if (ty < N - 1 - tz && tx < N - 1 && tz < N - 1) {
      for (int i3 = 0; i3 < Q; ++i3) {
        T w = qwts0_d<T, Q>[i3];
        T phi_0 = phi_0_N_s[(N - 2 - tz - ty)][i3][tx];
        f3x[tz][ty][tx] += w * phi_0 * f2x[tz][ty][i3];
        f3y[tz][ty][tx] += w * phi_0 * f2y[tz][ty][i3];
        f3z[tz][ty][tx] += w * phi_0 * f2z[tz][ty][i3];
      }
    }
  }
  __syncthreads();

  T yd = 0.;
  if (tz < N - 1 && ty < N - 1 && tx < N - 1)
    yd -= (f3x[tz][ty][tx] + f3y[tz][ty][tx] + f3z[tz][ty][tx]);
  if ((tz > 1 && tz < N) && ty < N - 1 && tx < N - 1)
    yd += f3x[tz - 1][ty][tx];
  if (tz < N - 1 && (ty > 1 && ty < N) && tx < N - 1)
    yd += f3y[tz][ty - 1][tx];
  if (tz < N - 1 && ty < N - 1 && (tx > 1 && tx < N))
    yd += f3z[tz][ty][tx - 1];

  if (tx < N - ty - tz && ty < N - tz && tz < N) {
    // Write back to global memory
    atomicAdd(&out_dofs[g_dof_idx], N * N * global_coefficient * yd);
  }
}

template <typename T, int nd, int nq>
__global__ void stiffness_operator3D_diagonal(
    T *__restrict__ out_dofs, const T *__restrict__ alpha_cells,
    const T *__restrict__ G_cells, const std::int32_t *__restrict__ dofmap,
    const T *__restrict__ dphi, T global_coefficient) {
  const int cell_idx = blockIdx.x;
  const int tx = threadIdx.x;
  const int gdof = dofmap[tx + cell_idx * nd];
  const T alpha = alpha_cells[cell_idx];

  auto dphix = [&](auto j, auto k) { return dphi[j * nd + k]; };
  auto dphiy = [&](auto j, auto k) { return dphi[nd * nq + j * nd + k]; };
  auto dphiz = [&](auto j, auto k) { return dphi[2 * nd * nq + j * nd + k]; };

  auto G_ = [&](int r, int ix) {
    return G_cells[nq * (cell_idx * 6 + r) + ix];
  };

  T val = 0.0;
  for (int iq = 0; iq < nq; ++iq) {
    val += G_(0, iq) * dphix(iq, tx) * dphix(iq, tx);
    val += G_(1, iq) * dphix(iq, tx) * dphiy(iq, tx);
    val += G_(2, iq) * dphix(iq, tx) * dphiz(iq, tx);
    val += G_(1, iq) * dphiy(iq, tx) * dphix(iq, tx);
    val += G_(3, iq) * dphiy(iq, tx) * dphiy(iq, tx);
    val += G_(4, iq) * dphiy(iq, tx) * dphiz(iq, tx);
    val += G_(2, iq) * dphiz(iq, tx) * dphix(iq, tx);
    val += G_(4, iq) * dphiz(iq, tx) * dphiy(iq, tx);
    val += G_(5, iq) * dphiz(iq, tx) * dphiz(iq, tx);
  }

  // Write back to global memory
  atomicAdd(&out_dofs[gdof], alpha * global_coefficient * val);
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