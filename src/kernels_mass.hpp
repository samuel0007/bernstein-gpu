#pragma once

template <typename T, int Q> __constant__ T qwts0_d[Q];
template <typename T, int Q> __constant__ T qpts0_d[Q];

template <typename T, int Q> __constant__ T qwts1_d[Q];
template <typename T, int Q> __constant__ T qpts1_d[Q];

template <int N> __constant__ int dof_reordering_d[(N + 1) * N / 2];
template <int N> __constant__ int dof_reordering3d_d[N * (N + 1) * (N + 2) / 6];

template <int ld0, int ld1, int ld2>
__device__ __forceinline__ int ijk(int i, int j, int k) {
  return i * ld0 + j * ld1 + k * ld2;
}

/// Apply mass matrix operator \int alpha(x) * inner(u, v) dx to in_dofs.
/// @param in_dofs input global dofs (x),   size ndofs
/// @param out_dofs output global dofs (y), size ndofs
/// @param alpha_cells DG0 alpha,           size ncells
/// @param detJ_cells det(J_K(dzeta_q)),    size Q * Q * ncells
/// @param dofmap global to local dofmap,   size K * ncells
/// @param N number of dofs on 1D interval
/// @param Q number of quadrature points on 1D interval
template <typename T, int N, int Q>
__launch_bounds__(Q *Q) __global__
    void mass_operator(const T *__restrict__ in_dofs, T *__restrict__ out_dofs,
                       const T *__restrict__ alpha_cells,
                       const T *__restrict__ detJ_cells,
                       const std::int32_t *__restrict__ dofmap) {
  auto triangle_ij = [](auto i, auto j) {
    return (i + j + 1) * (i + j) / 2 + j;
  }; // Maps 2d grid to triangle: TODO think about ordering

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int cell_idx = blockIdx.x;
  // constexpr int nq = Q * Q;
  // constexpr int nd = N * N;          // Number of dofs on square
  constexpr int K = (N + 1) * N / 2; // Number of dofs on triangle

  const int l_dof_idx =
      dof_reordering_d<N>[triangle_ij(ty, tx)]; // Only valid for tx < N - ty
  int g_dof_idx = -1;

  T alpha = alpha_cells[cell_idx]; // Load DG0 alpha coefficient

  // if(tx == 0) {
  //     for (int i = 0; i < Q; ++i) {
  //         printf("qwts0_d[%d] = %f, qwts1_d[%d] = %f\n", i, qwts0_d<T, Q>[i],
  //         i, qwts1_d<T, Q>[i]);
  //     }
  // }

  __shared__ T in_local_dofs[N][N];
  __shared__ T c1[N][Q];
  __shared__ T qvals[Q][Q];

  // Load geometry data
  T detJ_abs = fabs(detJ_cells[tx + ty * Q + cell_idx * Q * Q]);

  if (ty < N)
    c1[ty][tx] = 0.;
  if (tx < N && ty < N) {
    in_local_dofs[ty][tx] = 0.;
  }
  if (tx < N - ty && ty < N) {
    g_dof_idx = dofmap[l_dof_idx + cell_idx * K];
    in_local_dofs[ty][tx] = in_dofs[g_dof_idx];
    // printf("in_local_dofs[%d][%d] = %f\n", ty, tx, in_local_dofs[ty][tx]);
  }
  __syncthreads();

  // 1. Evaluate u(x_q) (dofs -> qvals)
  T p = qpts0_d<T, Q>[tx];
  T s = 1.0 - p;
  T r = p / s;
  T w = 1.;
  if (ty < N) {
    // T w = s^(N + 1 - ty);
    for (int i = 0; i < N - 1 - ty; ++i)
      w *= s;

    // printf("w[%d, %d]=%f\n", tx, ty, w);

    for (int alpha2 = 0; alpha2 < N - ty; ++alpha2) {
      // c1[ty][tx] += w;
      c1[ty][tx] += w * in_local_dofs[alpha2][ty];
      w *= r * (N - 1 - ty - alpha2) / (1 + alpha2);
    }
    // printf("c1[%d, %d]=%f\n", tx, ty, c1[ty][tx]);
  }

  __syncthreads();

  p = qpts1_d<T, Q>[tx];
  s = 1.0 - p;
  r = p / s;
  w = 1.;
  T qval = 0.;
  for (int i = 0; i < N - 1; ++i)
    w *= s;
  for (int alpha1 = 0; alpha1 < N; ++alpha1) {
    qval += w * c1[alpha1][ty];
    w *= r * (N - 1 - alpha1) / (1 + alpha1);
  }

  // 2. Apply geometry
  qvals[tx][ty] = alpha * qval * detJ_abs;
  // printf("qvals[%d, %d]=%f\n", tx, ty, qvals[tx][ty]);
  __syncthreads();

  // 3. Compute Moments (qvals -> dofs)
  T f1val = 0.;
  __shared__ T f1[N][Q];

  // tx = alpha1
  // ty = i2
  if (tx < N) {
    for (int i1 = 0; i1 < Q; ++i1) {
      T w = qwts1_d<T, Q>[i1];
      T p = qpts1_d<T, Q>[i1];

      T s = 1.0 - p;
      T r = p / s;
      // T ww = w * s ** n;
      for (int i = 0; i < N - 1; ++i) {
        w *= s;
      }
      // ww *= r * (n - alpha1) / (1 + alpha1)
      for (int i = 0; i < tx; ++i) {
        w *= r * (N - 1 - i) / (1 + i);
      }
      f1val += w * qvals[i1][ty];
    }

    f1[tx][ty] = f1val;
    // printf("f1[%d, %d]=%f\n", tx, ty, f1[tx][ty]);
  }

  __syncthreads();

  // tx = alpha1
  // ty = alpha2
  T f2val = 0.;
  if (tx < N && ty < N - tx) {
    for (int i2 = 0; i2 < Q; ++i2) {
      T w = qwts0_d<T, Q>[i2];
      T p = qpts0_d<T, Q>[i2];
      T s = 1.0 - p;
      T r = p / s;
      for (int i = 0; i < N - 1 - tx; ++i) {
        w *= s;
      }

      for (int i = 0; i < ty; ++i) {
        w *= r * (N - 1 - tx - i) / (1. + i);
      }

      f2val += w * f1[tx][i2];
    }

    // printf("out_dof[%d]=%f \n", g_dof_idx, f2val);
    atomicAdd(&out_dofs[g_dof_idx], f2val);
  };
}

template <typename T, int nd, int nq>
__global__ void
mass_operator_baseline(const T *__restrict__ in_dofs, T *__restrict__ out_dofs,
                       const T *__restrict__ alpha_cells,
                       const T *__restrict__ detJ_cells,
                       const std::int32_t *__restrict__ dofmap,
                       const T *__restrict__ phi, T global_coefficient) {
  const int tx = threadIdx.x;
  const int cell_idx = blockIdx.x;
  int g_dof_idx = -1;

  T alpha = alpha_cells[cell_idx]; // Load DG0 alpha coefficient

  __shared__ T in_local_dofs[nd];
  __shared__ T qvals[nq];

  if (tx < nd) {
    g_dof_idx = dofmap[tx + nd * cell_idx];
    in_local_dofs[tx] = in_dofs[g_dof_idx];
  }
  __syncthreads();

  // Load geometry data
  T detJ_abs = detJ_cells[tx + cell_idx * nq];

  // u_i(x_q)
  T qval = 0.;
  for (int i = 0; i < nd; ++i) {
    qval += phi[tx * nd + i] * in_local_dofs[i]; // Bank conflict probably
  }

  qvals[tx] = alpha * qval * detJ_abs;
  __syncthreads();
  // if(cell_idx == 0)
  //   printf("qvals[%d]=%f \n", tx, qvals[tx]);

  if (tx < nd) {
    T fval = 0.;
    for (int i = 0; i < nq; ++i) {
      fval += qvals[i] * phi[i * nd + tx];
    }

    // if(cell_idx == 0)
    //   printf("out_dof[%d]=%f \n", g_dof_idx, fval);
    atomicAdd(&out_dofs[g_dof_idx], fval * global_coefficient);
  }

  // printf("qvals[%d]=%f \n", tx, qval);
}

template <typename T, int nd, int nq>
__global__ void mass_operator_baseline_opti(
    const T *__restrict__ in_dofs, T *__restrict__ out_dofs,
    const T *__restrict__ alpha_cells, const T *__restrict__ detJ_cells,
    const std::int32_t *__restrict__ dofmap, const T *__restrict__ phi,
    T global_coefficient, int ncells) {
  const int tx = threadIdx.x;
  const int stride = blockDim.x;
  const int c_local = threadIdx.y;
  const int cells_pb = blockDim.y;
  const int cell_idx = blockIdx.x * cells_pb + c_local;
  const bool active = (cell_idx < ncells);

  // --- shared memory per cell: nd DOFs + nq qvals ---
  extern __shared__ T sh[];
  const int per_cell = nd + nq;
  T *in_local_dofs = sh + c_local * per_cell;
  T *qvals = in_local_dofs + nd;

  __syncthreads();

  // -------- load DOFs (stride over nd) --------
  if (active) {
    for (int i = tx; i < nd; i += stride) {
      int g_dof_idx = dofmap[cell_idx * nd + i];
      in_local_dofs[i] = in_dofs[g_dof_idx];
    }
  }
  __syncthreads();

  // -------- phase 1: compute qvals (stride over nq) --------
  if (active) {
    for (int iq = tx; iq < nq; iq += stride) {
      T qval = T(0);
      for (int i = 0; i < nd; ++i)
        qval += phi[iq * nd + i] * in_local_dofs[i];

      const T detJ_abs = detJ_cells[iq + cell_idx * nq];
      const T alpha = alpha_cells[cell_idx];

      qvals[iq] = alpha * qval * detJ_abs;
    }
  }
  __syncthreads();

  // -------- phase 2: accumulate back to DOFs (stride over nd) --------
  if (active) {
    for (int i = tx; i < nd; i += stride) {
      T fval = T(0);
      for (int iq = 0; iq < nq; ++iq)
        fval += qvals[iq] * phi[iq * nd + i];

      const int g_dof_idx = dofmap[cell_idx * nd + i];
      atomicAdd(&out_dofs[g_dof_idx], fval * global_coefficient);
    }
  }
}

template <typename T, int nd, int nq>
__global__ void mass_operator_baseline_optiT(
    const T *__restrict__ in_dofs,  T *__restrict__ out_dofs,
    const T *__restrict__ alpha_cells, const T *__restrict__ detJ_cells,
    const std::int32_t *__restrict__ dofmap,
    const T *__restrict__ phi,      //  layout:  [q][i]          (row‑major)
    const T *__restrict__ phiT,     //  layout:  [i][q] (transpose) – NEW
    T  global_coefficient, int ncells)
{
  const int tx        = threadIdx.x;
  const int stride    = blockDim.x;
  const int c_local   = threadIdx.y;
  const int cells_pb  = blockDim.y;
  const int cell_idx  = blockIdx.x * cells_pb + c_local;
  const bool active   = (cell_idx < ncells);

  /* shared: nd DOFs + nq qvals per cell */
  extern __shared__ T sh[];
  const int per_cell  = nd + nq;
  T *in_local_dofs = sh + c_local * per_cell;   // [0..nd)
  T *qvals         = in_local_dofs + nd;        // [0..nq)

  __syncthreads();

  /* ---------- load DOFs ---------- */
  if (active) {
    for (int i = tx; i < nd; i += stride) {
      int g = dofmap[cell_idx * nd + i];
      in_local_dofs[i] = in_dofs[g];
    }
  }
  __syncthreads();

  /* ---------- phase‑1 : q‑point loop ---------- */
  if (active) {
    for (int iq = tx; iq < nq; iq += stride) {
      T qval = T(0);
      for (int i = 0; i < nd; ++i)
        qval += phi[iq * nd + i] * in_local_dofs[i];

      const T detJ = detJ_cells[iq + cell_idx * nq];
      const T a    = alpha_cells[cell_idx];
      qvals[iq]    = a * qval * detJ;
    }
  }
  __syncthreads();

  /* ---------- phase‑2 : dof loop (now uses phiT for coalesced reads) ---------- */
  if (active) {
    for (int i = tx; i < nd; i += stride) {
      T fval = T(0);
      for (int iq = 0; iq < nq; ++iq)
        fval += qvals[iq] * phiT[i * nq + iq];

      int g = dofmap[cell_idx * nd + i];
      atomicAdd(&out_dofs[g], fval * global_coefficient);
    }
  }
}

template <typename T, int nd, int nq>
__global__ void mass_operator_baseline_optimem(
    const T *__restrict__ in_dofs, T *__restrict__ out_dofs,
    const T *__restrict__ alpha_cells, const T *__restrict__ detJ_cells,
    const int *__restrict__ dofmap,
    const T *__restrict__ phi, // size nq*nd (row‑major q)
    T global_coefficient, int ncells) {
  /* ---------------- block / thread indexing ---------------- */
  const int tx = threadIdx.x;
  const int stride = blockDim.x;
  const int c_local = threadIdx.y;
  const int cells_pb = blockDim.y;
  const int cell_idx = blockIdx.x * cells_pb + c_local;
  const bool active = (cell_idx < ncells);

  /* ---------------- shared‑memory layout -------------------
   *  [ per‑cell slices | shared copy of φ ]
   *  per‑cell : nd DOFs + nq qvals
   */
  extern __shared__ T sh[];
  const int slice_sz = nd + nq;               // per‑cell
  const int phi_offset = slice_sz * cells_pb; // start of φ copy
  const int per_cell_off = c_local * slice_sz;

  T *in_local_dofs = sh + per_cell_off; /* nd  */
  T *qvals = in_local_dofs + nd;        /* nq  */
  T *phi_s = sh + phi_offset;           /* nd*nq (for ALL cells) */

  /* ---------------- step 0 : copy φ to shared once ---------------- */
  // All threads across whole block cooperate (only once per kernel launch)
  for (int idx = threadIdx.y * stride + tx; idx < nd * nq;
       idx += blockDim.y * stride)
    phi_s[idx] = phi[idx];
  __syncthreads(); // φ ready for everybody

  /* ---------------- load DOFs ---------------- */
  if (active) {
    for (int i = tx; i < nd; i += stride) {
      const int gd = dofmap[cell_idx * nd + i];
      in_local_dofs[i] = in_dofs[gd];
    }
  }
  __syncthreads();

  /* ---------------- phase 1 : qvals ---------------- */
  if (active) {
    const T alpha = alpha_cells[cell_idx];
    for (int q = tx; q < nq; q += stride) {
      T acc = 0;
      for (int k = 0; k < nd; ++k)
        acc += phi_s[q * nd + k] * in_local_dofs[k];

      const T detJ = detJ_cells[q + cell_idx * nq];
      qvals[q] = alpha * acc * detJ;
    }
  }
  __syncthreads();

  /* ---------------- phase 2 : scatter back ---------------- */
  if (active) {
    for (int k = tx; k < nd; k += stride) {
      T f = 0;
      for (int q = 0; q < nq; ++q)
        f += qvals[q] * phi_s[q * nd + k];

      const int gd = dofmap[cell_idx * nd + k];
      atomicAdd(&out_dofs[gd], f * global_coefficient);
    }
  }
}

template <typename T, int nd, int nq>
__global__ void nonlinearmass_operator_baseline(
    const T *__restrict__ in_dofs1, const T *__restrict__ in_dofs2,
    T *__restrict__ out_dofs, const T *__restrict__ alpha_cells,
    const T *__restrict__ detJ_cells, const std::int32_t *__restrict__ dofmap,
    const T *__restrict__ phi, T global_coefficient) {
  const int tx = threadIdx.x;
  const int cell_idx = blockIdx.x;
  int g_dof_idx = -1;

  T alpha = alpha_cells[cell_idx]; // Load DG0 alpha coefficient

  __shared__ T in_local_dofs1[nd];
  __shared__ T in_local_dofs2[nd];

  __shared__ T qvals[nq];

  if (tx < nd) {
    g_dof_idx = dofmap[tx + nd * cell_idx];
    in_local_dofs1[tx] = in_dofs1[g_dof_idx];
    in_local_dofs2[tx] = in_dofs2[g_dof_idx];
  }
  __syncthreads();

  // Load geometry data
  T detJ_abs = detJ_cells[tx + cell_idx * nq];

  // Compute u1_i(x_q) and u2_i(x_q)
  T qval1 = 0.;
  T qval2 = 0.;
  for (int i = 0; i < nd; ++i) {
    qval1 += phi[tx * nd + i] * in_local_dofs1[i];
    qval2 += phi[tx * nd + i] * in_local_dofs2[i];
  }

  qvals[tx] = alpha * qval1 * qval2 * detJ_abs;
  __syncthreads();
  // if(cell_idx == 0)
  //   printf("qvals[%d]=%f \n", tx, qvals[tx]);

  if (tx < nd) {
    T fval = 0.;
    for (int i = 0; i < nq; ++i) {
      fval += qvals[i] * phi[i * nd + tx];
    }

    // if(cell_idx == 0)
    //   printf("out_dof[%d]=%f \n", g_dof_idx, fval);
    atomicAdd(&out_dofs[g_dof_idx], fval * global_coefficient);
  }

  // printf("qvals[%d]=%f \n", tx, qval);
}

/// Compute the facets mass operator
/// @param in_dofs input global dofs (x),   size ndofs
/// @param out_dofs output global dofs (y), size ndofs
/// @param cell_facet list of (cell, facet) to integrate over, size nf*2
/// @param detJ_facets det(J_f(dzeta_q)),    size (nc, n_faces, nq)
/// @param alpha_cells DG0 alpha,           size ncells
/// @param dofmap global to local dofmap,   size (ncells, K)
/// @param nfaces number of topological faces on tdim entitz
/// @param nd number of dofs on tdim - 1 entity
/// @param nq number of quadrature points on tdim - 1 entity
/// @param K number of dofs on tdim entity
template <typename T, int nd, int nq>
// __launch_bounds__(max(nq, nd)) what i want to do
__global__ void facets_mass_operator_baseline(
    const T *__restrict__ in_dofs, T *__restrict__ out_dofs,
    const std::int32_t *__restrict__ cell_facet,
    const T *__restrict__ detJ_facets, const T *__restrict__ alpha_cells,
    const std::int32_t *__restrict__ dofmap, const T *__restrict__ facets_phi,
    const std::int32_t *__restrict__ faces_dofs, int n_faces,
    T global_coefficient) {
  const int tx = threadIdx.x;
  const int cell_idx = cell_facet[2 * blockIdx.x];
  const int local_face_idx = cell_facet[2 * blockIdx.x + 1];

  T alpha = alpha_cells[cell_idx]; // Load DG0 alpha coefficient

  __shared__ T in_local_dofs[nd];
  __shared__ T qvals[nq];

  const T *phi = &facets_phi[local_face_idx * nd * nq];

  int g_dof_idx = -1;
  if (tx < nd) {
    g_dof_idx = dofmap[tx + nd * cell_idx];
    in_local_dofs[tx] = in_dofs[g_dof_idx];
  }
  __syncthreads();

  // u_i(x_q)
  if (tx < nq) {
    T qval = 0.;
    for (int i = 0; i < nd; ++i) {
      qval += phi[tx * nd + i] * in_local_dofs[i]; // Bank conflict probably
    }

    qvals[tx] = alpha * qval *
                detJ_facets[tx + local_face_idx * nq + cell_idx * n_faces * nq];
    // printf("qvals[%d]=%f %d %d \n", tx, qvals[tx], local_face_idx, cell_idx);
    // printf("detj[%d]=%f %d %d \n", tx,
    //        detJ_facets[tx + local_face_idx * nq + cell_idx * n_faces * nq],
    //        local_face_idx, cell_idx);

    // printf("alpha[%d]=%f %d %d \n", tx, alpha, local_face_idx, cell_idx);
  }

  __syncthreads();
  if (tx < nd) {
    T fval = 0.;
    for (int i = 0; i < nq; ++i) {
      fval += qvals[i] * phi[i * nd + tx];
    }
    atomicAdd(&out_dofs[g_dof_idx], global_coefficient * fval);
  }
}

template <typename T, int nd, int nq>
__global__ void mass_exterior_diagonal(
    T *__restrict__ out_dofs, const std::int32_t *__restrict__ cell_facet,
    const T *__restrict__ detJ_facets, const T *__restrict__ alpha_cells,
    const std::int32_t *__restrict__ dofmap, const T *__restrict__ facets_phi,
    const std::int32_t *__restrict__ faces_dofs, int n_faces,
    T global_coefficient) {
  const int tx = threadIdx.x;
  const int cell_idx = cell_facet[2 * blockIdx.x];
  const int local_face_idx = cell_facet[2 * blockIdx.x + 1];
  int g_dof_idx = dofmap[tx + nd * cell_idx];

  T alpha = alpha_cells[cell_idx]; // Load DG0 alpha coefficient
  const T *phi = &facets_phi[local_face_idx * nd * nq];

  T qval = 0.;
  for (int i = 0; i < nq; ++i) {
    T phi_l = phi[i * nd + tx];
    qval += phi_l * phi_l *
            detJ_facets[tx + local_face_idx * nq + cell_idx * n_faces * nq];
  }
  atomicAdd(&out_dofs[g_dof_idx], qval * alpha * global_coefficient);
}

template <typename T, int nd, int nq>
__global__ void mass_diagonal(T *__restrict__ out_dofs,
                              const T *__restrict__ alpha_cells,
                              const T *__restrict__ detJ_cells,
                              const std::int32_t *__restrict__ dofmap,
                              const T *__restrict__ phi, T global_coefficient) {
  const int tx = threadIdx.x;
  const int cell_idx = blockIdx.x;
  int g_dof_idx = dofmap[tx + nd * cell_idx];
  T alpha = alpha_cells[cell_idx]; // Load DG0 alpha coefficient

  T qval = 0.;
  for (int i = 0; i < nq; ++i) {
    T phi_l = phi[i * nd + tx];
    qval += phi_l * phi_l * detJ_cells[i + cell_idx * nq];
  }

  atomicAdd(&out_dofs[g_dof_idx], qval * alpha * global_coefficient);
}

template <typename T, int nd, int nq>
__global__ void
nonlinear_mass_diagonal(const T *__restrict__ in_dofs, T *__restrict__ out_dofs,
                        const T *__restrict__ alpha_cells,
                        const T *__restrict__ detJ_cells,
                        const std::int32_t *__restrict__ dofmap,
                        const T *__restrict__ phi, T global_coefficient) {
  const int tx = threadIdx.x;
  const int cell_idx = blockIdx.x;
  int g_dof_idx = dofmap[tx + nd * cell_idx];
  T alpha = alpha_cells[cell_idx]; // Load DG0 alpha coefficient
  T in_local_dof = in_dofs[g_dof_idx];

  T qval = 0.;
  for (int i = 0; i < nq; ++i) {
    T phi_l = phi[i * nd + tx];
    qval += phi_l * phi_l * detJ_cells[i + cell_idx * nq];
  }

  atomicAdd(&out_dofs[g_dof_idx],
            qval * alpha * global_coefficient * in_local_dof);
}

template <typename T, int Q> __constant__ T qwts2_d[Q];
template <typename T, int Q> __constant__ T qpts2_d[Q];

/// Apply 3D mass matrix operator on tets \int alpha(x) * inner(u, v) dx to
/// in_dofs.
/// @param in_dofs input global dofs (x),   size ndofs
/// @param out_dofs output global dofs (y), size ndofs
/// @param alpha_cells DG0 alpha,           size ncells
/// @param detJ_cells det(J_K(dzeta_q)),    size Q * Q * ncells
/// @param dofmap global to local dofmap,   size K * ncells
/// @param N number of dofs on 1D interval
/// @param Q number of quadrature points on 1D interval
template <typename T, int N, int Q>
__launch_bounds__(Q *Q *Q) __global__
    void mass_operator3D(const T *__restrict__ in_dofs,
                         T *__restrict__ out_dofs,
                         const T *__restrict__ alpha_cells,
                         const T *__restrict__ detJ_cells,
                         const std::int32_t *__restrict__ dofmap) {
  auto tet_ijk = [](int i, int j, int k) {
    int w = i + j + k;
    int s = j + k;
    return (w + 2) * (w + 1) * w / 6 + (s + 1) * s / 2 + k;
  }; // Maps 3d grid to tet using twice the cantor pairing function: TODO think
     // about ordering

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tz = threadIdx.z;

  constexpr int K = N * (N + 1) * (N + 2) / 6; // Number of dofs on tet

  const int cell_idx = blockIdx.x;

  const int l_dof_idx = dof_reordering3d_d<N>[tet_ijk(
      tz, ty, tx)]; // Only valid for ty < N - tz, tx < N - ty - tz
  // const int l_dof_idx = tet_ijk(tz, ty, tx); // Only valid for ty < N - tz,
  // tx < N - ty - tz
  int g_dof_idx = -1;

  T alpha = alpha_cells[cell_idx]; // Load DG0 alpha coefficient

  // Load geometry data
  T detJ_abs =
      fabs(detJ_cells[tx + ty * Q + tz * Q * Q + cell_idx * Q * Q * Q]);

  // if(tx == 0) {
  //     for (int i = 0; i < Q; ++i) {
  //         printf("qwts0_d[%d] = %f, qwts1_d[%d] = %f\n", i, qwts0_d<T, Q>[i],
  //         i, qwts1_d<T, Q>[i]);
  //     }
  // }

  __shared__ T c0[N][N][N]; // in_local_dofs
  __shared__ T c1[N][N][Q];
  __shared__ T c2[N][Q][Q];
  __shared__ T c3[Q][Q][Q];

  if (tz < N && ty < N && tx < N)
    c0[tz][ty][tx] = 0.;
  if (ty < N && tz < N)
    c1[tz][ty][tx] = 0.;
  if (tz < N)
    c2[tz][ty][tx] = 0.;

  if (tx < N - ty - tz && ty < N - tz && tz < N) {
    g_dof_idx = dofmap[l_dof_idx + cell_idx * K];
    // printf("l_dof_idx[%d][%d][%d] = %d\n", tz, ty, tx, l_dof_idx);

    c0[tz][ty][tx] = in_dofs[g_dof_idx];
    // if (cell_idx == 0)
    //   printf("in_local_dofs[%d][%d][%d] = %f\n", tz, ty, tx, c0[tz][ty][tx]);
  }
  __syncthreads();

  // 1. --- Evaluate u(x_q) (dofs -> qvals) ---
  // 1.1 c0[N][N][N] -> c1[N][N][Q]
  {
    T p = qpts2_d<T, Q>[tx];
    T s = 1.0 - p;
    T r = p / s;
    T w = 1.;
    if (ty < N - tz && tz < N) { // tz := alpha1, ty := alpha2, tx := i3
      // T w = s^(N - 1 - ty - tz);
      for (int i = 0; i < N - 1 - ty - tz; ++i)
        w *= s;

      // printf("w[%d, %d]=%f\n", tx, ty, w);

      for (int alpha3 = 0; alpha3 < N - ty - tz; ++alpha3) {
        // c1[ty][tx] += w;
        c1[tz][ty][tx] += w * c0[tz][ty][alpha3];
        w *= r * (N - 1 - tz - ty - alpha3) / (1 + alpha3);
      }
      // printf("c1[%d, %d]=%f\n", tx, ty, c1[ty][tx]);
    }
  }

  __syncthreads();

  // 1.2 c1[N][N][Q] -> c2[N][Q][Q]
  // tz := alpha1, ty := i2, tx := i3
  {
    T p = qpts1_d<T, Q>[ty];
    T s = 1.0 - p;
    T r = p / s;
    T w = 1.;

    if (tz < N) {
      // T w = s^(N - 1 - tz);
      for (int i = 0; i < N - 1 - tz; ++i)
        w *= s;

      // printf("w[%d, %d]=%f\n", tx, ty, w);

      for (int alpha2 = 0; alpha2 < N - tz; ++alpha2) {
        // c1[ty][tx] += w;
        c2[tz][ty][tx] += w * c1[tz][alpha2][tx];
        w *= r * (N - 1 - tz - alpha2) / (1 + alpha2);
      }
      // printf("c1[%d, %d]=%f\n", tx, ty, c1[ty][tx]);
    }
  }
  __syncthreads();

  // 1.3 c2[N][Q][Q] -> c3[Q][Q][Q]
  // tz := i1, ty := i2, tx := i3
  T qval = 0.;
  {
    T p = qpts0_d<T, Q>[tz];
    T s = 1.0 - p;
    T r = p / s;
    T w = 1.;
    for (int i = 0; i < N - 1; ++i)
      w *= s;
    for (int alpha1 = 0; alpha1 < N; ++alpha1) {
      qval += w * c2[alpha1][ty][tx];
      w *= r * (N - 1 - alpha1) / (1 + alpha1);
    }
  }

  // 2. Apply geometry
  c3[tz][ty][tx] = alpha * qval * detJ_abs;
  __syncthreads();

  // if (cell_idx == 0) {
  //   printf("qvals[%d, %d, %d]=%f\n", tz, ty, tx, c3[tz][ty][tx]);
  // }

  // 3. Compute Moments (qvals -> dofs)
  T(&f0)
  [Q][Q][Q] = c3; // qqq
  T(&f1)
  [N][Q][Q] = c2; // nqq
  T(&f2)
  [N][N][Q] = c1; // nnq
  T(&f3)
  [N][N][N] = c0; // nnn

  if (tz < N && ty < N && tx < N)
    f3[tz][ty][tx] = 0.;
  if (ty < N && tz < N)
    f2[tz][ty][tx] = 0.;
  if (tz < N)
    f1[tz][ty][tx] = 0.;
  __syncthreads();
  // 3.1 f0[Q][Q][Q] -> f1[N][Q][Q]
  // tz := alpha1, ty := i2, tx := i3
  {
    if (tz < N) {
      for (int i1 = 0; i1 < Q; ++i1) {
        T p = qpts2_d<T, Q>[i1];
        T s = 1.0 - p;
        T r = p / s;
        T w = qwts2_d<T, Q>[i1];
        // T ww = w * s ** n;
        for (int k = 0; k < N - 1; ++k)
          w *= s;
        // ww *= r * (n - alpha1) / (1 + alpha1)
        for (int k = 0; k < tz; ++k) {
          w *= r * (N - 1 - k) / (1 + k);
        }
        f1[tz][ty][tx] += w * f0[i1][ty][tx];
      }
      // printf("f1[%d, %d, %d]=%f\n", tz, ty, tx, f1[tz][ty][tx]);
    }
  }
  __syncthreads();

  // 3.2 f1[N][Q][Q] -> f2[N][N][Q]
  // tz := alpha1, ty := alpha2, tx := i3
  {
    if (tz < N && ty < N) {
      for (int i2 = 0; i2 < Q; ++i2) {
        T p = qpts1_d<T, Q>[i2];
        T s = 1.0 - p;
        T r = p / s;
        T w = qwts1_d<T, Q>[i2];
        // T ww = w * s ** n;
        for (int k = 0; k < N - 1 - tz; ++k)
          w *= s;
        // ww *= r * (n - alpha1) / (1 + alpha1)
        // for (int k = 0; k < ty - tz; ++k) { // this also works
        for (int k = 0; k < ty; ++k) {
          w *= r * (N - 1 - tz - k) / (1 + k);
        }
        f2[tz][ty][tx] += w * f1[tz][i2][tx];
      }
    }
  }
  __syncthreads();

  // 3.3 f2[N][N][Q] -> f3[N][N][N]
  // tz := alpha1, ty := alpha2, tx := alpha3
  {
    if (tz < N && ty < N && tx < N) {
      for (int i3 = 0; i3 < Q; ++i3) {
        T p = qpts0_d<T, Q>[i3];
        T s = 1.0 - p;
        T r = p / s;
        T w = qwts0_d<T, Q>[i3];
        // T ww = w * s ** n;
        for (int k = 0; k < N - 1 - tz - ty; ++k)
          w *= s;

        // for (int k = 0; k < tx - ty - tz; ++k) { // this also works
        for (int k = 0; k < tx; ++k) {
          w *= r * (N - 1 - tz - ty - k) / (1 + k);
        }
        f3[tz][ty][tx] += w * f2[tz][ty][i3];
      }
    }
  }
  __syncthreads();

  if (tx < N - ty - tz && ty < N - tz && tz < N) {
    // if(cell_idx == 0)
    //   printf("out_dof[%d]=%f \n", g_dof_idx, f3[tz][ty][tx]);
    atomicAdd(&out_dofs[g_dof_idx], f3[tz][ty][tx]);
  }
}

/// Apply mass matrix operator \int alpha(x) * inner(u, v) dx to in_dofs.
/// @param in_dofs input global dofs (x),   size ndofs
/// @param out_dofs output global dofs (y), size ndofs
/// @param alpha_cells DG0 alpha,           size ncells
/// @param detJ_cells det(J_K(dzeta_q)),    size Q * Q * ncells
/// @param dofmap global to local dofmap,   size K * ncells
/// @param phi_1 B_i^N(x_q),                size Q * N
/// @param phi_0_N B_j^i(x_q),              size N * Q * N
/// @param in_dofs input global dofs (x),   size ndofs
/// @param N number of dofs on 1D interval
/// @param Q number of quadrature points on 1D interval
template <typename T, int N, int Q>
__launch_bounds__(Q *Q) __global__
    void mass_operator_sf(const T *__restrict__ in_dofs,
                          T *__restrict__ out_dofs,
                          const T *__restrict__ alpha_cells,
                          const T *__restrict__ detJ_cells,
                          const std::int32_t *__restrict__ dofmap,
                          const T *__restrict__ phi_1,
                          const T *__restrict__ phi_0_N) {
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

  T alpha = alpha_cells[cell_idx]; // Load DG0 alpha coefficient
  // Load geometry data
  T detJ_abs = fabs(detJ_cells[tx + ty * Q + cell_idx * Q * Q]);

  __shared__ T c0[N][N];
  __shared__ T c1[N][Q];
  __shared__ T c2[Q][Q];

  if (ty < N)
    c1[ty][tx] = 0.;
  if (tx < N && ty < N) {
    c0[ty][tx] = 0.;
  }
  if (tx < N - ty && ty < N) {
    g_dof_idx = dofmap[l_dof_idx + cell_idx * K];
    c0[ty][tx] = in_dofs[g_dof_idx];
  }
  __syncthreads();
  // if (tx < N && ty < N) {
  //   printf("c0[%d, %d]=%f\n", ty, tx, c0[ty][tx]);
  // }
  // 1. Evaluate u(x_q) (dofs -> c2)
  // 1.1 c0[N][N] -> c1[N][Q]
  // ty := alpha1, tx := i2
  if (ty < N) {
    for (int alpha2 = 0; alpha2 < N - ty; ++alpha2) {
      // B_alpha2^{p - alpha1}(t0)
      c1[ty][tx] +=
          phi_0_N[(N - 1 - ty) * Q * N + tx * N + alpha2] * c0[ty][alpha2];
    }
    // printf("c1[%d, %d]=%f\n", ty, tx, c1[ty][tx]);
  }

  __syncthreads();

  T qval = 0.;
  // i1:= ty, i2 := tx
  for (int alpha1 = 0; alpha1 < N; ++alpha1) {
    // Load B_alpha1^N(t1)
    qval += phi_1[ty * N + alpha1] * c1[alpha1][tx];
  }

  // 2. Apply geometry
  c2[ty][tx] = alpha * qval * detJ_abs;
  // printf("qvals[%d, %d]=%f\n", ty, tx, c2[ty][tx]);
  // printf("qvals[%d, %d]=%f\n", ty, tx, qval);
  __syncthreads();

  // 3. Compute Moments (qvals -> dofs)
  T(&f0)
  [Q][Q] = c2; // qq
  T(&f1)
  [N][Q] = c1; // nq
  T(&f2)
  [N][N] = c0; // nn

  if (ty < N && tx < N)
    f2[ty][tx] = 0.;
  if (ty < N)
    f1[ty][tx] = 0.;
  __syncthreads();
  // 3.1 f0[Q][Q] -> f1[N][Q]
  // ty := alpha1, tx := i2
  {
    if (ty < N) {
      for (int i1 = 0; i1 < Q; ++i1) {
        T w = qwts1_d<T, Q>[i1];
        f1[ty][tx] += w * phi_1[i1 * N + ty] * f0[i1][tx];
      }
      // printf("f1[%d, %d]=%f\n", ty, tx, f1[ty][tx]);
    }
  }
  __syncthreads();

  // 3.1 f1[N][Q] -> f2[N][N]
  // ty := alpha1, tx := alpha2
  {
    if (ty < N && tx < N) {
      for (int i2 = 0; i2 < Q; ++i2) {
        T w = qwts0_d<T, Q>[i2];
        f2[ty][tx] +=
            w * phi_0_N[(N - 1 - ty) * Q * N + i2 * N + tx] * f1[ty][i2];
      }
    }
  }

  if (tx < N - ty && ty < N) {
    // printf("out_dof[%d]=%f \n", g_dof_idx, f3[tz][ty][tx]);
    atomicAdd(&out_dofs[g_dof_idx], f2[ty][tx]);
  }
}

/// Apply mass matrix operator \int alpha(x) * inner(u, v) dx to in_dofs.
/// @param in_dofs input global dofs (x),   size ndofs
/// @param out_dofs output global dofs (y), size ndofs
/// @param alpha_cells DG0 alpha,           size ncells
/// @param detJ_cells det(J_K(dzeta_q)),    size Q * Q * ncells
/// @param dofmap global to local dofmap,   size K * ncells
/// @param phi_1 B_i^N(x_q),                size Q * N
/// @param phi_0_N B_j^i(x_q),              size N * Q * N
/// @param in_dofs input global dofs (x),   size ndofs
/// @param N number of dofs on 1D interval
/// @param Q number of quadrature points on 1D interval
template <typename T, int N, int Q>
__launch_bounds__(Q *Q *Q) __global__ void mass_operator3D_sf(
    const T *__restrict__ in_dofs, T *__restrict__ out_dofs,
    const T *__restrict__ alpha_cells, const T *__restrict__ detJ_cells,
    const std::int32_t *__restrict__ dofmap, const T *__restrict__ phi_2,
    const T *__restrict__ phi_1_N, const T *__restrict__ phi_0_N) {
  auto tet_ijk = [](int i, int j, int k) {
    int w = i + j + k;
    int s = j + k;
    return (w + 2) * (w + 1) * w / 6 + (s + 1) * s / 2 + k;
  }; // Maps 3d grid to tet using twice the cantor pairing function

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tz = threadIdx.z;

  constexpr int K = N * (N + 1) * (N + 2) / 6; // Number of dofs on tet

  const int cell_idx = blockIdx.x;

  const int l_dof_idx = dof_reordering3d_d<N>[tet_ijk(
      tz, ty, tx)]; // Only valid for ty < N - tz, tx < N - ty - tz
  int g_dof_idx = -1;

  T alpha = alpha_cells[cell_idx]; // Load DG0 alpha coefficient

  // Load geometry data
  T detJ_abs =
      fabs(detJ_cells[tx + ty * Q + tz * Q * Q + cell_idx * Q * Q * Q]);

  __shared__ T scratch1[N * N * N];
  __shared__ T scratch2[N * N * N];

  // Load tables in shared memory
  __shared__ T phi_0_N_s[N][Q][N];
  __shared__ T phi_1_N_s[N][Q][N];
  __shared__ T phi_2_s[Q][N];

  if (tz < N && tx < N) {
    phi_0_N_s[tz][ty][tx] = phi_0_N[tz * Q * N + ty * N + tx];
    phi_1_N_s[tz][ty][tx] = phi_1_N[tz * Q * N + ty * N + tx];
  }
  if (tz == 0 && tx < N) {
    phi_2_s[ty][tx] = phi_2[ty * N + tx];
  }

  T(&c0)
  [N * N * N] = scratch1; // nnn

  if (tz < N && ty < N && tx < N)
    c0[ijk<N * N, N, 1>(tz, ty, tx)] = 0.;

  if (tx < N - ty - tz && ty < N - tz && tz < N) {
    g_dof_idx = dofmap[l_dof_idx + cell_idx * K];
    c0[ijk<N * N, N, 1>(tz, ty, tx)] = in_dofs[g_dof_idx];
  }
  __syncthreads();

  // 1. Evaluate u(x_q) (dofs -> c2)
  T(&c1)
  [N * N * N] = scratch2; // nnq
  // 1.1 c0[N][N][N] -> c1[N][N][Q], scratch1 -> scratch2
  if (ty < N - tz && tz < N) { // tz := alpha1, ty := alpha2, tx := i3
    T lc1 = 0.;
    for (int alpha3 = 0; alpha3 < N - ty; ++alpha3) {
      // B_alpha3^{p - alpha1 - alpha2}(t0)
      lc1 += phi_0_N_s[(N - 1 - tz - ty)][tx][alpha3] *
             c0[ijk<N * N, N, 1>(tz, ty, alpha3)];
    }
    c1[ijk<N * Q, Q, 1>(tz, ty, tx)] = lc1;
  }
  __syncthreads();

  // 1.2 c1[N][N][Q] -> c2[N][Q][Q], scratch2 -> scratch1
  T(&c2)
  [N * N * N] = scratch1; // nqq
  // tz := alpha1, ty := i2, tx := i3
  {
    if (tz < N) {
      T lc2 = 0.;
      for (int alpha2 = 0; alpha2 < N - tz; ++alpha2) {
        lc2 += phi_1_N_s[(N - 1 - tz)][ty][alpha2] *
               c1[ijk<N * Q, Q, 1>(tz, alpha2, tx)];
      }
      c2[ijk<Q * Q, Q, 1>(tz, ty, tx)] = lc2;
    }
  }
  __syncthreads();

  // 1.3 c2[N][Q][Q] -> c3[Q][Q][Q], scratch1 -> scratch2
  // tz := i1, ty := i2, tx := i3
  T(&c3)
  [N * N * N] = scratch2; // qqq
  T lc3 = 0.;
  for (int alpha1 = 0; alpha1 < N; ++alpha1) {
    lc3 += phi_2_s[tz][alpha1] * c2[ijk<Q * Q, Q, 1>(alpha1, ty, tx)];
  }

  // 2. Apply geometry
  c3[ijk<Q * Q, Q, 1>(tz, ty, tx)] = alpha * lc3 * detJ_abs;
  // printf("qvals[%d, %d]=%f\n", ty, tx, qvals[ty][tx]);
  // printf("detJ_abs[%d, %d, %d]=%f\n", tz, ty, tx, detJ_abs);
  __syncthreads();

  // 3. Compute Moments (qvals -> dofs)
  T(&f0)
  [N * N * N] = c3; // qqq

  // printf("f0[%d, %d, %d]=%f\n", tz, ty, tx, f0[ijk<Q * Q, Q, 1>(tz, ty,
  // tx)]); 3.1 f0[Q][Q][Q] -> f1[N][Q][Q], scratch2 -> scratch1
  T(&f1)
  [N * N * N] = scratch1; // nqq

  // tz := alpha1, ty := i2, tx := i3
  {
    if (tz < N) {
      T lf1 = 0.;
      for (int i1 = 0; i1 < Q; ++i1) {
        T w = qwts2_d<T, Q>[i1];
        lf1 += w * phi_2_s[i1][tz] * f0[ijk<Q * Q, Q, 1>(i1, ty, tx)];
      }
      f1[ijk<Q * Q, Q, 1>(tz, ty, tx)] = lf1;
      // printf("f1[%d, %d, %d]=%f\n", tz, ty, tx, f1[tz][ty][tx]);
    }
  }
  __syncthreads();

  // 3.2 f1[N][Q][Q] -> f2[N][N][Q], scratch1 -> scratch2
  T(&f2)
  [N * N * N] = scratch2; // nqq

  // tz := alpha1, ty := alpha2, tx := i3
  {
    if (tz < N && ty < N) {
      T lf2 = 0.;
      for (int i2 = 0; i2 < Q; ++i2) {
        T w = qwts1_d<T, Q>[i2];
        lf2 += w * phi_1_N_s[(N - 1 - tz)][i2][ty] *
               f1[ijk<Q * Q, Q, 1>(tz, i2, tx)];
      }
      f2[ijk<N * Q, Q, 1>(tz, ty, tx)] = lf2;
    }
  }
  __syncthreads();

  // 3.3 f2[N][N][Q] -> f3[N][N][N]
  // tz := alpha1, ty := alpha2, tx := alpha3
  if (tx < N - ty - tz && ty < N - tz && tz < N) {

    T lf3 = 0.;
    for (int i3 = 0; i3 < Q; ++i3) {
      T w = qwts0_d<T, Q>[i3];
      lf3 += w * phi_0_N_s[(N - 1 - tz - ty)][i3][tx] *
             f2[ijk<N * Q, Q, 1>(tz, ty, i3)];
    }
    // if(cell_idx == 0)
    // printf("out_dof[%d]=%f \n", g_dof_idx, lf3);
    // printf("out_dof[%d]=%f \n", g_dof_idx, f3[tz][ty][tx]);
    atomicAdd(&out_dofs[g_dof_idx], lf3);
  }
}
