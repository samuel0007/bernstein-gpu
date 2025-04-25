#pragma once

template <typename T, int Q> __constant__ T qwts0_d[Q];
template <typename T, int Q> __constant__ T qpts0_d[Q];

template <typename T, int Q> __constant__ T qwts1_d[Q];
template <typename T, int Q> __constant__ T qpts1_d[Q];

template <int N> __constant__ int dof_reordering_d[(N + 1) * N / 2];

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
__global__ void mass_operator_baseline(const T *__restrict__ in_dofs,
                                       T *__restrict__ out_dofs,
                                       const T *__restrict__ alpha_cells,
                                       const T *__restrict__ detJ_cells,
                                       const std::int32_t *__restrict__ dofmap,
                                       const T *__restrict__ phi) {
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

  if (tx < nd) {
    T fval = 0.;
    for (int i = 0; i < nq; ++i) {
      fval += qvals[i] * phi[i * nd + tx];
    }

    atomicAdd(&out_dofs[g_dof_idx], fval);
  }

  // printf("qvals[%d]=%f \n", tx, qval);
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

  const int l_dof_idx =
      tet_ijk(tz, ty, tx); // Only valid for ty < N - tz, tx < N - ty - tz
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
  // if (cell_idx == 0)
  // printf("qvals[%d, %d, %d]=%f\n", tz, ty, tx, qval);
  // printf("qvals[%d, %d, %d]=%f\n", tz, ty, tx, c3[tz][ty][tx]);
  __syncthreads();

  // 3. Compute Moments (qvals -> dofs)
  T(&f0)[Q][Q][Q] = c3; // qqq
  T(&f1)[N][Q][Q] = c2; // nqq
  T(&f2)[N][N][Q] = c1; // nnq
  T(&f3)[N][N][N] = c0; // nnn

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
    // printf("out_dof[%d]=%f \n", g_dof_idx, f3[tz][ty][tx]);
    atomicAdd(&out_dofs[g_dof_idx], f3[tz][ty][tx]);
  }
}