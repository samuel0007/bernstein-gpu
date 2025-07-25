

namespace kernels::fused {

// template <typename T, int nd, int nq, T beta = T(0.25), T gamma = T(0.5)>
// __global__ void fused_newmark(
//     T *__restrict__ out_dofs, const T *__restrict__ in_dofs,
//     const T *__restrict__ alpha1_cells, const T *__restrict__ alpha2_cells,
//     const T *__restrict__ alpha3_cells, const T *__restrict__ detJ_cells,
//     const T *__restrict__ G_cells, const std::int32_t *__restrict__ dofmap,
//     const T *__restrict__ dphi, T dt, T global_coefficient) {
//   const int cell_idx = blockIdx.x;
//   const int tx = threadIdx.x;
//   const int stride = blockDim.x;

//   int gdof = -1;

//   const T alpha1 = alpha1_cells[cell_idx]; // M
//   const T alpha2 = alpha2_cells[cell_idx]; // C
//   const T alpha3 = alpha3_cells[cell_idx]; // K

//   const T alpha_stiffness =
//       alpha2 * gamma * dt + alpha3 * beta * dt * dt; // C + K

//   T detJ_abs = detJ_cells[tx + cell_idx * nq];

//   __shared__ T scratch1[nq];
//   __shared__ T scratch2[nq];
//   __shared__ T scratch3[nq];
//   __shared__ T qvals[nq];

//   scratch1[tx] = 0.0;

//   for (int i = tx; i < nd; i += stride) {
//     gdof = dofmap[tx + cell_idx * nd];
//     scratch1[tx] = in_dofs[gdof];
//   }
//   __syncthreads();

//   auto phi = [&](auto j, auto k) { return dphi[j * nd + k]; };
//   auto dphix = [&](auto j, auto k) { return dphi[nd * nq + j * nd + k]; };
//   auto dphiy = [&](auto j, auto k) { return dphi[2 * nd * nq + j * nd + k]; };
//   auto dphiz = [&](auto j, auto k) { return dphi[3 * nd * nq + j * nd + k]; };

//   // Compute val{x,y} at quadrature points
//   T fw0 = 0, fw1 = 0, fw2 = 0;
//   T qval = 0.;
//   if (tx < nq) {
//     T val_x = 0.0;
//     T val_y = 0.0;
//     T val_z = 0.0;
//     for (int idof = 0; idof < nd; ++idof) {
//       qval += phi(tx, idof) * scratch1[idof];
//       val_x += dphix(tx, idof) * scratch1[idof];
//       val_y += dphiy(tx, idof) * scratch1[idof];
//       val_z += dphiz(tx, idof) * scratch1[idof];
//     }

//     // Apply geometric transformation to data at quadrature point
//     const int gid = tx + cell_idx * nq * 6;
//     const T G0 = G_cells[gid + nq * 0];
//     const T G1 = G_cells[gid + nq * 1];
//     const T G2 = G_cells[gid + nq * 2];
//     const T G3 = G_cells[gid + nq * 3];
//     const T G4 = G_cells[gid + nq * 4];
//     const T G5 = G_cells[gid + nq * 5];

//     fw0 = (G0 * val_x + G1 * val_y + G2 * val_z);
//     fw1 = (G1 * val_x + G3 * val_y + G4 * val_z);
//     fw2 = (G2 * val_x + G4 * val_y + G5 * val_z);
//   }
//   __syncthreads();

//   // Store values at quadrature points
//   qvals[tx] = alpha1 * qval * detJ_abs;
//   scratch1[tx] = fw0;
//   scratch2[tx] = fw1;
//   scratch3[tx] = fw2;

//   __syncthreads();

//   if (tx < nd) {
//     T y = 0.0;
//     T gradx = 0.;
//     T grady = 0.;
//     T gradz = 0.;
//     for (int iq = 0; iq < nq; ++iq) {
//       y += phi(iq, tx) * qvals[iq];
//       gradx += dphix(iq, tx) * scratch1[iq];
//       grady += dphiy(iq, tx) * scratch2[iq];
//       gradz += dphiz(iq, tx) * scratch3[iq];
//     }

//     T yd = y + alpha_stiffness * (gradx + grady + gradz);

//     // Sum
//     // Write back to global memory
//     atomicAdd(&out_dofs[gdof], global_coefficient * yd);
//   }
// }
// template <typename T, int nd, int nq, T beta = T(0.25), T gamma = T(0.5)>
// __launch_bounds__(nq)
// __global__ void fused_newmark(
//     T *__restrict__ out_dofs, const T *__restrict__ in_dofs,
//     const T *__restrict__ alpha1_cells, const T *__restrict__ alpha2_cells,
//     const T *__restrict__ alpha3_cells, const T *__restrict__ detJ_cells,
//     const T *__restrict__ G_cells, const std::int32_t *__restrict__ dofmap,
//     const T *__restrict__ dphi, T dt, T global_coefficient)
// {
//   const int cell_idx = blockIdx.x;
//   const int tx       = threadIdx.x;
//   const int stride   = blockDim.x;

//   const T alpha1 = alpha1_cells[cell_idx]; // M
//   const T alpha2 = alpha2_cells[cell_idx]; // C
//   const T alpha3 = alpha3_cells[cell_idx]; // K

//   const T alpha_stiffness =
//       alpha2 * gamma * dt + alpha3 * beta * dt * dt; // C + K

//   // --- shared ---
//   __shared__ T dofs_sh[nd];   // input dofs (never overwrite)
//   __shared__ T scratch1[nq];  // fw0
//   __shared__ T scratch2[nq];  // fw1
//   __shared__ T scratch3[nq];  // fw2
//   __shared__ T qvals[nq];     // mass part

//   // zero needed portions (optional if you write all elements anyway)
//   // for (int i = tx; i < nq; i += stride) {
//   //   scratch1[i] = scratch2[i] = scratch3[i] = qvals[i] = T(0);
//   // }

//   // load DOFs (strided -> covers nd even if stride < nd)
//   for (int i = tx; i < nd; i += stride) {
//     const int gdof = dofmap[cell_idx * nd + i];
//     dofs_sh[i] = in_dofs[gdof];
//   }
//   __syncthreads();

//   auto phi   = [&](int j, int k) { return dphi[j * nd + k]; };
//   auto dphix = [&](int j, int k) { return dphi[nd * nq     + j * nd + k]; };
//   auto dphiy = [&](int j, int k) { return dphi[2 * nd * nq + j * nd + k]; };
//   auto dphiz = [&](int j, int k) { return dphi[3 * nd * nq + j * nd + k]; };

//   // -------- phase 1: loop quadrature points (stride over nq) --------
//   for (int iq = tx; iq < nq; iq += stride) {
//     T qval = 0.0;
//     T val_x = 0.0, val_y = 0.0, val_z = 0.0;

//     for (int idof = 0; idof < nd; ++idof) {
//       const T u = dofs_sh[idof];
//       qval  += phi  (iq, idof) * u;
//       val_x += dphix(iq, idof) * u;
//       val_y += dphiy(iq, idof) * u;
//       val_z += dphiz(iq, idof) * u;
//     }

//     const int gid = iq + cell_idx * nq * 6;
//     const T G0 = G_cells[gid + nq * 0];
//     const T G1 = G_cells[gid + nq * 1];
//     const T G2 = G_cells[gid + nq * 2];
//     const T G3 = G_cells[gid + nq * 3];
//     const T G4 = G_cells[gid + nq * 4];
//     const T G5 = G_cells[gid + nq * 5];

//     scratch1[iq] = G0 * val_x + G1 * val_y + G2 * val_z;
//     scratch2[iq] = G1 * val_x + G3 * val_y + G4 * val_z;
//     scratch3[iq] = G2 * val_x + G4 * val_y + G5 * val_z;

//     const T detJ_abs = detJ_cells[iq + cell_idx * nq];
//     qvals[iq] = alpha1 * qval * detJ_abs;
//   }
//   __syncthreads();

//   // -------- phase 2: loop dofs (stride over nd) --------
//   for (int id = tx; id < nd; id += stride) {
//     T y = 0.0, gradx = 0.0, grady = 0.0, gradz = 0.0;

//     for (int iq = 0; iq < nq; ++iq) {
//       y     += phi  (iq, id) * qvals[iq];
//       gradx += dphix(iq, id) * scratch1[iq];
//       grady += dphiy(iq, id) * scratch2[iq];
//       gradz += dphiz(iq, id) * scratch3[iq];
//     }

//     const T yd = y + alpha_stiffness * (gradx + grady + gradz);
//     const int gdof = dofmap[cell_idx * nd + id];
//     atomicAdd(&out_dofs[gdof], global_coefficient * yd);
//   }
// }


template <typename T, int nd, int nq, T beta = T(0.25), T gamma = T(0.5)>
__global__ void fused_newmark(
    T *__restrict__ out_dofs, const T *__restrict__ in_dofs,
    const T *__restrict__ alpha1_cells, const T *__restrict__ alpha2_cells,
    const T *__restrict__ alpha3_cells, const T *__restrict__ detJ_cells,
    const T *__restrict__ G_cells, const std::int32_t *__restrict__ dofmap,
    const T *__restrict__ dphi, T dt, T global_coefficient,
    int ncells)                                      
{
  const int cell_local = threadIdx.y;              
  const int cells_per_block = blockDim.y;
  const int cell_idx = blockIdx.x * cells_per_block + cell_local;
  if (cell_idx >= ncells) return;

  const int tx     = threadIdx.x;
  const int stride = blockDim.x;

  const T alpha1 = alpha1_cells[cell_idx]; // M
  const T alpha2 = alpha2_cells[cell_idx]; // C
  const T alpha3 = alpha3_cells[cell_idx]; // K

  const T alpha_stiffness =
      alpha2 * gamma * dt + alpha3 * beta * dt * dt; // C + K

  // -------- dynamic shared memory layout --------
  // per cell: nd (dofs) + 4*nq (qvals + fw0/1/2)
  extern __shared__ T sh[];
  const int per_cell = nd + 4 * nq;
  T* dofs_sh   = sh + cell_local * per_cell;
  T* qvals     = dofs_sh + nd;
  T* scratch1  = qvals   + nq;
  T* scratch2  = scratch1 + nq;
  T* scratch3  = scratch2 + nq;

  // ---- load DOFs into shared (stride over nd) ----
  for (int i = tx; i < nd; i += stride) {
    const int gdof = dofmap[cell_idx * nd + i];
    dofs_sh[i] = in_dofs[gdof];
  }
  __syncthreads();

  auto phi   = [&](int j, int k) { return dphi[j * nd + k]; };
  auto dphix = [&](int j, int k) { return dphi[nd * nq     + j * nd + k]; };
  auto dphiy = [&](int j, int k) { return dphi[2 * nd * nq + j * nd + k]; };
  auto dphiz = [&](int j, int k) { return dphi[3 * nd * nq + j * nd + k]; };

  // ---- phase 1: loop over quadrature points ----
  for (int iq = tx; iq < nq; iq += stride) {
    T qval = 0.0;
    T val_x = 0.0, val_y = 0.0, val_z = 0.0;

    for (int idof = 0; idof < nd; ++idof) {
      const T u = dofs_sh[idof];
      qval  += phi  (iq, idof) * u;
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

    scratch1[iq] = G0 * val_x + G1 * val_y + G2 * val_z;
    scratch2[iq] = G1 * val_x + G3 * val_y + G4 * val_z;
    scratch3[iq] = G2 * val_x + G4 * val_y + G5 * val_z;

    const T detJ_abs = detJ_cells[iq + cell_idx * nq];
    qvals[iq] = alpha1 * qval * detJ_abs;
  }
  __syncthreads();

  // ---- phase 2: loop over dofs ----
  for (int id = tx; id < nd; id += stride) {
    T y = 0.0, gradx = 0.0, grady = 0.0, gradz = 0.0;

    for (int iq = 0; iq < nq; ++iq) {
      y     += phi  (iq, id) * qvals[iq];
      gradx += dphix(iq, id) * scratch1[iq];
      grady += dphiy(iq, id) * scratch2[iq];
      gradz += dphiz(iq, id) * scratch3[iq];
    }

    const T yd  = y + alpha_stiffness * (gradx + grady + gradz);
    const int g = dofmap[cell_idx * nd + id];
    atomicAdd(&out_dofs[g], global_coefficient * yd);
  }
}


} // namespace kernels::fused
