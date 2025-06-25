#pragma once

#include "types.hpp"
#include "cg_gpu.hpp"
#include "vector.hpp"

namespace freefus {
template <typename T, typename U>
auto create_solver(std::shared_ptr<fem::FunctionSpace<U>> V,
                   const UserConfig<U> config) {

  auto index_map = V->dofmap()->index_map;
  auto bs = V->dofmap()->index_map_bs();
  auto cg_p = std::make_unique<acc::CGSolver<acc::DeviceVector<T>>>(index_map, bs);
  cg_p->set_max_iterations(config.cg_max_steps);
  cg_p->set_tolerance(config.cg_tol);
  return cg_p;
}
} // namespace freefus