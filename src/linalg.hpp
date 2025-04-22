#pragma once

#include <algorithm>
#include <dolfinx/common/types.h>
#include <dolfinx.h>

namespace linalg {

using namespace dolfinx;

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
  if(rnorm0 < 1e-20) return 0;

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