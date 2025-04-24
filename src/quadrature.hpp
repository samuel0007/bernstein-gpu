#pragma once
#include "types.hpp"
#include <basix/quadrature.h>

// BASIX CODE

/// Evaluate the nth Jacobi polynomial and derivatives with weight
/// parameters (a, 0) at points x
/// @param[in] a Jacobi weight a
/// @param[in] n Order of polynomial
/// @param[in] nderiv Number of derivatives (if zero, just compute
/// polynomial itself)
/// @param[in] x Points at which to evaluate
/// @return Array of polynomial derivative values (rows) at points
/// (columns)
template <std::floating_point T>
mdarray_t<T, 2> compute_jacobi_deriv(T a, std::size_t n, std::size_t nderiv,
                                     std::span<const T> x) {
  std::vector<std::size_t> shape = {x.size()};
  mdarray_t<T, 3> J(nderiv + 1, n + 1, x.size());
  mdarray_t<T, 2> Jd(n + 1, x.size());
  for (std::size_t i = 0; i < nderiv + 1; ++i) {
    if (i == 0) {
      for (std::size_t j = 0; j < Jd.extent(1); ++j)
        Jd(0, j) = 1.0;
    } else {
      for (std::size_t j = 0; j < Jd.extent(1); ++j)
        Jd(0, j) = 0.0;
    }

    if (n > 0) {
      if (i == 0) {
        for (std::size_t j = 0; j < Jd.extent(1); ++j)
          Jd(1, j) = (x[j] * (a + 2.0) + a) * 0.5;
      } else if (i == 1) {
        for (std::size_t j = 0; j < Jd.extent(1); ++j)
          Jd(1, j) = a * 0.5 + 1;
      } else {
        for (std::size_t j = 0; j < Jd.extent(1); ++j)
          Jd(1, j) = 0.0;
      }
    }

    for (std::size_t j = 2; j < n + 1; ++j) {
      const T a1 = 2 * j * (j + a) * (2 * j + a - 2);
      const T a2 = (2 * j + a - 1) * (a * a) / a1;
      const T a3 = (2 * j + a - 1) * (2 * j + a) / (2 * j * (j + a));
      const T a4 = 2 * (j + a - 1) * (j - 1) * (2 * j + a) / a1;
      for (std::size_t k = 0; k < Jd.extent(1); ++k)
        Jd(j, k) = Jd(j - 1, k) * (x[k] * a3 + a2) - Jd(j - 2, k) * a4;
      if (i > 0) {
        for (std::size_t k = 0; k < Jd.extent(1); ++k)
          Jd(j, k) += i * a3 * J(i - 1, j - 1, k);
      }
    }

    for (std::size_t j = 0; j < Jd.extent(0); ++j)
      for (std::size_t k = 0; k < Jd.extent(1); ++k)
        J(i, j, k) = Jd(j, k);
  }

  mdarray_t<T, 2> result(nderiv + 1, x.size());
  for (std::size_t i = 0; i < result.extent(0); ++i)
    for (std::size_t j = 0; j < result.extent(1); ++j)
      result(i, j) = J(i, n, j);

  return result;
}

/// Computes the m roots of \f$P_{m}^{a,0}\f$ on [-1,1] by Newton's
/// method. The initial guesses are the Chebyshev points.  Algorithm
/// implemented from the pseudocode given by Karniadakis and Sherwin.
template <std::floating_point T>
std::vector<T> compute_gauss_jacobi_points(T a, int m) {
  constexpr T eps = 1.0e-8;
  constexpr int max_iter = 100;
  std::vector<T> x(m);
  for (int k = 0; k < m; ++k) {
    // Initial guess
    x[k] = -std::cos((2.0 * k + 1.0) * M_PI / (2.0 * m));
    if (k > 0)
      x[k] = 0.5 * (x[k] + x[k - 1]);

    int j = 0;
    while (j < max_iter) {
      T s = 0;
      for (int i = 0; i < k; ++i)
        s += 1.0 / (x[k] - x[i]);
      std::span<const T> _x(&x[k], 1);
      mdarray_t<T, 2> f = compute_jacobi_deriv<T>(a, m, 1, _x);
      T delta = f(0, 0) / (f(1, 0) - f(0, 0) * s);
      x[k] -= delta;
      if (std::abs(delta) < eps)
        break;
      ++j;
    }
  }

  return x;
}

/// @note Computes on [-1, 1]
template <std::floating_point T>
std::array<std::vector<T>, 2> compute_gauss_jacobi_rule(T a, int m) {
  std::vector<T> pts = compute_gauss_jacobi_points<T>(a, m);
  mdarray_t<T, 2> Jd = compute_jacobi_deriv<T>(a, m, 1, pts);
  T a1 = std::pow(2.0, a + 1.0);
  std::vector<T> wts(m);
  for (int i = 0; i < m; ++i) {
    T x = pts[i];
    T f = Jd(1, i);
    wts[i] = a1 / (1.0 - x * x) / (f * f);
  }

  return {pts, wts};
}

// Returns three different quadrature rules in pairs (first is points, second is
// weights) 0: 0-Gauss-Jacobi Quadrature on [0, 1] (a = 0, b = 0) 1:
// 1-Gauss-Jacobi Quadrature on [0, 1] (a = 1, b = 0) 2: Collapsed Gauss-Jacobi
// Quadrature on a triangle
//     Where 0-Gauss-Jacobi is used in in x and
//           1-Gauss-Jacobi is used in in y
template <typename T>
std::array<std::pair<std::vector<T>, std::vector<T>>, 3>
create_quadrature_triangle_duffy(int q) {
  const int p = q * 2 - 2;
  auto [ptsT, wtsT] = basix::quadrature::make_quadrature<T>(
      basix::quadrature::type::gauss_jacobi, basix::cell::type::triangle,
      basix::polyset::type::standard, p);
  assert(ptsT.size() == q * q * 2);

  // Note: this computes the quadrature nodes on [-1, 1]
  auto [pts0, wts0] = compute_gauss_jacobi_rule<T>(0.0, q);
  auto [pts1, wts1] = compute_gauss_jacobi_rule<T>(1.0, q);

  // Rescale to [0, 1]
  std::ranges::transform(wts0, wts0.begin(), [](auto w) { return 0.5 * w; });
  std::ranges::transform(pts0, pts0.begin(),
                         [](auto x) { return 0.5 * (x + 1.0); });
  std::ranges::transform(wts1, wts1.begin(), [](auto w) { return 0.25 * w; });
  std::ranges::transform(pts1, pts1.begin(),
                         [](auto x) { return 0.5 * (x + 1.0); });

  mdspan_t<T, 2> x(ptsT.data(), q * q, 2);

  // Check if collapsed triangle quadrature = uncollapsed 1d tensor product on
  // quad
  int c = 0;
  T eps = 1e-10;
  for (int i = 0; i < q; ++i) {
    for (int j = 0; j < q; ++j) {
      assert(std::abs(x(c, 0) - pts0[i] * (1.0 - pts1[j])) < eps);
      assert(std::abs(x(c, 1) - pts1[j]) < eps);
      assert(std::abs((wtsT[c] - wts0[i] * wts1[j])) < eps);
      ++c;
    }
  }

  return {{{pts0, wts0}, {pts1, wts1}, {ptsT, wtsT}}};
}

template<typename T> std::array<std::pair<std::vector<T>, std::vector<T>>, 4>
create_quadrature_tetrahedron_duffy(int q) {
  const int p = q * 2 - 2;
  auto [ptsT, wtsT] = basix::quadrature::make_quadrature<T>(
      basix::quadrature::type::gauss_jacobi, basix::cell::type::tetrahedron,
      basix::polyset::type::standard, p);
  assert(ptsT.size() == q * q * q * 3);

  // Note: this computes the quadrature nodes on [-1, 1]
  auto [pts0, wts0] = compute_gauss_jacobi_rule<T>(0.0, q);
  auto [pts1, wts1] = compute_gauss_jacobi_rule<T>(1.0, q);
  auto [pts2, wts2] = compute_gauss_jacobi_rule<T>(2.0, q);

  // Rescale to [0, 1]
  std::ranges::transform(wts0, wts0.begin(), [](auto w) { return 0.5 * w; });
  std::ranges::transform(pts0, pts0.begin(),
                         [](auto x) { return 0.5 * (x + 1.0); });
  std::ranges::transform(wts1, wts1.begin(), [](auto w) { return 0.25 * w; });
  std::ranges::transform(pts1, pts1.begin(),
                         [](auto x) { return 0.5 * (x + 1.0); });
  std::ranges::transform(wts2, wts2.begin(), [](auto w) { return 0.125 * w; });
  std::ranges::transform(pts2, pts2.begin(),
                         [](auto x) { return 0.5 * (x + 1.0); });

  mdspan_t<T, 2> x(ptsT.data(), q * q * q, 3);

  // Check if collapsed triangle quadrature = uncollapsed 1d tensor product on
  // quad
  int c = 0;
  T eps = 1e-10;
  for (int i = 0; i < q; ++i) {
    for (int j = 0; j < q; ++j) {
      for (int k = 0; k < q; ++k) {
        assert(std::abs(x(c, 0) - pts0[i] * (1.0 - pts1[j]) * (1.0 - pts2[k])) < eps);
        assert(std::abs(x(c, 1) - pts1[j] * (1.0 - pts2[k])) < eps);
        assert(std::abs(x(c, 2) - pts2[k]) < eps);
        assert(std::abs((wtsT[c] - wts0[i] * wts1[j] * wts2[k])) < eps);
        ++c;
      }
    }
  }

  return {{{pts0, wts0}, {pts1, wts1}, {pts2, wts2}, {ptsT, wtsT}}};
}