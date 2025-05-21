#pragma once

#include <algorithm>
#include <dolfinx/fem/FiniteElement.h>

using namespace dolfinx;

/// Function that takes in a simplex bernstein element, and returns a reordering
/// dofmap
template <int P, typename T>
std::vector<int>
get_tp_ordering2D(std::shared_ptr<const fem::FiniteElement<T>> element_p) {
  const std::vector<std::vector<std::vector<int>>> entity_dofs =
      element_p->entity_dofs();

  int v0 = entity_dofs[0][0][0];
  int v1 = entity_dofs[0][1][0];
  int v2 = entity_dofs[0][2][0];

  std::vector<int> e0 = entity_dofs[1][0];
  std::vector<int> e1 = entity_dofs[1][1];
  std::vector<int> e2 = entity_dofs[1][2];

  std::vector<int> c = entity_dofs[2][0];

  std::vector<int> dof_reordering;

  constexpr int N = P + 1;
  int c_count = 0;
  for (int row_idx = 0; row_idx < N; ++row_idx) {
    // bottom row
    if (row_idx == 0) {
      dof_reordering.push_back(v0);
      for (int i = 0; i < P - 1; ++i) {
        dof_reordering.push_back(e2[i]);
      }
      dof_reordering.push_back(v1);
      continue;
    }
    // top row
    if (row_idx == N - 1) {
      dof_reordering.push_back(v2);
      continue;
    }

    // inner section
    dof_reordering.push_back(e1[row_idx - 1]);
    for (int j = 0; j < P - row_idx - 1; ++j) {
      dof_reordering.push_back(c[c_count++]);
    }
    dof_reordering.push_back(e0[row_idx - 1]);
  }

  std::reverse(dof_reordering.begin(), dof_reordering.end());
  return dof_reordering;
}

/// Function that takes in a simplex bernstein element, and returns a reordering
/// dofmap
template <typename T>
std::vector<int>
get_tp_ordering1D(std::shared_ptr<basix::FiniteElement<T>> element_p, int p) {
  const std::vector<std::vector<std::vector<int>>> entity_dofs =
      element_p->entity_dofs();

  if (p == 0) {
    return {0};
  }

  int v0 = entity_dofs[0][0][0];
  int v1 = entity_dofs[0][1][0];

  std::vector<int> e0 = entity_dofs[1][0];
  std::vector<int> dof_reordering;
  dof_reordering.push_back(v0);
  for (int i = 0; i < e0.size(); ++i) {
    dof_reordering.push_back(e0[i]);
  }
  dof_reordering.push_back(v1);
  return dof_reordering;
}

// From Basix
//-----------------------------------------------------------------------------
std::vector<int> lex_dof_ordering(basix::element::family family,
                                  basix::cell::type cell, int degree) {
  std::vector<int> dof_ordering;
  std::vector<int> perm;

  switch (family) {
  case basix::element::family::P: {
    switch (cell) {
    case basix::cell::type::interval: {
      perm.push_back(0);
      if (degree > 0) {
        for (int i = 2; i <= degree; ++i)
          perm.push_back(i);
        perm.push_back(1);
      }
      break;
    }
    case basix::cell::type::quadrilateral: {
      perm.push_back(0);
      if (degree > 0) {
        int n = degree - 1;
        for (int i = 0; i < n; ++i)
          perm.push_back(4 + i);
        perm.push_back(1);
        for (int i = 0; i < n; ++i) {
          perm.push_back(4 + n + i);
          for (int j = 0; j < n; ++j)
            perm.push_back(4 + j + (4 + i) * n);
          perm.push_back(4 + 2 * n + i);
        }
        perm.push_back(2);
        for (int i = 0; i < n; ++i)
          perm.push_back(4 + 3 * n + i);
        perm.push_back(3);
      }
      assert((int)perm.size() == (degree + 1) * (degree + 1));
      break;
    }
    case basix::cell::type::triangle: {
      perm.push_back(0);
      if (degree > 0) {
        int n = degree - 1;
        for (int i = 0; i < n; ++i)
          perm.push_back(3 + 2 * n + i);
        perm.push_back(1);
        int dof = 3 + 3 * n;
        for (int i = 0; i < n; ++i) {
          perm.push_back(3 + n + i);
          for (int j = 0; j < n - 1 - i; ++j)
            perm.push_back(dof++);
          perm.push_back(3 + i);
        }
        perm.push_back(2);
      }

      assert((int)perm.size() == (degree + 1) * (degree + 2) / 2);
      break;
    }
    case basix::cell::type::hexahedron: {
      perm.push_back(0);
      if (degree > 0) {
        int n = degree - 1;
        for (int i = 0; i < n; ++i)
          perm.push_back(8 + i);
        perm.push_back(1);
        for (int i = 0; i < n; ++i) {
          perm.push_back(8 + n + i);
          for (int j = 0; j < n; ++j)
            perm.push_back(8 + 12 * n + n * i + j);
          perm.push_back(8 + 3 * n + i);
        }
        perm.push_back(2);
        for (int i = 0; i < n; ++i)
          perm.push_back(8 + 5 * n + i);
        perm.push_back(3);

        for (int i = 0; i < n; ++i) {
          perm.push_back(8 + 2 * n + i);
          for (int j = 0; j < n; ++j)
            perm.push_back(8 + 12 * n + n * n + n * i + j);
          perm.push_back(8 + 4 * n + i);
          for (int j = 0; j < n; ++j) {
            perm.push_back(8 + 12 * n + 2 * n * n + n * i + j);
            for (int k = 0; k < n; ++k)
              perm.push_back(8 + 12 * n + 6 * n * n + i * n * n + j * n + k);
            perm.push_back(8 + 12 * n + 3 * n * n + n * i + j);
          }
          perm.push_back(8 + 6 * n + i);
          for (int j = 0; j < n; ++j)
            perm.push_back(8 + 12 * n + 4 * n * n + n * i + j);
          perm.push_back(8 + 7 * n + i);
        }
        perm.push_back(4);
        for (int i = 0; i < n; ++i)
          perm.push_back(8 + 8 * n + i);
        perm.push_back(5);
        for (int i = 0; i < n; ++i) {
          perm.push_back(8 + 9 * n + i);
          for (int j = 0; j < n; ++j)
            perm.push_back(8 + 12 * n + 5 * n * n + n * i + j);
          perm.push_back(8 + 10 * n + i);
        }
        perm.push_back(6);
        for (int i = 0; i < n; ++i)
          perm.push_back(8 + 11 * n + i);
        perm.push_back(7);
      }

      assert((int)perm.size() == (degree + 1) * (degree + 1) * (degree + 1));
      break;
    }
    case basix::cell::type::tetrahedron: {
      perm.push_back(0);
      if (degree > 0) {
        int n = degree - 1;
        int face0 = 4 + 6 * n;
        int face1 = 4 + 6 * n + n * (n - 1) / 2;
        int face2 = 4 + 6 * n + n * (n - 1);
        int face3 = 4 + 6 * n + n * (n - 1) * 3 / 2;
        int interior = 4 + 6 * n + n * (n - 2) * 2;
        for (int i = 0; i < n; ++i)
          perm.push_back(4 + 5 * n + i);
        perm.push_back(1);
        for (int i = 0; i < n; ++i) {
          perm.push_back(4 + 4 * n + i);
          for (int j = 0; j < n - 1 - i; ++j)
            perm.push_back(face3++);
          perm.push_back(4 + 2 * n + i);
        }
        perm.push_back(2);
        for (int i = 0; i < n; ++i) {
          perm.push_back(4 + 3 * n + i);
          for (int j = 0; j < n - 1 - i; ++j)
            perm.push_back(face2++);
          perm.push_back(4 + n + i);
          for (int j = 0; j < n - 1 - i; ++j) {
            perm.push_back(face1++);
            for (int k = 0; k < n - 2 - i - j; ++k)
              perm.push_back(interior++);
            perm.push_back(face0++);
          }
          perm.push_back(4 + i);
        }
        perm.push_back(3);
      }

      assert((int)perm.size() ==
             (degree + 1) * (degree + 2) * (degree + 3) / 6);
      break;
    }
    default: {
    }
    }
    break;
  }
  default: {
  }
  }

  if (perm.size() == 0) {
    throw std::runtime_error(
        "Element does not have tensor product factorisation.");
  }
  dof_ordering.resize(perm.size());
  for (std::size_t i = 0; i < perm.size(); ++i)
    dof_ordering[perm[i]] = i;
  return dof_ordering;
}