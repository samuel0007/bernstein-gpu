#pragma once

#include <dolfinx/fem/FiniteElement.h>
#include <algorithm>

using namespace dolfinx;

/// Function that takes in a simplex bernstein element, and returns a reordering dofmap
template <int P, typename T>
std::vector<int> get_tp_ordering2D(std::shared_ptr<const fem::FiniteElement<T>> element_p) {
    const std::vector<std::vector<std::vector<int>>> entity_dofs = element_p->entity_dofs();

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
    for(int row_idx = 0; row_idx < N; ++row_idx) {
        // bottom row
        if(row_idx == 0) {
            dof_reordering.push_back(v0);
            for(int i = 0; i < P - 1; ++i) {
                dof_reordering.push_back(e2[i]);
            }
            dof_reordering.push_back(v1);
            continue;
        }
        // top row
        if(row_idx == N - 1) {
            dof_reordering.push_back(v2);
            continue;
        }

        // inner section
        dof_reordering.push_back(e1[row_idx - 1]);
        for(int j = 0; j < P - row_idx - 1; ++j) {
            dof_reordering.push_back(c[c_count++]);
        }
        dof_reordering.push_back(e0[row_idx - 1]);
    }

    std::reverse(dof_reordering.begin(), dof_reordering.end());   
    return dof_reordering;
}


/// Function that takes in a simplex bernstein element, and returns a reordering dofmap
template <typename T>
std::vector<int> get_tp_ordering1D(std::shared_ptr<basix::FiniteElement<T>> element_p, int p) {
    const std::vector<std::vector<std::vector<int>>> entity_dofs = element_p->entity_dofs();

    if(p == 0) {
        return {0};
    }
    
    int v0 = entity_dofs[0][0][0];
    int v1 = entity_dofs[0][1][0];

    std::vector<int> e0 = entity_dofs[1][0];
    std::vector<int> dof_reordering;
    dof_reordering.push_back(v0);
    for(int i = 0; i < e0.size(); ++i) {
        dof_reordering.push_back(e0[i]);
    }
    dof_reordering.push_back(v1);
    return dof_reordering;
}