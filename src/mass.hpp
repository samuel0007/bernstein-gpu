#pragma once

#include "vector.hpp"

namespace dolfinx::acc {


template <typename T>
class MatFreeMass {
public:
    using value_type = T;

    MatFreeMass(int degree, int nq): degree(degree), nq(nq) {

    }

    template <typename Vector>
    void operator()(const Vector& in, Vector& out)
    {
        // TODO
    }
    
private:
    int degree;
    
    // Number of quadrature points in 1D
    int nq;
    
    // Reference to on-device storage for constants, dofmap etc.
    std::span<const T> cell_constants;
    std::span<const std::int32_t> cell_dofmap;
};






} // namespace dolfinx::acc