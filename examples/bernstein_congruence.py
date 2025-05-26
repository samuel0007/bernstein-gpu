"""Test congruence of Bernstein and other Lagrange galerkin matrix for a poisson problem"""

import dolfinx
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, mesh 
from ufl import dx, grad, inner
from scipy.sparse.linalg import eigsh
import basix

bases = [basix.LagrangeVariant.gll_warped, basix.LagrangeVariant.bernstein]
degree = 4
c = 1500
dt = 0.01
print(f"Asserting congruence of Bernstein and other Lagrange elements, degree {degree}.")
for base in bases:
    print(f"Testing with base: {base}")
    element = basix.ufl.element(
        basix.ElementFamily.P,
        basix.CellType.triangle,
        degree,
        base,
    )

    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (2.0, 1.0)),
        n=(10, 10),
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(msh, element)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # this would arise from an implicit time-stepping scheme (like cranck-nicolson)
    # a = inner(u, v) * dx + (dt**2/4)*c**2*inner(grad(u), grad(v)) * dx
    a = inner(u, v) * dx
    # a = inner(grad(u), grad(v)) * dx
    

    a_form = dolfinx.fem.form(a)
    A = dolfinx.fem.assemble_matrix(a_form)
    A_sparse = A.to_scipy()

    k = A_sparse.shape[0] - 1
    vals, vecs = eigsh(A_sparse, k=k) 

    tol = 1e-10
    rank = np.sum(np.abs(vals) > tol)
    n_pos = np.sum(vals > tol)
    n_neg = np.sum(vals < -tol)
    n_zero = np.sum(np.abs(vals) <= tol)

    print(f"Rank: {rank}")
    print(f"Signature: ({n_pos} positive, {n_neg} negative, {n_zero} zero)")
    # Spectrum range and condition number
    min_val = np.min(vals)
    max_val = np.max(vals)
    print(f"Min eigenvalue: {min_val}")
    print(f"Max eigenvalue: {max_val}")
    print(f"Condition number: {np.abs(max_val/min_val)}")