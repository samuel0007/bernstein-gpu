import basix
from basix.ufl import element
from ufl import (Coefficient, FunctionSpace, Mesh, TrialFunction, TestFunction,
                 ds, dx, grad, inner, action)
import os

P = int(os.environ.get("polynomial_degree", 4))

# Define mesh and finite element
coord_element = element("Lagrange", "tetrahedron", 1, shape=(3, ))
mesh = Mesh(coord_element)
e = element(basix.ElementFamily.P, basix.CellType.tetrahedron, P,
    basix.LagrangeVariant.equispaced)
e_DG = element(basix.ElementFamily.P, basix.CellType.tetrahedron, 0,
    basix.LagrangeVariant.equispaced, basix.DPCVariant.unset, True)

# Define function spaces
V = FunctionSpace(mesh, e)
V_DG = FunctionSpace(mesh, e_DG)

c0 = Coefficient(V_DG)
rho0 = Coefficient(V_DG)

u_n = Coefficient(V)
v = TestFunction(V)

# Define forms
L = - inner(1/rho0*grad(u_n), grad(v)) * dx \
    + inner(1/rho0*u_n, v) * ds(1) \
    - inner(1/rho0/c0*u_n, v) * ds(2)
# L = - inner(1/rho0*grad(u_n), grad(v)) * dx
    
forms = [L]
