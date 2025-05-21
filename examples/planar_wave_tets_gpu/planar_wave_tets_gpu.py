import basix
from basix.ufl import element
from ufl import (Coefficient, FunctionSpace, Mesh, TrialFunction, TestFunction,
                 ds, dx, grad, inner, action)

P = 2  # Degree of polynomial basis

# Define mesh and finite element
coord_element = element("Lagrange", "tetrahedron", 1, shape=(3, ))
mesh = Mesh(coord_element)
e = element(basix.ElementFamily.P, basix.CellType.tetrahedron, P,
    basix.LagrangeVariant.bernstein)
e_DG = element(basix.ElementFamily.P, basix.CellType.tetrahedron, 0,
    basix.LagrangeVariant.bernstein, basix.DPCVariant.unset, True)

# Define function spaces
V = FunctionSpace(mesh, e)
V_DG = FunctionSpace(mesh, e_DG)

c0 = Coefficient(V_DG)
rho0 = Coefficient(V_DG)

u = Coefficient(V)
u_n = Coefficient(V)
v_n = Coefficient(V)
g = Coefficient(V)
v = TestFunction(V)


# Define forms
a = inner(u/rho0/c0/c0, v) * dx

L = - inner(1/rho0*grad(u_n), grad(v)) * dx \
    + inner(1/rho0*g, v) * ds(1) \
    - inner(1/rho0/c0*v_n, v) * ds(2)

forms = [a, L]
