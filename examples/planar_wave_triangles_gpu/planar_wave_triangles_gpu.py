import basix
from basix.ufl import element
from ufl import (Coefficient, FunctionSpace, Mesh, TrialFunction, TestFunction,
                 ds, dx, grad, inner, action)

P = 4  # Degree of polynomial basis

# Define mesh and finite element
coord_element = element("Lagrange", "triangle", 1, shape=(2, ))
mesh = Mesh(coord_element)
e = element(basix.ElementFamily.P, basix.CellType.triangle, P,
    basix.LagrangeVariant.bernstein)
e_DG = element(basix.ElementFamily.P, basix.CellType.triangle, 0,
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


# Bilinear form Version (this actually gives the same result as directly creating the linear form as above)
u_M = TrialFunction(V)
a_M = inner(u_M/rho0/c0/c0, v) * dx
# ui = Coefficient(V)
# M = action(a_M, ui)

# Define forms to compute the norm
Norm = inner(u_n, u_n) * dx

# forms = [a, L, M, Norm]
forms = [a, a_M, L, Norm]
