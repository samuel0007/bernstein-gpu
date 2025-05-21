import basix
from basix.ufl import element
from ufl import (Coefficient, FunctionSpace, Mesh, TrialFunction, TestFunction,
                 ds, dx, grad, inner, action)

P = 2  # Degree of polynomial basis
Q = 3  # Quadrature degree

# Define mesh and finite element
coord_element = element("Lagrange", "quadrilateral", 1, shape=(2, ))
mesh = Mesh(coord_element)
e = element(basix.ElementFamily.P, basix.CellType.quadrilateral, P,
    basix.LagrangeVariant.gll_warped)
e_DG = element(basix.ElementFamily.P, basix.CellType.quadrilateral, 0,
    basix.LagrangeVariant.gll_warped, basix.DPCVariant.unset, True)

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


# Map from quadrature points to basix quadrature degree
qdegree = {3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 12, 9: 14, 10: 16}
md = {"quadrature_rule": "GLL", "quadrature_degree": qdegree[Q]}

# Define forms
a = inner(u/rho0/c0/c0, v) * dx(metadata=md)

L = - inner(1/rho0*grad(u_n), grad(v)) * dx(metadata=md) \
    + inner(1/rho0*g, v) * ds(1, metadata=md) \
    - inner(1/rho0/c0*v_n, v) * ds(2, metadata=md)


# # Bilinear form Version (this actually gives the same result as directly creating the linear form as above)
# u_M = TrialFunction(V)
# a_M = inner(u_M/rho0/c0/c0, v) * dx(metadata=md)
# ui = Coefficient(V)
# M = action(a_M, ui)

# Define forms to compute the norm
Norm = inner(u_n, u_n) * dx

# forms = [a, L, M, Norm]
forms = [a, L, Norm]
