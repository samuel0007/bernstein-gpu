from basix.ufl import element
from basix import LagrangeVariant
from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    action,
    dx,
    inner,
)
import os

P = int(os.environ.get("polynomial_degree", 4))

coord_element = element("Lagrange", "tetrahedron", 1, shape=(3,))
mesh = Mesh(coord_element)

e = element("Lagrange", "tetrahedron", P, lagrange_variant=LagrangeVariant.bernstein)
V = FunctionSpace(mesh, e)

u = TrialFunction(V)
v = TestFunction(V)

f = Coefficient(V)

a = inner(u, v) * dx
L = inner(f, v) * dx

# Linear form representing the action of the form `a`` on the
# coefficient `ui`:`
ui = Coefficient(V)
M = action(a, ui)

forms = [M, L]
