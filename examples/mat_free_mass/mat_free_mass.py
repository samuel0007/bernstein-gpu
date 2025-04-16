# UFL input for the Matrix-free Poisson Demo

from basix.ufl import element
from basix import LagrangeVariant
from ufl import (
    Coefficient,
    Constant,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    action,
    dx,
    inner,
)

coord_element = element("Lagrange", "triangle", 1, lagrange_variant=LagrangeVariant.bernstein, shape=(2,))
mesh = Mesh(coord_element)

# Function Space
e = element("Lagrange", "triangle", 2, lagrange_variant=LagrangeVariant.bernstein)
# e = element("Lagrange", "triangle", 1)
V = FunctionSpace(mesh, e)

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Constant RHS
f = Constant(V)

# Bilinear and linear forms according to the variational
# formulation of the equations:
a = inner(u, v) * dx
L = inner(f, v) * dx

# Linear form representing the action of the form `a`` on the
# coefficient `ui`:`
ui = Coefficient(V)
M = action(a, ui)

# Form to compute the L2 norm of the error
usol = Coefficient(V)
uexact = Coefficient(V)
E = inner(usol - uexact, usol - uexact) * dx

forms = [M, L, E]
