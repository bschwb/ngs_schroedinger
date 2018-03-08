""" Simulate the time-independet Schrödingers Equation"""

from scipy.constants import m_e, hbar

import ngsolve as ngs
from netgen.geom2d import unit_square


mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))

fes = ngs.H1(mesh, order=1, dirichlet=[1,2,3,4], complex=True)

u = fes.TrialFunction()
v = fes.TestFunction()

a = ngs.BilinearForm(fes)
a += ngs.SymbolicBFI(hbar / 2 / m_e * grad(u) * grad(v))
a.Assemble()

m = ngs.BilinearForm(fes)
m += ngs.SymbolicBFI(u * v)
m.Assemble()

Draw(mesh)
