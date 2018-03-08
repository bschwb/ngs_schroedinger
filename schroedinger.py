""" Simulate the time-independet Schrödingers Equation"""

from scipy.constants import m_e, hbar

import ngsolve as ngs
from netgen.geom2d import unit_square

from random import random


mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.05))

fes = ngs.H1(mesh, order=1, dirichlet=[1,2,3,4], complex=True)

u = fes.TrialFunction()
v = fes.TestFunction()

a = ngs.BilinearForm(fes)
a += ngs.SymbolicBFI(hbar / 2 / m_e * grad(u) * grad(v))
a.Assemble()

m = ngs.BilinearForm(fes)
m += ngs.SymbolicBFI(u * v)
m.Assemble()

gf_psi = ngs.GridFunction(fes)

freedofs = fes.FreeDofs()
for i in range(len(gf_psi.vec)):
    gf_psi.vec[i] = random() if freedofs[i] else 0

Draw(gf_psi)
