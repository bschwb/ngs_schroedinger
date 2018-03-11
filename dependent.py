""" Simulate the time-dependent Schrödinger Equation"""

from scipy.constants import m_e, hbar

import ngsolve as ngs
from ngsolve import exp, x, y, grad
from netgen.geom2d import unit_square

from random import random


mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.05))

fes = ngs.H1(mesh, order=2, dirichlet=[1,2,3,4], complex=True)

u = fes.TrialFunction()
v = fes.TestFunction()

a = ngs.BilinearForm(fes)
a += ngs.SymbolicBFI(hbar / 2 / m_e * grad(u) * grad(v))
a.Assemble()

m = ngs.BilinearForm(fes)
m += ngs.SymbolicBFI(1j * hbar * u * v)
m.Assemble()

delta_x = 0.1
kx_0 = 2
ky_0 = 0
wave_packet = ngs.CoefficientFunction(exp(1j * (kx_0 * x + ky_0 * y)) *
                                      exp(-((x-0.5)**2+(y-0.5)**2)/2/delta_x**2))

gf_psi = ngs.GridFunction(fes)
gf_psi.Set(wave_packet)
gf_psi.vec.data = 1/ngs.Norm(gf_psi.vec) * gf_psi.vec

freedofs = fes.FreeDofs()
for i in range(len(gf_psi.vec)):
    if not freedofs[i]:
        gf_psi.vec[i] = 0

ngs.Draw(gf_psi)
