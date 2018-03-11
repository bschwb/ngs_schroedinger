""" Simulate the time-dependent Schrödinger Equation"""

import ngsolve as ngs
from ngsolve import exp, x, y, grad
from netgen.geom2d import unit_square


mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.005))

fes = ngs.H1(mesh, order=1, dirichlet=[1,2,3,4], complex=True)

u = fes.TrialFunction()
v = fes.TestFunction()

a = ngs.BilinearForm(fes)
a += ngs.SymbolicBFI(1/2 * grad(u) * grad(v))
a.Assemble()

m = ngs.BilinearForm(fes)
m += ngs.SymbolicBFI(1j * u * v)
m.Assemble()

## Initial condition
delta_x = 0.05
x0 = 0.2
y0 = 0.5
kx = 0
ky = 0
wave_packet = ngs.CoefficientFunction(
    exp(1j * (kx * x + ky * y)) * exp(-((x-x0)**2+(y-y0)**2)/4/delta_x**2))

gf_psi = ngs.GridFunction(fes)
gf_psi.Set(wave_packet)
gf_psi.vec.data = 1/ngs.Norm(gf_psi.vec) * gf_psi.vec

freedofs = fes.FreeDofs()
for i in range(len(gf_psi.vec)):
    if not freedofs[i]:
        gf_psi.vec[i] = 0

ngs.Draw(ngs.Norm(gf_psi), mesh, name='abs(psi)')

## Crank-Nicolson time step
max_time = 1
timestep = 0.0001
t = 0

mstar = m.mat.CreateMatrix()
mstar.AsVector().data = timestep / 2 * a.mat.AsVector() - m.mat.AsVector()
inv = mstar.Inverse(freedofs)

w = gf_psi.vec.CreateVector()
du = gf_psi.vec.CreateVector()
while t < max_time:
    t += timestep
    w.data = a.mat * gf_psi.vec
    du.data = inv * w
    gf_psi.vec.data -= timestep * du

    print('t: ', t, ' Norm(psi): ', ngs.Norm(gf_psi.vec))
    ngs.Redraw()
