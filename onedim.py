from math import pi
from time import sleep

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import ngsolve as ngs
import netgen.meshing as ngm
from ngsolve import x, grad, exp
from ngsolve.internal import viewoptions, visoptions
viewoptions.drawedges='1'
visoptions.scaledeform1='100'

## 1D Mesh
m = ngm.Mesh()
m.dim = 1

n_elems = 1000
x_min = -50
x_max = 50
length = x_max - x_min
pnums = []
xs = []
for i in range(0, n_elems+1):
    pnt_x = x_min + length * i / n_elems
    xs.append(pnt_x)
    pnums.append(m.Add(ngm.MeshPoint(ngm.Pnt(pnt_x, 0, 0))))

for i in range(0, n_elems):
    m.Add(ngm.Element1D([pnums[i], pnums[i+1]], index=1))

m.SetMaterial(1, 'material')

m.Add(ngm.Element0D(pnums[0], index=1))
m.Add(ngm.Element0D(pnums[n_elems], index=2))

## NGSolve
mesh = ngs.Mesh(m)
fes = ngs.H1(mesh, order=1, dirichlet=[1, 2], complex=True)

u = fes.TrialFunction()
v = fes.TestFunction()

## Potentials
### Potential barrier
barrier_w = 2
barrier_h = 1
potential = ngs.CoefficientFunction(ngs.IfPos(x, barrier_h, 0) - ngs.IfPos(x-barrier_w, barrier_h, 0))

### Square potential
# potential = ngs.CoefficientFunction(1/2*x*x-10)

### Zero potential
# potential = ngs.CoefficientFunction(0)

gf_potential = ngs.GridFunction(fes)
gf_potential.Set(potential)

a = ngs.BilinearForm(fes)
a += ngs.SymbolicBFI(1/2 * grad(u) * grad(v) + potential * u * v)
a.Assemble()

m = ngs.BilinearForm(fes)
m += ngs.SymbolicBFI(1j * u * v)
m.Assemble()

## Initial condition
### Gaussian wave packet
delta_x = 2
x0 = -20
kx = 2
wave_packet = ngs.CoefficientFunction(
    exp(1j * (kx * x)) * exp(-((x-x0)**2)/4/delta_x**2))

### Heaviside function
# wave_packet = ngs.CoefficientFunction(IfPos(x, 1, 0))

gf_psi = ngs.GridFunction(fes)
gf_psi.Set(wave_packet)
gf_psi.vec.data = 1/ngs.Norm(gf_psi.vec) * gf_psi.vec

freedofs = fes.FreeDofs()
for i in range(len(gf_psi.vec)):
    if not freedofs[i]:
        gf_psi.vec[i] = 0

# ngs.Draw(ngs.Norm(gf_psi), mesh, name='abs(psi)')
# ngs.Draw(gf_psi.real, mesh, name='psi.real')
# ngs.Draw(gf_psi.imag, mesh, name='psi.imag')

## Crank-Nicolson time step
max_time = 100
timestep = 0.1
t = 0

mstar = m.mat.CreateMatrix()
mstar.AsVector().data = timestep / 2 * a.mat.AsVector() - m.mat.AsVector()
inv = mstar.Inverse(freedofs)

w = gf_psi.vec.CreateVector()
du = gf_psi.vec.CreateVector()

fig = plt.figure()
ims = []
while t < max_time:
    t += timestep
    w.data = a.mat * gf_psi.vec
    du.data = inv * w
    gf_psi.vec.data -= timestep * du

    print('t: ', t, ' Norm(psi): ', ngs.Norm(gf_psi.vec))
    ims.append(plt.plot(xs, abs(gf_psi.vec.FV().NumPy()), 'g', xs, gf_potential.vec.FV().NumPy(), 'black'))
    # ngs.Redraw()

im_ani = animation.ArtistAnimation(fig, ims, interval=0, blit=True)
plt.show()
