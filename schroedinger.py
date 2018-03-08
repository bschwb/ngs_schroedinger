""" Simulate the time-independet Schrödingers Equation"""

import ngsolve as ngs
from netgen.geom2d import unit_square

mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))

Draw(mesh)
