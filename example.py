import numpy

# use tex
# plot residuals
from matplotlib import pyplot, rc

from krypy.linsys import Gmres, LinearSystem

# use beautiful style
from mpltools import style

linear_system = LinearSystem(
    A=numpy.diag([1e-3] + range(2, 101)), b=numpy.ones((100, 1))
)
sol = Gmres(linear_system)


rc("text", usetex=True)

style.use("ggplot")

pyplot.figure(figsize=(6, 4), dpi=100)
pyplot.xlabel("Iteration $i$")
pyplot.ylabel(r"Relative residual norm $\frac{\|r_i\|}{\|b\|}$")
pyplot.semilogy(sol.resnorms)

pyplot.savefig("example.png", bbox_inches="tight")
