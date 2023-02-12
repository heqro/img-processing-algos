from sympy import exp
from sympy.abc import x, y
from sympy.plotting import plot3d


def synth_f():
    return exp(-(x - .5) ** 2 - (y - .5) ** 2)


plot3d(synth_f(), (x, 0, 1), (y, 0, 1))
