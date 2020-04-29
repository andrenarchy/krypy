import numpy
import pytest

import krypy


@pytest.mark.parametrize(
    "method, ref",
    [
        (krypy.cg, [1004.1873775173957, 1000.0003174916551, 999.9999999997555]),
        (krypy.gmres, [1004.1873724888546, 1000.0003124630923, 999.999994971191]),
        (krypy.minres, [1004.187372488912, 1000.0003124632159, 999.9999949713145]),
    ],
)
def test_cg_matrix(method, ref):
    tol = 1.0e-11
    A = numpy.diag([1.0e-3] + list(range(2, 101)))

    # Make sure the shapes are alright
    b = numpy.ones((100, 1))
    sol, _ = method(A, b, inner_product=numpy.dot)
    assert sol.shape == b.shape

    b = numpy.ones(100)
    sol, _ = method(A, b, inner_product=numpy.dot)
    assert sol.shape == b.shape

    sol, _ = method(A, b, inner_product=numpy.dot)
    assert abs(numpy.sum(numpy.abs(sol)) - ref[0]) < tol * ref[0]
    assert abs(numpy.sqrt(numpy.dot(sol, sol)) - ref[1]) < tol * ref[1]
    assert abs(numpy.max(numpy.abs(sol)) - ref[2]) < tol * ref[2]


@pytest.mark.parametrize(
    "method, ref",
    [
        (krypy.cg, [1004.1873775173271, 1000.0003174918709, 1000.0]),
        (krypy.minres, [1004.1873774950692, 1000.0003174918709, 1000.0]),
        (krypy.gmres, [1004.1873774950692, 1000.0003174918709, 1000.0]),
    ],
)
def test_deflate_cg(method, ref):
    tol = 1.0e-11
    n = 100
    A = numpy.diag([1.0e-3] + list(range(2, n + 1)))
    b = numpy.ones(n)

    # deflate out the vector that belongs to the small eigenvalue
    U = numpy.zeros(n)
    U[0] = 1.0
    sol, _ = method(A, b, U=U)

    assert abs(numpy.sum(numpy.abs(sol)) - ref[0]) < tol * ref[0]
    assert abs(numpy.sqrt(numpy.dot(sol, sol)) - ref[1]) < tol * ref[1]
    assert abs(numpy.max(numpy.abs(sol)) - ref[2]) < tol * ref[2]


def test_custom_inner_product():
    tol = 1.0e-11
    n = 100
    A = numpy.diag([1.0e-3] + list(range(2, n + 1)))
    b = numpy.ones(n)

    def inner(a, b):
        return numpy.dot(a, b)

    sol, _ = krypy.cg(A, b, inner_product=inner)

    ref = 1004.1873775173957
    assert abs(numpy.sum(numpy.abs(sol)) - ref) < tol * ref
    ref = 1000.0003174916551
    assert abs(numpy.sqrt(numpy.dot(sol, sol)) - ref) < tol * ref
    ref = 999.9999999997555
    assert abs(numpy.max(numpy.abs(sol)) - ref) < tol * ref
