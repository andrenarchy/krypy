import krypy
import numpy
import pytest


@pytest.mark.parametrize(
    "method, ref",
    [
        (krypy.cg, [1004.1873775173957, 1000.0003174916551, 999.9999999997555]),
        (krypy.minres, [1004.187372488912, 1000.0003124632159, 999.9999949713145]),
        (krypy.gmres, [1004.1873724888546, 1000.0003124630923, 999.999994971191]),
    ],
)
def test_matrix(method, ref):
    A = numpy.diag([1.0e-3] + list(range(2, 101)))
    b = numpy.ones(100)

    out = method(A, b)
    print(out)

    assert abs(numpy.sum(numpy.abs(out.xk)) - ref[0]) < 1.0e-12 * ref[0]
    assert abs(numpy.sqrt(numpy.dot(out.xk, out.xk)) - ref[1]) < 1.0e-12 * ref[1]
    assert abs(numpy.max(numpy.abs(out.xk)) - ref[2]) < 1.0e-12 * ref[2]


@pytest.mark.parametrize(
    "method, ref",
    [
        (krypy.cg, [1004.1873775173271, 1000.0003174918709, 1000.0]),
        (krypy.minres, [1004.1873774950692, 1000.0003174918709, 1000.0]),
        (krypy.gmres, [1004.1873774950692, 1000.0003174918709, 1000.0]),
    ],
)
def test_gmres(method, ref):
    n = 100
    A = numpy.diag([1.0e-3] + list(range(2, n + 1)))
    b = numpy.ones(n)

    # deflate out the vector that belongs to the small eigenvalue
    U = numpy.zeros(n)
    U[0] = 1.0
    out = method(A, b, U=U)

    assert abs(numpy.sum(numpy.abs(out.xk)) - ref[0]) < 1.0e-12 * ref[0]
    assert abs(numpy.sqrt(numpy.dot(out.xk, out.xk)) - ref[1]) < 1.0e-12 * ref[1]
    assert abs(numpy.max(numpy.abs(out.xk)) - ref[2]) < 1.0e-12 * ref[2]


def test_custom_inner_product():
    n = 100
    A = numpy.diag([1.0e-3] + list(range(2, n + 1)))
    b = numpy.ones(n)

    def inner(a, b):
        return numpy.dot(a, b)

    out = krypy.cg(A, b, inner_product=inner)

    ref = 1004.1873775173957
    assert abs(numpy.sum(numpy.abs(out.xk)) - ref) < 1.0e-12 * ref
    ref = 1000.0003174916551
    assert abs(numpy.sqrt(numpy.dot(out.xk, out.xk)) - ref) < 1.0e-12 * ref
    ref = 999.9999999997555
    assert abs(numpy.max(numpy.abs(out.xk)) - ref) < 1.0e-12 * ref
