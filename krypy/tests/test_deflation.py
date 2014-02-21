import krypy
import krypy.tests.test_utils as test_utils
import numpy
import scipy.linalg
import itertools
from numpy.testing import assert_almost_equal, assert_array_almost_equal, \
    assert_array_equal, assert_equal


def test_Arnoldifyer():
    v = numpy.ones((10, 1))
    for matrix in test_utils.get_matrices():
        evecs, evals = numpy.linalg.eig(matrix)
        sort = numpy.argsort(numpy.abs(evals))
        evecs = evecs[:, sort]
        Us = [numpy.zeros((10, 0)),
              evecs[:, -2:],
              evecs[:, -2:] + 1e-5*numpy.random.rand(10, 2)
              ]
        Wt_sels = ['none', 'smallest', 'largest']
        for A, U, Wt_sel in itertools.product(test_utils.get_operators(matrix),
                                              Us,
                                              Wt_sels):
            yield run_Arnoldifyer, A, v, U, 5, Wt_sel


def run_Arnoldifyer(A, v, U, n, Wt_sel):
    N, d = U.shape
    # orthonormalize if U is not zero-dim
    if d > 0:
        U, _ = numpy.linalg.qr(U)

    # build projection
    AU = A.dot(U)
    P = krypy.utils.Projection(AU, U).operator_complement()

    # run Arnoldi
    V, H = krypy.utils.arnoldi(P*A, P*v, maxiter=n)

    # build matrices used in Arnoldifyer
    B_ = V.T.conj().dot(AU)
    C = U.T.conj().dot(A.dot(V[:, :n]))
    E = U.T.conj().dot(AU)

    VU = numpy.c_[V[:, :n], U]
    M = VU.T.conj().dot(A.dot(VU))
    rvals, rvecs = numpy.linalg.eig(M)
    sort = numpy.argsort(numpy.abs(rvals))
    rvecs = rvecs[sort]
    if Wt_sel == 'none':
        Wt = numpy.zeros((n+d, 0))
    elif Wt_sel == 'smallest':
        Wt = rvecs[:, :2]
    elif Wt_sel == 'largest':
        Wt = rvecs[:, -2:]

    # get Arnoldifyer instance
    arnoldifyer = krypy.deflation.Arnoldifyer(V, U, AU, H, B_, C, E,
                                              numpy.linalg.norm(P*v, 2),
                                              U.T.conj().dot(v))
    # arnoldify given Wt
    Hh, Rh, Vh, F = arnoldifyer.get(Wt, full=True)

    # perform checks
    W = VU.dot(Wt)
    PW = krypy.utils.Projection(A.dot(W), W).operator_complement()
    A = krypy.utils.get_linearoperator((N, N), A)
    At = PW*A
    F = krypy.utils.get_linearoperator((N, N), F)
    assert_almost_equal(numpy.linalg.norm((At+F).dot(Vh) - Vh.dot(Hh), 2)
                        / numpy.linalg.norm(M, 2),
                        0, 12)


if __name__ == '__main__':
    import nose
    nose.main()
