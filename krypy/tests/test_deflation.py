import krypy
import krypy.tests.test_utils as test_utils
import numpy
import scipy.linalg
import itertools
from numpy.testing import assert_almost_equal, assert_array_almost_equal, \
    assert_array_equal, assert_equal


def test_Arnoldifyer():
    vs = [numpy.ones((10, 1)),
          numpy.r_[numpy.ones((3, 1)), numpy.zeros((7, 1))]
          ]
    for matrix in test_utils.get_matrices():
        numpy.random.seed(0)
        evals, evecs = scipy.linalg.eig(matrix)
        sort = numpy.argsort(numpy.abs(evals))
        evecs = evecs[:, sort]
        Us = [numpy.zeros((10, 0)),
              evecs[:, -2:],
              evecs[:, -2:] + 1e-5*numpy.random.rand(10, 2)
              ]
        Wt_sels = ['none', 'smallest', 'largest']
        for A, v, U, Wt_sel in \
                itertools.product(test_utils.get_operators(matrix),
                                  vs, Us, Wt_sels):
            yield run_Arnoldifyer, A, v, U, 5, Wt_sel


def run_Arnoldifyer(A, v, U, maxiter, Wt_sel):
    N, d = U.shape
    # orthonormalize if U is not zero-dim
    if d > 0:
        U, _ = numpy.linalg.qr(U)

    # build projection
    AU = A.dot(U)
    P = krypy.utils.Projection(AU, U).operator_complement()

    # run Arnoldi
    V, H = krypy.utils.arnoldi(P*krypy.utils.get_linearoperator((N, N), A),
                               P*v, maxiter=maxiter, ortho='house')
    n = H.shape[1]

    # build matrices used in Arnoldifyer
    B_ = V.T.conj().dot(AU)
    C = U.T.conj().dot(A.dot(V[:, :n]))
    E = U.T.conj().dot(AU)

    VU = numpy.c_[V[:, :n], U]
    M = VU.T.conj().dot(A.dot(VU))
    rvals, rvecs = scipy.linalg.eig(M)
    sort = numpy.argsort(numpy.abs(rvals))
    rvecs = rvecs[sort]
    if Wt_sel == 'none':
        Wt = numpy.zeros((n+d, 0))
    elif Wt_sel == 'smallest':
        Wt = rvecs[:, :2]
    elif Wt_sel == 'largest':
        Wt = rvecs[:, -2:]

    k = Wt.shape[1]

    # get Arnoldifyer instance
    arnoldifyer = krypy.deflation.Arnoldifyer(V, U, AU, H, B_, C, E,
                                              numpy.linalg.norm(P*v, 2),
                                              U.T.conj().dot(v))
    # arnoldify given Wt
    Hh, Rh, vdiff_norm, Vh, F = arnoldifyer.get(Wt, full=True)

    # perform checks
    W = VU.dot(Wt)
    PW = krypy.utils.Projection(A.dot(W), W).operator_complement()
    A = krypy.utils.get_linearoperator((N, N), A)
    At = PW*A
    Fop = krypy.utils.get_linearoperator((N, N), F)

    # check arnoldi relation
    assert_almost_equal(numpy.linalg.norm((At+Fop).dot(Vh) - Vh.dot(Hh), 2)
                        / numpy.linalg.norm(M, 2),
                        0, 7)

    # check projection
    assert_almost_equal(numpy.linalg.norm(Vh.T.conj().dot((At+Fop).dot(Vh))
                                          - Hh, 2)
                        / numpy.linalg.norm(M, 2),
                        0, 7)

    # check orthonormality
    assert_almost_equal(numpy.linalg.norm(Vh.T.conj().dot(Vh)
                        - numpy.eye(n+d-k), 2),
                        0, 7)

    # check norm of perturbation
    if Rh.size > 0:
        #TODO: reenable once the numpy installation on travis is up-to-date!
        #assert_almost_equal(numpy.linalg.norm(Rh, 2), numpy.linalg.norm(F, 2),
        #                    8)
        pass



if __name__ == '__main__':
    import nose
    nose.main()