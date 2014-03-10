import krypy
import krypy.tests.test_utils as test_utils
import krypy.tests.test_linsys as test_linsys
import numpy
import scipy.linalg
import itertools
from numpy.testing import assert_almost_equal, assert_array_almost_equal, \
    assert_array_equal, assert_equal


def test_deflation_solver():
    for case in test_linsys.cases:
        for ls in test_linsys.linear_systems_generator(**case):
            solvers = [krypy.deflation.DeflatedGmres]
            if ls.self_adjoint:
                solvers.append(krypy.deflation.DeflatedMinres)
            if ls.positive_definite:
                solvers.append(krypy.deflation.DeflatedCg)
            for U in [None, numpy.eye(ls.N, 1)]:
                for solver in solvers:
                    yield test_linsys.run_solver, solver, ls, {
                        'U': U,
                        'x0': None,
                        'tol': 1e-6,
                        'maxiter': 15}


def test_Arnoldifyer():
    vs = [numpy.ones((10, 1)),
          numpy.r_[numpy.ones((3, 1)), numpy.zeros((7, 1))]
          ]
    for matrix in test_utils.get_matrices():
        A_norm = numpy.linalg.norm(matrix, 2)
        numpy.random.seed(0)
        evals, evecs = scipy.linalg.eig(matrix)
        sort = numpy.argsort(numpy.abs(evals))
        evecs = evecs[:, sort]
        Us = [numpy.zeros((10, 0)),
              evecs[:, -2:],
              evecs[:, -2:] + 1e-5*numpy.random.rand(10, 2)
              ]
        Wt_sels = ['none', 'smallest', 'largest']
        for A, v, U in \
                itertools.product(test_utils.get_operators(matrix),
                                  vs, Us,):
            ls = krypy.linsys.LinearSystem(A, v)
            for Wt_sel in Wt_sels:
                yield run_Arnoldifyer, ls, U, A_norm, Wt_sel


def run_Arnoldifyer(ls, U, A_norm, Wt_sel):
    try:
        deflated_solver = krypy.deflation.DeflatedGmres(ls, U=U,
                                                        store_arnoldi=True,
                                                        maxiter=5)
    except krypy.utils.ConvergenceError as e:
        deflated_solver = e.solver
    ritz = krypy.deflation.Ritz(deflated_solver)
    sort = numpy.argsort(numpy.abs(ritz.values))
    coeffs = ritz.coeffs[:, sort]
    if Wt_sel == 'none':
        Wt = numpy.zeros((coeffs.shape[0], 0))
    elif Wt_sel == 'smallest':
        Wt = coeffs[:, :2]
    elif Wt_sel == 'largest':
        Wt = coeffs[:, -2:]

    k = Wt.shape[1]
    if k > 0:
        Wt, _ = scipy.linalg.qr(Wt, mode='economic')

    # get Arnoldifyer instance
    arnoldifyer = krypy.deflation.Arnoldifyer(deflated_solver)

    # arnoldify given Wt
    Hh, Rh, q_norm, vdiff_norm, PWAW_norm, Vh, F = \
        arnoldifyer.get(Wt, full=True)

    # perform checks
    (n_, n) = deflated_solver.H.shape
    ls = deflated_solver.linear_system
    N = ls.N
    d = deflated_solver.P.U.shape[1]

    VU = numpy.c_[deflated_solver.V[:, :n], deflated_solver.P.U]
    W = VU.dot(Wt)
    PW = krypy.utils.Projection(ls.MlAMr*W, W).operator_complement()
    At = PW*ls.MlAMr
    Fop = krypy.utils.get_linearoperator((N, N), F)

    # check arnoldi relation
    assert_almost_equal(numpy.linalg.norm((At+Fop).dot(Vh) - Vh.dot(Hh), 2)
                        / A_norm,
                        0, 7)

    # check projection
    assert_almost_equal(numpy.linalg.norm(Vh.T.conj().dot((At+Fop).dot(Vh))
                                          - Hh, 2)
                        / A_norm,
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

    # check PWAW_norm
    assert_almost_equal(PWAW_norm, numpy.linalg.norm(PW*numpy.eye(N), 2))


if __name__ == '__main__':
    import nose
    nose.main()
