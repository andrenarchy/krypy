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
            for U, store_arnoldi in itertools.product(
                    [None, numpy.eye(ls.N, 1),
                     numpy.eye(ls.N, 1) + 1e-3*numpy.ones((ls.N, 1))],
                    [False, True]):
                for solver in solvers:
                    params = {
                        'U': U,
                        'x0': None,
                        'tol': 1e-6,
                        'maxiter': 15,
                        'store_arnoldi': True}

                    #if solver is krypy.deflation.DeflatedGmres:
                    #    params['ortho'] = 'dmgs'

                    yield run_deflation_solver, solver, ls, params


def run_deflation_solver(Solver, ls, params):
    sol = Solver(ls, **params)
    test_linsys.check_solver(sol, Solver, ls, params)

    if params['store_arnoldi']:
        (n_, n) = sol.H.shape

        # check E
        assert_array_almost_equal(
            sol.E,
            krypy.utils.inner(sol.projection.U, ls.MlAMr*sol.projection.U,
                              ip_B=ls.ip_B))

        # check C
        assert_array_almost_equal(
            sol.C,
            krypy.utils.inner(sol.projection.U, ls.MlAMr*sol.V[:, :n],
                              ip_B=ls.ip_B))

        # check B_
        assert_array_almost_equal(
            sol.B_,
            krypy.utils.inner(sol.V, sol.projection.AU,
                              ip_B=ls.ip_B))

        # check Ritz pairs
        check_Ritz(sol, ls)


def check_Ritz(solver, ls):
    (n_, n) = solver.H.shape
    m = solver.projection.U.shape[1]

    # compute Ritz pairs 'by hand'
    Z = numpy.c_[solver.V[:, :n], solver.projection.U]
    MMlAMrZ = ls.M * (ls.MlAMr * Z)

    inner_left = krypy.utils.inner(Z, MMlAMrZ, ip_B=ls.get_ip_Minv_B())
    inner_right = krypy.utils.inner(Z, Z, ip_B=ls.get_ip_Minv_B())

    if ls.self_adjoint:
        assert_array_almost_equal(inner_left, inner_left.T.conj())

    # inner_right should be identity if full orthogonalization is performed
    if isinstance(solver, krypy.linsys.Gmres) and 0 < n+m <= ls.N:
        assert_array_almost_equal(inner_right, numpy.eye(n+m), decimal=4)

    if 0 < n+m <= ls.N:
        if numpy.linalg.norm(inner_right - numpy.eye(n+m), 2) < 1e-8:
            # use eigh for self-adjoint problems
            eig = scipy.linalg.eigh if ls.self_adjoint else scipy.linalg.eig
            eig = scipy.linalg.eig

            cmp_values, cmp_coeffs = eig(inner_left, inner_right)
            cmp_sort = numpy.argsort(numpy.abs(cmp_values))
            cmp_values = cmp_values[cmp_sort]
            cmp_coeffs = cmp_coeffs[:, cmp_sort]
            # normalize coeffs
            for i in range(n+m):
                cmp_coeffs[:, [i]] /= numpy.linalg.norm(cmp_coeffs[:, [i]], 2)
            cmp_vectors = Z.dot(cmp_coeffs)
            cmp_res = ls.M*(ls.MlAMr*cmp_vectors) - cmp_vectors*cmp_values
            cmp_resnorms = [krypy.utils.norm(cmp_res[:, [i]],
                                             ip_B=ls.get_ip_Minv_B())
                            for i in range(n+m)]

            # get Ritz pairs via Ritz()
            ritz = krypy.deflation.Ritz(solver, mode='ritz')
            sort = numpy.argsort(numpy.abs(ritz.values))

            # check values
            assert_array_almost_equal(ritz.values[sort], cmp_values)

            # check vectors via inner products
            assert_array_almost_equal(numpy.diag(numpy.abs(
                krypy.utils.inner(ritz.get_vectors()[:, sort], cmp_vectors,
                                  ip_B=ls.get_ip_Minv_B()))),
                numpy.ones(m+n))

            # check resnorms
            #assert_array_almost_equal(ritz.resnorms[sort],
            #                          cmp_resnorms, decimal=4)


def test_Arnoldifyer():
    vs = [numpy.ones((10, 1)),
          numpy.r_[numpy.ones((3, 1)), numpy.zeros((7, 1))]
          ]
    for matrix in test_utils.get_matrices():
        A_norm = numpy.linalg.norm(matrix, 2)
        numpy.random.seed(0)
        Ms = [None, numpy.diag(range(1, 11))]
        Wt_sels = ['none', 'smallest', 'largest']
        for A, v, M, Wt_sel in \
                itertools.product(test_utils.get_operators(matrix),
                                  vs, Ms, Wt_sels):
            if M is None:
                Minv = None
            else:
                Minv = numpy.linalg.inv(M)
            ls = krypy.linsys.LinearSystem(A, v, M=M, Minv=Minv)

            evals, evecs = scipy.linalg.eig(ls.M*matrix)
            sort = numpy.argsort(numpy.abs(evals))
            evecs = evecs[:, sort]
            Us = [numpy.zeros((10, 0)),
                  evecs[:, -2:],
                  evecs[:, -2:] + 1e-2*numpy.random.rand(10, 2)
                  ]
            for U in Us:
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

    ip_Minv_B = ls.get_ip_Minv_B()

    # check orthonormality of computed residual basis (may fail if U is
    # close to an invariant subspace of MA due to round-off in rank-revealing
    # QR
    Z = arnoldifyer.Z
    assert_array_almost_equal(krypy.utils.inner(Z, Z, ip_B=ip_Minv_B),
                              numpy.eye(arnoldifyer.Z.shape[1]),
                              7)

    # arnoldify given Wt
    Hh, Rh, q_norm, vdiff_norm, PWAW_norm, Vh, F = \
        arnoldifyer.get(Wt, full=True)

    # perform checks
    (n_, n) = deflated_solver.H.shape
    N = ls.N
    d = deflated_solver.projection.U.shape[1]

    VU = numpy.c_[deflated_solver.V[:, :n], deflated_solver.projection.U]
    W = VU.dot(Wt)
    PW = krypy.utils.Projection(ls.MlAMr*W, W).operator_complement()
    At = ls.M*(PW*ls.MlAMr)

    # check arnoldi relation
    assert_almost_equal(numpy.linalg.norm((At+F)*Vh - Vh.dot(Hh), 2)
                        / A_norm,
                        0, 7)

    ## check projection
    assert_array_almost_equal(krypy.utils.inner(Vh, (At+F)*Vh, ip_B=ip_Minv_B),
                              Hh,
                              7)

    # check orthonormality of returned Vh
    assert_array_almost_equal(krypy.utils.inner(Vh, Vh, ip_B=ip_Minv_B),
                              numpy.eye(n+d-k),
                              7)

    # check norm of perturbation
    if Rh.size > 0:
        #TODO: reenable once the numpy installation on travis is up-to-date!
        #assert_almost_equal(numpy.linalg.norm(Rh, 2), numpy.linalg.norm(F, 2),
        #                    8)
        pass

    # check norm of vdiff
    #PWb = PW*ls.b
    #assert_array_almost_equal(krypy.utils.norm(ls.M*PW*,
    #                                           ip_B=ls.get_ip_Minv_B()),
    #                          vdiff_norm)

    # get orthonormal basis
    Q, _ = krypy.utils.qr(numpy.eye(N), ip_B=ls.get_ip_Minv_B())

    def _get_op_norm(op):
        return krypy.utils.norm(op*Q, ip_B=ls.get_ip_Minv_B())

    # check PWAW_norm
    assert_almost_equal(PWAW_norm,
                        _get_op_norm(ls.M*PW*ls.Minv)
                        )



if __name__ == '__main__':
    import nose
    nose.main()
