import krypy
import krypy.tests.test_utils as test_utils
import numpy
import itertools
from numpy.testing import assert_almost_equal, assert_array_almost_equal, \
    assert_array_equal, assert_equal


def dictproduct(d):
    '''enhance itertools product to process values of dicts

    example:
        d = {'a':[1,2],'b':[3,4]}
        then list(dictproduct(d)) ==
        [{'a':1,'b':3}, {'a':1,'b':4}, {'a':2,'b':3}, {'a':2,'b':4}]
    '''
    for p in itertools.product(*d.values()):
        yield dict(zip(d.keys(), p))


def dictpick(d):
    yield {}
    for k in d:
        for v in d[k]:
            yield {k: v}


def test_LinearSystem():
    A = numpy.diag(range(1, 11))
    exact_solution = numpy.ones((10, 1))
    b = A.dot(exact_solution)
    ls = krypy.linsys.LinearSystem(A, b,
                                   M=numpy.eye(10),
                                   Ml=numpy.eye(10),
                                   Mr=numpy.eye(10))
    # check that r=b for z=0
    Mr, r, rnorm = ls.get_residual(numpy.zeros((10, 1)), compute_norm=True)
    assert_almost_equal(r, b)
    assert_almost_equal(r, Mr)
    assert_almost_equal(rnorm, numpy.linalg.norm(b, 2))

    # check that r=0 for exact solution
    Mr, r, rnorm = ls.get_residual(exact_solution, compute_norm=True)
    assert_almost_equal(r, numpy.zeros((10, 1)))
    assert_almost_equal(r, Mr)
    assert_almost_equal(rnorm, 0)


def linear_systems_generator(A, **ls_kwargs):
    ip_Bs = [None, numpy.diag(range(1, 11))]
    xs = [numpy.ones((10, 1)), numpy.ones((10,)), (1+1j)*numpy.ones((10, 1)),
          numpy.zeros((10, 1))]
    for ip_B, x in itertools.product(ip_Bs, xs):
        if ip_B is not None and 'self_adjoint' in ls_kwargs and \
                ls_kwargs['self_adjoint']:
            # make A self-adjoint again if the inner product is changed
            A_new = numpy.linalg.inv(ip_B).dot(A)
        else:
            A_new = A

        preconditioners = {
            'M': [None],
            'Ml': [None, numpy.linalg.inv(A_new)],
            'Mr': [None, numpy.linalg.inv(A_new)]
            }

        if 'positive_definite' in ls_kwargs and ls_kwargs['positive_definite']:
            preconditioners['M'].append(numpy.linalg.inv(A_new))

        # if A is diagonal, ip_B and all o
        if numpy.linalg.norm(numpy.diag(numpy.diag(A_new))-A_new) == 0 \
                and ip_B is None:
            M = numpy.diag(numpy.linspace(1, 10, 10))
            preconditioners['M'].append(M)

        for exact_solution in [None, x]:
            for preconditioner in dictpick(preconditioners):
                kwargs = dict(ls_kwargs)
                kwargs.update(preconditioner)

                if 'M' in preconditioner and preconditioner['M'] is not None:
                    kwargs['Minv'] = numpy.linalg.inv(preconditioner['M'])

                yield krypy.linsys.LinearSystem(A_new, A_new.dot(x), ip_B=ip_B,
                                                exact_solution=exact_solution,
                                                **ls_kwargs)


def solver_params_generator(solver, ls):
    params_add = {}
    if solver is krypy.linsys.RestartedGmres:
        params_add = {'maxiter': [7], 'max_restarts': [20]}
    solver_params = {
        'x0': [None, numpy.zeros(ls.b.shape), numpy.ones(ls.b.shape)],
        'tol': [1e-13, 1e-2],
        'maxiter': [15],
        }
    if ls.exact_solution is not None:
        solver_params['x0'].append(ls.exact_solution)
    for params in dictproduct(dict(list(solver_params.items()) +
                              list(params_add.items()))):
        yield params


cases = [
    # spd
    {'A': test_utils.get_matrix_spd(),
     'normal': True, 'self_adjoint': True, 'positive_definite': True},
    # hpd
    {'A': test_utils.get_matrix_hpd(),
     'normal': True, 'self_adjoint': True, 'positive_definite': True},
    # symmetric indefinite
    {'A': test_utils.get_matrix_symm_indef(),
     'normal': True, 'self_adjoint': True},
    # hermitian indefinite
    {'A': test_utils.get_matrix_herm_indef(),
     'normal': True, 'self_adjoint': True},
    # nonsymm
    {'A': test_utils.get_matrix_nonsymm()},
    # nonsymm
    {'A': test_utils.get_matrix_comp_nonsymm()},
    ]


def test_solver():
    for case in cases:
        for ls in linear_systems_generator(**case):
            solvers = [krypy.linsys.Gmres, krypy.linsys.RestartedGmres]
            if ls.self_adjoint:
                solvers.append(krypy.linsys.Minres)
            if ls.positive_definite:
                solvers.append(krypy.linsys.Cg)
            for solver in solvers:
                for params in solver_params_generator(solver, ls):
                    yield run_solver, solver, ls, params


def run_solver(solver, ls, params):
    sol = solver(ls, **params)
    check_solver(sol, solver, ls, params)


def check_solver(sol, solver, ls, params):
    # pick out the interesting data
    b = krypy.utils.shape_vec(ls.b)
    xk = krypy.utils.shape_vec(sol.xk)
    N = len(b)
    shape = (N, N)
    A = krypy.utils.get_linearoperator(shape, ls.A)
    M = krypy.utils.get_linearoperator(shape, ls.M)
    Ml = krypy.utils.get_linearoperator(shape, ls.Ml)

    # maxiter respected?
    if not 'max_restarts' in params:
        assert(len(sol.resnorms)-1 <= params['maxiter'])
    else:
        assert(len(sol.resnorms)-1 <=
               params['maxiter']*(params['max_restarts']+1))

    # tolerance reached (if not near machine eps)?
    if params['tol'] > 1e-15:
        assert(sol.resnorms[-1] <= params['tol'])

    # final residual norm correct?
    # relresvec[-1] == ||M*Ml*(b-A*xk))||_{M^{-1}} / ||M*Ml*b||_{M^{-1}}
    # compute residual norm
    MMlrk, Mlrk, MMlrk_norm = ls.get_residual(xk, compute_norm=True)

    # finally: the assertion
    if ls.MMlb_norm == 0:
        assert(abs(sol.resnorms[-1]) == 0)
    else:
        assert_almost_equal(sol.resnorms[-1], MMlrk_norm/ls.MMlb_norm,
                            decimal=14)

    # final error norm correct?
    # (if exact_solution was provided)
    if ls.exact_solution is not None:
        assert_almost_equal(sol.errnorms[-1],
                            krypy.utils.norm(
                                krypy.utils.shape_vec(ls.exact_solution)
                                - krypy.utils.shape_vec(sol.xk),
                                ip_B=ls.ip_B))

    # if the preconditioner is the inverse, then check if convergence
    # occured after the first iteration
    if isinstance(A, numpy.ndarray) and \
            isinstance(params['M'], numpy.ndarray) and \
            numpy.linalg.norm(numpy.eye(*A.shape)
                              - numpy.dot(A, ls.M)) < 1e-15:
        assert(len(sol.resnorms) <= 2)

    # 0 iterations if initial guess was good enough?
    if params['x0'] is not None:
        r0 = b - A * krypy.utils.shape_vec(params['x0'])
        Mlr0 = Ml * r0
        MMlr0 = M * Mlr0
        norm_MMlr0 = numpy.sqrt(krypy.utils.inner(Mlr0, MMlr0,
                                                  ip_B=ls.ip_B))
        if ls.MMlb_norm != 0 and norm_MMlr0/ls.MMlb_norm < params['tol']:
            assert(len(sol.resnorms) == 1)

    # has gmres (without restarts) found the solution after max N iterations?
    # (cg or minres may take longer because of roundoff errors)
    if solver == krypy.linsys.Gmres and \
            ((not 'max_restarts' in params) or (params['max_restarts'] == 0)):
        assert (len(sol.resnorms)-1 <= ls.b.shape[0])

if __name__ == '__main__':
    import nose
    nose.main()
