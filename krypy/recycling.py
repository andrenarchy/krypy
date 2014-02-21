# -*- coding: utf8 -*-
import numpy
import scipy.linalg
from . import utils, linsys, deflation


class _RecyclingStrategy(object):
    def __init__(self, last_solver, last_P):
        pass


def _get_ritz(solver, P, mode='ritz', hermitian=False):
    n = solver.iter
    V = solver.V[:, :n+1]
    N = V.shape[0]
    H_ = solver.H[:n+1, :n]
    H = H_[:n, :]
    if P is not None and P.U is not None:
        U = P.U
        A = P.A
    else:
        U = numpy.zeros((N, 0))
    B_ = utils.inner(V, A*U, ip_B=solver.ip_B)
    B = B_[:-1, :]
    E = utils.inner(U, A*U, ip_B=solver.ip_B)
    C = utils.inner(U, A*V[:, :-1], ip_B=solver.ip_B)

    # build block matrix
    M = numpy.bmat([[H + B.dot(numpy.linalg.solve(E, C)), B],
                    [C, E]])

    # solve eigenvalue problem
    if hermitian:
        ritz_vals, Y = scipy.linalg.eigh(M)
    else:
        ritz_vals, Y = scipy.linalg.eig(M)

    ritz_vecs = numpy.c_[V[:, :-1], U].dot(Y)
    ritz_res = A * ritz_vecs - ritz_vecs * ritz_vals

    return ritz_vals, ritz_vecs, ritz_res


class _RecyclingSolver(object):
    '''Base class for recycling solvers.'''
    def __init__(self, Solver,
                 projection='oblique',
                 projection_kwargs=None,
                 ip_B=None,
                 d_max=None,
                 strategy='aggressive'
                 ):
        '''Initialize recycling solver base.

        :param ip_B: (optional) defines the inner product, see
          :py:meth:`~krypy.utils.inner`.
        :param d_max: maximum number of deflation vectors.
        '''
        self.Solver = Solver
        self.projection = projection
        self.projection_kwargs = {} if projection_kwargs is None \
            else projection_kwargs
        self.ip_B = ip_B
        self.d_max = d_max
        self.strategy = strategy

        # timings have to be filled after each run of solve():
        # a dictionary holding the time needed for applying ``A``, ``M``,
        # ``Ml``, ``Mr``, ``ip``, ``axpy``.
        self.timings = None

        # Solver instance of the last run
        self.last_solver = None

        # Projection that was used in the last run
        self.last_P = None

    def _get_deflation_basis(self, defl_extra_vecs, strategy_kwargs):
        if strategy_kwargs is None:
            strategy_kwargs = {}

        # deflation basis
        U = None

        # examine last run and store proposed deflation space in U
        if self.last_solver is not None and self.last_P is not None:
            # TODO
            pass

        # include extra vectors
        if defl_extra_vecs is not None:
            if U is None:
                U = defl_extra_vecs
            else:
                # TODO: check if defl_extra_vecs are in U and only add new ones
                U = numpy.c_[U, defl_extra_vecs]

        # check if U is a N-by-zero matrix
        if U is not None and U.shape[1] == 0:
            U = None

        return U

    def solve(self, A, b,
              defl_extra_vecs=None,
              strategy_kwargs=None,
              **kwargs):
        '''Solve the given linear system with recycling.

        The deflation vectors and the Krylov subspace of the last solve() call
        are examined and considered for deflation according to the provided
        ``strategy`` in the constructor of :py:class:`_RecyclingSolver`.
        Additionally, the vectors in ``defl_extra_vecs`` are included in the
        deflation subspace.

        :param A: the linear operator (has to be compatible with
          :py:meth:`~krypy.utils.get_linearoperator`).
        :param b: the right hand side with length ``N``.
        :param defl_extra_vecs: vectors to include for deflation additional
          to automatically determined vectors.
        :param strategy_kwargs: arguments for the deflation vector selection.

        All remaining keyword arguments are passed to the solver.

        :returns: the solver instance which was used to obtain the approximate
          solution. The approximate solution is available under the attribute
          ``xk``.
        '''

        # check and process parameters
        pre = linsys._KrylovSolver(A, b, **kwargs)

        # get deflation basis
        U = self._get_deflation_basis(defl_extra_vecs, strategy_kwargs)

        P = None
        # initialize deflation
        if U is not None:
            # get projection
            if self.projection == 'oblique':
                P = deflation.ObliqueProjection(pre.Ml*pre.A*pre.Mr, U,
                                                ip_B=pre.ip_B)
            elif self.projection == 'orthogonal':
                raise NotImplementedError('orthogonal projections are not yet '
                                          'implemented.')
            else:
                raise ValueError('projection {0} is unknown.'
                                 .format(self.projection))

        # set up deflation via right preconditioner and adapted initial guess
        solver_kwargs = dict(kwargs)
        if P is not None:
            solver_kwargs['Mr'] = pre.Mr * P.operator_complement()
            solver_kwargs['x0'] = P.get_x0(pre)
        solver_kwargs['store_arnoldi'] = True

        # solve deflated linear system
        solver = self.Solver(A, b, **solver_kwargs)

        # store solver and projection instance for later use
        self.last_solver = solver
        self.last_P = P

        # return solver instance
        return solver

    def _estimate_time(self, nsteps, d):
        '''Estimate time needed to run nsteps iterations with deflation

        :param nsteps: number of iterations.
        :param d: number of deflation vectors.
        '''
        if not self.timings is None:
            raise RuntimeError('timinigs not yet present. Called before first '
                               'run of solve()?')

        solver_ops = self.Solver.operations(nsteps)
        proj_ops = {'A': 1,
                    'M': 0,
                    'Ml': 1,
                    'Mr': 1,
                    'ip': d*(d+1)/2 + d*d + 2*d*solver_ops['Ml'],
                    'axpy': d*(d+1)/2 + d*d + (2*d+2)*solver_ops['Ml']
                    }
        time = 0.
        for ops in [solver_ops, proj_ops]:
            for op, count in ops.iteritems():
                time += self.timings[op]*count
        return time


class RecyclingCg(_RecyclingSolver):
    '''Recycling preconditioned CG method.'''
    def __init__(self, *args, **kwargs):
        super(RecyclingCg, self).__init__(linsys.Cg, *args, **kwargs)


class RecyclingMinres(_RecyclingSolver):
    '''Recycling preconditioned MINRES method.'''
    def __init__(self, *args, **kwargs):
        super(RecyclingMinres, self).__init__(linsys.Minres, *args, **kwargs)


class RecyclingGmres(_RecyclingSolver):
    '''Recycling preconditioned GMRES method.'''
    def __init__(self, *args, **kwargs):
        super(RecyclingGmres, self).__init__(linsys.Gmres, *args, **kwargs)
