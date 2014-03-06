# -*- coding: utf8 -*-
import numpy
import scipy.linalg
from .. import utils, linsys, deflation


class _RitzCandidatesGenerator(object):
    '''Abstract base class for candidate set generation.'''
    def generate(self, solver, P, ritz):
        raise NotImplementedError('abstract base class cannot be instanciated')


class SmallRitz(_RitzCandidatesGenerator):
    def __init__(self, max_vectors=numpy.Inf):
        self.max_vectors = max_vectors

    def generate(self, solver, P, ritz):
        sort = numpy.argsort(ritz.values)
        for i in range(min(self.max_vectors, len(sort))):
            yield sort[:i]


class _DeflationVectorFactory(object):
    '''Abstract base class for selectors.'''
    def get(self, solver, P):
        '''Get deflation vectors.

        :returns: numpy.array of shape ``(N,k)``
        '''
        raise NotImplementedError('abstract base class cannot be instanciated')


class RitzEvaluator(object):
    def __init__(self, Bound, tol):
        self.Bound = Bound
        self.tol = tol

    def evaluate(self, solver, P, ritz, candidate):
        indices = list(set(range(len(ritz.values))).difference(set(candidate)))
        bound = self.Bound(ritz.values[indices])
        return bound.get_steps(self.tol)


class RitzFactory(_DeflationVectorFactory):
    '''Select Ritz vectors.'''
    def __init__(self, candidate_generator, candidate_evaluator,
                 mode='ritz',
                 hermitian=False):
        self.candidate_generator = candidate_generator
        self.candidate_evaluator = candidate_evaluator
        self.mode = mode
        self.hermitian = hermitian

    def get(self, solver, P):
        ritz = Ritz(solver, P, mode=self.mode, hermitian=self.hermitian)
        return ritz.get_vectors(self._get_best_candidate(solver, P, ritz))

    def _get_best_candidate(self, solver, P, ritz):
        '''Return candidate set with least goal functional.'''
        evaluations = {}

        # evaluate candidate
        for candidate in self.candidate_generator.generate(solver, P, ritz):
            try:
                evaluations[frozenset(candidate)] = \
                    self.candidate_evaluator.evaluate(solver, P, ritz,
                                                      candidate)
            except utils.AssumptionError:
                # no evaluation possible -> move on
                pass

        if evaluations:
            return list(min(evaluations, key=evaluations.get))
        return []


class Ritz(object):
    def __init__(self, solver, P, mode='ritz', hermitian=False):
        self.solver = solver
        self.P = P

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

        # TODO: optimize
        B_ = utils.inner(V, A*U, ip_B=solver.ip_B)
        B = B_[:-1, :]
        E = utils.inner(U, A*U, ip_B=solver.ip_B)
        if hermitian:
            C = B.T.conj()
        else:
            C = utils.inner(U, A*V[:, :-1], ip_B=solver.ip_B)

        if mode == 'ritz':
            # build block matrix
            M = numpy.bmat([[H + B.dot(numpy.linalg.solve(E, C)), B],
                            [C, E]])

            # solve eigenvalue problem
            eig = scipy.linalg.eigh if hermitian else scipy.linalg.eig
            self.values, self.coeffs = eig(M)

            # TODO: compute residual norms

        elif mode == 'harmonic':
            # TODO
            raise NotImplementedError('mode {0} not implemented'.format(mode))
        else:
            raise ValueError('mode {0} not known'.format(mode))

    def get_vectors(self, indices=None):
        coeffs = self.coeffs if indices is None else self.coeffs[:, indices]
        return numpy.c_[self.solver.V[:, :-1], self.P.U].dot(coeffs)

    def get_explicit_residual(self):
        ritz_vecs = self.get_vecs()
        return self.solver.A * ritz_vecs - ritz_vecs * self.ritz_values


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

        After a run of the provided ``Solver``, the resulting ``Solver``
        instance is available in the attribute ``last_solver``.
        '''
        self.Solver = Solver
        self.projection = projection
        self.projection_kwargs = {} if projection_kwargs is None \
            else projection_kwargs
        self.ip_B = ip_B
        self.d_max = d_max
        self.strategy = strategy

        self.last_timings = None
        '''Timings from last run of :py:meth:`solve`.

        Timings have to be filled after each run of :py:meth:`solve` and has
        to be a dictionary holding the time needed for applying ``A``, ``M``,
        ``Ml``, ``Mr``, ``ip``, ``axpy``.'''

        # Solver instance of the last run
        self.last_solver = None
        '''``Solver`` instance from last run of :py:meth:`solve`.

        Instance of ``Solver`` that resulted from the last call to
        :py:meth:`solve`. Initialized with ``None`` before the first run.'''

        self.last_P = None
        '''Projection instance from last run of :py:meth:`solve`.

        Instance of :py:class:`~krypy.deflation.ObliqueProjection` (if
        ``projection=='oblique'``) that was used in the last call to
        :py:meth:`solve`. Initialized with ``None`` before the first run.'''

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

        # set deflation space to zero-dim deflation space if U is None
        if U is None:
            U = numpy.zeros((b.shape[0], 0))

        # initialize projection for deflation
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

        Uses timings from :py:attr:`last_timings`.

        :param nsteps: number of iterations.
        :param d: number of deflation vectors.
        '''
        if not self.last_timings is None:
            raise RuntimeError('last_timinigs not yet present. Called before '
                               'first run of solve()?')

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
                time += self.last_timings[op]*count
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
