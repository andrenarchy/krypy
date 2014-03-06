# -*- coding: utf8 -*-
import numpy
import scipy.linalg
from .. import utils, linsys, deflation


class _RecyclingSolver(object):
    '''Base class for recycling solvers.'''
    def __init__(self, Solver,
                 Projection=deflation.ObliqueProjection,
                 projection_kwargs=None
                 ):
        '''Initialize recycling solver base.

        :param ip_B: (optional) defines the inner product, see
          :py:meth:`~krypy.utils.inner`.

        After a run of the provided ``Solver``, the resulting ``Solver``
        instance is available in the attribute ``last_solver``.
        '''
        self._Solver = Solver
        if not issubclass(Projection, deflation._Projection):
            raise ValueError('Projection invalid')
        self._Projection = Projection
        self._projection_kwargs = {} if projection_kwargs is None \
            else projection_kwargs

        self.last_timings = None
        '''Timings from last run of :py:meth:`solve`.

        Timings have to be filled after each run of :py:meth:`solve` and has
        to be a dictionary holding the time needed for applying ``A``, ``M``,
        ``Ml``, ``Mr``, ``ip``, ``axpy``.'''

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
              deflation_vector_factory=None,
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
        :param deflation_vector_factory: (optional) An instance of a subclass
          of :py:class:`krypy.recycling.factories._DeflationVectorFactory`.

        All remaining keyword arguments are passed to the solver.

        :returns: the solver instance which was used to obtain the approximate
          solution. The approximate solution is available under the attribute
          ``xk``.
        '''
        # check and process parameters
        pre = linsys._KrylovSolver(A, b, **kwargs)

        # get deflation vectors
        if self.last_solver is None or deflation_vector_factory is None:
            U = numpy.zeros((b.shape[0], 0))
        else:
            U = deflation_vector_factory(self.last_solver, self.last_P)

        # initialize projection for deflation
        P = self._Projection(pre.Ml*pre.A*pre.Mr, U,
                             ip_B=pre.ip_B)

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
