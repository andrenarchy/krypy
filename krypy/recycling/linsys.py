# -*- coding: utf8 -*-
import numpy
from .. import utils, deflation



class _RecyclingSolver(object):
    '''Base class for recycling solvers.'''
    def __init__(self, DeflatedSolver):
        '''Initialize recycling solver base.

        :param DeflatedSolver: a deflated solver from
          :py:mod:`~krypy.deflation`.

        After a run of the provided ``DeflatedSolver`` via :py:meth:`solve`,
        the resulting instance of the ``DeflatedSolver`` is available in the
        attribute ``last_solver``.
        '''
        self._DeflatedSolver = DeflatedSolver

        self.last_solver = None
        '''``DeflatedSolver`` instance from last run of :py:meth:`solve`.

        Instance of ``DeflatedSolver`` that resulted from the last call to
        :py:meth:`solve`. Initialized with ``None`` before the first run.'''

    def solve(self, linear_system,
              vector_factory=None,
              *args, **kwargs):
        '''Solve the given linear system with recycling.

        The deflation vectors and the Krylov subspace of the last solve() call
        are examined and considered for deflation according to the provided
        ``deflation_vector_factory``.

        :param linear_system: the :py:class:`~krypy.linsys.LinearSystem` that
          is about to be solved.
        :param vector_factory: (optional) An instance of a subclass
          of :py:class:`krypy.recycling.factories._DeflationVectorFactory`.

        All remaining arguments are passed to the ``DeflatedSolver``.

        :returns: instance of ``DeflatedSolver`` which was used to obtain the
          approximate solution. The approximate solution is available under the
          attribute ``xk``.
        '''
        # get deflation vectors
        if self.last_solver is None or vector_factory is None:
            U = numpy.zeros((linear_system.N, 0))
        else:
            U = vector_factory.get(self.last_solver)

        # solve deflated linear system
        self.last_solver = self._DeflatedSolver(linear_system,
                                                U=U,
                                                store_arnoldi=True,
                                                *args, **kwargs)

        # return solver instance
        return self.last_solver


class RecyclingCg(_RecyclingSolver):
    '''Recycling preconditioned CG method.

    See :py:class:`~krypy.recycling.linsys._RecyclingSolver` for the
    documentation of the available parameters.
    '''
    def __init__(self, *args, **kwargs):
        super(RecyclingCg, self).__init__(deflation.DeflatedCg,
                                          *args, **kwargs)


class RecyclingMinres(_RecyclingSolver):
    '''Recycling preconditioned MINRES method.

    See :py:class:`~krypy.recycling.linsys._RecyclingSolver` for the
    documentation of the available parameters.
    '''
    def __init__(self, *args, **kwargs):
        super(RecyclingMinres, self).__init__(deflation.DeflatedMinres,
                                              *args, **kwargs)


class RecyclingGmres(_RecyclingSolver):
    '''Recycling preconditioned GMRES method.

    See :py:class:`~krypy.recycling.linsys._RecyclingSolver` for the
    documentation of the available parameters.
    '''
    def __init__(self, *args, **kwargs):
        super(RecyclingGmres, self).__init__(deflation.DeflatedGmres,
                                             *args, **kwargs)
