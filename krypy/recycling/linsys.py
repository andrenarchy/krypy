# -*- coding: utf8 -*-
import numpy
from .. import utils, deflation, linsys
from . import factories, evaluators


class _RecyclingSolver(object):
    '''Base class for recycling solvers.'''
    def __init__(self, DeflatedSolver,
                 vector_factory=None
                 ):
        '''Initialize recycling solver base.

        :param DeflatedSolver: a deflated solver from
          :py:mod:`~krypy.deflation`.
        :param vector_factory: (optional) An instance of a subclass of
          :py:class:`krypy.recycling.factories._DeflationVectorFactory`
          that constructs deflation vectors for recycling. Defaults to `None`
          which means that no recycling is used.

          Also the following strings are allowed as shortcuts:

           * ``'RitzApproxKrylov'``: uses the approximate Krylov subspace bound
             evaluator :py:class:`krypy.recycling.evaluators.RitzApproxKrylov`.
           * ``'RitzAprioriCg'``: uses the CG :math:`\kappa`-bound
             (:py:class:`krypy.utils.BoundCG`) as an a priori bound with
             :py:class:`krypy.recycling.evaluators.RitzApriori`.
           * ``'RitzAprioriMinres'``: uses the MINRES bound
             (:py:class:`krypy.utils.BoundMinres`) as an a priori bound with
             :py:class:`krypy.recycling.evaluators.RitzApriori`.

        After a run of the provided ``DeflatedSolver`` via :py:meth:`solve`,
        the resulting instance of the ``DeflatedSolver`` is available in the
        attribute ``last_solver``.
        '''
        self._DeflatedSolver = DeflatedSolver

        self._vector_factory = vector_factory

        self.timings = utils.Timings()
        '''Timings from last run of :py:meth:`solve`.

        Timings of the vector factory runs and the actual solution processes.
        '''

        self.last_solver = None
        '''``DeflatedSolver`` instance from last run of :py:meth:`solve`.

        Instance of ``DeflatedSolver`` that resulted from the last call to
        :py:meth:`solve`. Initialized with ``None`` before the first run.'''

    def solve(self, linear_system,
              vector_factory=None,
              *args, **kwargs):
        '''Solve the given linear system with recycling.

        The provided `vector_factory` determines which vectors are used for
        deflation.

        :param linear_system: the :py:class:`~krypy.linsys.LinearSystem` that
          is about to be solved.
        :param vector_factory: (optional) see description in constructor.

        All remaining arguments are passed to the ``DeflatedSolver``.

        :returns: instance of ``DeflatedSolver`` which was used to obtain the
          approximate solution. The approximate solution is available under the
          attribute ``xk``.
        '''

        # replace linear_system with equivalent TimedLinearSystem on demand
        if not isinstance(linear_system, linsys.TimedLinearSystem):
            linear_system = linsys.ConvertedTimedLinearSystem(linear_system)

        with self.timings['vector_factory']:
            if vector_factory is None:
                vector_factory = self._vector_factory

            # construct vector_factory if strings are provided
            if vector_factory == 'RitzApproxKrylov':
                vector_factory = factories.RitzFactory(
                    subset_evaluator=evaluators.RitzApproxKrylov()
                    )
            elif vector_factory == 'RitzAprioriCg':
                vector_factory = factories.RitzFactory(
                    subset_evaluator=evaluators.RitzApriori(
                        Bound=utils.BoundCG
                        )
                    )
            elif vector_factory == 'RitzAprioriMinres':
                vector_factory = factories.RitzFactory(
                    subset_evaluator=evaluators.RitzApriori(
                        Bound=utils.BoundMinres
                        )
                    )

            # get deflation vectors
            if self.last_solver is None or vector_factory is None:
                U = numpy.zeros((linear_system.N, 0))
            else:
                U = vector_factory.get(self.last_solver)

        with self.timings['solve']:
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
