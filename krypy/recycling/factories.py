import numpy
from .. import utils, deflation

#__all__ = ['_DeflationVectorFactory', 'RitzFactory']


class _DeflationVectorFactory(object):
    '''Abstract base class for selectors.'''
    def get(self, solver):
        '''Get deflation vectors.

        :returns: numpy.array of shape ``(N,k)``
        '''
        raise NotImplementedError('abstract base class cannot be instanciated')


class RitzFactory(_DeflationVectorFactory):
    '''Select Ritz vectors.'''
    def __init__(self, candidate_generator, candidate_evaluator,
                 mode='ritz'):
        self.candidate_generator = candidate_generator
        self.candidate_evaluator = candidate_evaluator
        self.mode = mode

    def get(self, solver):
        ritz = deflation.Ritz(solver, mode=self.mode)
        return ritz.get_vectors(self._get_best_candidate(solver, ritz))

    def _get_best_candidate(self, solver, ritz):
        '''Return candidate set with least goal functional.'''
        evaluations = {}

        # evaluate candidate
        for candidate in self.candidate_generator.generate(solver, ritz):
            try:
                evaluations[frozenset(candidate)] = \
                    self.candidate_evaluator.evaluate(solver, ritz,
                                                      candidate)
            except utils.AssumptionError:
                # no evaluation possible -> move on
                pass

        if evaluations:
            return list(min(evaluations, key=evaluations.get))
        return []


class RitzFactorySimple(_DeflationVectorFactory):
    def __init__(self, mode='ritz', n_vectors=0, which='smallest_abs'):
        '''Selects a fixed number of Ritz or harmonic Ritz vectors
        with respect to a prescribed criterion.

        :param mode: See ``mode`` parameter of
          :py:class:`~krypy.deflation.Ritz`.
        :param n_vectors: number of vectors that are chosen. Actual number of
          deflation vectors may be lower if the number of Ritz pairs is less
          than ``n_vectors``.
        :param which: the ``n_vectors`` Ritz vectors are chosen with:

          * ``smallest_abs``: smallest magnitude of corresponding Ritz values.
          * ``largest_abs``: largest magnitude of corresponding Ritz values.
          * ``smallest_res``: smallest Ritz residual norms.
        '''
        self.mode = mode
        self.n_vectors = n_vectors
        self.which = which

    def get(self, solver):
        ritz = deflation.Ritz(solver, mode=self.mode)

        # get indices according to criterion
        if self.which == 'smallest_abs':
            indices = numpy.argsort(numpy.abs(ritz.values))[:self.n_vectors]
        elif self.which == 'largest_abs':
            indices = numpy.argsort(numpy.abs(ritz.values))[-self.n_vectors:]
        elif self.which == 'smallest_res':
            indices = numpy.argsort(ritz.resnorms)[:self.n_vectors]
        else:
            raise ValueError(r'value \'{0}\' of which is unknown'
                             .format(self.which))
        return ritz.get_vectors(indices)
