from .. import utils, deflation

#__all__ = ['_DeflationVectorFactory', 'RitzFactory']


class _DeflationVectorFactory(object):
    '''Abstract base class for selectors.'''
    def get(self, solver, P):
        '''Get deflation vectors.

        :returns: numpy.array of shape ``(N,k)``
        '''
        raise NotImplementedError('abstract base class cannot be instanciated')


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
        ritz = deflation.Ritz(solver, P,
                              mode=self.mode, hermitian=self.hermitian)
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
