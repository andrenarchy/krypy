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
        :param which: the ``n_vectors`` Ritz vectors are chosen such that the
          corresponding Ritz values are the ones with

          * ``lm``: largest magnitude.
          * ``sm``: smallest magnitude.
          * ``lr``: largest real part.
          * ``sr``: smallest real part.
          * ``li``: largest imaginary part.
          * ``si``: smallest imaginary part.
          * ``smallest_res``: smallest Ritz residual norms.
        '''
        self.mode = mode
        self.n_vectors = n_vectors
        self.which = which

    def get(self, solver):
        ritz = deflation.Ritz(solver, mode=self.mode)

        values = ritz.values
        which = self.which
        n_vectors = self.n_vectors

        # get indices according to criterion
        if which == 'lm':
            indices = numpy.argsort(numpy.abs(values))[-n_vectors:]
        elif which == 'sm':
            indices = numpy.argsort(numpy.abs(values))[:n_vectors]
        elif which == 'lr':
            indices = numpy.argsort(numpy.real(values))[-n_vectors:]
        elif which == 'sr':
            indices = numpy.argsort(numpy.real(values))[:n_vectors]
        elif which == 'li':
            indices = numpy.argsort(numpy.imag(values))[-n_vectors:]
        elif which == 'si':
            indices = numpy.argsort(numpy.imag(values))[:n_vectors]
        elif which == 'smallest_res':
            indices = numpy.argsort(ritz.resnorms)[:n_vectors]
        else:
            raise utils.ArgumentError(
                'Invalid value \'{0}\' for \'which\'. '.format(which)
                + 'Valid are lm, sm, lr, sr, li, si and smallest_res.')
        return ritz.get_vectors(indices)
