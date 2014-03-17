import numpy
from .. import utils, deflation
from . import generators, evaluators

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
    def __init__(self,
                 subsets_generator=None,
                 subset_evaluator=None,
                 mode='ritz'
                 ):
        if subsets_generator is None:
            subsets_generator = generators.RitzExtremal()
        self.subsets_generator = subsets_generator
        self.subset_evaluator = subset_evaluator
        self.mode = mode

    def get(self, deflated_solver):
        ritz = deflation.Ritz(deflated_solver, mode=self.mode)
        return ritz.get_vectors(self._get_best_subset(ritz))

    def _get_best_subset(self, ritz):
        '''Return candidate set with smallest goal functional.'''

        # (c,\omega(c)) for all considered subsets c
        overall_evaluations = {}

        def evaluate(_subset, _evaluations):
            try:
                _evaluations[_subset] = \
                    self.subset_evaluator.evaluate(ritz, _subset)
            except utils.AssumptionError:
                # no evaluation possible -> move on
                pass

        # I in algo
        current_subset = frozenset()

        # evaluate empty set
        evaluate(current_subset, overall_evaluations)

        while True:
            # get a list of subset candidates for inclusion in current_subset
            # (S in algo)
            remaining_subset = set(range(len(ritz.values))) \
                .difference(current_subset)
            subsets = self.subsets_generator.generate(ritz, remaining_subset)

            # no more candidates to check?
            if len(subsets) == 0:
                break

            # evaluate candidates
            evaluations = {}
            for subset in subsets:
                eval_subset = current_subset.union(subset)
                evaluate(eval_subset, evaluations)

            if len(evaluations) > 0:
                current_subset = min(evaluations, key=evaluations.get())
            else:
                # fallback: pick the subset with smallest residual
                # note: only a bound is used if the subset consists of more
                #       than one index.
                resnorms = [numpy.sum(ritz.resnorms[list(subset)])
                            for subset in subsets]
                subset = subsets[numpy.argmin(resnorms)]
                current_subset = current_subset.union(subset)

            overall_evaluations.update(evaluations)

        # if there was a successfull evaluation: pick the best one
        if len(overall_evaluations) > 0:
            return list(min(overall_evaluations, key=evaluations.get))

        # otherwise: return empty list
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


class UnionFactory(_DeflationVectorFactory):
    def __init__(self, factories):
        '''Combine a list of factories.

        :param factories: a list of factories derived from
          :py:class:`_DeflationVectorFactory`.
        '''
        self._factories = factories

    def get(self, solver):
        vectors = [factory.get(solver) for factory in self._factories]
        return numpy.asarray(numpy.bmat(vectors))
