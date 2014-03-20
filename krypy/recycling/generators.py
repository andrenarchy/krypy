import numpy


class _RitzSubsetsGenerator(object):
    '''Abstract base class for the generation of subset generation.'''
    def generate(self, ritz, remaining_subset):
        '''Returns a list of subsets with indices of Ritz vectors that are
        considered for deflation.'''
        raise NotImplementedError('abstract base class cannot be instanciated')


class RitzSmall(_RitzSubsetsGenerator):
    '''Successively returns the Ritz value of smallest magnitude.'''
    def __init__(self, max_vectors=numpy.Inf):
        self.max_vectors = max_vectors

    def generate(self, ritz, remaining_subset):
        remaining = list(remaining_subset)
        if len(remaining) <= 1 or \
                len(ritz.values) - len(remaining) >= self.max_vectors:
            return []
        sort = numpy.argsort(numpy.abs(ritz.values[remaining]))
        return [set([remaining[sort[0]]])]


class RitzExtremal(_RitzSubsetsGenerator):
    '''Successively returns the extremal Ritz values.

    For self-adjoint problems, the indices of the minimal negative, maximal
    negative, minimal positive and maximal positive Ritz values are returned.

    For non-self-adjoint problems, only the indices of the Ritz values of
    smallest and largest magnitude are returned.
    '''
    def __init__(self, max_vectors=numpy.Inf):
        self.max_vectors = max_vectors

    def generate(self, ritz, remaining_subset):
        remaining = numpy.array(list(remaining_subset))
        if len(remaining) <= 1 or \
                len(ritz.values) - len(remaining) >= self.max_vectors:
            return []
        remaining_values = ritz.values[remaining]

        def get_minmax_indices(values):
            if len(values) == 0:
                return []
            sort = numpy.argsort(values)
            indices = [sort[0]]
            if len(values) > 1:
                indices.append(sort[-1])
            return indices

        if ritz._deflated_solver.linear_system.self_adjoint:
            indices = []

            # handle negative and positive Ritz values seperately
            negative = numpy.where(remaining_values < 0)[0]
            positive = numpy.where(remaining_values > 0)[0]
            for selection in [negative, positive]:
                indices += list(remaining[selection][
                    get_minmax_indices(remaining_values[selection])])
        else:
            # pick index for smallest and largest magnitude value
            indices = remaining[get_minmax_indices(
                numpy.abs(remaining_values))]

        # return list of sets where each set contains one element
        return [set([i]) for i in indices]
