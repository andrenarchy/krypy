import numpy


class _RitzCandidateGenerator(object):
    '''Abstract base class for candidate set generation.'''
    def generate(self, solver, P, ritz):
        raise NotImplementedError('abstract base class cannot be instanciated')


class SmallRitz(_RitzCandidateGenerator):
    def __init__(self, max_vectors=numpy.Inf):
        self.max_vectors = max_vectors

    def generate(self, solver, P, ritz):
        sort = numpy.argsort(ritz.values)
        for i in range(min(self.max_vectors, len(sort))):
            yield sort[:i]
