from .. import utils


class _RitzSubsetEvaluator(object):
    def evaluator(self, ritz, subset):
        '''Returns a list of subsets with indices of Ritz vectors that are
        considered for deflation.'''
        raise NotImplementedError('abstract base class cannot be instanciated')


class RitzApriori(object):
    def __init__(self,
                 tol=None,
                 Bound=None,
                 strategy='simple'
                 ):
        '''Evaluates a choice of Ritz vectors with an a-priori bound for
        self-adjoint problems.

        :param tol: (optional) the tolerance for the stopping criterion, see
            :py:class:`~krypy.linsys._KrylovSolver`. If `None` is provided
            (default), then the tolerance is retrieved from
            `ritz._deflated_solver.tol` in the call to :py:meth:`evaluate`.
        :param Bound: (optional) the a-priori bound which is used for
          estimating the convergence behavior.
        :param strategy: (optional) the following strategies are available

          * `simple`: (default) uses the Ritz values that are complementary to
            the deflated ones for the evaluation of the bound.
          * `intervals`: uses intervals around the Ritz values that are
            considered with `simple`. The intervals incorporate possible
            changes in the operators.
        '''
        self.tol = tol
        self.Bound = Bound
        self.strategy = strategy

    def evaluate(self, ritz, subset):
        tol = self.tol
        if tol is None:
            tol = ritz._deflated_solver.tol

        Bound = self.Bound
        if Bound is None:
            # TODO: determine a bound for the used deflated_solver
            pass

        # indices of the Ritz values that are _not_ considered for deflation
        indices_remaining = list(set(range(len(ritz.values)))
                                 .difference(subset))
        if self.strategy == 'simple':
            bound = self.Bound(ritz.values[indices_remaining])
            return bound.get_steps(self.tol)
        elif self.strategy == 'intervals':
            # TODO
            pass
        else:
            raise utils.ArgumentError(
                'Invalid value \'{0}\' for argument \'strategy\'. '
                .format(self.strategy)
                + 'Valid are simple and intervals.')
