import numpy
from .. import deflation, linsys, utils


class _RitzSubsetEvaluator(object):
    def evaluator(self, ritz, subset):
        '''Returns a list of subsets with indices of Ritz vectors that are
        considered for deflation.'''
        raise NotImplementedError('abstract base class cannot be instanciated')


class RitzApriori(object):
    def __init__(self,
                 Bound,
                 tol=None,
                 strategy='simple'
                 ):
        '''Evaluates a choice of Ritz vectors with an a-priori bound for
        self-adjoint problems.

        :param Bound: the a-priori bound which is used for estimating the
          convergence behavior.
        :param tol: (optional) the tolerance for the stopping criterion, see
            :py:class:`~krypy.linsys._KrylovSolver`. If `None` is provided
            (default), then the tolerance is retrieved from
            `ritz._deflated_solver.tol` in the call to :py:meth:`evaluate`.
        :param strategy: (optional) the following strategies are available

          * `simple`: (default) uses the Ritz values that are complementary to
            the deflated ones for the evaluation of the bound.
          * `intervals`: uses intervals around the Ritz values that are
            considered with `simple`. The intervals incorporate possible
            changes in the operators.
        '''
        self.Bound = Bound
        self.tol = tol
        self.strategy = strategy

    def evaluate(self, ritz, subset):
        if not ritz._deflated_solver.linear_system.self_adjoint:
            from warnings import warn
            warn('RitzApriori is designed for self-adjoint problems but '
                 'the provided LinearSystem is not marked as self-adjoint.')
        tol = self.tol
        if tol is None:
            tol = ritz._deflated_solver.tol

        # indices of Ritz values that are considered for deflation
        indices = list(subset)

        # indices of Ritz values that are _not_ considered for deflation
        indices_remaining = list(set(range(len(ritz.values)))
                                 .difference(subset))
        if self.strategy == 'simple':
            bound = self.Bound(ritz.values[indices_remaining])
            return bound.get_step(tol)
        elif self.strategy == 'intervals':
            intervals = self._estimate_eval_intervals(ritz,
                                                      indices,
                                                      indices_remaining)
            bound = self.Bound(intervals)
            return bound.get_step(tol)
        else:
            raise utils.ArgumentError(
                'Invalid value \'{0}\' for argument \'strategy\'. '
                .format(self.strategy)
                + 'Valid are simple and intervals.')

    def _estimate_eval_intervals(self, ritz, indices, indices_remaining,
                                 eps_min=0,
                                 eps_max=0,
                                 eps_res=None):
        '''Estimate evals based on eval inclusion theorem + heuristic.

        :returns: Intervals object with inclusion intervals for eigenvalues
        '''
        if len(indices) == 0:
            return utils.Intervals(
                [utils.Interval(mu-resnorm, mu+resnorm)
                 for mu, resnorm in zip(ritz.values, ritz.resnorms)])
        if len(ritz.values) == len(indices):
            raise utils.AssumptionError(
                'selection of all Ritz pairs does not allow estimation.')
        if eps_res is None:
            eps_res = numpy.max(numpy.abs([eps_min, eps_max]))

        # compute quantities for bound
        delta_sel = numpy.linalg.norm(ritz.resnorms[indices], 2)
        delta_non_sel = numpy.linalg.norm(ritz.resnorms[indices_remaining], 2)
        delta = utils.gap(ritz.values[indices],
                          ritz.values[indices_remaining])
        mu_ints = utils.Intervals(
            [utils.Interval(mu+eps_min, mu+eps_max)
             for mu in ritz.values[indices]])
        mu_min = mu_ints.min_abs()

        # check gap assumption
        #if delta_sel + delta_non_sel + eps_max - eps_min >= delta:
        if delta_sel + eps_max - eps_min >= delta:
            raise utils.AssumptionError(
                'delta_sel + delta_non_sel + eps_max - eps_min >= delta'
                + '({0} >= {1}'.format(
                    delta_sel + delta_non_sel + eps_max - eps_min,
                    delta)
                )

        # check assumption on mu_min
        if mu_min == 0:
            raise utils.AssumptionError('mu_min == 0 not allowed')

        # compute eta
        #eta = (delta_sel+eps_res)**2 * (
        #    1/(delta-delta_non_sel-eps_max+eps_min)
        #    + 1/mu_min
        #    )
        #left = - delta_non_sel + eps_min - eta
        #right = delta_non_sel + eps_max + eta
        eta = (delta_sel+eps_res)**2 * (
            1/(delta-eps_max+eps_min)
            + 1/mu_min
            )
        left = eps_min - eta
        right = eps_max + eta

        # return bound
        return utils.Intervals(
            [utils.Interval(mu+left, mu+right)
             for mu in ritz.values[indices_remaining]])


class RitzApproxKrylov(object):
    def __init__(self,
                 pseudospectra=False):
        '''Evaluates a choice of Ritz vectors with a tailored approximate
        Krylov subspace method.

        :param pseudospectra: (optional) should pseudospectra be computed
          for the given problem?
        '''
        self._arnoldifyer = None

    def evaluate(self, ritz, subset):
        if self._arnoldifyer is not None and \
                self._arnoldifyer._deflated_solver is ritz._deflated_solver:
            arnoldifyer = self._arnoldifyer
        else:
            arnoldifyer = deflation.Arnoldifyer(ritz._deflated_solver)

        Wt = ritz.coeffs[:, list(subset)]
        #Hh, Rh, q_norm, bdiff_norm, PWAW_norm = arnoldifyer.get(Wt)
        bound_pseudo = deflation.bound_pseudo(
            arnoldifyer, Wt, ritz._deflated_solver.linear_system.MMlb_norm)
        from matplotlib import pyplot
        print(' check')
        pyplot.semilogy(bound_pseudo)
        pyplot.show()
        return numpy.Inf
