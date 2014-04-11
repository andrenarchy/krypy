import numpy
from .. import deflation, utils


class _RitzSubsetEvaluator(object):
    def evaluate(self, ritz, subset):
        '''Returns a list of subsets with indices of Ritz vectors that are
        considered for deflation.'''
        raise NotImplementedError('abstract base class cannot be instanciated')


class RitzApriori(_RitzSubsetEvaluator):
    def __init__(self,
                 Bound,
                 tol=None,
                 strategy='simple',
                 deflweight=1.0
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
        self.deflweight = deflweight

    def evaluate(self, ritz, subset):
        ndefl = len(subset)
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
            nsteps = bound.get_step(tol)
        elif self.strategy == 'intervals':
            intervals = self._estimate_eval_intervals(ritz,
                                                      indices,
                                                      indices_remaining)
            bound = self.Bound(intervals)
            nsteps = bound.get_step(tol)
        else:
            raise utils.ArgumentError(
                'Invalid value \'{0}\' for argument \'strategy\'. '
                .format(self.strategy)
                + 'Valid are simple and intervals.')

        return ritz._deflated_solver.estimate_time(
            nsteps, ndefl, deflweight=self.deflweight)

    @staticmethod
    def _estimate_eval_intervals(ritz, indices, indices_remaining,
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


class RitzApproxKrylov(_RitzSubsetEvaluator):
    def __init__(self,
                 mode='extrapolate',
                 tol=None,
                 pseudospectra=False,
                 bound_pseudo_kwargs=None,
                 deflweight=1.0):
        '''Evaluates a choice of Ritz vectors with a tailored approximate
        Krylov subspace method.

        :param mode: (optional) determines how the number of iterations
          is estimated. Must be one of the following:

          * ``extrapolate`` (default): use the iteration count where the
            extrapolation of the smallest residual reduction over all steps
            drops below the tolerance.
          * ``direct``: use the iteration count where the predicted residual
            bound drops below the tolerance. May result in severe
            underestimation if ``pseudospectra==False``.

        :param pseudospectra: (optional) should pseudospectra be computed
          for the given problem? With ``pseudospectra=True``, a prediction
          may not be possible due to unfulfilled assumptions for the
          computation of the pseudospectral bound.
        :param bound_pseudo_kwargs: (optional) a dictionary with arguments
          that are passed to :py:meth:`~krypy.deflation.bound_pseudo`.
        :param deflweight: (optional) see
          :py:meth:`~krypy._DeflationMixin.estimate_time`. Defaults to 1.
        '''
        self._arnoldifyer = None
        self.mode = mode
        self.tol = tol
        self.pseudospectra = pseudospectra
        if bound_pseudo_kwargs is None:
            bound_pseudo_kwargs = {}
        self.bound_pseudo_kwargs = bound_pseudo_kwargs
        self.deflweight = deflweight

    def evaluate(self, ritz, subset):
        # number of deflation vectors
        ndefl = len(subset)

        # get tolerance
        if self.tol is None:
            tol = ritz._deflated_solver.tol
        else:
            tol = self.tol

        if self._arnoldifyer is not None and \
                self._arnoldifyer._deflated_solver is ritz._deflated_solver:
            arnoldifyer = self._arnoldifyer
        else:
            arnoldifyer = deflation.Arnoldifyer(ritz._deflated_solver)
            self._arnoldifyer = arnoldifyer

        Wt = ritz.coeffs[:, list(subset)]
        bound_pseudo = deflation.bound_pseudo(
            arnoldifyer, Wt,
            tol=tol,
            pseudo_type='omit' if not self.pseudospectra else 'auto',
            **self.bound_pseudo_kwargs
            )

        #from matplotlib import pyplot
        #pyplot.semilogy(bound_pseudo)

        if len(bound_pseudo) <= 1:
            raise utils.AssumptionError('no bound computed')

        if self.mode == 'direct':
            if (bound_pseudo > self.tol).all():
                raise utils.AssumptionError(
                    'tolerance not reached with mode==`direct`.')
            else:
                nsteps = (bound_pseudo > tol).sum()
        elif self.mode == 'extrapolate':
            # compute minimal overall residual reduction
            alpha = numpy.max(
                (bound_pseudo[1:]/bound_pseudo[0])
                ** (1./numpy.array(range(1, len(bound_pseudo))))
                )
            if alpha >= 1 or alpha==0:
                raise utils.AssumptionError(
                    'Cannot compute bound because alpha == {0} >= 1'.format(
                        alpha))
            nsteps = numpy.log(tol/bound_pseudo[0])/numpy.log(alpha)

        else:
            raise utils.ArgumentError(
                'Invalid value `{0}` for argument `omode`. '.format(self.mode)
                + 'Valid are `direct` and `extrapolate`.')

        return ritz._deflated_solver.estimate_time(
            nsteps, ndefl, deflweight=self.deflweight)
