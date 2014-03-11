class RitzEvaluator(object):
    def __init__(self, Bound, tol):
        self.Bound = Bound
        self.tol = tol

    def evaluate(self, solver, P, ritz, candidate):
        indices = list(set(range(len(ritz.values))).difference(set(candidate)))
        bound = self.Bound(ritz.values[indices])
        return bound.get_steps(self.tol)
