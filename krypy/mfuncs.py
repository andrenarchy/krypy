from . import utils
import numpy
import scipy.linalg
import scipy.sparse
import numbers

__all__ = [
    "MatrixFunction",
    "MatrixExponential",
    "MatrixInverse",
    "MatrixFunctionSystem",
    "MatrixFunctionUpdate"
]

# Sparse matrix??
class MatrixFunction:
    """DOCSTRING?"""
    def __init__(
            self,
            f,
            f_description=None,
            implementation="scipy"
    ):
        """DOCSTRING?"""
        self.f = numpy.vectorize(f)
        self.f_description = f_description
        self.implementation = implementation

    def _evaluate_general(self, A):
        if self.implementation == "scipy":
            return scipy.linalg.funm(A, self.f)
        else:
            raise NotImplementedError("Unknown implementation:"
                                      + " {}".format(self.implementation))

    def _evaluate_hermitian(self, A, is_positive_semidefinite=False):
        """From scipy notes*** NEED TO CITE"""
        w, v = scipy.linalg.eigh(A, check_finite=True)
        if is_positive_semidefinite:
            w = numpy.maximum(w, 0)
        w = self.f(w)
        return (v * w) @ v.conj().T

    def _evaluate_general_sparse(self, A_sp):
        raise NotImplementedError("Unknown sparse implementation:"
                                  + " {}".format(self.implementation))

    def _evaluate_hermitian_sparse(self, A_sp, is_positive_semidefinite=False):
        raise NotImplementedError("Unknown sparse implementation:"
                                  + " {}".format(self.implementation))

    def __call__(self, A, is_hermitian=False, is_positive_semidefinite=False):
        """Compute matrix function of current matrix with defined method"""
        is_sparse = scipy.sparse.issparse(A)
        if is_hermitian and is_sparse:
            return self._evaluate_hermitian_sparse(A, is_positive_semidefinite)
        elif is_hermitian and not is_sparse:
            return self._evaluate_hermitian(A, is_positive_semidefinite)
        elif not is_hermitian and is_sparse:
            return self._evaluate_general_sparse(A)
        else:
            return self._evaluate_general(A)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.f_description:
            return str(type(self).__name__) + "({})".format(self.f_description)
        else:
            return str(type(self).__name__) + "({})".format(str(self.f))


class MatrixExponential(MatrixFunction):
    f_description = "numpy.exp(x)"
    f = numpy.exp

    def __init__(self,
                 display_error_estimate=False,
                 implementation="scipy"):
        self.implementation = implementation

    def _evaluate_general(self, A):
        if self.implementation == "scipy":
            return scipy.linalg.expm(A)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))

    def _evaluate_hermitian(self, A, is_positive_semidefinite=False):
        """From scipy notes*** NEED TO CITE"""
        if self.implementation == "scipy":
            return scipy.linalg.expm(A)
        elif self.implementation == "diagonalizable":
            super()._evaluate_hermitian(A, is_positive_semidefinite)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))

    def _evaluate_general_sparse(self, A_sp):
        return scipy.sparse.linalg.expm(A_sp)

    def _evaluate_hermitian_sparse(self, A_sp, is_positive_semidefinite=False):
        raise NotImplementedError("Unknown sparse implementation:"
                                  + " {}".format(self.implementation))


class MatrixInverse(MatrixFunction):
    f_description = "1/x"
    def f(x: any) -> any: return 1/x

    def __init__(self,
                 implementation="scipy"):
        self.implementation = implementation

    def _evaluate_general(self, A):
        if self.implementation == "scipy":
            return scipy.linalg.inv(A)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))

    def _evaluate_hermitian(self, A, is_positive_semidefinite=False):
        """From scipy notes*** NEED TO CITE"""
        if self.implementation == "scipy":
            return scipy.linalg.inv(A)
        elif self.implementation == "diagonalizable":
            super()._evaluate_hermitian(A, is_positive_semidefinite)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))


class MatrixFunctionSystem:
    pass


class MatrixFunctionUpdate:
    pass
