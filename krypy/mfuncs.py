from .utils import Arnoldi, ArgumentError
import numpy
import scipy.linalg
import scipy.sparse
from collections import deque


__all__ = [
    "MatrixFunction",
    "MatrixExponential",
    "MatrixInverse",
    "MatrixFunctionSystem",
    "RankOneUpdate"
]


class MatrixFunction:
    """DOCSTRING?"""
    def __init__(
            self,
            f,
            f_description=None,
            implementation="scipy",
            dropping_tol=1e-14  # How to implement? 
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

    def _evaluate_hermitian(self, A, is_pos_semidefinite=False):
        """From scipy notes*** NEED TO CITE"""
        w, v = scipy.linalg.eigh(A, check_finite=True)
        if is_pos_semidefinite:
            w = numpy.maximum(w, 0)
        w = self.f(w)
        return (v * w) @ v.conj().T

    def _evaluate_general_sparse(self, A_sp):
        raise NotImplementedError("Unknown sparse implementation:"
                                  + " {}".format(self.implementation))

    def _evaluate_hermitian_sparse(self, A_sp, is_pos_semidefinite=False):
        raise NotImplementedError("Unknown sparse implementation:"
                                  + " {}".format(self.implementation))

    def __call__(self, A, is_hermitian=False, is_pos_semidefinite=False):
        """Compute matrix function of current matrix with defined method"""
        is_sparse = scipy.sparse.issparse(A)
        if is_hermitian and is_sparse:
            return self._evaluate_hermitian_sparse(A, is_pos_semidefinite)
        elif is_hermitian and not is_sparse:
            return self._evaluate_hermitian(A, is_pos_semidefinite)
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
                 implementation="scipy"):
        self.implementation = implementation

    def _evaluate_general(self, A):
        if self.implementation == "scipy":
            return scipy.linalg.expm(A)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))

    def _evaluate_hermitian(self, A, is_pos_semidefinite=False):
        """From scipy notes*** NEED TO CITE"""
        if self.implementation == "scipy":
            return scipy.linalg.expm(A)
        elif self.implementation == "diagonalizable":
            super()._evaluate_hermitian(A, is_pos_semidefinite)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))

    def _evaluate_general_sparse(self, A_sp):
        return scipy.sparse.linalg.expm(A_sp)

    def _evaluate_hermitian_sparse(self, A_sp, is_pos_semidefinite=False):
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

    def _evaluate_hermitian(self, A, is_pos_semidefinite=False):
        """From scipy notes*** NEED TO CITE"""
        if self.implementation == "scipy":
            return scipy.linalg.inv(A)
        elif self.implementation == "diagonalizable":
            super()._evaluate_hermitian(A, is_pos_semidefinite)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))

    def _evaluate_general_sparse(self, A_sp):
        return scipy.sparse.linalg.inv(A_sp)

    def _evaluate_hermitian_sparse(self, A_sp, is_pos_semidefinite=False):
        raise NotImplementedError("Unknown evaluation algorithm:"
                                  + " {}".format(self.implementation))


class MatrixFunctionSystem:

    def __init__(self,
                 A,
                 mfunc,
                 fA=None,
                 diag_only=False,
                 normal=False,
                 hermitian=False,
                 pos_semidefinite=False):
        self.is_sparse = scipy.sparse.issparse(A)
        self.A = A
        self.mfunc = mfunc
        self.fA = fA if fA else mfunc(A,  # *args for MatrixFunction?
                                      is_hermitian=hermitian,
                                      is_pos_semidefinite=pos_semidefinite)
        self.diag_only = diag_only
        if diag_only and len(self.fA.shape) > 1:
            self.fA = (numpy.diag(self.fA) if (not self.is_sparse)
                       else numpy.diag(self.fA.toarray()))

        self.normal = True if hermitian or normal else False
        self.hermitian = hermitian
        self.pos_semidefinite = pos_semidefinite

    def get_mat(self):
        return self.A

    def get_mfunc(self):
        return self.mfunc

    def get_computed_mfunc(self):
        return self.fA

    def __repr__(self):
        s = "MatrixFunctionSystem object\n"
        num_nonzero = (self.A.nnz if self.is_sparse 
                       else sum(sum(self.A == numpy.zeros(self.A.shape))))
        s += "    A: has dimensions {} and {} non-zero elements\n".format(
            self.A.shape,
            num_nonzero
            )
        s += "    f: {}\n".format(self.mfunc.f_description)
        s += ("    f(A) storage: " + self.diag_only * "diagonal elements\n"
              + (not self.diag_only) * "all elements\n")
        s += "    Normal: " + str(self.normal) + "\n"
        s += "    Hermitian:" + str(self.hermitian) + "\n"
        s += "    Positive-semidefinite: " + str(self.pos_semidefinite)
        return s


class RankOneUpdate:

    def __init__(self,
                 mfunc_system,
                 b,
                 c=None,
                 sign=1,
                 maxiter=None,
                 rel_tol=1e-5,
                 d=2,
                 ortho="dmgs",
                 explicit_soln=None,
                 dropping_tol=None,
                 store_res=True):
        #  Clean params, initialize system
        self.N = N = b.shape[0]
        if not isinstance(mfunc_system, MatrixFunctionSystem):
            raise TypeError("Unrecognized type {}. ".format(type(mfunc_system))
                            + "Should be a MatrixFunctionSystem.")
        self.mfunc_system = mfunc_system
        self.A = mfunc_system.get_mat()
        self.fA = mfunc_system.get_computed_mfunc()
        self.mfunc = mfunc_system.get_mfunc()
        self.diag_only = mfunc_system.diag_only
        self.b = b
        self.c = c if c else b
        self.sign = sign
        self.maxiter = maxiter if maxiter else N - 1
        self.rel_tol = rel_tol
        self.d = d
        self.explicit_soln = explicit_soln
        self.dropping_tol = dropping_tol
        self.ortho = ortho

        #  Construct important properties
        self.hermitian = min([numpy.all(b == c), mfunc_system.hermitian])
        self.l_arnoldi = Arnoldi(self.A,
                                       self.b,
                                       self.maxiter,
                                       self.ortho)
        if not self.hermitian:
            self.r_arnoldi = Arnoldi(self.A.conj().T,
                                           self.sign * self.c,
                                           self.maxiter,
                                           self.ortho)
        else:
            self.r_arnoldi = self.l_arnoldi
        self.res = numpy.zeros(N) if store_res else None
        self.Xkf = None
        self.oldXkfs = deque()

    def advance(self):
        self.l_arnoldi.advance()

        k = self.l_arnoldi.iter
        if k > 1 and len(self.oldXkfs) < self.d:
            self.oldXkfs.append(self.Xkf)
        elif k > 1 and len(self.oldXkfs) >= self.d:
            self.oldXkfs.popleft(0)
            self.oldXkfs.append(self.Xkf)

        if self.hermitian:
            self.Xkf = self._compute_Xkf_hermitian()
        else:
            self.r_arnoldi.advance()
            self.Xkf = self._compute_Xkf_nonhermitian()

        self.res[k-1] = self._estimate_residual()

    def _compute_Xkf_hermitian(self):
        iter = self.get_iter()
        U, G, V, H = self.get_arnoldi()
        size_b = numpy.linalg.norm(self.b, 2)
        e1 = numpy.zeros((iter, 1))
        e1[0, 0] = 1
        M1 = self.mfunc(
            G[:iter, :iter] + (self.sign * size_b**2 * e1) @ e1.conj().T
            )
        M2 = self.mfunc(G[:iter, :iter])
        Xkf = M1 - M2
        return Xkf

    def _compute_Xkf_nonhermitian(self):
        k = self.get_iter()
        U, G, V, H = self.get_arnoldi()

        size_b = numpy.linalg.norm(self.b, 2)
        size_c = numpy.linalg.norm(self.c, 2)
        e1 = numpy.zeros((self.N, 1))
        e1[0, 0] = 1
        M1 = G[:k, :k]
        M2 = (size_b * size_c * e1) @ e1.conj().T
        M3 = numpy.zeros((k, k))
        M4 = H[:k, :k] + (size_c * (V.conj().T @ self.b)) @ e1.conj().T

        F = self.mfunc(numpy.block(
            [[M1, M2],
             [M3, M4]]
        ))
        Xkf = F[:k, k:2*k]
        return Xkf

    def _estimate_residual(self):
        if not self.Xkf:
            return float("inf")
        elif len(self.oldXkfs) < self.d:
            return numpy.linalg.norm(self.Xkf)
        else:
            k_plus_d = self.Xkf.shape[0]
            k = k_plus_d - self.d
            d = self.d
            R = self.Xkf
            R = R - numpy.block([[self.oldXkfs[0],     numpy.zeros((k, d))],
                                 [numpy.zeros((d, k)), numpy.zeros((d, d))]])
            return numpy.linalg.norm(R, 2)

    def _diagonal_update(self):
        raise NotImplementedError("No diagonal update defined yet.")

    def _full_update(self):
        U, G, V, H = self.get_arnoldi()
        return U @ self.Xkf @ V.conj().T



    def get_update(self):
        if not self.Xkf:
            raise ArgumentError("No iterations for"
                                      + " low-rank update performed yet")
        
        if self.diag_only:
            return self._diagonal_update()
        else:
            return self._full_update()


    def get_updated_matrix(self):
        return self.A + (self.sign * self.b) @ self.c.conj().T

    def get_new_system(self, maintains_psd=False, maintains_normality=False):
        return MatrixFunctionSystem(self.get_updated_matrix(),
                                    self.mfunc,
                                    self.fA + self.get_update(),
                                    self.diag_only,
                                    maintains_normality,
                                    self.hermitian,
                                    maintains_psd)

    def get_iter(self):
        return self.l_arnoldi.iter

    def get_arnoldi(self):
        if self.hermitian:
            U, G = self.l_arnoldi.get()
            V, H = U, G
        else:
            U, G = self.l_arnoldi.get()
            V, H = self.r_arnoldi.get()
        return U, G, V, H
