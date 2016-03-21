# -*- coding: utf8 -*-
'''
Collection of standard functions.

This method provides functions like inner products, norms, ...
'''

import numpy
import warnings
import time
import scipy.linalg
from scipy.sparse import isspmatrix
#from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.sputils import isintlike
from collections import defaultdict

# for Givens rotations
import scipy.linalg.blas as blas

__all__ = ['ArgumentError', 'AssumptionError', 'ConvergenceError',
           'LinearOperatorError', 'InnerProductError', 'RuntimeError',
           'Arnoldi', 'BoundCG', 'BoundMinres',
           'ConvergenceError', 'Givens', 'House', 'IdentityLinearOperator',
           'LinearOperator', 'MatrixLinearOperator',
           'NormalizedRootsPolynomial', 'Projection', 'Timer', 'angles',
           'arnoldi', 'arnoldi_res', 'arnoldi_projected',
           'bound_perturbed_gmres', 'gap', 'get_linearoperator', 'hegedus',
           'inner', 'ip_euclid', 'norm', 'norm_MMlr', 'norm_squared',
           'orthonormality', 'qr', 'ritz', 'shape_vec', 'shape_vecs',
           'strakos']


class ArgumentError(Exception):
    '''Raised when an argument is invalid.

    Analogue to ``ValueError`` which is not used here in order to be able
    to distinguish between built-in errors and ``krypy`` errors.
    '''


class AssumptionError(Exception):
    '''Raised when an assumption is not satisfied.

    Differs from :py:class:`ArgumentError` in that all passed arguments are
    valid but computations reveal that assumptions are not satisfied and
    the result cannot be computed.
    '''
    pass


class ConvergenceError(Exception):
    '''Raised when a method did not converge.

    The ``ConvergenceError`` holds a message describing the error and
    the attribute ``solver`` through which the last approximation and other
    relevant information can be retrieved.
    '''
    def __init__(self, msg, solver):
        super(ConvergenceError, self).__init__(msg)
        self.solver = solver


class LinearOperatorError(Exception):
    '''Raised when a :py:class:`LinearOperator` cannot be applied.'''


class InnerProductError(Exception):
    '''Raised when the inner product is indefinite.'''


class RuntimeError(Exception):
    '''Raised for errors that do not fit in any other exception.'''


def find_common_dtype(*args):
    '''Returns common dtype of numpy and scipy objects.

    Recognizes ndarray, spmatrix and LinearOperator. All other objects are
    ignored (most notably None).'''
    dtypes = []
    for arg in args:
        if type(arg) is numpy.ndarray or  \
                isspmatrix(arg) or \
                isinstance(arg, LinearOperator):
            if hasattr(arg, 'dtype'):
                dtypes.append(arg.dtype)
            else:
                warnings.warn('object %s does not have a dtype.'
                              % arg.__repr__)
    return numpy.find_common_type(dtypes, [])


def shape_vec(x):
    '''Take a (n,) ndarray and return it as (n,1) ndarray.'''
    return numpy.reshape(x, (x.shape[0], 1))


def shape_vecs(*args):
    '''Reshape all ndarrays with ``shape==(n,)`` to ``shape==(n,1)``.

    Recognizes ndarrays and ignores all others.'''
    ret_args = []
    flat_vecs = True
    for arg in args:
        if type(arg) is numpy.ndarray:
            if len(arg.shape) == 1:
                arg = shape_vec(arg)
            else:
                flat_vecs = False
        ret_args.append(arg)
    return flat_vecs, ret_args


def ip_euclid(X, Y):
    '''Euclidean inner product.

    numpy.vdot only works for vectors and numpy.dot does not use the conjugate
    transpose.

    :param X: numpy array with ``shape==(N,m)``
    :param Y: numpy array with ``shape==(N,n)``

    :return: numpy array :math:`X^* Y` with ``shape==(m,n)``.
    '''
    return numpy.dot(X.T.conj(), Y)


def inner(X, Y, ip_B=None):
    '''Euclidean and non-Euclidean inner product.

    numpy.vdot only works for vectors and numpy.dot does not use the conjugate
    transpose.

    :param X: numpy array with ``shape==(N,m)``
    :param Y: numpy array with ``shape==(N,n)``
    :param ip_B: (optional) May be one of the following

        * ``None``: Euclidean inner product.
        * a self-adjoint and positive definite operator :math:`B` (as
          ``numpy.array`` or ``LinearOperator``). Then :math:`X^*B Y` is
          returned.
        * a callable which takes 2 arguments X and Y and returns
          :math:`\\langle X,Y\\rangle`.

    **Caution:** a callable should only be used if necessary. The choice
    potentially has an impact on the round-off behavior, e.g. of projections.

    :return: numpy array :math:`\\langle X,Y\\rangle` with ``shape==(m,n)``.
    '''
    if ip_B is None or isinstance(ip_B, IdentityLinearOperator):
        return numpy.dot(X.T.conj(), Y)
    (N, m) = X.shape
    (_, n) = Y.shape
    try:
        B = get_linearoperator((N, N), ip_B)
    except TypeError:
        return ip_B(X, Y)
    if m > n:
        return numpy.dot((B*X).T.conj(), Y)
    else:
        return numpy.dot(X.T.conj(), B*Y)


def norm_squared(x, Mx=None, inner_product=ip_euclid):
    '''Compute the norm^2 w.r.t. to a given scalar product.'''
    assert(len(x.shape) == 2)
    if Mx is None:
        rho = inner_product(x, x)
    else:
        assert(len(Mx.shape) == 2)
        rho = inner_product(x, Mx)

    if rho.shape == (1, 1):
        if abs(rho[0, 0].imag) > abs(rho[0, 0])*1e-10 or rho[0, 0].real < 0.0:
            raise InnerProductError(('<x,Mx> = %g. Is the inner product '
                                     'indefinite?') % rho[0, 0])

    return numpy.linalg.norm(rho, 2)


def norm(x, y=None, ip_B=None):
    r'''Compute norm (Euclidean and non-Euclidean).

    :param x: a 2-dimensional ``numpy.array``.
    :param y: a 2-dimensional ``numpy.array``.
    :param ip_B: see :py:meth:`inner`.

    Compute :math:`\sqrt{\langle x,y\rangle}` where the inner product is
    defined via ``ip_B``.
    '''
    if y is None:
        y = x
    # Euclidean inner product?
    if y is None and (ip_B is None
                      or isinstance(ip_B, IdentityLinearOperator)):
        return numpy.linalg.norm(x, 2)
    ip = inner(x, y, ip_B=ip_B)
    nrm_diag = numpy.linalg.norm(numpy.diag(ip), 2)
    nrm_diag_imag = numpy.linalg.norm(numpy.imag(numpy.diag(ip)), 2)
    if nrm_diag_imag > nrm_diag*1e-10:
        raise InnerProductError('inner product defined by ip_B not positive '
                                'definite? ||diag(ip).imag||/||diag(ip)||={0}'
                                .format(nrm_diag_imag/nrm_diag))
    return numpy.sqrt(numpy.linalg.norm(ip, 2))


def get_linearoperator(shape, A, timer=None):
    """Enhances aslinearoperator if A is None."""
    ret = None
    import scipy.sparse.linalg as scipylinalg
    if isinstance(A, LinearOperator):
        ret = A
    elif A is None:
        ret = IdentityLinearOperator(shape)
    elif isinstance(A, numpy.ndarray) or isspmatrix(A):
        ret = MatrixLinearOperator(A)
    elif isinstance(A, numpy.matrix):
        ret = MatrixLinearOperator(numpy.atleast_2d(numpy.asarray(A)))
    elif isinstance(A, scipylinalg.LinearOperator):
        if not hasattr(A, 'dtype'):
            raise ArgumentError('scipy LinearOperator has no dtype.')
        ret = LinearOperator(A.shape, dot=A.matvec, dot_adj=A.rmatvec,
                             dtype=A.dtype)
    else:
        raise TypeError('type not understood')

    # set up timer if requested
    if A is not None and not isinstance(A, IdentityLinearOperator) \
            and timer is not None:
        ret = TimedLinearOperator(ret, timer)

    # check shape
    if shape != ret.shape:
        raise LinearOperatorError('shape mismatch')

    return ret


def norm_MMlr(M, Ml, A, Mr, b, x0, yk, inner_product=ip_euclid):
    xk = x0 + Mr * yk
    r = b - A * xk
    Mlr = Ml * r
    # normalize residual before applying the preconditioner here.
    # otherwise MMlr can become 0 exactly (pyamg doesnt respect relative
    # residual)
    # TODO for testing: 2-norm
    norm_Mlr = norm(Mlr)
    if norm_Mlr == 0:
        MMlr = numpy.zeros(Mlr.shape)
        norm_MMlr = 0
    else:
        nMlr = Mlr / norm_Mlr
        nMMlr = M * nMlr
        MMlr = nMMlr * norm_Mlr
        norm_MMlr = norm(Mlr, MMlr, inner_product=inner_product)
    # return xk and ||M*Ml*(b-A*(x0+Mr*yk))||_{M^{-1}}
    return xk, Mlr, MMlr, norm_MMlr


def orthonormality(V, ip_B=None):
    """Measure orthonormality of given basis.

    :param V: a matrix :math:`V=[v_1,\ldots,v_n]` with ``shape==(N,n)``.
    :param ip_B: (optional) the inner product to use, see :py:meth:`inner`.

    :return: :math:`\\| I_n - \\langle V,V \\rangle \\|_2`.
    """
    return norm(numpy.eye(V.shape[1]) - inner(V, V, ip_B=ip_B))


def arnoldi_res(A, V, H, ip_B=None):
    """Measure Arnoldi residual.

    :param A: a linear operator that can be used with scipy's aslinearoperator
      with ``shape==(N,N)``.
    :param V: Arnoldi basis matrix with ``shape==(N,n)``.
    :param H: Hessenberg matrix: either :math:`\\underline{H}_{n-1}` with
      ``shape==(n,n-1)`` or :math:`H_n` with ``shape==(n,n)`` (if the Arnoldi
      basis spans an A-invariant subspace).
    :param ip_B: (optional) the inner product to use, see :py:meth:`inner`.

    :returns: either :math:`\\|AV_{n-1} - V_n \\underline{H}_{n-1}\\|` or
      :math:`\\|A V_n - V_n H_n\\|` (in the invariant case).
    """
    N = V.shape[0]
    invariant = H.shape[0] == H.shape[1]
    A = get_linearoperator((N, N), A)
    if invariant:
        res = A*V - numpy.dot(V, H)
    else:
        res = A*V[:, :-1] - numpy.dot(V, H)
    return norm(res, ip_B=ip_B)


class House:
    def __init__(self, x):
        """Compute Householder transformation for given vector.

        Initialize Householder transformation :math:`H` such that
        :math:`Hx = \\alpha \|x\|_2 e_1` with :math:`|\\alpha|=1`

        The algorithm is a combination of Algorithm 5.1.1 on page 236
        and the treatment of the complex case in Section 5.1.13 on page 243
        in Golub, Van Loan. Matrix computations. Fourth Edition. 2013.
        """
        # make sure that x is a vector ;)
        if len(x.shape) != 2 or x.shape[1] != 1:
            raise ArgumentError('x is not a vector of dim (N,1)')

        v = x.copy()

        gamma = numpy.asscalar(v[0])
        v[0] = 1
        if x.shape[0] == 1:
            sigma = 0
            xnorm = numpy.abs(gamma)
            beta = 0
            alpha = 1 if gamma == 0 else gamma/xnorm
        else:
            sigma = numpy.linalg.norm(v[1:], 2)
            xnorm = numpy.sqrt(numpy.abs(gamma)**2+sigma**2)

            # is x the multiple of first unit vector?
            if sigma == 0:
                beta = 0
                xnorm = numpy.abs(gamma)
                alpha = 1 if gamma == 0 else gamma/xnorm
            else:
                beta = 2
                if gamma == 0:
                    v[0] = -sigma
                    alpha = 1
                else:
                    v[0] = gamma + gamma/numpy.abs(gamma) * xnorm
                    alpha = - gamma/numpy.abs(gamma)

        self.xnorm = xnorm
        self.v = v/numpy.sqrt(numpy.abs(v[0])**2 + sigma**2)
        self.alpha = alpha
        self.beta = beta

    def apply(self, x):
        """Apply Householder transformation to vector x.

        Applies the Householder transformation efficiently to the given vector.
        """
        # make sure that x is a (N,*) matrix
        if len(x.shape) != 2:
            raise ArgumentError('x is not a matrix of shape (N,*)')
        if self.beta == 0:
            return x
        return x - self.beta * self.v * numpy.dot(self.v.T.conj(), x)

    def matrix(self):
        """Build matrix representation of Householder transformation.

        Builds the matrix representation
        :math:`H = I - \\beta vv^*`.

        **Use with care!** This routine may be helpful for testing purposes but
        should not be used in production codes for high dimensions since
        the resulting matrix is dense.
        """
        n = self.v.shape[0]
        return numpy.eye(n, n) - self.beta * numpy.dot(self.v, self.v.T.conj())


def hessenberg(a, calc_q=False, overwrite_a=False, check_finite=True):
    """
    Compute Hessenberg form of a matrix.

    The Hessenberg decomposition is::

        A = Q H Q^H

    where `Q` is unitary/orthogonal and `H` has only zero elements below
    the first sub-diagonal.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to bring into Hessenberg form.
    calc_q : bool, optional
        Whether to compute the transformation matrix.  Default is False.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.
        Default is False.
    check_finite : boolean, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    H : (M, M) ndarray
        Hessenberg form of `a`.
    Q : (M, M) ndarray
        Unitary/orthogonal similarity transformation matrix ``A = Q H Q^H``.
        Only returned if ``calc_q=True``.

    """
    from numpy import asarray_chkfinite, asarray
    from scipy.linalg import calc_lwork
    from scipy.linalg.misc import _datacopied
    from scipy.linalg.lapack import get_lapack_funcs
    from scipy.linalg.blas import get_blas_funcs

    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    if len(a1.shape) != 2 or (a1.shape[0] != a1.shape[1]):
        raise ValueError('expected square matrix')
    overwrite_a = overwrite_a or (_datacopied(a1, a))

    # if 2x2 or smaller: already in Hessenberg form
    if a1.shape[0] <= 2:
        if calc_q:
            return a1, numpy.eye(a1.shape[0])
        return a1

    gehrd, gebal = get_lapack_funcs(('gehrd', 'gebal'), (a1,))
    ba, lo, hi, pivscale, info = gebal(a1, permute=0, overwrite_a=overwrite_a)
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal gebal '
                         '(hessenberg)' % -info)
    n = len(a1)
    lwork = calc_lwork.gehrd(gehrd.typecode, n, lo, hi)
    hq, tau, info = gehrd(ba, lo=lo, hi=hi, lwork=lwork, overwrite_a=1)
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal gehrd '
                         '(hessenberg)' % -info)

    if not calc_q:
        for i in range(lo, hi):
            hq[i+2:hi+1, i] = 0.0
        return hq

    # XXX: Use ORGHR routines to compute q.
    typecode = hq.dtype
    ger, gemm = get_blas_funcs(('ger', 'gemm'), dtype=typecode)
    q = None
    for i in range(lo, hi):
        if tau[i] == 0.0:
            continue
        v = numpy.zeros(n, dtype=typecode)
        v[i+1] = 1.0
        v[i+2:hi+1] = hq[i+2:hi+1, i]
        hq[i+2:hi+1, i] = 0.0
        h = ger(-tau[i], v, v,
                a=numpy.diag(numpy.ones(n, dtype=typecode)), overwrite_a=1)
        if q is None:
            q = h
        else:
            q = gemm(1.0, q, h)
    if q is None:
        q = numpy.diag(numpy.ones(n, dtype=typecode))
    return hq, q


class Givens:
    def __init__(self, x):
        """Compute Givens rotation for provided vector x.

        Computes Givens rotation
        :math:`G=\\begin{bmatrix}c&s\\\\-\\overline{s}&c\\end{bmatrix}`
        such that
        :math:`Gx=\\begin{bmatrix}r\\\\0\\end{bmatrix}`.
        """
        # make sure that x is a vector ;)
        if x.shape != (2, 1):
            raise ArgumentError('x is not a vector of shape (2,1)')

        a = numpy.asscalar(x[0])
        b = numpy.asscalar(x[1])
        # real vector
        if numpy.isreal(x).all():
            a = numpy.real(a)
            b = numpy.real(b)
            c, s = blas.drotg(a, b)
        # complex vector
        else:
            c, s = blas.zrotg(a, b)

        self.c = c
        self.s = s
        self.r = c*a + s*b
        self.G = numpy.array([[c, s], [-numpy.conj(s), c]])

    def apply(self, x):
        """Apply Givens rotation to vector x."""
        return numpy.dot(self.G, x)


class Projection(object):
    def __init__(self, X,
                 Y=None,
                 ip_B=None,
                 orthogonalize=True,
                 iterations=2
                 ):
        """Generic projection.

        This class can represent any projection (orthogonal and oblique)
        on a N-dimensional Hilbert space. A projection is a linear operator
        :math:`P` with :math:`P^2=P`. A projection is uniquely defined by its
        range :math:`\\mathcal{V}:=\\operatorname{range}(P)` and its kernel
        :math:`\\mathcal{W}:=\\operatorname{ker}(P)`; this projection is called
        :math:`P_{\\mathcal{V},\\mathcal{W}}`.

        Let X and Y be two full rank arrays with ``shape==(N,k)`` and let
        :math:`\\mathcal{X}\\oplus\\mathcal{Y}^\\perp=\\mathbb{C}^N` where
        :math:`\\mathcal{X}:=\\operatorname{colspan}(X)` and
        :math:`\\mathcal{Y}:=\\operatorname{colspan}(Y)`.
        Then this class constructs the projection
        :math:`P_{\\mathcal{X},\\mathcal{Y}^\\perp}`.
        The requirement
        :math:`\\mathcal{X}\\oplus\\mathcal{Y}^\\perp=\\mathbb{C}^N`
        is equivalent to ``\\langle X,Y\\rangle`` being nonsingular.

        :param X: array with ``shape==(N,k)`` and
            :math:`\\operatorname{rank}(X)=k`.
        :param Y: (optional) ``None`` or array with ``shape==(N,k)`` and
            :math:`\\operatorname{rank}(X)=k`. If Y is ``None`` then Y is
            set to X which means that the resulting projection is orthogonal.
        :param ip_B: (optional) inner product, see :py:meth:`inner`. ``None``,
            a ``numpy.array`` or a ``LinearOperator`` is preferred due to the
            applicability of the proposed algorithms in [Ste11]_, see below.
        :param orthogonalize: (optional) `True` orthogonalizes the bases
            provided in `X` and `Y` with respect to the inner product defined
            by `ip_B`. Defaults to `True` as the orthogonalization is
            suggested by the round-off error analysis in [Ste11]_.
        :param iterations: (optional) number of applications of the projection.
            It was suggested in [Ste11]_ to use 2 iterations (default) in order
            to achieve high accuracy ("twice is enough" as in the orthogonal
            case).

        This projection class makes use of the round-off error analysis of
        oblique projections in the work of Stewart [Ste11]_ and implements the
        algorithms that are considered as the most stable ones (e.g., the XQRY
        representation in [Ste11]_).
        """
        # check and store input
        self.ip_B = ip_B
        if iterations < 1:
            raise ArgumentError('iterations < 1 not allowed')
        self.orthogonalize = orthogonalize
        self.iterations = iterations

        Y = X if Y is None else Y   # default: orthogonal projection

        if len(X.shape) != 2:
            raise ArgumentError('X does not have shape==(N,k)')
        if X.shape != Y.shape:
            raise ArgumentError('X and Y have different shapes')

        # set N-by-zero vectors if input is N-by-zero
        # (results in zero operator)
        if X.shape[1] == 0:
            self.V = self.W = numpy.zeros(X.shape)
            self.VR = self.WR = self.Q = self.R = None
            return

        # orthogonalize X
        if orthogonalize:
            self.V, self.VR = qr(X, ip_B=ip_B)
        else:
            self.V = X
            self.VR = None

        if Y is X and orthogonalize:  # orthogonal projection
            self.W, self.WR = self.V, self.VR
            self.Q, self.R = None, None
        else:  # general case
            if orthogonalize:
                self.W, self.WR = qr(Y, ip_B=ip_B)
            else:
                self.W = Y
                self.WR = None
            M = inner(self.W, self.V, ip_B=ip_B)
            self.Q, self.R = scipy.linalg.qr(M)

    def _apply(self, a, return_Ya=False):
        r'''Single application of the projection.

        :param a: array with ``a.shape==(N,m)``.
        :param return_inner: (optional) should the inner product
          :math:`\langle Y,a\rangle` be returned?
        :return:

          * :math:`P_{\mathcal{X},\mathcal{Y}^\perp} a =
            X \langle Y,X\rangle^{-1} \langle Y, a\rangle`.
          * :math:`\langle Y,a\rangle` if ``return_inner==True``.
        '''
        # is projection the zero operator?
        if self.V.shape[1] == 0:
            Pa = numpy.zeros(a.shape)
            if return_Ya:
                return Pa, numpy.zeros((0, a.shape[1]))
            return Pa
        c = inner(self.W, a, ip_B=self.ip_B)

        if return_Ya:
            Ya = c.copy()
            if self.WR is not None:
                Ya = self.WR.T.conj().dot(Ya)

        if self.Q is not None and self.R is not None:
            c = scipy.linalg.solve_triangular(self.R, self.Q.T.conj().dot(c))
        Pa = self.V.dot(c)
        if return_Ya:
            return Pa, Ya
        return Pa

    def _apply_adj(self, a):
        # is projection the zero operator?
        if self.V.shape[1] == 0:
            return numpy.zeros(a.shape)
        '''Single application of the adjoint projection.'''
        c = inner(self.V, a, ip_B=self.ip_B)
        if self.Q is not None and self.R is not None:
            c = self.Q.dot(scipy.linalg.solve_triangular(self.R.T.conj(), c,
                                                         lower=True))
        return self.W.dot(c)

    def apply(self, a, return_Ya=False):
        r"""Apply the projection to an array.

        The computation is carried out without explicitly forming the
        matrix corresponding to the projection (which would be an array with
        ``shape==(N,N)``).

        See also :py:meth:`_apply`.
        """
        # is projection the zero operator?
        if self.V.shape[1] == 0:
            Pa = numpy.zeros(a.shape)
            if return_Ya:
                return Pa, numpy.zeros((0, a.shape[1]))
            return Pa
        if return_Ya:
            x, Ya = self._apply(a, return_Ya=return_Ya)
        else:
            x = self._apply(a)
        for i in range(self.iterations-1):
            z = a - x
            w = self._apply(z)
            x = x + w
        if return_Ya:
            return x, Ya
        return x

    def apply_adj(self, a):
        # is projection the zero operator?
        if self.V.shape[1] == 0:
            return numpy.zeros(a.shape)
        x = self._apply_adj(a)
        for i in range(self.iterations-1):
            z = a - x
            w = self._apply_adj(z)
            x = x + w
        return x

    def apply_complement(self, a, return_Ya=False):
        """Apply the complementary projection to an array.

        :param z: array with ``shape==(N,m)``.

        :return: :math:`P_{\\mathcal{Y}^\\perp,\\mathcal{X}}z =
            z - P_{\\mathcal{X},\\mathcal{Y}^\\perp} z`.
        """
        # is projection the zero operator? --> complement is identity
        if self.V.shape[1] == 0:
            if return_Ya:
                return a.copy(), numpy.zeros((0, a.shape[1]))
            return a.copy()
        if return_Ya:
            x, Ya = self._apply(a, return_Ya=True)
        else:
            x = self._apply(a)
        z = a - x
        for i in range(self.iterations-1):
            w = self._apply(z)
            z = z - w
        if return_Ya:
            return z, Ya
        return z

    def apply_complement_adj(self, a):
        # is projection the zero operator? --> complement is identity
        if self.V.shape[1] == 0:
            return a.copy()
        x = self._apply_adj(a)
        z = a - x
        for i in range(self.iterations-1):
            w = self._apply_adj(z)
            z = z - w
        return z

    def _get_operator(self, fun, fun_adj):
        N = self.V.shape[0]
        t = numpy.find_common_type([self.V.dtype, self.W.dtype], [])
        return LinearOperator((N, N), t, fun, fun_adj)

    def operator(self):
        """Get a ``LinearOperator`` corresponding to apply().

        :return: a LinearOperator that calls apply().
        """
        # is projection the zero operator?
        if self.V.shape[1] == 0:
            N = self.V.shape[0]
            return ZeroLinearOperator((N, N))
        return self._get_operator(self.apply, self.apply_adj)

    def operator_complement(self):
        """Get a ``LinearOperator`` corresponding to apply_complement().

        :return: a LinearOperator that calls apply_complement().
        """
        # is projection the zero operator? --> complement is identity
        if self.V.shape[1] == 0:
            N = self.V.shape[0]
            return IdentityLinearOperator((N, N))
        return self._get_operator(self.apply_complement,
                                  self.apply_complement_adj)

    def matrix(self):
        """Builds matrix representation of projection.

        Builds the matrix representation
        :math:`P = X \\langle Y,X\\rangle^{-1} \\langle Y, I_N\\rangle`.

        **Use with care!** This routine may be helpful for testing purposes but
        should not be used in production codes for high dimensions since
        the resulting matrix is dense.
        """
        return self.apply(numpy.eye(self.V.shape[0]))


def qr(X, ip_B=None, reorthos=1):
    """QR factorization with customizable inner product.

    :param X: array with ``shape==(N,k)``
    :param ip_B: (optional) inner product, see :py:meth:`inner`.
    :param reorthos: (optional) numer of reorthogonalizations. Defaults to
      1 (i.e. 2 runs of modified Gram-Schmidt) which should be enough in most
      cases (TODO: add reference).

    :return: Q, R where :math:`X=QR` with :math:`\\langle Q,Q \\rangle=I_k` and
      R upper triangular.
    """
    if ip_B is None and X.shape[1] > 0:
        return scipy.linalg.qr(X, mode='economic')
    else:
        (N, k) = X.shape
        Q = X.copy()
        R = numpy.zeros((k, k), dtype=X.dtype)
        for i in range(k):
            for reortho in range(reorthos+1):
                for j in range(i):
                    alpha = inner(Q[:, [j]], Q[:, [i]], ip_B=ip_B)[0, 0]
                    R[j, i] += alpha
                    Q[:, [i]] -= alpha * Q[:, [j]]
            R[i, i] = norm(Q[:, [i]], ip_B=ip_B)
            if R[i, i] >= 1e-15:
                Q[:, [i]] /= R[i, i]
        return Q, R


def angles(F, G, ip_B=None, compute_vectors=False):
    """Principal angles between two subspaces.

    This algorithm is based on algorithm 6.2 in `Knyazev, Argentati. Principal
    angles between subspaces in an A-based scalar product: algorithms and
    perturbation estimates. 2002.` This algorithm can also handle small angles
    (in contrast to the naive cosine-based svd algorithm).

    :param F: array with ``shape==(N,k)``.
    :param G: array with ``shape==(N,l)``.
    :param ip_B: (optional) angles are computed with respect to this
      inner product. See :py:meth:`inner`.
    :param compute_vectors: (optional) if set to ``False`` then only the angles
      are returned (default). If set to ``True`` then also the principal
      vectors are returned.

    :return:

      * ``theta`` if ``compute_vectors==False``
      * ``theta, U, V`` if ``compute_vectors==True``

      where

      * ``theta`` is the array with ``shape==(max(k,l),)`` containing the
        principal angles
        :math:`0\\leq\\theta_1\\leq\\ldots\\leq\\theta_{\\max\\{k,l\\}}\\leq
        \\frac{\\pi}{2}`.
      * ``U`` are the principal vectors from F with
        :math:`\\langle U,U \\rangle=I_k`.
      * ``V`` are the principal vectors from G with
        :math:`\\langle V,V \\rangle=I_l`.

    The principal angles and vectors fulfill the relation
    :math:`\\langle U,V \\rangle = \
    \\begin{bmatrix} \
    \\cos(\\Theta) & 0_{m,l-m} \\\\ \
    0_{k-m,m} & 0_{k-m,l-m} \
    \\end{bmatrix}`
    where :math:`m=\\min\\{k,l\\}` and
    :math:`\\cos(\\Theta)=\\operatorname{diag}(\\cos(\\theta_1),\\ldots,\\cos(\\theta_m))`.
    Furthermore,
    :math:`\\theta_{m+1}=\\ldots=\\theta_{\\max\\{k,l\\}}=\\frac{\\pi}{2}`.
    """
    # make sure that F.shape[1]>=G.shape[1]
    reverse = False
    if F.shape[1] < G.shape[1]:
        reverse = True
        F, G = G, F

    QF, _ = qr(F, ip_B=ip_B)
    QG, _ = qr(G, ip_B=ip_B)

    # one or both matrices empty? (enough to check G here)
    if G.shape[1] == 0:
        theta = numpy.ones(F.shape[1])*numpy.pi/2
        U = QF
        V = QG
    else:
        Y, s, Z = scipy.linalg.svd(inner(QF, QG, ip_B=ip_B))
        Vcos = numpy.dot(QG, Z.T.conj())
        n_large = numpy.flatnonzero((s**2) < 0.5).shape[0]
        n_small = s.shape[0] - n_large
        theta = numpy.r_[
            numpy.arccos(s[n_small:]),  # [-i:] does not work if i==0
            numpy.ones(F.shape[1]-G.shape[1])*numpy.pi/2]
        if compute_vectors:
            Ucos = numpy.dot(QF, Y)
            U = Ucos[:, n_small:]
            V = Vcos[:, n_small:]

        if n_small > 0:
            RG = Vcos[:, :n_small]
            S = RG - numpy.dot(QF, inner(QF, RG, ip_B=ip_B))
            _, R = qr(S, ip_B=ip_B)
            Y, u, Z = scipy.linalg.svd(R)
            theta = numpy.r_[
                numpy.arcsin(u[::-1][:n_small]),
                theta]
            if compute_vectors:
                RF = Ucos[:, :n_small]
                Vsin = numpy.dot(RG, Z.T.conj())
                # next line is hand-crafted since the line from the paper does
                # not seem to work.
                Usin = numpy.dot(RF, numpy.dot(
                    numpy.diag(1/s[:n_small]),
                    numpy.dot(Z.T.conj(), numpy.diag(s[:n_small]))))
                U = numpy.c_[Usin, U]
                V = numpy.c_[Vsin, V]

    if compute_vectors:
        if reverse:
            U, V = V, U
        return theta, U, V
    else:
        return theta


def hegedus(A, b, x0, M=None, Ml=None, ip_B=None):
    """Rescale initial guess appropriately (Hegedüs trick).

    The Hegedüs trick rescales the initial guess to :math:`\\gamma_{\\min} x_0`
    such that

    .. math ::

      \\|r_0\\|_{M^{-1}}
      = \\| M M_l (b - A \\gamma_{\\min} x_0) \\|_{M^{-1}}
      = \\min_{\\gamma\\in\\mathbb{C}}
        \\| M M_l (b - A \\gamma x_0) \\|_{M^{-1}}
      \\leq \\| M M_l b \\|_{M^{-1}}.

    This is achieved by
    :math:`\\gamma_{\\min} = \\frac{\\langle z, M M_l b \\rangle_{M^{-1}}}{\\|z\\|_{M^{-1}}^2}` for
    :math:`z=M M_l A x_0` because then :math:`r_0=P_{z^\\perp}b`. (Note that
    the right hand side of formula (5.8.16) in [LieS13]_ has to be complex
    conjugated.)

    The parameters are the parameters you want to pass to
    :py:meth:`~krypy.linsys.gmres`,
    :py:meth:`~krypy.linsys.minres` or
    :py:meth:`~krypy.linsys.cg`.

    :return: the adapted initial guess with the above property.
    """
    N = len(b)
    shape = (N, N)
    A = get_linearoperator(shape, A)
    M = get_linearoperator(shape, M)
    Ml = get_linearoperator(shape, Ml)

    MlAx0 = Ml*(A*x0)
    z = M*MlAx0
    znorm2 = inner(z, MlAx0, ip_B=ip_B)
    if znorm2 <= 1e-15:
        return numpy.zeros((N, 1))
    gamma = inner(z, Ml*b, ip_B=ip_B) / znorm2
    return gamma*x0


class Arnoldi(object):
    def __init__(self, A, v,
                 maxiter=None,
                 ortho='mgs',
                 M=None,
                 Mv=None,
                 Mv_norm=None,
                 ip_B=None
                 ):
        """Arnoldi algorithm.

        Computes V and H such that :math:`AV_n=V_{n+1}\\underline{H}_n`.  If
        the Krylov subspace becomes A-invariant then V and H are truncated such
        that :math:`AV_n = V_n H_n`.

        :param A: a linear operator that can be used with scipy's
          aslinearoperator with ``shape==(N,N)``.
        :param v: the initial vector with ``shape==(N,1)``.
        :param maxiter: (optional) maximal number of iterations. Default: N.
        :param ortho: (optional) orthogonalization algorithm: may be one of

            * ``'mgs'``: modified Gram-Schmidt (default).
            * ``'dmgs'``: double Modified Gram-Schmidt.
            * ``'lanczos'``: Lanczos short recurrence.
            * ``'house'``: Householder.
        :param M: (optional) a self-adjoint and positive definite
          preconditioner. If ``M`` is provided, then also a second basis
          :math:`P_n` is constructed such that :math:`V_n=MP_n`. This is of
          importance in preconditioned methods. ``M`` has to be ``None`` if
          ``ortho=='house'`` (see ``B``).
        :param ip_B: (optional) defines the inner product to use. See
          :py:meth:`inner`.

          ``ip_B`` has to be ``None`` if ``ortho=='house'``. It's unclear to me
          (andrenarchy), how a variant of the Householder QR algorithm can be
          used with a non-Euclidean inner product. Compare
          http://math.stackexchange.com/questions/433644/is-householder-orthogonalization-qr-practicable-for-non-euclidean-inner-products
        """
        N = v.shape[0]

        # save parameters
        self.A = get_linearoperator((N, N), A)
        self.maxiter = N if maxiter is None else maxiter
        self.ortho = ortho
        self.M = get_linearoperator((N, N), M)
        if isinstance(self.M, IdentityLinearOperator):
            self.M = None
        self.ip_B = ip_B

        self.dtype = find_common_dtype(A, v, M)
        # number of iterations
        self.iter = 0
        # Arnoldi basis
        self.V = numpy.zeros((N, self.maxiter+1), dtype=self.dtype)
        if self.M is not None:
            self.P = numpy.zeros((N, self.maxiter+1), dtype=self.dtype)
        # Hessenberg matrix
        self.H = numpy.zeros((self.maxiter+1, self.maxiter),
                             dtype=self.dtype)
        # flag indicating if Krylov subspace is invariant
        self.invariant = False

        if ortho == 'house':
            if (self.M is not None
                    and not isinstance(self.M, IdentityLinearOperator)) or \
                    (not isinstance(self.ip_B, IdentityLinearOperator) and
                     self.ip_B is not None):
                raise ArgumentError('Only euclidean inner product allowed '
                                    'with Householder orthogonalization')
            self.houses = [House(v)]
            self.vnorm = numpy.linalg.norm(v, 2)
        elif ortho in ['mgs', 'dmgs', 'lanczos']:
            self.reorthos = 0
            if ortho == 'dmgs':
                self.reorthos = 1
            if self.M is not None:
                p = v
                if Mv is None:
                    v = self.M*p
                else:
                    v = Mv
                if Mv_norm is None:
                    self.vnorm = norm(p, v, ip_B=ip_B)
                else:
                    self.vnorm = Mv_norm
                if self.vnorm > 0:
                    self.P[:, [0]] = p / self.vnorm
            else:
                if Mv_norm is None:
                    self.vnorm = norm(v, ip_B=ip_B)
                else:
                    self.vnorm = Mv_norm
        else:
            raise ArgumentError(
                'Invalid value \'{0}\' for argument \'ortho\'. '.format(ortho)
                + 'Valid are house, mgs, dmgs and lanczos.')
        if self.vnorm > 0:
            self.V[:, [0]] = v / self.vnorm
        else:
            self.invariant = True

    def advance(self):
        """Carry out one iteration of Arnoldi."""
        if self.iter >= self.maxiter:
            raise ArgumentError('Maximum number of iterations reached.')
        if self.invariant:
            raise ArgumentError('Krylov subspace was found to be invariant '
                                'in the previous iteration.')

        N = self.V.shape[0]
        k = self.iter

        # the matrix-vector multiplication
        Av = self.A * self.V[:, [k]]

        if self.ortho == 'house':
            # Householder
            for j in range(k+1):
                Av[j:] = self.houses[j].apply(Av[j:])
                Av[j] *= numpy.conj(self.houses[j].alpha)
            if k+1 < N:
                house = House(Av[k+1:])
                self.houses.append(house)
                Av[k+1:] = house.apply(Av[k+1:]) * numpy.conj(house.alpha)
                self.H[:k+2, [k]] = Av[:k+2]
            else:
                self.H[:k+1, [k]] = Av[:k+1]
            # next line is safe due to the multiplications with alpha
            self.H[k+1, k] = numpy.abs(self.H[k+1, k])
            if self.H[k+1, k] / numpy.linalg.norm(self.H[:k+2, :k+1], 2)\
                    <= 1e-14:
                self.invariant = True
            else:
                vnew = numpy.zeros((N, 1), dtype=self.dtype)
                vnew[k+1] = 1
                for j in range(k+1, -1, -1):
                    vnew[j:] = self.houses[j].apply(vnew[j:])
                self.V[:, [k+1]] = vnew * self.houses[-1].alpha
        else:
            # determine vectors for orthogonalization
            start = 0

            # Lanczos?
            if self.ortho == 'lanczos':
                start = k
                if k > 0:
                    self.H[k-1, k] = self.H[k, k-1]
                    if self.M is not None \
                            and not isinstance(self.M, IdentityLinearOperator):
                        Av -= self.H[k, k-1] * self.P[:, [k-1]]
                    else:
                        Av -= self.H[k, k-1] * self.V[:, [k-1]]

            # (double) modified Gram-Schmidt
            for reortho in range(self.reorthos+1):
                # orthogonalize
                for j in range(start, k+1):
                    alpha = inner(self.V[:, [j]], Av, ip_B=self.ip_B)[0, 0]
                    if self.ortho == 'lanczos':
                        # check if alpha is real
                        if abs(alpha.imag) > 1e-10:
                            warnings.warn(
                                'Iter {0}: abs(alpha.imag) = {1} > 1e-10. '
                                'Is your operator self-adjoint in the '
                                'provided inner product?'
                                .format(self.iter, abs(alpha.imag)))
                        alpha = alpha.real
                    self.H[j, k] += alpha
                    if self.M is not None:
                        Av -= alpha * self.P[:, [j]]
                    else:
                        Av -= alpha * self.V[:, [j]]
            if self.M is not None:
                MAv = self.M * Av
                self.H[k+1, k] = norm(Av, MAv, ip_B=self.ip_B)
            else:
                self.H[k+1, k] = norm(Av, ip_B=self.ip_B)
            if self.H[k+1, k] / numpy.linalg.norm(self.H[:k+2, :k+1], 2)\
                    <= 1e-14:
                self.invariant = True
            else:
                if self.M is not None:
                    self.P[:, [k+1]] = Av / self.H[k+1, k]
                    self.V[:, [k+1]] = MAv / self.H[k+1, k]
                else:
                    self.V[:, [k+1]] = Av / self.H[k+1, k]

        # increase iteration counter
        self.iter += 1

    def get(self):
        k = self.iter
        if self.invariant:
            V, H = self.V[:, :k], self.H[:k, :k]
            if self.M:
                return V, H, self.P[:, :k]
            return V, H
        else:
            V, H = self.V[:, :k+1], self.H[:k+1, :k]
            if self.M:
                return V, H, self.P[:, :k+1]
            return V, H

    def get_last(self):
        k = self.iter
        if self.invariant:
            V, H = None, self.H[:k, [k-1]]
            if self.M:
                return V, H, None
            return V, H
        else:
            V, H = self.V[:, [k]], self.H[:k+1, [k-1]]
            if self.M:
                return V, H, self.P[:, [k]]
            return V, H


def arnoldi(*args, **kwargs):
    _arnoldi = Arnoldi(*args, **kwargs)
    while _arnoldi.iter < _arnoldi.maxiter and not _arnoldi.invariant:
        _arnoldi.advance()
    return _arnoldi.get()


def arnoldi_projected(H, P, k, ortho='mgs'):
    """Compute (perturbed) Arnoldi relation for projected operator.

    Assume that you have computed an Arnoldi relation

    .. math ::

        A V_n = V_{n+1} \\underline{H}_n

    where :math:`V_{n+1}\\in\\mathbb{C}^{N,n+1}` has orthogonal columns
    (with respect to an inner product :math:`\\langle\\cdot,\\cdot\\rangle`)
    and :math:`\\underline{H}_n\\in\\mathbb{C}^{n+1,n}` is an extended
    upper Hessenberg matrix.

    For :math:`k<n` you choose full rank matrices
    :math:`X\\in\\mathbb{C}^{n-1,k}` and :math:`Y\\in\\mathbb{C}^{n,k}` and
    define :math:`\\tilde{X}:=A V_{n_1}X = V_n \\underline{H}_{n-1} X` and
    :math:`\\tilde{Y}:=V_n Y` such that
    :math:`\\langle \\tilde{Y}, \\tilde{X} \\rangle = Y^*\\underline{H}_{n-1} X`
    is invertible. Then the projections :math:`P` and :math:`\\tilde{P}`
    characterized by

    * :math:`\\tilde{P}x = x -
      \\tilde{X} \\langle \\tilde{Y},\\tilde{X} \\rangle^{-1}
      \\langle\\tilde{Y},x\\rangle`
    * :math:`P = I - \\underline{H}_{n-1}X (Y^*\\underline{H}_{n-1}X)^{-1}Y^*`

    are well defined and :math:`\\tilde{P}V_{n+1} = [V_n P, v_{n+1}]` holds.

    This method computes for :math:`i<n-k` the Arnoldi relation

    .. math ::

        (\\tilde{P}A + E_i) W_i
        = W_{i+1} \\underline{G}_i

    where :math:`W_{i+1}=V_n U_{i+1}` has orthogonal columns with respect
    to :math:`\\langle\\cdot,\\cdot\\rangle`,
    :math:`\\underline{G}_i` is an extended upper Hessenberg matrix
    and :math:`E_i x = v_{n+1} F_i \\langle W_i,x\\rangle` with
    :math:`F_i=[f_1,\ldots,f_i]\\in\\mathbb{C}^{1,i}`.

    The perturbed Arnoldi relation can also be generated with the operator
    :math:`P_{V_n} \\tilde{P} A`:

    .. math ::

        P_{V_n} \\tilde{P} A W_i
        = W_{i+1} \\underline{G}_i.

    In a sense the perturbed Arnoldi relation is the best prediction for the
    behavior of the Krylov subspace :math:`K_i(\\tilde{P}A,\\tilde{P}v_1)`
    that can be generated only with the data from :math:`K_{n+1}(A,v_1)` and
    without carrying out further matrix-vector multiplications with A.

    :param H: the extended upper Hessenberg matrix
      :math:`\\underline{H}_n` with ``shape==(n+1,n)``.
    :param P: the projection
      :math:`P:\\mathbb{C}^n\\longrightarrow\\mathbb{C}^n` (has to be
      compatible with :py:meth:`get_linearoperator`).
    :param k: the dimension of the null space of P.
    :returns: U, G, F where

      * U is the coefficient matrix :math:`U_{i+1}` with ``shape==(n,i+1)``,
      * G is the extended upper Hessenberg matrix :math:`\\underline{G}_i`
        with ``shape==(i+1,i)``,
      * F is the error matrix :math:`F_i` with ``shape==(1,i)``.
    """
    n = H.shape[1]
    dtype = find_common_dtype(H, P)
    invariant = H.shape[0] == n
    hlast = 0 if invariant else H[-1, -1]
    H = get_linearoperator((n, n), H if invariant else H[:-1, :])
    P = get_linearoperator((n, n), P)
    v = P * numpy.eye(n, 1)
    maxiter = n-k+1
    F = numpy.zeros((1, maxiter), dtype=dtype)
    PH = lambda x: P*(H*x)
    PH = LinearOperator((n, n), dtype, PH)
    _arnoldi = Arnoldi(PH, v, maxiter=maxiter, ortho=ortho)
    while _arnoldi.iter < _arnoldi.maxiter and not _arnoldi.invariant:
        u, _ = _arnoldi.get_last()
        F[0, _arnoldi.iter] = hlast * u[-1, 0]
        _arnoldi.advance()
    U, G = _arnoldi.get()
    return U, G, F[[0], :_arnoldi.iter]


def ritz(H, V=None, hermitian=False, type='ritz'):
    """Compute several kinds of Ritz pairs from an Arnoldi/Lanczos relation.

    This function computes Ritz, harmonic Ritz or improved harmonic Ritz values
    and vectors with respect to the Krylov subspace :math:`K_n(A,v)` from the
    extended Hessenberg matrix :math:`\\underline{H}_n` generated with n
    iterations the Arnoldi algorithm applied to A and v.

    :param H: Hessenberg matrix from Arnoldi/Lanczos algorithm.
    :param V: (optional) Arnoldi/Lanczos vectors,
      :math:`V\\in\\mathbb{C}^{N,n+1}`. If provided, the Ritz vectors are
      also returned. The Arnoldi vectors have to form an orthonormal basis
      with respect to an inner product.

      **Caution:** if you are using the Lanzcos or Gram-Schmidt Arnoldi
      algorithm without reorthogonalization, then the orthonormality of the
      basis is usually lost. For accurate results it is advisable to use
      the Householder Arnoldi (``ortho='house'``) or modified Gram-Schmidt
      with reorthogonalization (``ortho='dmgs'``).
    :param hermitian: (optional) if set to ``True`` the matrix :math:`H_n` must
      be Hermitian. A Hermitian matrix :math:`H_n` allows for faster and
      often more accurate computation of Ritz pairs.
    :param type: (optional) type of Ritz pairs, may be one of ``'ritz'``,
      ``'harmonic'`` or ``'harmonic_like'``. Two choices of Ritz pairs fit
      in the following description:

      `Given two n-dimensional subspaces`
      :math:`X,Y\subseteq \\mathbb{C}^N`,
      `find a basis`
      :math:`z_1,\ldots,z_n`
      `of`
      :math:`X`
      `and` :math:`\\theta_1,\ldots,\\theta_n\\in\\mathbb{C}`
      such that
      :math:`A z_i - \\theta_i z_i \\perp Y`
      for all :math:`i\\in\{1,\ldots,n\}`.

      In this setting the choices are

      * ``'ritz'``: regular Ritz pairs, i.e. :math:`X=Y=K_n(A,v)`.
      * ``'harmonic'``: harmonic Ritz pairs, i.e.
          :math:`X=K_n(A,v)` and :math:`Y=AK_n(A,v)`.
      * ``'harmonic_improved'``: the returned vectors ``U`` (and ``V``, if
        requested) are the same as with ``type='harmonic'``. The ``theta``
        array contains the improved Ritz values
        :math:`\\theta_i = u_i^* H_n u_i`, cf. section 2 in *Morgan, Zeng.
        Harmonic Projection Methods for Large Non-symmetric Eigenvalue
        Problems. 1998.* It can be shown that the residual norm of improved
        Ritz pairs is always less than or equal to the residual norm of the
        harmonic Ritz pairs. However, the improved Ritz pairs do not fit into
        the framework above since the orthogonality condition is lost.

    :return:

      * If V is not ``None`` then ``theta, U, resnorm, Z`` is returned.
      * If V is ``None`` then ``theta, U, resnorm`` is returned.

      Where

      * ``theta`` are the Ritz values :math:`[\\theta_1,\ldots,\\theta_n]`.
      * ``U`` are the coefficients of the Ritz vectors in the Arnoldi basis,
        i.e. :math:`z_i=Vu_i` where :math:`u_i` is the i-th column of U.
      * ``resnorm`` is a residual norm vector.
      * ``Z`` are the actual Ritz vectors, i.e. ``Z=dot(V,U)``.
    """
    # TODO: enhance ritz to accept an augmented space

    n = H.shape[1]
    if V is not None and V.shape[1] != H.shape[0]:
        raise ArgumentError('shape mismatch with V and H')
    if not H.shape[0] in [n, n+1]:
        raise ArgumentError('H not of shape (n+1,n) or (n,n)')
    symmres = numpy.linalg.norm(H[:n, :] - H[:n, :].T.conj())
    if hermitian and symmres >= 5e-14:
        warnings.warn('Hessenberg matrix is not symmetric: |H-H^*|={0}'
                      .format(symmres))

    # choose eig for Hermitian or non-Hermitian matrices
    eig = scipy.linalg.eigh if hermitian else scipy.linalg.eig

    if type == 'ritz':
        theta, U = eig(H[:n, :])
        beta = 0 if H.shape[0] == n else H[-1, -1]
        resnorm = numpy.abs(beta * U[-1, :])
    elif type == 'harmonic':
        theta, U = eig(H[:n, :].T.conj(), numpy.dot(H.T.conj(), H))
        theta = 1/theta
        resnorm = []
        for i in range(n):
            U[:, i] /= numpy.linalg.norm(U[:, i], 2)
            resi = numpy.dot(H, U[:, i])
            if resi.dtype != numpy.complex and theta.dtype == numpy.complex:
                resi = numpy.array(resi, dtype=numpy.complex)
            resi[:n] -= theta[i]*U[:, i]
            resnorm.append(numpy.linalg.norm(resi, 2))
        resnorm = numpy.array(resnorm)
    elif type == 'harmonic_improved':
        theta, U = eig(H[:n, :].T.conj(), numpy.dot(H.T.conj(), H))
        rho = []
        for i in range(n):
            U[:, i] /= numpy.linalg.norm(U[:, i], 2)
            rho.append(numpy.dot(U[:, i].T.conj(),
                                 numpy.dot(H[:n, :], U[:, i])))
        theta = numpy.array(rho)
        resnorm = []
        for i in range(n):
            resi = numpy.dot(H, U[:, i])
            resi[:n] -= theta[i]*U[:, i]
            resnorm.append(numpy.linalg.norm(resi, 2))
        resnorm = numpy.array(resnorm)
        pass
    else:
        raise ArgumentError('unknown Ritz type {0}'.format(type))

    if V is not None:
        return theta, U, resnorm, numpy.dot(V[:, :n], U)

    return theta, U, resnorm


class Timer(list):
    """Measure execution time of multiple code blocks with ``with``.

    Example: ::

        t = Timer()
        with t:
            print('time me!')
        print('don\\\'t time me!')
        with t:
            print('time me, too!')
        print(t)

    Result: ::

        time me!
        don't time me!
        time me, too!
        [6.389617919921875e-05, 6.008148193359375e-05]

    """
    def __init__(self):
        super(Timer, self).__init__()

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, a, b, c):
        self.append(time.time() - self.tstart)


class Timings(defaultdict):
    '''Manages several timers.

    If you want to measure different types of code blocks you can use ::

        tm = Timings()
        with tm['class1']:
            print('code that belongs to class1')
        with tm['class2']:
            print('code that belongs to class2')
        with tm['class1']:
            print('code that belongs to class1')
        print(tm)

    Result: ::

        code that belongs to class1
        code that belongs to class2
        code that belongs to class1
        defaultdict(<class krypy.utils.Timer at 0x23810b8>, {'class2': [2.9087066650390625e-05], 'class1': [4.696846008300781e-05, 3.2901763916015625e-05]})
    '''
    def __init__(self):
        super(Timings, self).__init__(Timer)

    def get(self, key):
        '''Return timings for `key`. Returns 0 if not present.'''
        if key in self and len(self[key]) > 0:
            return min(self[key])
        else:
            return 0

    def get_ops(self, ops):
        '''Return timings for dictionary ops holding the operation names as
        keys and the number of applications as values.'''
        time = 0.
        for op, count in ops.items():
            time += self.get(op) * count
        return time

    def __repr__(self):
        return 'Timings(' + ', '.join(
            ['{0}: {1}'.format(key, self.get(key)) for key in self]
            ) + ')'


class LinearOperator(object):
    """Linear operator.

    Is partly based on the LinearOperator from scipy (BSD License).
    """
    def __init__(self, shape, dtype, dot=None, dot_adj=None):
        if len(shape) != 2 or not isintlike(shape[0]) \
                or not isintlike(shape[1]):
            raise LinearOperatorError('shape must be (m,n) with m and n '
                                      'integer')
        self.shape = shape
        self.dtype = numpy.dtype(dtype)  # defaults to float64
        if dot is None and dot_adj is None:
            raise LinearOperatorError('dot or dot_adj have to be defined')
        self._dot = dot
        self._dot_adj = dot_adj

    def dot(self, X):
        X = numpy.asanyarray(X)
        m, n = self.shape
        if X.shape[0] != n:
            raise LinearOperatorError('dimension mismatch')
        if self._dot is None:
            raise LinearOperatorError('dot undefined')
        if X.shape[1] == 0:
            return numpy.zeros(X.shape)
        return self._dot(X)

    def dot_adj(self, X):
        X = numpy.asanyarray(X)
        m, n = self.shape
        if X.shape[0] != m:
            raise LinearOperatorError('dimension mismatch')
        if self._dot_adj is None:
            raise LinearOperatorError('dot_adj undefined')
        if X.shape[1] == 0:
            return numpy.zeros(X.shape)
        return self._dot_adj(X)

    @property
    def adj(self):
        return _AdjointLinearOperator(self)

    def __mul__(self, X):
        try:
            if isinstance(X, IdentityLinearOperator):
                return self
            elif isinstance(self, IdentityLinearOperator):
                return X
            elif isinstance(X, LinearOperator):
                return _ProductLinearOperator(self, X)
            elif numpy.isscalar(X):
                return _ScaledLinearOperator(self, X)
            else:
                return self.dot(X)
        except LinearOperatorError:
            return NotImplemented

    def __rmul__(self, X):
        try:
            return _ScaledLinearOperator(self, X)
        except LinearOperatorError:
            return NotImplemented

    def __pow__(self, X):
        try:
            return _PowerLinearOperator(self, X)
        except LinearOperatorError:
            return NotImplemented

    def __add__(self, X):
        try:
            return _SumLinearOperator(self, X)
        except LinearOperatorError:
            return NotImplemented

    def __neg__(self):
        try:
            return _ScaledLinearOperator(self, -1)
        except LinearOperatorError as e:
            return NotImplemented

    def __sub__(self, X):
        return self + (-X)

    def __repr__(self):
        m, n = self.shape
        return '<%dx%d %s with dtype=%s>' \
            % (m, n, self.__class__.__name__, str(self.dtype))


def _get_dtype(operators, dtypes=None):
    if dtypes is None:
        dtypes = []
    for obj in operators:
        if obj is not None and hasattr(obj, 'dtype'):
            dtypes.append(obj.dtype)
    return numpy.find_common_type(dtypes, [])


class _SumLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or \
                not isinstance(B, LinearOperator):
            raise LinearOperatorError('both operands have to be a '
                                      'LinearOperator')
        if A.shape != B.shape:
            raise LinearOperatorError('shape mismatch')
        self.args = (A, B)
        super(_SumLinearOperator, self).__init__(A.shape, _get_dtype([A, B]),
                                                 self._dot, self._dot_adj)

    def _dot(self, X):
        return self.args[0].dot(X) + self.args[1].dot(X)

    def _dot_adj(self, X):
        return self.args[0].dot_adj(X) + self.args[1].dot_adj(X)


class _ProductLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or \
                not isinstance(B, LinearOperator):
            raise LinearOperatorError('both operands have to be a '
                                      'LinearOperator')
        if A.shape[1] != B.shape[0]:
            raise LinearOperatorError('shape mismatch')
        self.args = (A, B)
        super(_ProductLinearOperator, self).__init__((A.shape[0], B.shape[1]),
                                                     _get_dtype([A, B]),
                                                     self._dot, self._dot_adj)

    def _dot(self, X):
        return self.args[0].dot(self.args[1].dot(X))

    def _dot_adj(self, X):
        return self.args[1].dot_adj(self.args[0].dot_adj(X))


class _ScaledLinearOperator(LinearOperator):
    def __init__(self, A, alpha):
        if not isinstance(A, LinearOperator):
            raise LinearOperatorError('LinearOperator expected as A')
        if not numpy.isscalar(alpha):
            raise LinearOperatorError('scalar expected as alpha')
        self.args = (A, alpha)
        super(_ScaledLinearOperator, self).__init__(
            A.shape, _get_dtype([A], [type(alpha)]), self._dot, self._dot_adj)

    def _dot(self, X):
        return self.args[1] * self.args[0].dot(X)

    def _dot_adj(self, X):
        return numpy.conj(self.args[1]) * self.args[0].dot_adj(X)


class _PowerLinearOperator(LinearOperator):
    def __init__(self, A, p):
        if not isinstance(A, LinearOperator):
            raise LinearOperatorError('LinearOperator expected as A')
        if A.shape[0] != A.shape[1]:
            raise LinearOperatorError('square LinearOperator expected as A')
        if not isintlike(p):
            raise LinearOperatorError('integer expected as p')
        self.args = (A, p)
        super(_PowerLinearOperator, self).__init__(A.shape, A.dtype,
                                                   self._dot, self._dot_adj)

    def _power(self, fun, X):
        res = X.copy()
        for i in range(self.args[1]):
            res = fun(res)
        return res

    def _dot(self, X):
        return self._power(self.args[0].dot, X)

    def _dot_adj(self, X):
        return self._power(self.args[0]._dot_adj, X)


class _AdjointLinearOperator(LinearOperator):
    def __init__(self, A):
        if not isinstance(A, LinearOperator):
            raise LinearOperatorError('LinearOperator expected as A')
        self.args = (A,)
        m, n = A.shape
        super(_AdjointLinearOperator, self).__init__((n, m), A.dtype,
                                                     A._dot_adj, A._dot)


class IdentityLinearOperator(LinearOperator):
    def __init__(self, shape):
        super(IdentityLinearOperator, self).__init__(shape, numpy.dtype(None),
                                                     self._dot, self._dot_adj)

    def _dot(self, X):
        return X

    def _dot_adj(self, X):
        return X


class ZeroLinearOperator(LinearOperator):
    def __init__(self, shape):
        super(ZeroLinearOperator, self).__init__(shape, numpy.dtype(None),
                                                 self._dot, self._dot_adj)

    def _dot(self, X):
        return numpy.zeros(X.shape)

    def _dot_adj(self, X):
        return numpy.zeros(X.shape)


class MatrixLinearOperator(LinearOperator):
    def __init__(self, A):
        super(MatrixLinearOperator, self).__init__(A.shape, A.dtype,
                                                   self._dot, self._dot_adj)
        self._A = A
        self._A_adj = None

    def _dot(self, X):
        return self._A.dot(X)

    def _dot_adj(self, X):
        if self._A_adj is None:
            self._A_adj = self._A.T.conj()
        return self._A_adj.dot(X)

    def __repr__(self):
        return self._A.__repr__()


class TimedLinearOperator(LinearOperator):
    def __init__(self, linear_operator, timer=None):
        self._linear_operator = linear_operator
        super(TimedLinearOperator, self).__init__(
            shape=linear_operator.shape,
            dtype=linear_operator.dtype,
            dot=linear_operator.dot,
            dot_adj=linear_operator.dot_adj
            )

        if timer is None:
            timer = Timer()
        self._timer = timer

    def dot(self, X):
        k = X.shape[1]
        if k == 0:
            return self._linear_operator.dot(X)
        with self._timer:
            ret = self._linear_operator.dot(X)
        self._timer[-1] /= k
        return ret

    def dot_adj(self, X):
        k = X.shape[1]
        if k == 0:
            return self._linear_operator.dot(X)
        k = X.shape[1]
        with self._timer:
            ret = self._linear_operator.dot_adj(X)
        self._timer[-1] /= k
        return ret


def strakos(n, l_min=0.1, l_max=100, rho=0.9):
    """Return the Strakoš matrix.

    See [Str92]_.
    """
    d = [l_min + (i-1)*1./(n-1)*(l_max-l_min)*(rho**(n-i))
         for i in range(1, n+1)]
    return numpy.diag(d)


def gap(lamda, sigma, mode='individual'):
    """Compute spectral gap.

    Useful for eigenvalue/eigenvector bounds. Computes the gap
    :math:`\delta\geq 0` between two sets of real numbers ``lamda`` and
    ``sigma``. The gap can be computed in several ways and may not exist, see
    the ``mode`` parameter.

    :param lamda: a non-empty set
      :math:`\Lambda=\{\lambda_1,\ldots,\lambda_n\}` given as a single real
      number or a list or ``numpy.array`` with real numbers.
    :param sigma: a non-empty set :math:`\Sigma=\{\sigma_1,\ldots,\sigma_m\}`.
      See ``lamda``.
    :param mode: (optional). Defines how the gap should be computed. May be one
      of

      * ``'individual'`` (default):
        :math:`\delta=\min_{\substack{i\in\{1,\ldots,n\}\\\\j\in\{1,\ldots,m\}}} |\lambda_i - \sigma_j|`.
        With this mode, the gap is always be defined.
      * ``'interval'``: determine the maximal :math:`\delta` such that
        :math:`\Sigma\subset\mathbb{R}\setminus[\min_{\lambda\in\Lambda}\lambda-\delta,\max_{\lambda\in\Lambda}\lambda+\delta]`.
        If the gap does not exists, ``None`` is returned.

    :return: :math:`\delta` or ``None``.
    """
    # sanitize input
    if numpy.isscalar(lamda):
        lamda = [lamda]
    lamda = numpy.array(lamda)
    if numpy.isscalar(sigma):
        sigma = [sigma]
    sigma = numpy.array(sigma)

    if not numpy.isreal(lamda).all() or not numpy.isreal(sigma).all():
        raise ArgumentError('complex spectra not yet implemented')

    if mode == 'individual':
        return numpy.min(numpy.abs(numpy.reshape(lamda, (len(lamda), 1))
                         - numpy.reshape(sigma, (1, len(sigma)))))
    elif mode == 'interval':
        lamda_min, lamda_max = numpy.min(lamda), numpy.max(lamda)
        # determine all values in sigma<lamda_min or >lamda_max
        sigma_lo = sigma <= lamda_min
        sigma_hi = sigma >= lamda_max
        # is a sigma value in lamda interval?
        if not numpy.all(sigma_lo + sigma_hi):
            return None
        delta = numpy.Infinity
        if numpy.any(sigma_lo):
            delta = lamda_min - numpy.max(sigma[sigma_lo])
        if numpy.any(sigma_hi):
            delta = numpy.min([delta, numpy.min(sigma[sigma_hi]) - lamda_max])

        return delta


class Interval(object):
    '''A closed interval on the real line (may also be a single point).'''
    def __init__(self, left, right=None):
        if right is None:
            right = left
        if left > right:
            raise ArgumentError('left > right not allowed.')
        self.left = left
        self.right = right

    def __and__(self, other):
        '''Return intersection interval or None'''
        left = numpy.max([self.left, other.left])
        right = numpy.min([self.right, other.right])
        if left <= right:
            return Interval(left, right)
        return None

    def __or__(self, other):
        '''Return union of intervals if they intersect or None.'''
        if self & other:
            left = numpy.min([self.left, other.left])
            right = numpy.max([self.right, other.right])
            return Interval(left, right)
        return None

    def __repr__(self):
        return '[{0},{1}]'.format(self.left, self.right)

    def contains(self, alpha):
        '''Returns True if alpha is an element of the interval.'''
        return self.left <= alpha and alpha <= self.right

    def distance(self, other):
        '''Returns the distance to other (0 if intersection is nonempty).'''
        if self & other:
            return 0
        return numpy.max([other.left-self.right, self.left-other.right])


class Intervals(object):
    '''A set of non-intersecting intervals.'''
    def __init__(self, intervals=None):
        self.intervals = set()
        if intervals is not None:
            for interval in intervals:
                self.add(interval)

    def add(self, new):
        intersecting = {interval for interval in self.intervals
                        if interval & new}
        # compute union
        for interval in intersecting:
            new = new | interval
        self.intervals = self.intervals.difference(intersecting)
        self.intervals.add(new)

    def contains(self, alpha):
        for interval in self.intervals:
            if interval.contains(alpha):
                return True
        return False

    def get_endpoints(self):
        endpoints = []
        for interval in self.intervals:
            if interval.left == interval.right:
                endpoints += [interval.left]
            else:
                endpoints += [interval.left, interval.right]
        endpoints.sort()
        return endpoints

    def __len__(self):
        return len(self.intervals)

    def __iter__(self):
        return self.intervals.__iter__()

    def __repr__(self):
        return ', '.join([interval.__repr__() for interval in self.intervals])

    def min(self):
        if self.__len__() == 0:
            return ArgumentError('empty set has no minimum.')
        return numpy.min(list(map(lambda i: i.left, self.intervals)))

    def max(self):
        if self.__len__() == 0:
            return ArgumentError('empty set has no maximum.')
        return numpy.max(list(map(lambda i: i.right, self.intervals)))

    def min_pos(self):
        '''Returns minimal positive value or None.'''
        if self.__len__() == 0:
            return ArgumentError('empty set has no minimum positive value.')
        if self.contains(0):
            return None
        positive = [interval for interval in self.intervals
                    if interval.left > 0]
        if len(positive) == 0:
            return None
        return numpy.min(list(map(lambda i: i.left, positive)))

    def max_neg(self):
        '''Returns maximum negative value or None.'''
        if self.__len__() == 0:
            return ArgumentError('empty set has no maximum negative value.')
        if self.contains(0):
            return None
        negative = [interval for interval in self.intervals
                    if interval.right < 0]
        if len(negative) == 0:
            return None
        return numpy.max(list(map(lambda i: i.right, negative)))

    def min_abs(self):
        '''Returns minimum absolute value.'''
        if self.__len__() == 0:
            return ArgumentError('empty set has no minimum absolute value.')
        if self.contains(0):
            return 0
        return numpy.min([numpy.abs(val)
                          for val in [self.max_neg(), self.min_pos()]
                          if val is not None])

    def max_abs(self):
        '''Returns maximum absolute value.'''
        if self.__len__() == 0:
            return ArgumentError('empty set has no maximum absolute value.')
        return numpy.max(numpy.abs([self.max(), self.min()]))


class BoundCG(object):
    r'''CG residual norm bound.

    Computes the :math:`\kappa`-bound for the CG error :math:`A`-norm when the
    eigenvalues of the operator are given, see [LieS13]_.

    :param evals: an array of eigenvalues
      :math:`\lambda_1,\ldots,\lambda_N\in\mathbb{R}`. The eigenvalues will be
      sorted internally such that
      :math:`0=\lambda_1=\ldots=\lambda_{t-1}<\lambda_t\leq\ldots\lambda_N`
      for :math:`t\in\mathbb{N}`.
    :param steps: (optional) the number of steps :math:`k` to compute the bound
      for. If steps is ``None`` (default), then :math:`k=N` is used.

    :return: array :math:`[\eta_0,\ldots,\eta_k]` with

      .. math::

         \eta_n = 2 \left(
           \frac{\sqrt{\kappa_{\text{eff}}} - 1}
           {\sqrt{\kappa_{\text{eff}}} + 1}
         \right)^n
         \quad\text{for}\quad
         n\in\{0,\ldots,k\}

      where :math:`\kappa_{\text{eff}}=\frac{\lambda_N}{\lambda_t}`.
    '''
    def __init__(self, evals, exclude_zeros=False):
        '''Initialize with array/list of eigenvalues or Intervals object.'''
        if isinstance(evals, Intervals):
            evals = [evals.min(), evals.max()]
            if evals[0] <= 0:
                raise AssumptionError(
                    'non-positive eigenvalues not allowed with intervals')

        # empty spectrum?
        if len(evals) == 0:
            raise AssumptionError('empty spectrum not allowed')

        # all evals real?
        if not numpy.isreal(evals).all():
            raise AssumptionError('non-real eigenvalues not allowed')

        # sort
        evals = numpy.sort(numpy.array(evals, dtype=numpy.float))

        # normalize
        evals /= evals[-1]

        if exclude_zeros is False and not (evals > 1e-15).all():
            raise AssumptionError(
                'non-positive eigenvalues not allowed (use exclude_zeros?)')

        # check that all are non-negative
        assert(evals[0] > -1e-15)

        # compute effective condition number
        kappa = 1/numpy.min(evals[evals > 1e-15])
        self.base = (numpy.sqrt(kappa)-1) / (numpy.sqrt(kappa)+1)

    def eval_step(self, step):
        '''Evaluate bound for given step.'''
        return 2 * self.base**step

    def get_step(self, tol):
        '''Return step at which bound falls below tolerance.'''
        return numpy.log(tol/2.)/numpy.log(self.base)


class BoundMinres(object):
    r'''MINRES residual norm bound.

    Computes a bound for the MINRES residual norm when the eigenvalues of the
    operator are given, see [Gre97]_.

    :param evals: an array of eigenvalues
      :math:`\lambda_1,\ldots,\lambda_N\in\mathbb{R}`. The eigenvalues will be
      sorted internally such that
      :math:`\lambda_1\leq\ldots\lambda_s<0=\lambda_{s+1}=\ldots=\lambda_{s+t-1}<\lambda_t\leq\ldots\lambda_N`
      for :math:`s,t\in\mathbb{N}` and :math:`s<t`.
    :param steps: (optional) the number of steps :math:`k` to compute the bound
      for. If steps is ``None`` (default), then :math:`k=N` is used.

    :return: array :math:`[\eta_0,\ldots,\eta_k]` with

      .. math::

         \eta_n = 2 \left(
         \frac{ \sqrt{|\lambda_1\lambda_N|} - \sqrt{|\lambda_s\lambda_t|}}
         { \sqrt{|\lambda_1\lambda_N|} + \sqrt{|\lambda_s\lambda_t|}}
         \right)^{\left[\frac{n}{2}\right]}
         \quad\text{for}\quad
         n\in\{0,\ldots,k\}

      if :math:`s>0`. If :math:`s=0`, i.e., if the eigenvalues are
      non-negative, then the result of :py:meth:`bound_cg` is returned.
    '''
    def __new__(cls, evals):
        '''Use BoundCG if all eigenvalues are non-negative.'''
        pos = False
        if isinstance(evals, Intervals):
            if evals.min() > 0:
                pos = True
        elif (numpy.array(evals) > -1e-15).all():
            pos = True
        if pos:
            return BoundCG(evals)
        return super(BoundMinres, cls).__new__(cls, evals)

    def __init__(self, evals):
        '''Initialize with array/list of eigenvalues or Intervals object.'''
        if isinstance(evals, Intervals):
            if evals.contains(0):
                raise AssumptionError(
                    'zero eigenvalues not allowed with intervals')
            evals = [val for val in [evals.min(), evals.max_neg(),
                                     evals.min_pos(), evals.max()]
                     if val is not None]

        # empty spectrum?
        if len(evals) == 0:
            raise AssumptionError('empty spectrum not allowed')

        # all evals real?
        if not numpy.isreal(evals).all():
            raise AssumptionError('non-real eigenvalues not allowed')

        # sort
        evals = numpy.sort(numpy.array(evals, dtype=numpy.float))

        # normalize and categorize evals
        evals /= numpy.max(numpy.abs(evals))
        negative = evals < -1e-15
        positive = evals > 1e-15

        lambda_1 = numpy.min(evals[negative])
        lambda_s = numpy.max(evals[negative])
        lambda_t = numpy.min(evals[positive])
        lambda_N = numpy.max(evals[positive])

        a = numpy.sqrt(numpy.abs(lambda_1*lambda_N))
        b = numpy.sqrt(numpy.abs(lambda_s*lambda_t))

        self.base = (a-b) / (a+b)

    def eval_step(self, step):
        '''Evaluate bound for given step.'''
        return 2 * self.base**numpy.floor(step/2.)

    def get_step(self, tol):
        '''Return step at which bound falls below tolerance. '''
        return 2 * numpy.log(tol/2.)/numpy.log(self.base)


def bound_perturbed_gmres(pseudo, p, epsilon, deltas):
    '''Compute GMRES perturbation bound based on pseudospectrum

    Computes the GMRES bound from [SifEM13]_.
    '''
    if not numpy.all(numpy.array(deltas) > epsilon):
        raise ArgumentError('all deltas have to be greater than epsilon')

    bound = []
    for delta in deltas:
        # get boundary paths
        paths = pseudo.contour_paths(delta)

        # get vertices on boundary
        vertices = paths.vertices()

        # evaluate polynomial
        supremum = numpy.max(numpy.abs(p(vertices)))

        # compute bound
        bound.append(epsilon/(delta-epsilon)
                     * paths.length()/(2*numpy.pi*delta)
                     * supremum)
    return bound


class NormalizedRootsPolynomial(object):
    def __init__(self, roots):
        r'''A polynomial with specified roots and p(0)=1.

        Represents the polynomial

        .. math::

            p(\lambda) = \prod_{i=1}^n \left(1-\frac{\lambda}{\theta_i}\right).

        :param roots: array with roots :math:`\theta_1,\dots,\theta_n` of the
          polynomial and ``roots.shape==(n,)``.
        '''
        # check input
        roots = numpy.asarray(roots)
        if len(roots.shape) != 1:
            raise ArgumentError('one-dimensional array of roots expected.')
        self.roots = roots

    def minmax_candidates(self):
        '''Get points where derivative is zero.

        Useful for computing the extrema of the polynomial over an interval if
        the polynomial has real roots. In this case, the maximum is attained
        for one of the interval endpoints or a point from the result of this
        function that is contained in the interval.
        '''
        from numpy.polynomial import Polynomial as P
        p = P.fromroots(self.roots)
        return p.deriv(1).roots()

    def __call__(self, points):
        '''Evaluate polyonmial at given points.

        :param points: a point :math:`x` or array of points
          :math:`x_1,\dots,x_m` with ``points.shape==(m,)``.
        :returns: :math:`p(x)` or array of shape ``(m,)`` with
          :math:`p(x_1),\dots,p(x_m)`.
        '''
        # check input
        p = numpy.asarray(points)
        if len(p.shape) > 1:
            raise ArgumentError('scalar or one-dimensional array of points '
                                'expected.')
        n = self.roots.shape[0]
        vals = 1 - p/self.roots.reshape(n, 1)

        # prevent under/overflow by multiplying interlaced large and small
        # values
        for j in range(vals.shape[1]):
            sort_tmp = numpy.argsort(numpy.abs(vals[:, j]))
            sort = numpy.zeros((n,), dtype=numpy.int)
            mid = numpy.ceil(float(n)/2)
            sort[::2] = sort_tmp[:mid]
            sort[1::2] = sort_tmp[mid:][::-1]
            vals[:, j] = vals[sort, j]

        # form product of each column
        vals = numpy.prod(vals, axis=0)

        if numpy.isscalar(points):
            return numpy.asscalar(vals)
        return vals


def get_residual_norms(H, self_adjoint=False):
    '''Compute relative residual norms from Hessenberg matrix.

    It is assumed that the initial guess is chosen as zero.'''
    H = H.copy()
    n_, n = H.shape
    y = numpy.eye(n_, 1, dtype=H.dtype)
    resnorms = [1.]
    for i in range(n_-1):
        G = Givens(H[i:i+2, [i]])
        if self_adjoint:
            H[i:i+2, i:i+3] = G.apply(H[i:i+2, i:i+3])
        else:
            H[i:i+2, i:] = G.apply(H[i:i+2, i:])
        y[i:i+2] = G.apply(y[i:i+2])
        resnorms.append(numpy.abs(y[i+1, 0]))
    if n_ == n:
        resnorms.append(0.)
    return numpy.array(resnorms)
