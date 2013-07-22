# -*- coding: utf8 -*-
'''
Collection of standard functions.

This method provides functions like inner products, norms, ...
'''

import numpy
import warnings
import time
import scipy.linalg
from scipy.sparse import issparse, isspmatrix
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.sputils import upcast

# for Givens rotations
try:
    # scipy < 0.12, compare
    # http://docs.scipy.org/doc/scipy-dev/reference/release.0.12.0.html#fblas-and-cblas
    import scipy.linalg.blas as blas
    blas = blas.cblas
except ImportError:
    import scipy.linalg.blas as blas

# ===================================================================
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
                warnings.warn('object %s does not have a dtype.' % arg.__repr__)
    return numpy.find_common_type(dtypes, [])

# ===================================================================
def shape_vec(x):
    '''Take a (n,) ndarray and return it as (n,1) ndarray.'''
    return numpy.reshape(x, (x.shape[0], 1))

# ===================================================================
def shape_vecs(*args):
    '''Reshape all ndarrays with ``shape==(n,)`` to ``shape==(n,1)``.

    Recognizes ndarrays and ignores all others.'''
    ret_args = []
    flat_vecs = True
    for arg in args:
        if type(arg) is numpy.ndarray:
            if len(arg.shape)==1:
                arg = shape_vec(arg)
            else:
                flat_vecs = False
        ret_args.append(arg)
    return flat_vecs, ret_args

# ===================================================================
def ip_euclid(X,Y):
    '''Euclidean inner product.

    numpy.vdot only works for vectors and numpy.dot does not use the conjugate
    transpose.

    :param X: numpy array with ``shape==(N,m)``
    :param Y: numpy array with ``shape==(N,n)``

    :return: numpy array :math:`X^* Y` with ``shape==(m,n)``.
    '''
    return numpy.dot(X.T.conj(), Y)

# ===================================================================
def norm_squared( x, Mx = None, inner_product = ip_euclid ):
    '''Compute the norm^2 w.r.t. to a given scalar product.'''
    assert( len(x.shape)==2 )
    if Mx is None:
        rho = inner_product(x, x)
    else:
        assert( len(Mx.shape)==2 )
        rho = inner_product(x, Mx)

    if rho.shape==(1,1):
        if abs(rho[0,0].imag) > abs(rho[0,0]) *1e-10 or rho[0,0].real < 0.0:
            raise ValueError( '<x,Mx> = %g. M not positive definite?' % rho[0,0] )

    return numpy.linalg.norm(rho, 2)


# ===================================================================
def norm( x, Mx = None, inner_product = ip_euclid ):
    '''Compute the norm w.r.t. to a given scalar product.'''
    if Mx is None and inner_product==ip_euclid:
        return numpy.linalg.norm(x, 2)
    else:
        return numpy.sqrt(norm_squared( x, Mx = Mx, inner_product = inner_product ) )

# ===================================================================
def get_linearoperator(shape, A):
    """Enhances aslinearoperator if A is None."""
    if A is None:
        identity = lambda x: x
        return LinearOperator(shape, identity, identity, identity, numpy.double)
    else:
        return aslinearoperator(A)

# ===================================================================
def norm_MMlr(M, Ml, A, Mr, b, x0, yk, inner_product = ip_euclid):
    xk = x0 + Mr * yk
    r = b - A * xk
    Mlr = Ml * r
    # normalize residual before applying the preconditioner here.
    # otherwise MMlr can become 0 exactly (pyamg doesnt respect relative
    # residual)
    # TODO for testing: 2-norm
    norm_Mlr = norm(Mlr)
    if norm_Mlr == 0:
        MMlr = numpy.zeros( Mlr.shape )
        norm_MMlr = 0
    else:
        nMlr = Mlr / norm_Mlr
        nMMlr = M * nMlr
        MMlr = nMMlr * norm_Mlr
        norm_MMlr = norm(Mlr, MMlr, inner_product=inner_product)
    # return xk and ||M*Ml*(b-A*(x0+Mr*yk))||_{M^{-1}}
    return xk, Mlr, MMlr, norm_MMlr

# ===================================================================
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
        if len(x.shape)!=2 or x.shape[1]!=1:
            raise ValueError('x is not a vector of dim (N,1)')

        v = x.copy()

        gamma = numpy.asscalar(v[0])
        v[0] = 1
        if x.shape[0]==1:
            sigma = 0
            xnorm = numpy.abs(gamma)
            beta = 0
            alpha = 1 if gamma==0 else gamma/xnorm
        else:
            sigma = numpy.linalg.norm(v[1:],2)
            xnorm = numpy.sqrt(numpy.abs(gamma)**2+sigma**2)

            # is x the multiple of first unit vector?
            if sigma==0:
                beta = 0
                xnorm = numpy.abs(gamma)
                alpha = 1 if gamma==0 else gamma/xnorm
            else:
                beta = 2
                if gamma==0:
                    v[0] = -sigma
                    alpha = 1
                else:
                    v[0] = gamma + gamma/numpy.abs(gamma) * xnorm
                    alpha = - gamma/numpy.abs(gamma)

        self.xnorm = xnorm
        self.v = v/numpy.sqrt( numpy.abs(v[0])**2 + sigma**2)
        self.alpha = alpha
        self.beta = beta

    def apply(self, x):
        """Apply Householder transformation to vector x.

        Applies the Householder transformation efficiently to the given vector.
        """
        # make sure that x is a (N,*) matrix
        if len(x.shape)!=2:
            raise ValueError('x is not a matrix of shape (N,*)')
        if self.beta==0:
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
        return numpy.eye(n,n) - self.beta * numpy.dot(self.v, self.v.T.conj())

# ===================================================================
class Givens:
    def __init__(self, x):
        """Compute Givens rotation for provided vector x.

        Computes Givens rotation
        :math:`G=\\begin{bmatrix}c&s\\\\-\\overline{s}&c\\end{bmatrix}`
        such that
        :math:`Gx=\\begin{bmatrix}r\\\\0\\end{bmatrix}`.
        """
        # make sure that x is a vector ;)
        if x.shape!=(2,1):
            raise ValueError('x is not a vector of shape (2,1)')

        a = numpy.asscalar(x[0])
        b = numpy.asscalar(x[1])
        # real vector
        if numpy.isrealobj(x):
            c, s = blas.drotg(a,b)
        # complex vector
        else:
            c, s = blas.zrotg(a,b)

        self.c = c
        self.s = s
        self.r = c*a + s*b
        self.G = numpy.array([[c, s], [-numpy.conj(s), c]])

    def apply(self, x):
        """Apply Givens rotation to vector x."""
        return numpy.dot(self.G, x)

# ===================================================================
class Projection:
    def __init__(self, X,
            Y = None,
            inner_product = ip_euclid,
            ipYX = None,
            ipYXinv = 'explicit'):
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
        is equivalent to ``inner_product(X,Y)`` being nonsingular.

        :param X: array with ``shape==(N,k)`` and
            :math:`\\operatorname{rank}(X)=k`.
        :param Y: (optional) ``None`` or array with ``shape==(N,k)`` and
            :math:`\\operatorname{rank}(X)=k`. If Y is ``None`` then Y is
            set to X which means that the resulting projection is orthogonal.
        :param inner_product: (optional) the inner product to use, default is
            :py:meth:`ip_euclid`.
        :param ipYX: (optional) ``None`` or array with precomputed
            ``inner_product(Y,X)`` (``shape==(k,k)``).
        :param ipYXinv: (optional) may be one of

            * ``'explicit'``: the inverse of ``ipYX`` is explicitly computed.
            * ``'ondemand'``: ``numpy.linalg.solve()`` is called for each
                application of the projection.
            * array with precomputed inverse of ``ipYX``.
        """
        Y = X if Y is None else Y   # default: orthogonal projection

        if len(X.shape) != 2:
            raise ValueError('X does not have shape==(N,k)')
        if X.shape != Y.shape:
            raise ValueError('X and Y have different shapes')

        (N, k) = X.shape
        self.inner_product = inner_product
        # special case of orthogonal projection in Euclidean inner product
        # (the case of non-Euclidean inner products is handled in the general
        # case below)
        if Y is X and inner_product==ip_euclid:
            X = scipy.linalg.orth(X)
            Y = X
            ipYX = None
            ipYXinv = None
        # general case
        else:
            ipYX = inner_product(Y, X) if ipYX is None else ipYX
            if ipYX.shape != (k,k):
                raise ValueError('ipYX does not have shape==(k,k)')

            if ipYXinv == 'explicit':
                ipYXinv = numpy.linalg.inv(ipYX)
            elif ipYXinv == 'ondemand':
                pass
            elif type(ipYXinv) == numpy.ndarray:
                if ipYXinv.shape != (k,k):
                    raise ValueError('ipYXinv does not have shape==(k,k)')
            else:
                raise ValueError('unknown value for ipYXinv: {0}'.format(ipYXinv))

        # save computed quantities in object
        self.X = X
        self.Y = Y
        self.inner_product = inner_product
        self.ipYX = ipYX
        self.ipYXinv = ipYXinv

    def apply(self, z):
        """Apply the projection to an array.

        The computation is carried out without explicitly forming the
        matrix corresponding to the projection (which would be an array with
        ``shape==(N,N)``).

        :param z: array with ``shape==(N,m)``.

        :return: :math:`P_{\\mathcal{X},\\mathcal{Y}^\\perp} z =
            X \\langle Y,X\\rangle^{-1} \\langle Y, z\\rangle`.
        """

        z = self.inner_product(self.Y, z)
        if self.ipYXinv is not None:
            if type(self.ipYXinv) == numpy.ndarray:
                z = numpy.dot(self.ipYXinv, z)
            else: # ondemand
                z = numpy.linalg.solve(self.ipYX, z)
        return numpy.dot(self.X, z)

    def apply_complement(self, z):
        """Apply the complementary projection to an array.

        :param z: array with ``shape==(N,m)``.

        :return: :math:`P_{\\mathcal{Y}^\\perp,\\mathcal{X}}z =
            z - P_{\\mathcal{X},\\mathcal{Y}^\\perp} z`.
        """
        return z - self.apply(z)

    def matrix(self):
        """Builds matrix representation of projection.

        Builds the matrix representation
        :math:`P = X \\langle Y,X\\rangle^{-1} \\langle Y, I_N\\rangle`.

        **Use with care!** This routine may be helpful for testing purposes but
        should not be used in production codes for high dimensions since
        the resulting matrix is dense.
        """
        return self.apply(numpy.eye(self.X.shape[0]))

# ===================================================================
def qr(X, inner_product = ip_euclid, reorthos=1):
    """QR factorization with customizable inner product.

    :param X: array with ``shape==(N,k)``
    :param inner_product: (optional) inner product in which the resulting Q
      matrix should be orthogonal in. Defaults to :py:meth:`ip_euclid`.
    :param reorthos: (optional) numer of reorthogonalizations. Defaults to
      1 (i.e. 2 runs of modified Gram-Schmidt) which should be enough in most
      cases (TODO: add reference).

    :return: Q, R where :math:`X=QR` with :math:`\\langle Q,Q \\rangle=I_k` and
      R upper triangular.
    """
    if inner_product==ip_euclid and X.shape[1]>0:
        return scipy.linalg.qr(X, mode='economic')
    else:
        (N,k) = X.shape
        Q = X.copy()
        R = numpy.zeros((k,k), dtype=X.dtype)
        for i in range(k):
            for reortho in range(reorthos+1):
                for j in range(i):
                    alpha = inner_product(Q[:, [j]], Q[:,[i]])[0,0]
                    R[j,i] += alpha
                    Q[:,[i]] -= alpha * Q[:,[j]]
            R[i,i] = norm(Q[:,[i]], inner_product=inner_product)
            if R[i,i] >= 1e-15:
                Q[:,[i]] /= R[i,i]
        return Q, R

# ===================================================================
def angles(F, G, inner_product = ip_euclid, compute_vectors=False):
    """Principal angles between two subspaces.

    This algorithm is based on algorithm 6.2 in `Knyazev, Argentati. Principal
    angles between subspaces in an A-based scalar product: algorithms and
    perturbation estimates. 2002.` This algorithm can also handle small angles
    (in contrast to the naive cosine-based svd algorithm).

    :param F: array with ``shape==(N,k)``.
    :param G: array with ``shape==(N,l)``.
    :param inner_product: (optional) angles are computed with respect to this
      inner product. Defaults to :py:meth:`ip_euclid`.
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
        reverse=True
        F, G = G, F

    QF, _ = qr(F, inner_product=inner_product)
    QG, _ = qr(G, inner_product=inner_product)

    # one or both matrices empty? (enough to check G here)
    if G.shape[1]==0:
        theta = numpy.ones(F.shape[1])*numpy.pi/2
        U = QF
        V = QG
    else:
        Y, s, Z = scipy.linalg.svd( inner_product(QF, QG) )
        Vcos = numpy.dot(QG, Z.T.conj())
        n_large = numpy.flatnonzero( (s**2) < 0.5).shape[0]
        n_small = s.shape[0] - n_large
        theta = numpy.r_[
                numpy.arccos(s[n_small:]), # [-i:] does not work if i==0
                numpy.ones(F.shape[1]-G.shape[1])*numpy.pi/2 ]
        if compute_vectors:
            Ucos = numpy.dot(QF, Y)
            U = Ucos[:, n_small:]
            V = Vcos[:, n_small:]

        if n_small>0:
            RG = Vcos[:,:n_small]
            S = RG - numpy.dot(QF, inner_product(QF, RG))
            _, R = qr(S, inner_product=inner_product)
            Y, u, Z = scipy.linalg.svd( R )
            theta = numpy.r_[
                    numpy.arcsin( u[::-1][:n_small]),
                    theta ]
            if compute_vectors:
                RF = Ucos[:,:n_small]
                Vsin = numpy.dot(RG, Z.T.conj())
                # next line is hand-crafted since the line from the paper does
                # not seem to work.
                Usin = numpy.dot(RF, numpy.dot( numpy.diag(1/s[:n_small]), numpy.dot(Z.T.conj(), numpy.diag(s[:n_small]))))
                U = numpy.c_[Usin, U]
                V = numpy.c_[Vsin, V]

    if compute_vectors:
        if reverse:
            U, V = V, U
        return theta, U, V
    else:
        return theta

# ===================================================================
def hegedus(A, b, x0, M=None, Ml=None, inner_product=ip_euclid):
    """Rescale initial guess appropriately (Hegedüs trick).

    The Hegedüs trick rescales the initial guess to :math:`\\gamma_{\\min} x_0`
    such that

    .. math ::

      \\|r_0\\|_{M^{-1}}
      = \\| M M_l (b - A \\gamma_{\\min} x_0) \\|_{M^{-1}}
      = \\min_{\\gamma\\in\\mathbb{C}} \\| M M_l (b - A \\gamma x_0) \\|_{M^{-1}}
      \\leq \\| M M_l b \\|_{M^{-1}}.

    This is achieved by
    :math:`\\gamma_{\\min} = \\frac{\\langle z, M M_l b \\rangle_{M^{-1}}}{\\|z\\|_{M^{-1}}^2}` for
    :math:`z=M M_l A x_0` because then :math:`r_0=P_{z^\\perp}b`. (Note that
    the right hand side of formula (5.8.16) in *Liesen, Strakos. Krylov
    subspace methods. 2013.* has to be complex conjugated.)

    The parameters are the parameters you want to pass to
    :py:meth:`~krypy.linsys.gmres`,
    :py:meth:`~krypy.linsys.minres` or
    :py:meth:`~krypy.linsys.cg`.

    :return: the adapted initial guess with the above property.
    """
    N = len(b)
    shape = (N,N)
    A = get_linearoperator(shape, A)
    M = get_linearoperator(shape, M)
    Ml = get_linearoperator(shape, Ml)

    MlAx0 = Ml*(A*x0)
    z = M*MlAx0
    znorm2 = inner_product(z, MlAx0)
    if znorm2 <= 1e-15:
        return numpy.zeros((N,1))
    gamma = inner_product(z, Ml*b) / znorm2
    return gamma*x0

# ===================================================================
def arnoldi(A, v, maxiter=None, ortho='mgs', inner_product=ip_euclid):
    """Arnoldi algorithm.

    Computes V and H such that :math:`AV_n=V_{n+1}\\underline{H}_n`.
    If the Krylov subspace becomes A-invariant then V and H are truncated such
    that :math:`AV_n = V_n H_n`.

    :param A: a linear operator that can be used with scipy's aslinearoperator
        with ``shape==(N,N)``.
    :param v: the initial vector with ``shape==(N,1)``.
    :param maxiter: (optional) maximal number of iterations. Default: N.
    :param ortho: (optional) orthogonalization algorithm: may be one of

        * ``'mgs'``: modified Gram-Schmidt (default).
        * ``'dmgs'``: double Modified Gram-Schmidt.
        * ``'house'``: Householder.
    :param inner_product: (optional) the inner product to use (has to be the
        Euclidean inner product if ``ortho=='house'``). It's unclear to me
        (andrenarchy), how a variant of the Householder QR algorithm can be
        used with a non-Euclidean inner product. Compare
        http://math.stackexchange.com/questions/433644/is-householder-orthogonalization-qr-practicable-for-non-euclidean-inner-products
    """
    dtype = find_common_dtype(A, v)
    N = v.shape[0]
    A = get_linearoperator((N,N), A)
    V = numpy.zeros((N, maxiter+1), dtype=dtype) # Arnoldi basis
    H = numpy.zeros((maxiter+1, maxiter), dtype=dtype) # Hessenberg matrix

    if ortho=='house':
        if inner_product!=ip_euclid:
            raise ValueError('Only euclidean inner product allowed with Householder orthogonalization')
        houses = [House(v)]
        V[:,[0]] = v / numpy.linalg.norm(v, 2)
    elif ortho in ['mgs','dmgs']:
        reorthos = 0
        if ortho=='dmgs':
            reorthos = 1
        V[:,[0]] = v / norm(v, inner_product=inner_product)
    else:
        raise ValueError('Unknown orthogonalization method "%s"' % ortho)

    invariant = False
    for k in range(maxiter):
        Av = A * V[:,[k]]

        if ortho=='house':
            # Householder
            for j in range(k+1):
                Av[j:] = houses[j].apply(Av[j:])
                Av[j] *= numpy.conj(houses[j].alpha)
            if k+1<N:
                houses.append( House(Av[k+1:]) )
                Av[k+1:] = houses[-1].apply(Av[k+1:]) * numpy.conj(houses[-1].alpha)
                H[:k+2,[k]] = Av[:k+2]
            else:
                H[:k+1,[k]] = Av[:k+1]
            H[k+1,k] = numpy.abs(H[k+1,k]) # safe due to the multiplications with alpha
            if H[k+1,k] <= 1e-15:
                invariant = True
                break

            vnew = numpy.zeros((N,1), dtype=dtype)
            vnew[k+1] = 1
            for j in range(k+1,-1,-1):
                vnew[j:] = houses[j].apply(vnew[j:])
            V[:,[k+1]] = vnew * houses[-1].alpha
        else:
            # (double) modified Gram-Schmidt
            for reortho in range(reorthos+1):
                for j in range(k+1):
                    alpha = inner_product(V[:, [j]], Av)[0,0]
                    H[j,k] += alpha
                    Av -= alpha * V[:,[j]]
            H[k+1,k] = norm(Av, inner_product=inner_product)
            if H[k+1,k] <= 5e-14:
                invariant = True
                break
            V[:,[k+1]] = Av / H[k+1,k]

    # if V is A-invariant: shorten Arnoldi vectors and Hessenberg matrix
    if invariant:
        V = V[:,:k+1]
        H = H[:k+1,:k+1]

    return V, H

# ===================================================================
def ritz(H, V=None, hermitian=False, type='ritz'):
    """Compute several kinds of Ritz pairs from an Arnoldi/Lanczos relation.

    This function computes Ritz, harmonic Ritz or improved harmonic Ritz values and
    vectors with respect to the Krylov subspace :math:`K_n(A,v)` from the
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
    if V is not None and V.shape[1]!=H.shape[0]:
        raise ValueError('shape mismatch with V and H')
    if not H.shape[0] in [n, n+1]:
        raise ValueError('H not of shape (n+1,n) or (n,n)')
    symmres = numpy.linalg.norm(H[:n,:] - H[:n,:].T.conj())
    if hermitian and symmres >= 5e-14:
        warnings.warn('Hessenberg matrix is not symmetric: |H-H^*|={0}'.format(symmres))

    # choose eig for Hermitian or non-Hermitian matrices
    eig = scipy.linalg.eigh if hermitian else scipy.linalg.eig

    if type=='ritz':
        theta, U = eig(H[:n,:])
        beta = 0 if H.shape[0]==n else H[-1,-1]
        resnorm = numpy.abs( beta * U[-1,:] )
    elif type=='harmonic':
        theta, U = eig(H[:n,:].T.conj(), numpy.dot(H.T.conj(),H))
        theta = 1/theta
        resnorm = []
        for i in range(n):
            U[:,i] /= numpy.linalg.norm(U[:,i], 2)
            resi = numpy.dot(H, U[:,i])
            if resi.dtype!=numpy.complex and theta.dtype==numpy.complex:
                resi = numpy.array(resi, dtype=numpy.complex)
            resi[:n] -= theta[i]*U[:,i]
            resnorm.append( numpy.linalg.norm( resi, 2) )
        resnorm = numpy.array(resnorm)
    elif type=='harmonic_improved':
        theta, U = eig(H[:n,:].T.conj(), numpy.dot(H.T.conj(),H))
        rho = []
        for i in range(n):
            U[:,i] /= numpy.linalg.norm(U[:,i], 2)
            rho.append( numpy.dot( U[:,i].T.conj(), numpy.dot(H[:n,:], U[:,i])) )
        theta = numpy.array(rho)
        resnorm = []
        for i in range(n):
            resi = numpy.dot(H, U[:,i])
            resi[:n] -= theta[i]*U[:,i]
            resnorm.append( numpy.linalg.norm( resi, 2) )
        resnorm = numpy.array(resnorm)
        pass
    else:
        raise ValueError('unknown Ritz type {0}'.format(type))

    if V is not None:
        return theta, U, resnorm, numpy.dot(V[:,:n], U)

    return theta, U, resnorm

class Timer:
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

    If you want to measure different types of code blocks you can use
    ``defaultdict`` as a timer manager: ::

        from collections import defaultdict

        tm = defaultdict(Timer)
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
    """
    def __init__(self):
        self.times = []
    def __enter__(self):
        self.tstart = time.time()
    def __exit__(self, a,b,c):
        self.times.append( time.time() - self.tstart )
    def __repr__(self):
        return self.times.__repr__()
