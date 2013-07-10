# -*- coding: utf8 -*-
'''
Collection of standard functions.

This method provides functions like inner products, norms, ...
'''

import numpy
import warnings
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
def get_projection(b, W, AW,
        # TODO: AW optional, supply A
        x0 = None, 
        inner_product = ip_euclid):
    """Get projection and appropriate initial guess for use in deflated methods.

    Arguments:
        W:  the basis vectors used for deflation (Nxk array).
        AW: A*W, where A is the operator of the linear algebraic system to be
            deflated. A has to be self-adjoint w.r.t. inner_product. Do not
            include the positive-definite preconditioner (argument M in MINRES)
            here. Let N be the dimension of the vector space the operator is
            defined on.
        b:  the right hand side of the linear system (array of length N).
        x0: the initial guess (array of length N).
        inner_product: the inner product also used for the deflated iterative
            method.

    Returns:
        P:  the projection to be used as _right_ preconditioner (e.g. Mr=P in
            MINRES). The preconditioned operator A*P is self-adjoint w.r.t.
            inner_product.
            P(x)=x - W*inner_product(W, A*W)^{-1}*inner_product(A*W, x)
        x0new: an adapted initial guess s.t. the deflated iterative solver
            does not break down (in exact arithmetics).
        AW: AW=A*W. This is returned in order to reduce the total number of
            matrix-vector multiplications with A.

    For nW = W.shape[1] = AW.shape[1] the computational cost is
    cost(get_projection): 2*cost(Pfun) + (nW^2)*IP
    cost(Pfun): nW*IP + (2/3)*nW^3 + nW*AXPY
    """
    # --------------------------------------------------------------------------
    def Pfun(x):
        '''Computes x - W * E\<AW,x>.'''
        return x - numpy.dot(W, numpy.dot(Einv, inner_product(AW, x)))
    # --------------------------------------------------------------------------

    # cost: (nW^2)*IP
    E = inner_product(W, AW)
    Einv = numpy.linalg.inv(E)

    # cost: nW*IP + (2/3)*nW^3
    EWb = numpy.dot(Einv, inner_product(W, b))

    # Define projection operator.
    N = len(b)
    if x0 is None:
        x0 = numpy.zeros((N,1))
    dtype = upcast(W.dtype, AW.dtype, b.dtype, x0.dtype)
    P = LinearOperator( [N,N], Pfun, matmat=Pfun,
                                            dtype=dtype)
    # Get updated x0.
    # cost: nW*AXPY + cost(Pfun)
    x0new = P*x0 +  numpy.dot(W, EWb)

    return P, x0new

# ===================================================================
def ritzh(Vfull, Hfull,
               W = None,
               AW = None,
               A = None,
               M = None, 
               Minv = None,
               inner_product = ip_euclid,
               ):
    """Compute Ritz pairs from a (possibly deflated) Lanczos procedure.

    Arguments
        W:  a Nxk array. W's columns must be orthonormal w.r.t. the
            M-inner-product (inner_product(M^{-1} W, W) = I_k).
        AW: contains the result of A applied to W (passed in order to reduce #
            of matrix-vector multiplications with A).
        Vfull: a Nxn array. Vfull's columns must be orthonormal w.r.t. the
            M-inner-product. Vfull and Hfull must be created with a (possibly
            deflated) Lanczos procedure (e.g. CG/MINRES). For example, Vfull
            and Hfull can be obtained from MINRES applied to a linear system
            with the operator A, the inner product inner_product, the HPD
            preconditioner M and the right preconditioner Mr set to the
            projection obtained with get_projection(W, AW, ...).
        Hfull: see Vfull.
        M:  The preconditioner used in the Lanczos procedure.

        The arguments thus have to fulfill the following equations:
            AW = A*W.
            M*A*Mr*Vfull[:,0:-1] = Vfull*Hfull,
                 where Mr=get_projection(W, AW,...,inner_product).
            inner_product( M^{-1} [W,Vfull], [W,Vfull] ) = I_{k+n}.

    Returns:
        ritz_vals: an array with n+k Ritz values.
        ritz_coeffs: a (n+k)x(n+k) array where the ritz_coeffs[:,i] are the
            coefficients (in the basis [W,Vfull]) of the vector corresponding
            the Ritz value ritz_vals[i]. The Ritz vectors then also are
            orthonormal w.r.t. the M-inner-product, that is inner_product(
            M^{-1}*ritz_vecs, ritz_vecs ) = I_{k+n}.
        ritz_res_norm: an array with n+k residual norms. ritz_res_norm[i] is
            the M^{-1}-norm of the residual
                M*A*ritz_vecs[:,i] - ritz_vals[i]*ritz_vecs[:,i].
            ritz_vals, ritz_vecs and norm_ritz_res are sorted s.t. the
            residual norms are ascending.

    Under the above assumptions, [W, Vfull] is orthonormal w.r.t. the
    M-inner-product. Then the Ritz pairs w.r.t. the operator M*A, the basis [W,
    Vfull[:,0:-1]] and the M-inner-product are computed. Also the M-norm of the
    Ritz pair residual is computed. The computation of the residual norms do
    not need the application of the operator A, but the preconditioner has to
    be applied to the basis W. The computation of the residual norm may be
    unstable (it seems as if residual norms below 1e-8 cannot be achieved...
    note that the actual residual may be lower!).
    """
    N = Vfull.shape[0]
    nVfull = Vfull.shape[1]
    nW = 0 if W is None else W.shape[1]
    # total number ritz vectors
    nritz = nW + nVfull - 1

    if issparse(Hfull):
        Hfull = Hfull.todense()

    if nW > 0:
        B1 = inner_product(AW, Vfull)   # can (and should) be obtained from projection
        
        E = inner_product(W, AW)        # ~
        Einv = numpy.linalg.inv(E) # can (and should) be obtained from earlier computation
        # Apply preconditioner to AW (I don't see a way to get rid of this! -- André).
        # cost: nW*APPLM
        MAW = M * AW
    else:
        W = numpy.zeros((N,0))
        AW = numpy.zeros((N,0))
        B1 = numpy.zeros((0,nVfull))
        E = numpy.zeros((0, 0))
        Einv = numpy.zeros((0, 0))
        MAW = numpy.zeros((W.shape[0], 0))
    
    B = B1[:, 0:-1]

    # Stack matrices appropriately: [E, B;
    #                                B', Hfull(1:end-1,:) + B'*Einv*B].
    ritzmat = numpy.r_[ numpy.c_[E,          B],
                     numpy.c_[B.T.conj(), Hfull[0:-1,:] + numpy.dot(B.T.conj(), numpy.dot(Einv, B))]
                   ]
    # Compute Ritz values / vectors.
    ritz_values, ritz_coeffs = scipy.linalg.eigh(ritzmat)

    # Calculate the Ritz-residuals in a smart way.
    ritz_res_norm = numpy.zeros(nritz)
    D = inner_product(AW, MAW)
    for i in range(nritz):
        w = ritz_coeffs[:W.shape[1],[i]]
        v = ritz_coeffs[W.shape[1]:,[i]]
        mu = ritz_values[i]

        # The following computes  <z, CC*z>_2  with
        #
        #z = numpy.r_[ -mu*w,
        #            w + numpy.dot(Einv, numpy.dot(B, v)),
        #            numpy.dot(Hfull, v) - numpy.r_[mu*v, numpy.zeros((1,1))] ]
        #z = numpy.reshape(z, (z.shape[0],1))
        # cost: (nW^2)*IP
        #zeros = numpy.zeros((nW,nVfull))
        #CC = numpy.r_[ numpy.c_[ numpy.eye(nW),     E,           zeros],
        #            numpy.c_[ E.T.conj(),     D,           B1],
        #            numpy.c_[ zeros.T.conj(), B1.T.conj(), numpy.eye(nVfull)]
        #          ]
        #
        # Attention:
        # CC should be HPD (and thus $<z, CC*z>_2 > 0$).
        # However, this only holds if the preconditioner M was solved exactly
        # (then inner_product(Minv*W,W)=I).
        # In this case, the computation seems to be stable.
        z0 = -mu * w
        z1 = w + numpy.dot(Einv, numpy.dot(B, v))
        z2 = numpy.dot(Hfull, v) - numpy.r_[mu*v, numpy.zeros((1,1))]
        res_ip = ip_euclid(z0, z0) + 2 * ip_euclid(z0, numpy.dot(E, z1)).real + ip_euclid(z1, numpy.dot(D, z1)) \
               + 2 * ip_euclid(z1, numpy.dot(B1, z2)).real + ip_euclid(z2, z2)
        res_ip = res_ip[0,0]

        if res_ip.imag > 1e-13:
            warnings.warn('res_ip.imag = %g > 1e-13. Preconditioner not symmetric?' % res_ip.imag)
        if res_ip.real < -1e-10:
            warnings.warn('res_ip.real = %g > 1e-10. Make sure that the basis is linearly independent and the preconditioner is solved \'exactly enough\'.' % res_ip.real)
        ritz_res_norm[i] = numpy.sqrt(abs(res_ip))

        # Explicit computation of residual (this part only works for M=I)
        #X = numpy.c_[W, Vfull[:,0:-1]]
        #V = numpy.dot(X, numpy.r_[w,v])
        #MAV = _apply(M,_apply(A, V))
        #res_explicit = MAV - ritz_values[i]*V
        #zz = inner_product(_apply(Minv, res_explicit), res_explicit)[0,0]
        #assert( zz.imag<1e-13 )
        #print abs(norm_ritz_res[i] - numpy.sqrt(abs(zz)))
        #print 'Explicit residual: %g' % numpy.sqrt(abs(zz))

    return ritz_values, ritz_coeffs, ritz_res_norm

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

def angles(X, Y, inner_product = ip_euclid):
    #TODO!
    pass

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
            houses.append( House(Av[k+1:]) )
            Av[k+1:] = houses[-1].apply(Av[k+1:]) * numpy.conj(houses[-1].alpha)
            H[:k+2,[k]] = Av[:k+2]
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

def ritz(H, V=None, hermitian=False, type='ritz'):
    """Compute several kinds of Ritz pairs from an Arnoldi/Lanczos relation.

    **WARNING: this method is still under construction!**

    This function computes Ritz, harmonic Ritz or harmonic-like Ritz values and
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
      ``'harmonic'`` or ``'harmonic_like'``. All choices of Ritz pairs fit
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
        Problems. 1998.*

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
        theta, U = eig(H[:n,:], numpy.dot(H.T.conj(),H))
        theta = 1/theta
        resnorm = []
        for i in range(n):
            U[:,i] /= numpy.linalg.norm(U[:,i], 2)
            resi = numpy.dot(H, U[:,i])
            resi[:n] -= theta[i]*U[:,i]
            resnorm.append( numpy.linalg.norm( resi, 2) )
        resnorm = numpy.array(resnorm)
    elif type=='harmonic_improved':
        theta, U = eig(H[:n,:], numpy.dot(H.T.conj(),H))
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
