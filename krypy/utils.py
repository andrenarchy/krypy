# -*- coding: utf8 -*-
'''
Collection of standard functions.

This method provides functions like inner products, norms, ...
'''

import numpy
import warnings
from scipy.sparse import issparse, isspmatrix
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.sputils import upcast
from scipy.linalg import eigh

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
    '''returns common dtype of numpy and scipy objects

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
    '''Take a (n,) ndarray and return it as (n,1) ndarray'''
    return numpy.reshape(x, (x.shape[0], 1))

# ===================================================================
def shape_vecs(*args):
    '''Reshape all ndarrays with shape (n,) to shape (n,1)

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
    transpose. In Octave/MATLAB notation ip_euclid(X,Y) == X'*Y.

    Arguments:
        X:  array of shape [N,m]
        Y:  array of shape [N,n]

    Returns:
        ip_euclid: array of shape [m,n] with X^H * Y
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
    ritz_values, ritz_coeffs = eigh(ritzmat)

    # Calculate the Ritz-residuals in a smart way.
    ritz_res_norm = numpy.empty(nritz)
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
        """Compute Householder transformation for given vector

        Initialize Householder transformation :math:`H` such that 
        :math:`Hx = \\alpha e_1` with :math:`|\\alpha|=\|x\|_2`

        The algorithm is a combination of Algorithm 5.1.1 on page 236
        and the treatment of the complex case in Section 5.1.13 on page 243
        in Golub, Van Loan. Matrix computations. Fourth Edition. 2013.
        """
        # make sure that x is a vector ;)
        if len(x.shape)!=2 or x.shape[1]!=1:
            raise ValueError('x is not a vector of dim (N,1)')

        v = x.copy()

        gamma = numpy.asscalar(v[0])
        sigma = numpy.linalg.norm(v[1:],2)
        self.xnorm = numpy.sqrt(numpy.abs(gamma)**2+sigma**2)

        v[0] = 1
        # is x the multiple of first unit vector?
        if sigma==0:
            beta = 0
        else:
            if gamma==0:
                v[0] = -sigma
            else:
                v[0] = gamma + gamma/numpy.abs(gamma) * self.xnorm
            beta = 2

        self.v = v/numpy.sqrt( numpy.abs(v[0])**2 + sigma**2)
        self.beta = beta

    def apply(self, x):
        # make sure that x is a (N,*) matrix
        if len(x.shape)!=2:
            raise ValueError('x is not a matrix of shape (N,*)')
        return x - self.beta * self.v * numpy.dot(self.v.T.conj(), x)

    def norm(self):
        return self.xnorm

    def matrix(self):
        n = self.v.shape[0]
        return numpy.eye(n,n) - self.beta * numpy.dot(self.v, self.v.T.conj())

# ===================================================================
class Givens:
    def __init__(self, x):
        """Compute Givens rotation for given vector

        Computes Givens rotation G such that
            G*x=[c        s] [a] = [r]
                [-conj(s) c] [b]   [0]
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
        return numpy.dot(self.G, x)

def arnoldi(A, v, maxiter=None, ortho='mgs', inner_product=ip_euclid):
    """Compute Arnoldi relation

    Computes V and H such that :math:`AV_{n}=V_{n+1}\\underline{H}_n`

    :param ortho: may be 'mgs' (Modified Gram-Schmidt), 'dmgs' (double Modified Gram-Schmidt), 'house' (Householder)

    If the Householder orthogonalization is used, the inner product has to be
    the Euclidean inner product.  It's unclear to me (andrenarchy), how a
    variant of the Householder QR algorithm can be used with a non-Euclidean
    inner product. Compare
    http://math.stackexchange.com/questions/433644/is-householder-orthogonalization-qr-practicable-for-non-euclidean-inner-products
    """
    dtype = find_common_dtype(A, v)
    N = v.shape[0]
    A = get_linearoperator((N,N), A)
    V = numpy.zeros([N, maxiter+1], dtype=dtype) # Arnoldi basis
    H = numpy.zeros([maxiter+1, maxiter], dtype=dtype) # Hessenberg matrix

    if ortho=='house':
        if inner_product!=ip_euclid:
            raise ValueError('Only euclidean inner product allowed with Householder orthogonalization')
        houses = [House(v)]
        V[:,[0]] = houses[0].apply(numpy.eye(N,1, dtype=dtype))
    elif ortho in ['mgs','dmgs']:
        reorthos = 0
        if ortho=='dmgs':
            reorthos = 1
        V[:,[0]] = v / norm(v, inner_product=inner_product)
    else:
        raise ValueError('Unknown orthogonalization method "%s"' % ortho)

    for k in range(maxiter):
        Av = A * V[:,[k]]

        if ortho=='house':
            # Householder
            for j in range(k+1):
                Av[j:] = houses[j].apply(Av[j:])
            houses.append( House(Av[k+1:]) )
            Av[k+1:] = houses[-1].apply(Av[k+1:])
            H[:k+2,[k]] = Av[:k+2]

            vnew = numpy.zeros((N,1), dtype=dtype)
            vnew[k+1] = 1
            for j in range(k+1,-1,-1):
                vnew[j:] = houses[j].apply(vnew[j:])
            V[:,[k+1]] = vnew
        else:
            # (double) modified Gram-Schmidt
            for reortho in range(reorthos+1):
                for j in range(k+1):
                    alpha = inner_product(V[:, [j]], Av)[0,0]
                    H[j,k] += alpha
                    Av -= alpha * V[:,[j]]
            H[k+1,k] = norm(Av, inner_product=inner_product)
            V[:,[k+1]] = Av / H[k+1,k]

    return V, H
