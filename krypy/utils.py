# -*- coding: utf8 -*-
'''
Collection of standard functions.

This method provides functions like inner products, norms, ...
'''

import numpy
import warnings
from scipy.sparse import issparse, isspmatrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.sputils import upcast
from scipy.linalg import eigh

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
def ip(X,Y):
    '''Euclidean inner product.

    numpy.vdot only works for vectors and numpy.dot does not use the conjugate
    transpose. In Octave/MATLAB notation ip(X,Y) == X'*Y.

    Arguments:
        X:  array of shape [N,m]
        Y:  array of shape [N,n]

    Returns:
        ip: array of shape [m,n] with X^H * Y
    '''
    return numpy.dot(X.T.conj(), Y)

# ===================================================================
def norm_squared( x, Mx = None, inner_product = ip ):
    '''Compute the norm^2 w.r.t. to a given scalar product.'''
    assert( len(x.shape)==2 )
    assert( x.shape[1]==1 )
    if Mx is None:
        rho = inner_product(x, x)[0,0]
    else:
        assert( len(Mx.shape)==2 )
        assert( Mx.shape[1]==1 )
        rho = inner_product(x, Mx)[0,0]

#    if rho.imag != 0.0: #abs(rho.imag) > abs(rho) * 1.0e-10:
    if abs(rho.imag) > abs(rho) *1e-10:
        raise ValueError( 'M not positive definite?' )

    rho = rho.real

    if rho < 0.0:
        raise ValueError( '<x,Mx> = %g. M not positive definite?' % rho )

    return rho

# ===================================================================
def norm( x, Mx = None, inner_product = ip ):
    '''Compute the norm w.r.t. to a given scalar product.'''
    return numpy.sqrt(norm_squared( x, Mx = Mx, inner_product = inner_product ) )

# ===================================================================
def apply( A, x ):
    '''Implement A*x for different types of linear operators.'''
    if A is None:
        return x
    elif isinstance( A, numpy.ndarray ):
        return numpy.dot( A, x )
    elif isspmatrix(A):
        return A * x
    elif isinstance( A, LinearOperator ):
        return A * x
    else:
        raise ValueError( 'Unknown operator type "%s".' % type(A) )

# ===================================================================
def norm_MMlr(M, Ml, A, Mr, b, x0, yk, inner_product = ip):
    xk = x0 + apply(Mr, yk)
    r = b - apply(A, xk)
    Mlr = apply(Ml, r)
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
        nMMlr = apply(M, nMlr)
        MMlr = nMMlr * norm_Mlr
        norm_MMlr = norm(Mlr, MMlr, inner_product=inner_product)
    # return xk and ||M*Ml*(b-A*(x0+Mr*yk))||_{M^{-1}}
    return xk, Mlr, MMlr, norm_MMlr

# ===================================================================
def givens(a, b):
    '''Givens rotation
    [   c       s    ] * [a] = [r]
    [-conj(s) conj(c)]   [b]   [0]
    r real and non-negative.'''
    if abs(b) == 0:
        r = abs(a)
        c = a.conjugate() / r
        s = 0
    elif abs(a) == 0:
        r = abs(b)
        c = 0
        s = b.conjugate() / r
    elif abs(b) > abs(a):
        absb = abs(b)
        t = a.conjugate() / absb
        u = numpy.sqrt(1 + t.real**2 + t.imag**2)
        c = t / u
        s = (b.conjugate()/absb) / u
        r = absb * u
    else:
        absa = abs(a)
        t = b.conjugate()/absa
        u = numpy.sqrt(1 + t.real**2 + t.imag**2)
        c = (a.conjugate()/absa)/u
        s = t/u
        r = absa*u
    return numpy.array([[c, s],
                     [-s.conjugate(), c.conjugate()]])

# ===================================================================
def get_projection(b, W, AW,
        # TODO: AW optional, supply A
        x0 = None, 
        inner_product = ip):
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
               inner_product = ip,
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
        MAW = apply(M, AW)
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
        res_ip = ip(z0, z0) + 2 * ip(z0, numpy.dot(E, z1)).real + ip(z1, numpy.dot(D, z1)) \
               + 2 * ip(z1, numpy.dot(B1, z2)).real + ip(z2, z2)
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
