# -*- coding: utf8 -*-
import numpy
import warnings
from scipy.sparse.sputils import upcast
from scipy.sparse import issparse, isspmatrix
from . import utils

def cg(A, b,
       x0 = None,
       tol = 1.0e-5,
       maxiter = None,
       M = None,
       Ml = None,
       Mr = None,
       inner_product = utils.ip,
       explicit_residual = False,
       return_basis = False,
       full_reortho = False,
       exact_solution = None
       ):
    '''Conjugate gradient method with different inner product.
    '''
    if return_basis:
        raise RuntimeError('return_basis not implemented for CG.')

    info = 0
    N = len(b)

    if x0 is None:
        x0 = numpy.zeros((N,1))

    xtype = upcast( A.dtype, b.dtype, x0.dtype )
    if M is not None:
        xtype = upcast( xtype, M.dtype )
    if Ml is not None:
        xtype = upcast( xtype, Ml.dtype )
    if Mr is not None:
        xtype = upcast( xtype, Mr.dtype )

    x = xtype(x0.copy())
    # If len(x)==1, then xtype strips off the numpy.array frame around the value.
    # This is needed for utils.apply, though.
    if len(x0) == 1:
        x = numpy.array( [x] )

    # Compute M-norm of M*Ml*b.
    Mlb = utils.apply(Ml, b)
    MMlb = utils.apply(M, Mlb)
    norm_MMlb = utils.norm(Mlb, MMlb, inner_product = inner_product)

    # Init Lanczos and CG
    r0 = b - utils.apply(A, x0)
    Mlr0 = utils.apply(Ml, r0)
    MMlr0 = utils.apply(M, Mlr0)
    norm_MMlr0 = utils.norm(Mlr0, MMlr0, inner_product = inner_product)

    # initial relative residual norm
    relresvec = [norm_MMlr0 / norm_MMlb]

    # compute error?
    if exact_solution is not None:
        errvec = [utils.norm(exact_solution - x0, inner_product = inner_product)]

    # Allocate and initialize the 'large' memory blocks.
    if return_basis or full_reortho:
        raise RuntimeError('return_basis/full_reortho not implemented for CG.')

    # resulting approximation is xk = x0 + Mr*yk
    yk = numpy.zeros((N,1), dtype=xtype)

    if maxiter is None:
        maxiter = N

    out = {}
    out['info'] = 0

    rho_old = norm_MMlr0**2

    Mlr = Mlr0.copy()
    MMlr = MMlr0.copy()
    p = MMlr.copy()

    if exact_solution is not None:
        out['errorvec'] = numpy.empty(maxiter+1)
        out['errorvec'][0] = utils.norm_squared(x-exact_solution,
                                           inner_product = inner_product
                                           )

    k = 0
    while relresvec[k] > tol and k < maxiter:
        if k > 0:
            # update the search direction
            p = MMlr + rho_new/rho_old * p
            rho_old = rho_new
        Ap = utils.apply(Mr, p)
        Ap = utils.apply(A, Ap)
        Ap = utils.apply(Ml, Ap)

        # update current guess and residual
        alpha = rho_old / inner_product( p, Ap )
        if abs(alpha.imag) > 1e-12:
            warnings.warn('Iter %d: abs(alpha.imag) = %g > 1e-12' % (k+1, abs(alpha.imag)))
        alpha = alpha.real
        yk += alpha * p

        if exact_solution is not None:
            xk = x0 + utils.apply(Mr, yk)
            errvec.append(utils.norm(exact_solution - xk, inner_product=inner_product))

        if explicit_residual:
            xk, Mlr, MMlr, norm_MMlr = utils.norm_MMlr(M, Ml, A, Mr, b, x0, yk, inner_product=inner_product)
            relresvec.append( norm_MMlr / norm_MMlb )
            rho_new = norm_MMlr**2
        else:
            Mlr -= alpha * Ap
            MMlr = utils.apply(M, Mlr)
            rho_new = utils.norm_squared( Mlr, MMlr, inner_product = inner_product )
            relresvec.append( numpy.sqrt(rho_new) / norm_MMlb )

        # Compute residual explicitly if updated residual is below tolerance.
        if relresvec[-1] <= tol or k+1 == maxiter:
            norm_r_upd = relresvec[-1]
            # Compute the exact residual norm (if not yet done above)
            if not explicit_residual:
                xk, Mlr, MMlr, norm_MMlr = utils.norm_MMlr(M, Ml, A, Mr, b, x0, yk, inner_product = inner_product)
                relresvec[-1] = norm_MMlr / norm_MMlb
            # No convergence of explicit residual?
            if relresvec[-1] > tol:
                # Was this the last iteration?
                if k+1 == maxiter:
                    warnings.warn('Iter %d: No convergence! expl. res = %e >= tol =%e in last iter. (upd. res = %e)' \
                        % (k+1, relresvec[-1], tol, norm_r_upd))
                    info = 1
                else:
                    warnings.warn(('Iter %d: Updated residual is below tolerance, '
                                + 'explicit residual is NOT!\n  (resEx=%g > tol=%g >= '
                                + 'resup=%g)\n' ) % (k+1, relresvec[-1], tol, norm_r_upd))

        k += 1

    ret = { 'xk': xk,
            'info': info,
            'relresvec': relresvec
            }
    if exact_solution is not None:
        ret['errvec'] = errvec

    return ret

# ==============================================================================
def minres(A, b,
           x0 = None,
           tol = 1e-5,
           maxiter = None,
           M = None,
           Ml = None,
           Mr = None,
           inner_product = utils.ip,
           explicit_residual = False,
           return_basis = False,
           full_reortho = False,
           exact_solution = None,
           timer = False
           ):
    '''Preconditioned MINRES

    This MINRES solves M*Ml*A*Mr*y = M*Ml*b,  x=Mr*y
    where Ml and Mr have to be such that Ml*A*Mr is self-adjoint in the
    inner_product. M has to be self-adjoint and positive-definite w.r.t.
    inner_product.

    Details:
    The Lanczos procedure is used with the operator M*Ml*A*Mr and the
    inner product defined by inner_product(M^{-1}x,y). The initial vector
    for Lanczos is r0 = M*Ml*(b - A*x0) -- note that Mr is not used for
    the initial vector!

    Stopping criterion is
    ||M*Ml*(b-A*(x0+Mr*yk))||_{M^{-1}} / ||M*Ml*b||_{M^{-1}} <= tol
    '''
    info = 0
    N = len(b)
    if x0 is None:
        x0 = numpy.zeros((N,1))
    if maxiter is None:
        maxiter = N

    if timer:
        import time
        times = {'setup': numpy.empty(1),
                 'apply Ml*A*Mr': numpy.empty(maxiter),
                 'Lanczos': numpy.empty(maxiter),
                 'reortho': numpy.empty(maxiter),
                 'apply prec': numpy.empty(maxiter),
                 'extend Krylov': numpy.empty(maxiter),
                 'construct full basis': numpy.empty(maxiter),
                 'implicit QR': numpy.empty(maxiter),
                 'update solution': numpy.empty(maxiter),
                 'update residual': numpy.empty(maxiter)
                 }
    else:
        times = None

    if timer:
        start = time.time()

    xtype = upcast( A.dtype, b.dtype, x0.dtype )
    if isinstance(M, numpy.ndarray) or isspmatrix(M):
        xtype = upcast( xtype, M.dtype )
    if Ml is not None:
        xtype = upcast( xtype, Ml.dtype )
    if Mr is not None:
        xtype = upcast( xtype, Mr.dtype )

    # Compute M-norm of M*Ml*b.
    Mlb = utils.apply(Ml, b)
    MMlb = utils.apply(M, Mlb)
    norm_MMlb = utils.norm(Mlb, MMlb, inner_product = inner_product)

    # --------------------------------------------------------------------------
    # Init Lanczos and MINRES
    r0 = b - utils.apply(A, x0)
    Mlr0 = utils.apply(Ml, r0)
    MMlr0 = utils.apply(M, Mlr0)
    norm_MMlr0 = utils.norm(Mlr0, MMlr0, inner_product = inner_product)

    # initial relative residual norm
    relresvec = [norm_MMlr0 / norm_MMlb]

    # compute error?
    if exact_solution is not None:
        errvec = [utils.norm(exact_solution - x0, inner_product = inner_product)]

    # --------------------------------------------------------------------------
    # Allocate and initialize the 'large' memory blocks.
    if return_basis or full_reortho:
        Vfull = numpy.c_[MMlr0 / norm_MMlr0, numpy.zeros([N,maxiter], dtype=xtype)]
        Pfull = numpy.c_[Mlr0 / norm_MMlr0, numpy.zeros([N,maxiter], dtype=xtype)]
        Hfull = numpy.zeros((maxiter+1,maxiter)) #scipy.sparse.lil_matrix( (maxiter+1,maxiter) )
    # Last and current Lanczos vector:
    V = numpy.c_[numpy.zeros(N), MMlr0 / norm_MMlr0]
    # M*v[i] = P[1], M*v[i-1] = P[0]
    P = numpy.c_[numpy.zeros(N), Mlr0 / norm_MMlr0]
    # Necessary for efficient update of yk:
    W = numpy.c_[numpy.zeros(N), numpy.zeros(N)]
    # some small helpers
    ts = 0.0           # (non-existing) first off-diagonal entry (corresponds to pi1)
    y  = [norm_MMlr0, 0] # first entry is (updated) residual
    G2 = numpy.eye(2)     # old givens rotation
    G1 = numpy.eye(2)     # even older givens rotation ;)
    k = 0

    # resulting approximation is xk = x0 + Mr*yk
    yk = numpy.zeros((N,1), dtype=xtype)
    xk = x0.copy()

    if timer:
        times['setup'][0] = time.time()-start

    # --------------------------------------------------------------------------
    # Lanczos + MINRES iteration
    # --------------------------------------------------------------------------
    while relresvec[-1] > tol and k < maxiter:
        # ---------------------------------------------------------------------
        # Lanczos
        if timer:
            start = time.time()
        tsold = ts
        z  = utils.apply(Mr, V[:,[1]])
        z  = utils.apply(A, z)
        z  = utils.apply(Ml, z)
        if timer:
            times['apply Ml*A*Mr'][k] = time.time()-start

        if timer:
            start = time.time()
        # tsold = inner_product(V[:,[0]], z)[0,0]
        z  = z - tsold * P[:,[0]]
        # Should be real! (diagonal element):
        td = inner_product(V[:,[1]], z)[0,0]
        if abs(td.imag) > 1.0e-12:
            warnings.warn('Iter %d: abs(td.imag) = %g > 1e-12' % (k+1, abs(td.imag)))
        td = td.real
        z  = z - td * P[:,[1]]
        if timer:
            times['Lanczos'][k] = time.time()-start

        ## local reorthogonalization
        #tsold2 = inner_product(V[0], z)
        #z   = z - tsold2 * P[0]
        #td2 = inner_product(V[1], z)
        #td  = td + td2
        #z   = z - td2*P[1]
        #tsold = tsold + tsold2

        if timer:
            start = time.time()
        # double reortho
        for l in range(0,2):
            # full reortho?
            # cost: (k+1)*(IP + AXPY)
            if full_reortho:
                # here we can (and should) orthogonalize against ALL THE
                # vectors (thus k+1).
                # http://memegenerator.net/instance/13779948
                #
                for i in range(0,k+1):
                    ip = inner_product(Vfull[:,[i]], z)[0,0]
                    if abs(ip) > 1.0e-9:
                        warnings.warn('Iter %d: abs(ip) = %g > 1.0e-9: The Krylov basis has become linearly dependent. Maxiter (%d) too large and tolerance too severe (%g)? dim = %d.' % (k+1, abs(ip), maxiter, tol, len(x0)))
                    z -= ip * Pfull[:,[i]]
            ## ortho against additional (deflation) vectors?
            ## cost: ndeflW*(IP + AXPY)
            #if deflW is not None:
            #    for i in range(0,deflW.shape[1]):
            #        ip = inner_product(deflW[:,[i]], z)[0,0]
            #        if abs(ip) > 1.0e-9:
            #            print('Warning (iter %d): abs(ip) = %g > 1.0e-9: The Krylov basis has lost orthogonality to deflated space (deflW).' % (k+1, abs(ip)))
            #        z = z - ip * deflW[:,[i]]
        if timer:
            times['reortho'][k] = time.time()-start

        # needed for QR-update:
        R = utils.apply(G1, [0, tsold])
        R = numpy.append(R, [0.0, 0.0])

        # Apply the preconditioner.
        if timer:
            start = time.time()
        v  = utils.apply(M, z)
        alpha = inner_product(z, v)[0,0]
        if abs(alpha.imag)>1e-12:
            warnigs.warn('Iter %d: abs(alpha.imag) = %g > 1e-12' % (k+1, abs(alpha.imag)))
        alpha = alpha.real
        if alpha<0.0:
            warnings.warn('Iter %d: alpha = %g < 0' % (k+1, alpha))
            alpha = 0.0
        ts = numpy.sqrt( alpha )
        if timer:
            times['apply prec'][k] = time.time()-start

        if timer:
            start = time.time()
        if ts > 0.0:
            P  = numpy.c_[P[:,[1]], z / ts]
            V  = numpy.c_[V[:,[1]], v / ts]
        else:
            P  = numpy.c_[P[:,[1]], numpy.zeros(N)]
            V  = numpy.c_[V[:,[1]], numpy.zeros(N)]
        if timer:
            times['extend Krylov'][k] = time.time()-start


        # store new vectors in full basis
        if timer:
            start = time.time()
        if return_basis or full_reortho:
            if ts>0.0:
                Vfull[:,[k+1]] = v / ts
                Pfull[:,[k+1]] = z / ts
            Hfull[k,k] = td        # diagonal
            Hfull[k+1,k] = ts      # subdiagonal
            if k+1 < maxiter:
                Hfull[k,k+1] = ts  # superdiagonal
        if timer:
            times['construct full basis'][k] = time.time()-start

        # ----------------------------------------------------------------------
        # (implicit) update of QR-factorization of Lanczos matrix
        if timer:
            start = time.time()
        R[2:4] = [td, ts]
        R[1:3] = utils.apply(G2, R[1:3])
        G1 = G2.copy()
        # compute new givens rotation.
        gg = numpy.linalg.norm( R[2:4] )
        gc = R[2] / gg
        gs = R[3] / gg
        G2 = numpy.array([ [gc,  gs],
                        [-gs, gc] ])
        R[2] = gg
        R[3] = 0.0
        y = utils.apply(G2, y)
        if timer:
            times['implicit QR'][k] = time.time()-start

        # ----------------------------------------------------------------------
        # update solution
        if timer:
            start = time.time()
        z  = (V[:,0:1] - R[0]*W[:,0:1] - R[1]*W[:,1:2]) / R[2]
        W  = numpy.c_[W[:,1:2], z]
        yk = yk + y[0] * z
        y  = [y[1], 0]
        if timer:
            times['update solution'][k] = time.time()-start

        # ----------------------------------------------------------------------
        # update residual
        if timer:
            start = time.time()
        if exact_solution is not None:
            xk = x0 + utils.apply(Mr, yk)
            errvec.append(utils.norm(exact_solution - xk, inner_product=inner_product))

        if explicit_residual:
            xk, _, _, norm_MMlr = utils.norm_MMlr(M, Ml, A, Mr, b, x0, yk, inner_product = inner_product)
            relresvec.append( norm_MMlr / norm_MMlb )
        else:
            relresvec.append( abs(y[0]) / norm_MMlb )

        # Compute residual explicitly if updated residual is below tolerance.
        if relresvec[-1] <= tol or k+1 == maxiter:
            norm_r_upd = relresvec[-1]
            # Compute the exact residual norm (if not yet done above)
            if not explicit_residual:
                xk, _, _, norm_MMlr = utils.norm_MMlr(M, Ml, A, Mr, b, x0, yk, inner_product = inner_product)
                relresvec[-1] = norm_MMlr / norm_MMlb
            # No convergence of explicit residual?
            if relresvec[-1] > tol:
                # Was this the last iteration?
                if k+1 == maxiter:
                    warnings.warn('Iter %d: No convergence! expl. res = %e >= tol =%e in last iter. (upd. res = %e)' \
                        % (k+1, relresvec[-1], tol, norm_r_upd))
                    info = 1
                else:
                    warnings.warn( ( 'Info (iter %d): Updated residual is below tolerance, '
                          + 'explicit residual is NOT!\n  (resEx=%g > tol=%g >= '
                          + 'resup=%g)\n' \
                          ) % (k+1, relresvec[-1], tol, norm_r_upd) )

        if timer:
            times['update residual'][k] = time.time()-start

        # limit relative residual to machine precision (an exact 0 is rare but
        # seems to occur with pyamg...).
        relresvec[-1] = max(numpy.finfo(float).eps, relresvec[-1])
        k += 1
    # end MINRES iteration
    # --------------------------------------------------------------------------

    ret = { 'xk': xk,
            'info': info,
            'relresvec': relresvec
            }
    if exact_solution is not None:
        ret['errvec'] = errvec

    if return_basis:
        ret['Vfull'] = Vfull[:,0:k+1]
        ret['Pfull'] = Pfull[:,0:k+1]
        ret['Hfull'] = Hfull[0:k+1,0:k]
    if timer:
        # properly cut down times
        for key in times:
            times[key] = times[key][:k]
        ret['times'] = times
    return ret

# ==============================================================================
def gmres( A, b,
           x0 = None,
           tol = 1e-5,
           maxiter = None,
           M = None,
           Ml = None,
           Mr = None,
           inner_product = utils.ip,
           explicit_residual = False,
           return_basis = False,
           exact_solution = None
         ):
    '''Preconditioned GMRES

    Solves   M*Ml*A*Mr*y = M*Ml*b,  x=Mr*y.
    M has to be self-adjoint and positive-definite w.r.t. inner_product.

    Stopping criterion is
    ||M*Ml*(b-A*(x0+Mr*yk))||_{M^{-1}} / ||M*Ml*b||_{M^{-1}} <= tol

    Memory consumption is about maxiter+1 vectors for the Arnoldi basis.
    If M is used the memory consumption is 2*(maxiter+1).
    '''
    # TODO(Andre): errvec for exact_solution!=None
    # --------------------------------------------------------------------------
    def _compute_explicit_xk(H, V, y):
        '''Compute approximation xk to the solution.'''
        yy = numpy.linalg.solve(H, y)
        u  = utils.apply(Mr, numpy.dot(V, yy))
        xk = x0 + u
        return xk
    # --------------------------------------------------------------------------
    def _compute_explicit_residual( xk ):
        '''Compute residual explicitly.'''
        rk  = b - utils.apply(A, xk)
        rk  = utils.apply(Ml, rk)
        Mrk  = utils.apply(M, rk);
        norm_Mrk = utils.norm(rk, Mrk, inner_product=inner_product)
        return Mrk, norm_Mrk
    # --------------------------------------------------------------------------

    N = len(b)
    if x0 is None:
        x0 = numpy.zeros((N,1))
    if not maxiter:
        maxiter = N


    xtype = upcast( b.dtype, x0.dtype )
    if isinstance(A, numpy.ndarray) or isspmatrix(A):
        xtype = upcast( xtype, A.dtype)
    if isinstance(M, numpy.ndarray) or isspmatrix(M):
        xtype = upcast( xtype, M.dtype )
    if Ml is not None:
        xtype = upcast( xtype, Ml.dtype )
    if Mr is not None:
        xtype = upcast( xtype, Mr.dtype )

    out = {}
    out['info'] = 0

    # get memory for working variables
    V = numpy.zeros([N, maxiter+1], dtype=xtype) # Arnoldi basis
    H = numpy.zeros([maxiter+1, maxiter], dtype=xtype) # Hessenberg matrix

    if M is not None:
        P = numpy.zeros([N,maxiter+1], dtype=xtype) # V=M*P

    if return_basis:
        Horig = numpy.zeros([maxiter+1,maxiter], dtype=xtype)

    # initialize working variables
    Mlb = utils.apply(Ml, b)
    MMlb = utils.apply(M, Mlb)
    norm_MMlb = utils.norm(Mlb, MMlb, inner_product=inner_product)
    # This may only save us the application of Ml to the same vector again if
    # x0 is the zero vector.
    norm_x0 = utils.norm(x0, inner_product=inner_product)
    if norm_x0 > numpy.finfo(float).eps:
        r0 = b - utils.apply(A, x0)
        Mlr0 = utils.apply(Ml, r0)
        MMlr0 = utils.apply(M, Mlr0);
        norm_MMlr0 = utils.norm(Mlr0, MMlr0, inner_product=inner_product)
    else:
        x0 = numpy.zeros( (N,1) )
        Mlr0 = Mlb.copy()
        MMlr0 = MMlb.copy()
        norm_MMlr0 = norm_MMlb

    out['relresvec'] = numpy.empty(maxiter+1)

    V[:, [0]] = MMlr0 / norm_MMlr0
    if M is not None:
        P[:, [0]] = Mlr0 / norm_MMlr0
    out['relresvec'][0] = norm_MMlr0 / norm_MMlb
    # Right hand side of projected system:
    y = numpy.zeros( (maxiter+1,1), dtype=xtype )
    y[0] = norm_MMlr0
    # Givens rotations:
    G = []

    if exact_solution is not None:
        out['errorvec'] = numpy.empty(maxiter+1)
        out['errorvec'][0] = utils.norm(x0-exact_solution,
                                   inner_product=inner_product
                                   )

    k = 0
    while out['relresvec'][k] > tol and k < maxiter:
        # Apply operator Ml*A*Mr
        z = utils.apply(Ml, utils.apply(A, utils.apply(Mr, V[:, [k]])))

        # orthogonalize (MGS)
        for i in range(k+1):
            if M is not None:
                H[i, k] += inner_product(V[:, [i]], z)[0,0]
                z -= H[i, k] * P[:, [i]]
            else:
                H[i, k] += inner_product(V[:, [i]], z)[0,0]
                z -= H[i, k] * V[:, [i]]
        Mz = utils.apply(M, z);
        H[k+1, k] = utils.norm(z, Mz, inner_product=inner_product)
        if M is not None:
            P[:, [k+1]] = z / H[k+1, k]
        V[:, [k+1]] = Mz / H[k+1, k]
        if return_basis:
            Horig[0:k+2, [k]] = H[0:k+2, [k]]

        # Apply previous Givens rotations.
        for i in range(k):
            H[i:i+2, k] = utils.apply(G[i], H[i:i+2, k])

        # Compute and apply new Givens rotation.
        G.append(utils.givens(H[k, k], H[k+1, k]))
        H[k:k+2, k] = utils.apply(G[k], H[k:k+2, k])
        y[k:k+2] = utils.apply(G[k], y[k:k+2])

        # Update residual norm.
        if explicit_residual:
            xk = _compute_explicit_xk(H[:k+1, :k+1], V[:, :k+1], y[:k+1])
            Mrk, norm_Mrk = _compute_explicit_residual( xk )
            out['relresvec'][k+1] = norm_Mrk / norm_MMlb
        else:
            out['relresvec'][k+1] = abs(y[k+1]) / norm_MMlb

        # convergence of updated residual or maxiter reached?
        if out['relresvec'][k+1] < tol or k+1 == maxiter:
            norm_ur = out['relresvec'][k+1]

            if not explicit_residual:
                xk = _compute_explicit_xk(H[:k+1, :k+1], V[:, :k+1], y[:k+1])
                Mrk, norm_Mrk = _compute_explicit_residual( xk )
                out['relresvec'][k+1] = norm_Mrk / norm_MMlb

            # No convergence of expl. residual?
            if out['relresvec'][k+1] >= tol:
                # Was this the last iteration?
                if k+1 == maxiter:
                    warnings.warn('Iter %d: No convergence! expl. res = %e >= tol =%e in last it. (upd. res = %e)' \
                        % (k+1, out['relresvec'][k+1], tol, norm_ur))
                    out['info'] = 1
                else:
                    warnigs.warn('Iter %d: Expl. res = %e >= tol = %e > upd. res = %e.' \
                        % (k+1, out['relresvec'][k+1], tol, norm_ur))

        k += 1

    out['relresvec'] = out['relresvec'][:k+1]
    out['xk'] = _compute_explicit_xk(H[:k,:k], V[:,:k], y[:k])
    if return_basis:
        out['Vfull'] = V[:, :k+1]
        out['Hfull'] = Horig[:k+1, :k]
        if M is not None:
            out['Pfull'] = P[:, :k+1]
    return out
