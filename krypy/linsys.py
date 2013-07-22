# -*- coding: utf8 -*-
import numpy
import warnings
from scipy.sparse.linalg import aslinearoperator
from . import utils

def cg(A, b,
       x0 = None,
       tol = 1.0e-5,
       maxiter = None,
       M = None,
       Ml = None,
       Mr = None,
       inner_product = utils.ip_euclid,
       explicit_residual = False,
       exact_solution = None
       ):
    '''Preconditioned CG method.

    The *preconditioned conjugate gradient method* can be used to solve a
    system of linear algebraic equations where the linear operator is
    self-adjoint and positive definite. Let the following linear algebraic
    system be given:

    .. math::

      M M_l A M_r y = M M_l b,

    where :math:`x=M_r y` and :math:`M_l A M_r` is self-adjoint and
    positive definite with respect to the inner product
    :math:`\\langle \\cdot,\\cdot \\rangle` defined by ``inner_product``.
    The preconditioned CG method then computes (in exact arithmetics!)
    iterates :math:`x_k \\in x_0 + M_r K_k` with
    :math:`K_k:= K_k(M M_l A M_r, r_0)` such that

    .. math::

      \\|x - x_k\\|_A = \\min_{z \\in x_0 + M_r K_k} \\|x - z\\|_A.

    The Lanczos alorithm is used with the operator
    :math:`M M_l A M_r` and the inner product defined by
    :math:`\\langle x,y \\rangle_{M^{-1}} = \\langle M^{-1}x,y \\rangle`.
    The initial vector for Lanczos is
    :math:`r_0 = M M_l (b - Ax_0)` - note that :math:`M_r` is not used for
    the initial vector.

    Memory consumption is:

    * if ``return_basis==False``: 3 vectors or 6 vectors if :math:`M` is used.
    * if ``return_basis==True``: about maxiter+1 vectors for the Lanczos basis.
      If :math:`M` is used the memory consumption is 2*(maxiter+1).

    **Caution:** CG's convergence may be delayed significantly due to round-off
    errors, cf. chapter 5.9 in *Liesen, Strakos. Krylov subspace Methods.
    2013.*

    :param A:
      a linear operator on :math:`\\mathbb{C}^N`. Note that
      :math:`M_l A M_r` has to be self-adjoint and positive definite in the
      inner product.
    :param b:
      a vector in :math:`\\mathbb{C}^N`.
    :param x0: (optional) the initial guess to use. Defaults to zero vector.
      Unless you have a good reason to use a nonzero initial guess you should
      use the zero vector, cf. chapter 5.8.3 in *Liesen, Strakos. Krylov
      subspace methods. 2013*. See also :py:meth:`~krypy.utils.hegedus`.
    :param tol: (optional) the tolerance for the stopping criterion with respect to
      the relative residual norm:

      .. math::

         \\frac{ \\| M M_l (b-A (x_0+M_r y_k))\\|_{M^{-1}} }{ \\|M M_l b\\|_{M^{-1}}}
         \leq \\text{tol}

    :param maxiter: (optional) maximum number of iterations. Defaults to N.
    :param M: (optional)
      a self-adjoint and positive definite preconditioner, linear operator on
      :math:`\\mathbb{C}^N` with respect to ``inner_product``. This
      preconditioner changes the inner product used for orthogonalization to
      :math:`\\langle x,y\\rangle_M = \\langle Mx,y\\rangle` where
      :math:`\\langle \\cdot,\\cdot\\rangle` is the inner product defined
      by the parameter ``inner_product``. Defaults to the identity.
    :param Ml: (optional) left preconditioner, linear operator on
      :math:`\\mathbb{C}^N`. Defaults to the identity.
    :param Mr: (optional) right preconditioner, linear operator on
      :math:`\\mathbb{C}^N`. Defaults to the identity.
    :param inner_product: (optional) a function that takes two arguments and
      computes the (block-) inner product of the arguments. Defaults to
      :py:meth:`~krypy.utils.ip_euclid`.
    :param explicit_residual: (optional)
      if set to ``False`` (default), the updated residual norm from the CG
      iteration is used in each iteration. If set to ``True``, the residual is
      computed explicitly in each iteration and thus requires an additional
      matrix-vector multiplication in each iteration.
    :param exact_solution: (optional)
      if the solution vector :math:`x` is passed then the error norm
      :math:`\|x-x_k\|` will be computed in each iteration (with respect
      to ``inner_product``) and returned as a list in the result dictionary
      with the key ``errvec``. Defaults to ``None``, which means that no
      errors are computed.

    :return:
      a dictionary with the following keys:

      * ``xk``: the approximate solution :math:`x_k`.
      * ``info``: convergence flag (0 if converged, 1 otherwise).
      * ``relresvec``: relative residual norms of all iterations, see
        parameter ``tol``.
    '''

    N = len(b)
    shape = (N,N)
    A = utils.get_linearoperator(shape, A)
    if maxiter is None:
        maxiter = N
    flat_vecs, (b, x0, exact_solution) = utils.shape_vecs(b, x0, exact_solution)
    if x0 is None:
        x0 = numpy.zeros((N,1))
    M = utils.get_linearoperator(shape, M)
    Ml = utils.get_linearoperator(shape, Ml)
    Mr = utils.get_linearoperator(shape, Mr)
    cdtype = utils.find_common_dtype(A, b, x0, M, Ml, Mr)

    # Compute M-norm of M*Ml*b.
    Mlb = Ml * b
    MMlb = M * Mlb
    norm_MMlb = utils.norm(Mlb, MMlb, inner_product = inner_product)

    # Init Lanczos and CG
    r0 = b - A * x0
    Mlr0 = Ml * r0
    MMlr0 = M * Mlr0
    norm_MMlr0 = utils.norm(Mlr0, MMlr0, inner_product = inner_product)

    # if rhs is exactly(!) zero, return zero solution.
    if norm_MMlb==0:
        x0 = numpy.zeros((N,1))
        relresvec = [0.0]
    else:
        # initial relative residual norm
        relresvec = [norm_MMlr0 / norm_MMlb]

    # compute error?
    if exact_solution is not None:
        errvec = [utils.norm(exact_solution - x0, inner_product = inner_product)]

    # resulting approximation is xk = x0 + Mr*yk
    yk = numpy.zeros((N,1), dtype=cdtype)
    xk = x0.copy()

    rho_old = norm_MMlr0**2

    Mlr = Mlr0.copy()
    MMlr = MMlr0.copy()
    p = MMlr.copy()

    k = 0
    while relresvec[k] > tol and k < maxiter:
        if k > 0:
            # update the search direction
            p = MMlr + rho_new/rho_old * p
            rho_old = rho_new
        Ap = Mr * p
        Ap = A * Ap
        Ap = Ml * Ap

        # update current guess and residual
        alpha = rho_old / inner_product( p, Ap )
        if abs(alpha.imag) > 1e-12:
            warnings.warn('Iter %d: abs(alpha.imag) = %g > 1e-12' % (k+1, abs(alpha.imag)))
        alpha = alpha.real
        yk += alpha * p

        if exact_solution is not None:
            xk = x0 + Mr * yk
            errvec.append(utils.norm(exact_solution - xk, inner_product=inner_product))

        if explicit_residual:
            xk, Mlr, MMlr, norm_MMlr = utils.norm_MMlr(M, Ml, A, Mr, b, x0, yk, inner_product=inner_product)
            relresvec.append( norm_MMlr / norm_MMlb )
            rho_new = norm_MMlr**2
        else:
            Mlr -= alpha * Ap
            MMlr = M * Mlr
            rho_new = utils.norm_squared( Mlr, MMlr, inner_product = inner_product )
            relresvec.append( numpy.sqrt(rho_new) / norm_MMlb )

        # Compute residual explicitly if updated residual is below tolerance.
        if relresvec[-1] <= tol or k+1 == maxiter:
            norm_r_upd = relresvec[-1]
            # Compute the exact residual norm (if not yet done above)
            if not explicit_residual:
                xk, Mlr, MMlr, norm_MMlr = utils.norm_MMlr(M, Ml, A, Mr, b, x0, yk,
                        inner_product = inner_product)
                relresvec[-1] = norm_MMlr / norm_MMlb
            # No convergence of explicit residual?
            if relresvec[-1] > tol:
                # Was this the last iteration?
                if k+1 == maxiter:
                    warnings.warn('Iter %d: No convergence! expl. res = %e >= tol =%e in last iter. (upd. res = %e)' \
                        % (k+1, relresvec[-1], tol, norm_r_upd))
                else:
                    warnings.warn(('Iter %d: Updated residual is below tolerance, '
                                + 'explicit residual is NOT!\n  (resEx=%g > tol=%g >= '
                                + 'resup=%g)\n' ) % (k+1, relresvec[-1], tol, norm_r_upd))

        k += 1

    ret = { 'xk': xk if not flat_vecs else numpy.ndarray.flatten(xk),
            'info': relresvec[-1] <= tol,
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
           inner_product = utils.ip_euclid,
           explicit_residual = False,
           return_basis = False,
           full_reortho = False,
           exact_solution = None
           ):
    '''Preconditioned MINRES method.

    The *preconditioned minimal residual method* can be used to solve a
    system of linear algebraic equations where the linear operator is
    self-adjoint. Let the following linear algebraic
    system be given:

    .. math::

      M M_l A M_r y = M M_l b,

    where :math:`x=M_r y` and :math:`M_l A M_r` is self-adjoint with respect
    to the inner product
    :math:`\\langle \\cdot,\\cdot \\rangle` defined by ``inner_product``.
    The preconditioned MINRES method then computes (in exact arithmetics!)
    iterates :math:`x_k \\in x_0 + M_r K_k` with
    :math:`K_k:= K_k(M M_l A M_r, r_0)` such that

    .. math::

      \\|M M_l(b - A x_k)\\|_{M^{-1}} =
      \\min_{z \\in x_0 + M_r K_k} \\|M M_l (b - A z)\\|_{M^{-1}}.

    The Lanczos alorithm is used with the operator
    :math:`M M_l A M_r` and the inner product defined by
    :math:`\\langle x,y \\rangle_{M^{-1}} = \\langle M^{-1}x,y \\rangle`.
    The initial vector for Lanczos is
    :math:`r_0 = M M_l (b - Ax_0)` - note that :math:`M_r` is not used for
    the initial vector.

    Memory consumption is:

    * if ``return_basis==False``: 3 vectors or 6 vectors if :math:`M` is used.
    * if ``return_basis==True``: about maxiter+1 vectors for the Lanczos basis.
      If :math:`M` is used the memory consumption is 2*(maxiter+1).

    **Caution:** MINRES' convergence may be delayed significantly or even
    stagnate due to round-off errors, cf. chapter 5.9 in *Liesen, Strakos.
    Krylov subspace Methods. 2013.*

    :param A:
      a linear operator on :math:`\\mathbb{C}^N`. Note that
      :math:`M_l A M_r` has to be self-adjoint in the inner product.
    :param b:
      a vector in :math:`\\mathbb{C}^N`.
    :param x0: (optional) the initial guess to use. Defaults to zero vector.
      Unless you have a good reason to use a nonzero initial guess you should
      use the zero vector, cf. chapter 5.8.3 in *Liesen, Strakos. Krylov
      subspace methods. 2013*. See also :py:meth:`~krypy.utils.hegedus`.
    :param tol: (optional) the tolerance for the stopping criterion with
      respect to the relative residual norm:

      .. math::

         \\frac{ \\| M M_l (b-A (x_0+M_r y_k))\\|_{M^{-1}} }{ \\|M M_l b\\|_{M^{-1}}}
         \leq \\text{tol}

    :param maxiter: (optional) maximum number of iterations. Defaults to N.
    :param M: (optional)
      a self-adjoint and positive definite preconditioner, linear operator on
      :math:`\\mathbb{C}^N` with respect to ``inner_product``. This
      preconditioner changes the inner product used for orthogonalization to
      :math:`\\langle x,y\\rangle_M = \\langle Mx,y\\rangle` where
      :math:`\\langle \cdot,\cdot\\rangle` is the inner product defined by the
      parameter ``inner_product``. Defaults to the identity.
    :param Ml: (optional) left preconditioner, linear operator on
      :math:`\\mathbb{C}^N`. Defaults to the identity.
    :param Mr: (optional) right preconditioner, linear operator on
      :math:`\\mathbb{C}^N`. Defaults to the identity.
    :param inner_product: (optional) a function that takes two arguments and
      computes the (block-) inner product of the arguments. Defaults to
      :py:meth:`~krypy.utils.ip_euclid`.
    :param explicit_residual: (optional)
      if set to ``False`` (default), the updated residual norm from the MINRES
      iteration is used in each iteration. If set to ``True``, the residual is
      computed explicitly in each iteration and thus requires an additional
      matrix-vector multiplication in each iteration.
    :param return_basis: (optional)
      if set to ``True`` then the computed Lanczos basis and the tridiagonal
      matrix are returned in the result dictionary with the keys ``V``
      and ``H``. Defaults to ``False``.
    :param exact_solution: (optional)
      if the solution vector :math:`x` is passed then the error norm
      :math:`\|x-x_k\|` will be computed in each iteration (with respect to
      ``inner_product``) and returned as a list in the result dictionary with
      the key ``errvec``. Defaults to ``None``, which means that no errors are
      computed.

    :return:
      a dictionary with the following keys:

      * ``xk``: the approximate solution :math:`x_k`.
      * ``info``: convergence flag (0 if converged, 1 otherwise).
      * ``relresvec``: relative residual norms of all iterations, see
        parameter ``tol``.
      * ``V``: present if ``return_basis=True``. The Arnoldi basis
        vectors.
      * ``H``: present if ``return_basis=True``. The Hessenberg matrix.
      * ``P``: present if ``return_basis=True`` and ``M`` is provided.
        The matrix :math:`P` fulfills :math:`V=MP`.
    '''
    N = len(b)
    shape = (N,N)
    A = utils.get_linearoperator(shape, A)
    if maxiter is None:
        maxiter = N
    flat_vecs, (b, x0, exact_solution) = utils.shape_vecs(b, x0, exact_solution)
    if x0 is None:
        x0 = numpy.zeros((N,1))
    M = utils.get_linearoperator(shape, M)
    Ml = utils.get_linearoperator(shape, Ml)
    Mr = utils.get_linearoperator(shape, Mr)
    cdtype = utils.find_common_dtype(A, b, x0, M, Ml, Mr)

    # Compute M-norm of M*Ml*b.
    Mlb = Ml * b
    MMlb = M * Mlb
    norm_MMlb = utils.norm(Mlb, MMlb, inner_product = inner_product)

    # Init Lanczos and MINRES
    r0 = b - A * x0
    Mlr0 = Ml * r0
    MMlr0 = M * Mlr0
    norm_MMlr0 = utils.norm(Mlr0, MMlr0, inner_product = inner_product)

    # if rhs is exactly(!) zero, return zero solution.
    if norm_MMlb==0:
        x0 = numpy.zeros((N,1))
        relresvec = [0.0]
    else:
        # initial relative residual norm
        relresvec = [norm_MMlr0 / norm_MMlb]
    xk = x0.copy()

    # compute error?
    if exact_solution is not None:
        errvec = [utils.norm(exact_solution - x0, inner_product = inner_product)]

    # initialize only if needed
    if relresvec[-1] > tol:
        # Allocate and initialize the 'large' memory blocks.
        if return_basis or full_reortho:
            Vfull = numpy.c_[MMlr0 / norm_MMlr0, numpy.zeros([N,maxiter], dtype=cdtype)]
            Pfull = numpy.c_[Mlr0 / norm_MMlr0, numpy.zeros([N,maxiter], dtype=cdtype)]
            Hfull = numpy.zeros((maxiter+1,maxiter), dtype=numpy.float)
        # Last and current Lanczos vector:
        V = numpy.c_[numpy.zeros(N, dtype=cdtype), MMlr0 / norm_MMlr0]
        # M*v[i] = P[1], M*v[i-1] = P[0]
        P = numpy.c_[numpy.zeros(N, dtype=cdtype), Mlr0 / norm_MMlr0]
        # Necessary for efficient update of yk:
        W = numpy.c_[numpy.zeros(N, dtype=cdtype), numpy.zeros(N)]
        # some small helpers
        ts = 0.0           # (non-existing) first off-diagonal entry (corresponds to pi1)
        y  = [norm_MMlr0, 0] # first entry is (updated) residual
        G2 = None            # old givens rotation
        G1 = None            # even older givens rotation ;)
        k = 0

        # resulting approximation is xk = x0 + Mr*yk
        yk = numpy.zeros((N,1), dtype=cdtype)

    # --------------------------------------------------------------------------
    # Lanczos + MINRES iteration
    # --------------------------------------------------------------------------
    while relresvec[-1] > tol and k < maxiter:
        # ---------------------------------------------------------------------
        # Lanczos
        tsold = ts
        z  = Mr * V[:,[1]]
        z  = A * z
        z  = Ml * z
        z  = z - tsold * P[:,[0]]

        # Should be real! (diagonal element):
        td = inner_product(V[:,[1]], z)[0,0]
        if abs(td.imag) > 1.0e-12:
            warnings.warn('Iter %d: abs(td.imag) = %g > 1e-12' % (k+1, abs(td.imag)))
        td = td.real
        z  = z - td * P[:,[1]]

        # double reortho
        for l in range(0,2):
            # full reortho?
            # cost: (k+1)*(IP + AXPY)
            if full_reortho:
                # here we can (and should) orthogonalize against ALL THE
                # vectors (thus k+1).
                # http://memegenerator.net/instance/13779948
                for i in range(0,k+1):
                    ip = inner_product(Vfull[:,[i]], z)[0,0]
                    if abs(ip) > 1.0e-9:
                        warnings.warn('Iter %d: abs(ip) = %g > 1.0e-9: The Krylov basis has become linearly dependent. Maxiter (%d) too large and tolerance too severe (%g)? dim = %d.' % (k+1, abs(ip), maxiter, tol, len(x0)))
                    z -= ip * Pfull[:,[i]]

        # needed for QR-update:
        R = numpy.zeros( (4,1) )
        R[1] = tsold
        if G1 is not None:
            R[:2] = G1.apply(R[:2])

        # Apply the preconditioner.
        v  = M * z
        ts = utils.norm(z, Mx=v, inner_product=inner_product)


        if ts > 0.0:
            P  = numpy.c_[P[:,[1]], z / ts]
            V  = numpy.c_[V[:,[1]], v / ts]
        else:
            P  = numpy.c_[P[:,[1]], numpy.zeros(N)]
            V  = numpy.c_[V[:,[1]], numpy.zeros(N)]


        # store new vectors in full basis
        if return_basis or full_reortho:
            if ts>0.0:
                Vfull[:,[k+1]] = v / ts
                Pfull[:,[k+1]] = z / ts
            Hfull[k,k] = td        # diagonal
            Hfull[k+1,k] = ts      # subdiagonal
            if k+1 < maxiter:
                Hfull[k,k+1] = ts  # superdiagonal

        # ----------------------------------------------------------------------
        # (implicit) update of QR-factorization of Lanczos matrix
        R[2:4,0] = [td, ts]
        if G2 is not None:
            R[1:3] = G2.apply(R[1:3])
        G1 = G2
        # compute new givens rotation.
        G2 = utils.Givens(R[2:4])
        R[2] = G2.r
        R[3] = 0.0
        y = G2.apply(y)

        # ----------------------------------------------------------------------
        # update solution
        z  = (V[:,0:1] - R[0]*W[:,0:1] - R[1]*W[:,1:2]) / R[2]
        W  = numpy.c_[W[:,1:2], z]
        yk = yk + y[0] * z
        y  = [y[1], 0]

        # update residual
        if exact_solution is not None:
            xk = x0 + Mr * yk
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
                else:
                    warnings.warn( ( 'Info (iter %d): Updated residual is below tolerance, '
                          + 'explicit residual is NOT!\n  (resEx=%g > tol=%g >= '
                          + 'resup=%g)\n' \
                          ) % (k+1, relresvec[-1], tol, norm_r_upd) )


        # limit relative residual to machine precision (an exact 0 is rare but
        # seems to occur with pyamg...).
        # relresvec[-1] = max(numpy.finfo(float).eps, relresvec[-1])
        k += 1
    # end MINRES iteration

    ret = { 'xk': xk if not flat_vecs else numpy.ndarray.flatten(xk),
            'info': relresvec[-1] <= tol,
            'relresvec': relresvec
            }
    if exact_solution is not None:
        ret['errvec'] = errvec
    if return_basis:
        ret['V'] = Vfull[:,0:k+1]
        ret['P'] = Pfull[:,0:k+1]
        ret['H'] = Hfull[0:k+1,0:k]
    return ret

# ==============================================================================
def gmres( A, b,
           x0 = None,
           tol = 1e-5,
           maxiter = None,
           M = None,
           Ml = None,
           Mr = None,
           inner_product = utils.ip_euclid,
           explicit_residual = False,
           return_basis = False,
           full_reortho = True,
           exact_solution = None,
           max_restarts = 0
         ):
    '''Preconditioned GMRES method.

    The *preconditioned generalized minimal residual method* can be used to
    solve a system of linear algebraic equations. Let the following linear
    algebraic system be given:

    .. math::

      M M_l A M_r y = M M_l b,

    where :math:`x=M_r y`.
    The preconditioned GMRES method then computes (in exact arithmetics!)
    iterates :math:`x_k \\in x_0 + M_r K_k` with
    :math:`K_k:= K_k(M M_l A M_r, r_0)` such that

    .. math::

      \\|M M_l(b - A x_k)\\|_{M^{-1}} =
      \\min_{z \\in x_0 + M_r K_k} \\|M M_l (b - A z)\\|_{M^{-1}}.

    The Arnoldi alorithm is used with the operator
    :math:`M M_l A M_r` and the inner product defined by
    :math:`\\langle x,y \\rangle_{M^{-1}} = \\langle M^{-1}x,y \\rangle`.
    The initial vector for Arnoldi is
    :math:`r_0 = M M_l (b - Ax_0)` - note that :math:`M_r` is not used for
    the initial vector.

    Memory consumption is about maxiter+1 vectors for the Arnoldi basis.
    If :math:`M` is used the memory consumption is 2*(maxiter+1).

    If the operator :math:`M_l A M_r` is self-adjoint then consider using
    the MINRES method :py:meth:`minres`.

    :param A:
      a linear operator on :math:`\\mathbb{C}^N`.
    :param b:
      a vector in :math:`\\mathbb{C}^N`.
    :param x0: (optional) the initial guess to use. Defaults to zero vector.
      Unless you have a good reason to use a nonzero initial guess you should
      use the zero vector, cf. chapter 5.8.3 in *Liesen, Strakos. Krylov
      subspace methods. 2013*. See also :py:meth:`~krypy.utils.hegedus`.
    :param tol: (optional) the tolerance for the stopping criterion with
      respect to the relative residual norm:

      .. math::

         \\frac{ \\| M M_l (b-A (x_0+M_r y_k))\\|_{M^{-1}} }{ \\|M M_l b\\|_{M^{-1}}}
         \leq \\text{tol}

    :param maxiter: (optional)
      maximum number of iterations per restart cycle, see ``max_restarts``.
      Has to fulfill :math:`\\text{maxiter}\leq N`. Defaults to N.
    :param M: (optional)
      a self-adjoint and positive definite preconditioner, linear operator on
      :math:`\\mathbb{C}^N` with respect to ``inner_product``. This
      preconditioner changes the inner product used for orthogonalization to
      :math:`\\langle x,y\\rangle_M = \\langle Mx,y\\rangle` where
      :math:`\\langle \cdot,\cdot\\rangle` is the inner product defined by the
      parameter ``inner_product``. Defaults to the identity.
    :param Ml: (optional) left preconditioner, linear operator on
      :math:`\\mathbb{C}^N`. Defaults to the identity.
    :param Mr: (optional) right preconditioner, linear operator on
      :math:`\\mathbb{C}^N`.  Defaults to the identity.
    :param inner_product: (optional) a function that takes two arguments and
      computes the (block-) inner product of the arguments. Defaults to
      :py:meth:`~krypy.utils.ip_euclid`.
    :param explicit_residual: (optional)
      if set to ``False`` (default), the updated residual norm from the GMRES
      iteration is used in each iteration. If set to ``True``, the residual is
      computed explicitly in each iteration and thus requires an additional
      matrix-vector multiplication in each iteration.
    :param return_basis: (optional)
      if set to ``True`` then the computed Arnoldi basis and the Hessenberg
      matrix are returned in the result dictionary with the keys ``V``
      and ``H``. Defaults to ``False``.
    :param exact_solution:
      if the solution vector :math:`x` is passed then the error norm
      :math:`\|x-x_k\|` will be computed in each iteration (with respect to
      ``inner_product``) and returned as a list in the result dictionary with
      the key ``errvec``. Defaults to ``None``, which means that no errors
      are computed.
    :param max_restarts: the maximum number of restarts. The maximum number of
      iterations is ``(max_restarts+1)*maxiter``.

    :return:
      a dictionary with the following keys:

      * ``xk``: the approximate solution :math:`x_k`.
      * ``info``: convergence flag (0 if converged, 1 otherwise).
      * ``relresvec``: relative residual norms of all iterations, see
        parameter ``tol``.
      * ``V``: present if ``return_basis=True``. The Arnoldi basis
        vectors.
      * ``H``: present if ``return_basis=True``. The Hessenberg matrix.
      * ``P``: present if ``return_basis=True`` and ``M`` is provided.
        The matrix :math:`P` fulfills :math:`V=MP`.
    '''
    relresvec = [numpy.Inf]
    if exact_solution is not None:
        errvec = [numpy.Inf]
    sols = []
    xk = x0
    restart = 0
    while relresvec[-1] > tol and restart <= max_restarts:
        sol = _gmres( A, b,
                xk,
                tol,
                maxiter,
                M,
                Ml,
                Mr,
                inner_product,
                explicit_residual,
                return_basis,
                full_reortho,
                exact_solution,
                # enable warnings in last restart
                conv_warning = (restart==max_restarts))
        xk = sol['xk']
        del relresvec[-1]
        relresvec += sol['relresvec']
        if exact_solution is not None:
            del errvec[-1]
            errvec += sol['errvec']
        sols.append(sol)
        restart += 1
    ret = {
            'xk': xk,
            'info': relresvec[-1] <= tol,
            'relresvec': relresvec
            }
    if exact_solution is not None:
        ret['errvec'] = errvec
    if max_restarts == 0:
        if return_basis:
            ret['V'] = sol['V']
            ret['H'] = sol['H']
            if M is not None:
                ret['P'] = sol['P']
    return ret

def _gmres( A, b,
        x0,
        tol,
        maxiter,
        M,
        Ml,
        Mr,
        inner_product,
        explicit_residual,
        return_basis,
        full_reortho,
        exact_solution,
        conv_warning):
    if not full_reortho:
        raise RuntimeError('full_reortho=False not allowed in GMRES')

    N = len(b)
    shape = (N,N)
    A = utils.get_linearoperator(shape, A)
    if not maxiter:
        maxiter = N
    flat_vecs, (b, x0, exact_solution) = utils.shape_vecs(b, x0, exact_solution)
    if x0 is None:
        x0 = numpy.zeros((N,1))
    M = utils.get_linearoperator(shape, M)
    Ml = utils.get_linearoperator(shape, Ml)
    Mr = utils.get_linearoperator(shape, Mr)
    cdtype = utils.find_common_dtype(A, b, x0, M, Ml, Mr)

    # --------------------------------------------------------------------------
    def _compute_explicit_xk(H, V, y):
        '''Compute approximation xk to the solution.'''
        if (H.shape[0]>0):
            yy = numpy.linalg.solve(H, y)
            u  = Mr * numpy.dot(V, yy)
            return x0+u
        return x0
    # --------------------------------------------------------------------------
    def _compute_explicit_residual( xk ):
        '''Compute residual explicitly.'''
        rk  = b - A * xk
        rk  = Ml * rk
        Mrk  = M * rk
        norm_Mrk = utils.norm(rk, Mrk, inner_product=inner_product)
        return Mrk, norm_Mrk
    # --------------------------------------------------------------------------

    # get memory for working variables
    V = numpy.zeros([N, maxiter+1], dtype=cdtype) # Arnoldi basis
    H = numpy.zeros([maxiter+1, maxiter], dtype=cdtype) # Hessenberg matrix
    y = numpy.zeros( (maxiter+1,1), dtype=cdtype )

    if M is not None:
        P = numpy.zeros([N,maxiter+1], dtype=cdtype) # V=M*P

    if return_basis:
        Horig = numpy.zeros([maxiter+1,maxiter], dtype=cdtype)

    # initialize working variables
    Mlb = Ml * b
    MMlb = M * Mlb
    norm_MMlb = utils.norm(Mlb, MMlb, inner_product=inner_product)
    # This may only save us the application of Ml to the same vector again if
    # x0 is the zero vector.
    norm_x0 = utils.norm(x0, inner_product=inner_product)
    if norm_x0 > numpy.finfo(float).eps:
        r0 = b - A * x0
        Mlr0 = Ml * r0
        MMlr0 = M * Mlr0
        norm_MMlr0 = utils.norm(Mlr0, MMlr0, inner_product=inner_product)
    else:
        x0 = numpy.zeros( (N,1) )
        Mlr0 = Mlb.copy()
        MMlr0 = MMlb.copy()
        norm_MMlr0 = norm_MMlb

    # if rhs is exactly(!) zero, return zero solution.
    if norm_MMlb==0:
        x0 = numpy.zeros((N,1))
        relresvec = [0.0]
    else:
        # initial relative residual norm
        relresvec = [norm_MMlr0 / norm_MMlb]

    # compute error?
    if exact_solution is not None:
        errvec = [utils.norm(exact_solution - x0, inner_product = inner_product)]

    # initialize only if needed
    if relresvec[-1] > tol:
        V[:, [0]] = MMlr0 / norm_MMlr0
        if M is not None:
            P[:, [0]] = Mlr0 / norm_MMlr0
        # Right hand side of projected system:
        y[0] = norm_MMlr0
        # Givens rotations:
        G = []

    k = 0
    while relresvec[-1] > tol and k < maxiter:
        # Apply operator Ml*A*Mr
        z = Ml * (A * (Mr * V[:, [k]]))

        # orthogonalize (MGS)
        for i in range(k+1):
            if M is not None:
                H[i, k] += inner_product(V[:, [i]], z)[0,0]
                z -= H[i, k] * P[:, [i]]
            else:
                H[i, k] += inner_product(V[:, [i]], z)[0,0]
                z -= H[i, k] * V[:, [i]]
        Mz = M * z
        norm_Mz = utils.norm(z, Mz, inner_product=inner_product)
        H[k+1,k] = norm_Mz
        if return_basis:
            Horig[0:k+2, [k]] = H[0:k+2, [k]]

        # Apply previous Givens rotations.
        for i in range(k):
            H[i:i+2, k] = G[i].apply(H[i:i+2, k])

        # Compute and apply new Givens rotation.
        G.append(utils.Givens(H[k:k+2, [k]]))
        H[k:k+2, k] = G[k].apply(H[k:k+2, k])
        y[k:k+2] = G[k].apply(y[k:k+2])

        if exact_solution is not None:
            xk = _compute_explicit_xk(H[:k+1, :k+1], V[:, :k+1], y[:k+1])
            errvec.append(utils.norm(exact_solution - xk, inner_product=inner_product))

        # Update residual norm.
        if explicit_residual:
            xk = _compute_explicit_xk(H[:k+1, :k+1], V[:, :k+1], y[:k+1])
            Mrk, norm_Mrk = _compute_explicit_residual( xk )
            relresvec.append(norm_Mrk / norm_MMlb)
        else:
            relresvec.append(abs(y[k+1,0]) / norm_MMlb)

        # convergence of updated residual or maxiter reached?
        if relresvec[-1] < tol or k+1 == maxiter:
            norm_ur = relresvec[-1]

            if not explicit_residual:
                xk = _compute_explicit_xk(H[:k+1, :k+1], V[:, :k+1], y[:k+1])
                Mrk, norm_Mrk = _compute_explicit_residual( xk )
                relresvec[-1] = norm_Mrk / norm_MMlb

            # No convergence of expl. residual?
            if relresvec[-1] >= tol:
                # Was this the last iteration?
                if k+1 == maxiter:
                    if conv_warning:
                        warnings.warn('Iter %d: No convergence! expl. res = %e >= tol =%e in last it. (upd. res = %e)' \
                            % (k+1, relresvec[-1], tol, norm_ur))
                else:
                    if conv_warning:
                        warnings.warn('Iter %d: Expl. res = %e >= tol = %e > upd. res = %e.' \
                            % (k+1, relresvec[-1], tol, norm_ur))

        if norm_Mz < 1e-14 and relresvec[-1] > tol:
            warnings.warn('subdiagonal element is (close to) zero (%e) => breakdown in iteration %d' % (norm_Mz, k))
        if M is not None:
            P[:, [k+1]] = z / norm_Mz
        V[:, [k+1]] = Mz / norm_Mz

        k += 1

    xk = _compute_explicit_xk(H[:k,:k], V[:,:k], y[:k])
    ret = { 'xk': xk if not flat_vecs else numpy.ndarray.flatten(xk),
            'info': relresvec[-1] <= tol,
            'relresvec': relresvec
            }
    if exact_solution is not None:
        ret['errvec'] = errvec
    if return_basis:
        ret['V'] = V[:, :k+1]
        ret['H'] = Horig[:k+1, :k]
        if M is not None:
            ret['P'] = P[:, :k+1]
    return ret
