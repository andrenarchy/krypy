# -*- coding: utf8 -*-
import numpy
import warnings
from . import utils
import scipy.linalg

__all__ = ['Cg', 'Minres', 'Gmres']


class ConvergenceError(RuntimeError):
    '''Convergence error.

    The ``ConvergenceError`` holds a message describing the error and
    the attribute ``solver`` through which the last approximation and other
    relevant information can be retrieved.
    '''
    def __init__(self, msg, solver):
        super(ConvergenceError, self).__init__(msg)
        self.solver = solver


class _Solver(object):
    def __init__(self, A, b,
                 x0=None,
                 tol=1e-5,
                 maxiter=None,
                 M=None,
                 Ml=None,
                 Mr=None,
                 ip_B=None,
                 explicit_residual=False,
                 store_basis=False,
                 reortho='None',
                 exact_solution=None
                 ):
        '''Init standard attributes and perform checks.'''
        N = len(b)
        shape = (N, N)

        # init operators
        self.A = utils.get_linearoperator(shape, A)
        self.M = utils.get_linearoperator(shape, M)
        self.Ml = utils.get_linearoperator(shape, Ml)
        self.Mr = utils.get_linearoperator(shape, Mr)
        self.ip_B = ip_B

        # sanitize right hand side, initial guess, exact_solution
        self.maxiter = N if maxiter is None else maxiter
        self.flat_vecs, (self.b, self.x0, self.exact_solution) = \
            utils.shape_vecs(b, x0, exact_solution)
        if self.x0 is None:
            self.x0 = numpy.zeros((N, 1))
        self.tol = tol
        self.xk = None
        self.cdtype = utils.find_common_dtype(self.A, self.b, self.x0, self.M,
                                              self.Ml, self.Mr, self.ip_B)

        self.explicit_residual = explicit_residual
        self.store_basis = False
        # TODO: reortho

        # init vectors for convergence measurement
        self.resnorms = []
        if exact_solution is not None:
            self.errnorms = []

    def _prepare(self, norm_MMlr0):
        '''Prepare after Arnoldi/Lanczos was initialized.

        :param norm_MMlr0: norm of the initial residual (as computed by
          Arnoldi/Lanczos in the initialization).
        '''
        self.norm_MMlr0 = norm_MMlr0
        N = self.b.shape[0]

        # Compute M^{-1}-norm of M*Ml*b.
        Mlb = self.Ml*self.b
        MMlb = self.M*Mlb
        self.norm_MMlb = utils.norm(Mlb, MMlb, ip_B=self.ip_B)

        # if rhs is exactly(!) zero, return zero solution.
        if self.norm_MMlb == 0:
            self.x0 = numpy.zeros((N, 1))
            self.resnorms.append(0.)
        else:
            # initial relative residual norm
            self.resnorms.append(norm_MMlr0 / self.norm_MMlb)

        # compute error?
        if self.exact_solution is not None:
            self.errnorms.append(utils.norm(self.exact_solution - self.x0,
                                            ip_B=self.ip_B))

    def _compute_xk(self, yk):
        return self.x0 + self.Mr * yk

    def _compute_rkn(self, xk):
        rk = self.b - self.A*xk
        self.Mlrk = self.Ml*rk
        self.MMlrk = self.M*self.Mlrk
        return utils.norm(self.Mlrk, self.MMlrk, ip_B=self.ip_B)

    def _compute_norms(self, yk, resnorm):
        '''Compute solution, error norm and residual norm if required.

        :return: the residual norm or ``None``.
        '''
        self.xk = None
        # compute error norm if asked for
        if self.exact_solution is not None:
            self.xk = self._compute_xk(yk)
            self.errnorms.append(utils.norm(self.exact_solution - self.xk,
                                            ip_B=self.ip_B))

        rkn = None

        # compute explicit residual if asked for or if the updated residual
        # is below the tolerance or if this is the last iteration
        if self.explicit_residual or resnorm/self.norm_MMlb <= self.tol \
                or self.iter+1 == self.maxiter:
            # compute xk if not yet done
            if self.xk is None:
                self.xk = self._compute_xk(yk)

            # compute residual norm
            rkn = self._compute_rkn(self.xk)

            # store relative residual norm
            self.resnorms.append(rkn/self.norm_MMlb)

            # no convergence?
            if self.resnorms[-1] > self.tol:
                # no convergence in last iteration -> raise exception
                # (approximate solution can be obtained from exception)
                if self.iter+1 == self.maxiter:
                    raise ConvergenceError(
                        'No convergence in last iteration.', self)
                # updated residual was below but explicit is not: warn
                elif not self.explicit_residual \
                        and resnorm/self.norm_MMlb <= self.tol:
                    warnings.warn(
                        'updated residual is below tolerance, explicit '
                        'residual is NOT! (upd={0} <= tol={1} < exp={2})'
                        .format(resnorm, self.tol, self.resnorms[-1]))
        else:
            # only store updated residual
            self.resnorms.append(resnorm/self.norm_MMlb)

        return rkn


class Cg(_Solver):
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
    :param tol: (optional) the tolerance for the stopping criterion with
      respect to the relative residual norm:

      .. math::

         \\frac{ \\| M M_l (b-A (x_0+M_r y_k))\\|_{M^{-1}} }
         { \\|M M_l b\\|_{M^{-1}}}
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
    def __init__(self, A, b, ortho='lanczos', **kwargs):
        super(Cg, self).__init__(A, b, **kwargs)
        self._solve(ortho=ortho)

    def _solve(self, ortho):
        N = self.b.shape[0]

        # compute norm of r0
        r0 = self.b - self.A * self.x0
        Mlr0 = self.Ml * r0
        MMlr0 = self.M * Mlr0
        norm_MMlr0 = utils.norm(Mlr0, MMlr0, ip_B=self.ip_B)

        # prepare
        self._prepare(norm_MMlr0)

        # resulting approximation is xk = x0 + Mr*yk
        yk = numpy.zeros((N, 1), dtype=self.cdtype)

        # square of the old residual norm
        rho_old = norm_MMlr0**2

        # will be updated by _compute_rkn if explicit_residual is True
        self.Mlrk = Mlr0.copy()
        self.MMlrk = MMlr0.copy()

        # search direction
        p = self.MMlrk.copy()
        self.iter = 0
        rho_new = 0  # will be set at end of iteration

        # iterate
        while self.resnorms[-1] > self.tol and self.iter < self.maxiter:
            if self.iter > 0:
                # update the search direction
                p = self.MMlrk + rho_new/rho_old * p
                rho_old = rho_new
            # apply operators
            Ap = (self.Ml * (self.A * (self.Mr * p)))

            # compute inner product
            alpha = rho_old / utils.inner(p, Ap, ip_B=self.ip_B)

            # check if alpha is real
            if abs(alpha.imag) > 1e-12:
                warnings.warn(
                    'Iter {0}: abs(alpha.imag) = {1} > 1e-12. '
                    'Is your operator self-adjoint in the provided inner '
                    'product?'
                    .format(self.iter, abs(alpha.imag)))
            alpha = alpha.real

            # update solution
            yk += alpha * p

            # update residual
            self.Mlrk -= alpha * Ap

            # apply preconditioner
            self.MMlrk = self.M * self.Mlrk

            # compute norm and rho_new
            norm_MMlrk = utils.norm(self.Mlrk, self.MMlrk, ip_B=self.ip_B)
            rho_new = norm_MMlrk**2

            # compute norms
            # if explicit_residual: compute Mlrk and MMlrk here
            # (with preconditioner application)
            rkn = self._compute_norms(yk, norm_MMlrk)

            # update rho_new if it was updated in _compute_norms
            if rkn is not None:
                # new rho
                rho_new = rkn**2

            self.iter += 1

        # compute solution if not yet done
        if self.xk is None:
            self.xk = self._compute_xk(yk)

    @staticmethod
    def operations(nsteps):
        '''Returns the number of operations needed for nsteps of CG'''
        return {'A': 1 + nsteps,
                'M': 2 + nsteps,
                'Ml': 2 + nsteps,
                'Mr': 1 + nsteps,
                'ip': 2 + 2*nsteps,
                'axpy': 2 + 2*nsteps
                }


class Minres(_Solver):
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

         \\frac{ \\| M M_l (b-A (x_0+M_r y_k))\\|_{M^{-1}} }
         { \\|M M_l b\\|_{M^{-1}}}
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

    Upon convergence, the ``Minres`` instance contains the following
    attributes:

      * ``xk``: the approximate solution :math:`x_k`.
      * ``resnorms``: relative residual norms of all iterations, see
        parameter ``tol``.
      * ``errnorms``: the error norms of all iterations if ``exact_solution``
        was provided.
      * ``lanczos``: the Arnoldi instance.

    If MINRES does not converge, a :py:class:`ConvergenceError` is thrown.
    '''
    def __init__(self, A, b, ortho='lanczos', **kwargs):
        super(Minres, self).__init__(A, b, **kwargs)
        self._solve(ortho=ortho)

    def _solve(self, ortho):
        N = self.b.shape[0]

        # initialize Lanczos
        self.lanczos = utils.Arnoldi(self.Ml*self.A*self.Mr,
                                     self.Ml*(self.b - self.A*self.x0),
                                     maxiter=self.maxiter,
                                     ortho=ortho,
                                     M=self.M,
                                     ip_B=self.ip_B
                                     )

        # prepare
        self._prepare(self.lanczos.vnorm)

        # Necessary for efficient update of yk:
        W = numpy.c_[numpy.zeros(N, dtype=self.cdtype), numpy.zeros(N)]
        # some small helpers
        y = [self.norm_MMlr0, 0]  # first entry is (updated) residual
        G2 = None                 # old givens rotation
        G1 = None                 # even older givens rotation ;)

        # resulting approximation is xk = x0 + Mr*yk
        yk = numpy.zeros((N, 1), dtype=self.cdtype)

        # iterate Lanczos
        while self.resnorms[-1] > self.tol \
                and self.lanczos.iter < self.lanczos.maxiter \
                and not self.lanczos.invariant:
            k = self.iter = self.lanczos.iter
            self.lanczos.advance()
            V, H = self.lanczos.V, self.lanczos.H

            # needed for QR-update:
            R = numpy.zeros((4, 1))  # real because Lanczos matrix is real
            R[1] = H[k-1, k].real
            if G1 is not None:
                R[:2] = G1.apply(R[:2])

            # (implicit) update of QR-factorization of Lanczos matrix
            R[2:4, 0] = [H[k, k].real, H[k+1, k].real]
            if G2 is not None:
                R[1:3] = G2.apply(R[1:3])
            G1 = G2
            # compute new givens rotation.
            G2 = utils.Givens(R[2:4])
            R[2] = G2.r
            R[3] = 0.0
            y = G2.apply(y)

            # update solution
            z = (V[:, [k]] - R[0, 0]*W[:, [0]] - R[1, 0]*W[:, [1]]) / R[2, 0]
            W = numpy.c_[W[:, [1]], z]
            yk = yk + y[0] * z
            y = [y[1], 0]

            self._compute_norms(yk, numpy.abs(y[0]))

        # compute solution if not yet done
        if self.xk is None:
            self.xk = self._compute_xk(yk)

    @staticmethod
    def operations(nsteps):
        '''Returns the number of operations needed for nsteps of MINRES'''
        return {'A': 1 + nsteps,
                'M': 2 + nsteps,
                'Ml': 2 + nsteps,
                'Mr': 1 + nsteps,
                'ip': 2 + 2*nsteps,
                'axpy': 4 + 8*nsteps
                }


class _RestartedSolver(_Solver):
    def __init__(self, Solver, A, b, max_restarts=0, **kwargs):
        super(_RestartedSolver, self).__init__(A, b, **kwargs)

        # start with initial guess
        self.xk = self.x0

        # work on own copy of args in order to include proper initial guesses
        kwargs = dict(kwargs)

        # append dummy values for first run
        self.resnorms.append(numpy.Inf)
        if self.exact_solution is not None:
            self.errnorms.append(numpy.Inf)

        restart = 0
        while self.resnorms[-1] > self.tol and restart <= max_restarts:
            try:
                # use last approximate solution as initial guess
                kwargs.update({'x0': self.xk})

                # try to solve
                sol = Solver(A, b, **kwargs)
            except ConvergenceError as e:
                # use solver of exception
                sol = e.solver

            # set last approximate solution
            self.xk = sol.xk
            print(sol.xk)

            # concat resnorms / errnorms
            del self.resnorms[-1]
            self.resnorms += sol.resnorms
            if self.exact_solution is not None:
                del self.errnorms[-1]
                self.errnorms += sol.errnorms

            restart += 1

        if self.resnorms[-1] > self.tol:
            raise ConvergenceError(
                'No convergence after {0} restarts.'.format(max_restarts),
                self)


class RestartedGmres(_RestartedSolver):
    def __init__(self, A, b, **kwargs):
        super(RestartedGmres, self).__init__(Gmres, A, b, **kwargs)


class Gmres(_Solver):
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

         \\frac{ \\| M M_l (b-A (x_0+M_r y_k))\\|_{M^{-1}} }
         { \\|M M_l b\\|_{M^{-1}}}
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
    def __init__(self, A, b, ortho='mgs', **kwargs):
        super(Gmres, self).__init__(A, b, **kwargs)
        self._solve(ortho=ortho)

    def _compute_xk(self, y):
        k = self.arnoldi.iter
        if k > 0:
            yy = scipy.linalg.solve_triangular(self.R[:k, :k], y)
            return self.x0 + self.Mr * self.V[:, :k].dot(yy)
        return self.x0

    def _solve(self, ortho):
        # initialize Arnoldi
        self.arnoldi = utils.Arnoldi(self.Ml*self.A*self.Mr,
                                     self.Ml*(self.b - self.A*self.x0),
                                     maxiter=self.maxiter,
                                     ortho=ortho,
                                     M=self.M,
                                     ip_B=self.ip_B
                                     )

        # prepare
        self._prepare(self.arnoldi.vnorm)

        # Givens rotations:
        G = []
        # QR decomposition of Hessenberg matrix via Givens and R
        self.R = numpy.zeros([self.maxiter+1, self.maxiter],
                             dtype=numpy.complex)
        y = numpy.zeros((self.maxiter+1, 1), dtype=numpy.complex)
        # Right hand side of projected system:
        y[0] = self.norm_MMlr0

        # iterate Arnoldi
        while self.resnorms[-1] > self.tol \
                and self.arnoldi.iter < self.arnoldi.maxiter \
                and not self.arnoldi.invariant:
            k = self.iter = self.arnoldi.iter
            self.arnoldi.advance()

            # Copy new column from Arnoldi
            self.V = self.arnoldi.V
            self.R[:k+2, k] = self.arnoldi.H[:k+2, k]

            # Apply previous Givens rotations.
            for i in range(k):
                self.R[i:i+2, k] = G[i].apply(self.R[i:i+2, k])

            # Compute and apply new Givens rotation.
            G.append(utils.Givens(self.R[k:k+2, [k]]))
            self.R[k:k+2, k] = G[k].apply(self.R[k:k+2, k])
            y[k:k+2] = G[k].apply(y[k:k+2])

            self._compute_norms(y[:k+1], abs(y[k+1, 0]))

        # compute solution if not yet done
        if self.xk is None:
            self.xk = self._compute_xk(y[:self.arnoldi.iter])

    @staticmethod
    def operations(nsteps):
        '''Returns the number of operations needed for nsteps of GMRES'''
        return {'A': 1 + nsteps,
                'M': 2 + nsteps,
                'Ml': 2 + nsteps,
                'Mr': 1 + nsteps,
                'ip': 2 + nsteps + nsteps*(nsteps+1)/2,
                'axpy': 4 + 2*nsteps + nsteps*(nsteps+1)/2
                }
