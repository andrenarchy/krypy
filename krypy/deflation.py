# -*- coding: utf8 -*-
import numpy
import scipy.linalg
from . import utils, linsys


__all__ = ['DeflatedCg', 'DeflatedMinres', 'DeflatedGmres', '_DeflationMixin',
           'ObliqueProjection', '_Projection', 'Ritz', 'Arnoldifyer',
           'bound_pseudo']


class _Projection(utils.Projection):
    def __init__(self, linear_system, U, **kwargs):
        '''Abstract base class of a projection for deflation.

        :param A: the :py:class:`~krypy.linsys.LinearSystem`.
        :param U: basis of the deflation space with ``U.shape == (N, d)``.

        All parameters of :py:class:`~krypy.utils.Projection` are valid except
        ``X`` and ``Y``.
        '''
        raise NotImplementedError('abstract base class cannot be instanciated')


class ObliqueProjection(_Projection):
    def __init__(self, linear_system, U, **kwargs):
        '''Oblique projection for left deflation.'''
        # preprocess and store input
        self.linear_system = linear_system
        (N, d) = U.shape
        U, _ = utils.qr(U, ip_B=linear_system.ip_B)

        self.U = U
        '''An orthonormalized basis of the deflation space ``U`` with respect
        to provided inner product.'''

        # apply operator to U
        self.AU = linear_system.MlAMr*U
        '''Result of application of operator to deflation space, i.e.,
        :math:`M_lAM_rU`.'''

        # call Projection constructor
        super(_Projection, self).__init__(self.AU, self.U, **kwargs)

    def correct(self, z):
        '''Correct the given approximate solution ``z`` with respect to the
        linear system ``linear_system`` and the deflation space defined by
        ``U``.'''
        c = self.linear_system.Ml*(
            self.linear_system.b - self.linear_system.A*z)
        c = utils.inner(self.W, c, ip_B=self.ip_B)
        if self.Q is not None and self.R is not None:
            c = scipy.linalg.solve_triangular(self.R, self.Q.T.conj().dot(c))
        if self.WR is not self.VR:
            c = self.WR.dot(scipy.linalg.solve_triangular(self.VR, c))
        return z + self.W.dot(c)


class _DeflationMixin(object):
    '''Mixin class for deflation in Krylov subspace methods.

    Can be used to add deflation functionality to any solver from
    :py:mod:`~krypy.linsys`.

    :param linear_system: the :py:class:`~krypy.linsys.LinearSystem` that
      should be solved.
    :param U: a basis of the deflation space with ``U.shape == (N, k)``.

    All other parameters are passed through to the underlying solver from
    :py:mod:`~krypy.linsys`.
    '''
    def __init__(self, linear_system, U=None, *args, **kwargs):
        if U is None:
            U = numpy.zeros((linear_system.N, 0))
        projection = ObliqueProjection(linear_system, U)
        self.projection = projection
        '''Projection that is used for deflation.'''

        if projection.Q is None and projection.R is None:
            E = numpy.eye(U.shape[1])
        else:
            E = projection.Q.dot(projection.R)
        if projection.VR is not None and projection.WR is not None:
            E = projection.WR.T.conj().dot(E.dot(projection.VR))
        self.E = E
        r''':math:`E=\langle U,M_lAM_rU\rangle`.'''

        self.C = numpy.zeros((U.shape[1], 0))
        r''':math:`C=\langle U,M_lAM_rV_n\rangle`.

        This attribute is updated while the Arnoldi/Lanczos method proceeds.
        See also :py:meth:`_apply_projection`.
        '''

        self._B_ = None

        super(_DeflationMixin, self).__init__(linear_system,
                                              *args, **kwargs)

    def _solve(self):
        N = self.linear_system.N
        P = utils.LinearOperator((N, N), self.projection.AU.dtype,
                                 self._apply_projection)
        self.MlAMr = P*self.linear_system.MlAMr
        super(_DeflationMixin, self)._solve()

    def _apply_projection(self, Av):
        '''Apply the projection and store inner product.

        :param v: the vector resulting from an application of :math:`M_lAM_r`
          to the current Arnoldi vector. (CG needs special treatment, here).
        '''
        PAv, UAv = self.projection.apply_complement(Av, return_Ya=True)
        self.C = numpy.c_[self.C, UAv]
        return PAv

    def _get_initial_residual(self, x0):
        '''Return the projected initial residual.

        Returns :math:`MPM_l(b-Ax_0)`.
        '''
        if x0 is None:
            Mlr = self.linear_system.Mlb
        else:
            r = self.linear_system.b - self.linear_system.A*x0
            Mlr = self.linear_system.Ml*r

        PMlr, self.UMlr = self.projection.apply_complement(Mlr, return_Ya=True)
        MPMlr = self.linear_system.M*PMlr
        MPMlr_norm = utils.norm(PMlr, MPMlr, ip_B=self.linear_system.ip_B)
        return MPMlr, PMlr, MPMlr_norm

    def _get_xk(self, yk):
        xk = super(_DeflationMixin, self)._get_xk(yk)
        return self.projection.correct(xk)

    @property
    def B_(self):
        r''':math:`\underline{B}=\langle V_{n+1},M_lAM_rU\rangle`.

        This property is obtained from :math:`C` if the operator is
        self-adjoint. Otherwise, the inner products have to be formed
        explicitly.'''
        (n_, n) = self.H.shape
        ls = self.linear_system
        if self._B_ is None or self._B_.shape[1] < n_:
            # compute B_
            if ls.self_adjoint:
                self._B_ = self.C.T.conj()
                if n_ > n:
                    self._B_ = numpy.r_[self._B_,
                                        utils.inner(self.V[:, [-1]],
                                                    self.projection.AU,
                                                    ip_B=ls.ip_B)]
            else:
                self._B_ = utils.inner(self.V, self.projection.AU,
                                       ip_B=ls.ip_B)
        return self._B_


class DeflatedCg(_DeflationMixin, linsys.Cg):
    '''Deflated preconditioned CG method.

    See :py:class:`_DeflationMixin` and :py:class:`~krypy.linsys.Cg` for
    the documentation of the available parameters.
    '''
    def __init__(self, *args, **kwargs):
        self._UAps = []
        super(DeflatedCg, self).__init__(*args, **kwargs)

    def _apply_projection(self, Av):
        r'''Computes :math:`\langle C,M_lAM_rV_n\rangle` efficiently with a
        three-term recurrence.'''
        PAv, UAp = self.projection.apply_complement(Av, return_Ya=True)
        self._UAps.append(UAp)
        c = UAp.copy()
        rhos = self.rhos
        if self.iter > 0:
            c -= (1 + rhos[-1]/rhos[-2])*self._UAps[-2]
        if self.iter > 1:
            c += rhos[-2]/rhos[-3]*self._UAps[-3]
        c *= ((-1)**self.iter) / numpy.sqrt(rhos[-1])
        if self.iter > 0:
            c -= numpy.sqrt(rhos[-2]/rhos[-1]) * self.C[:, [-1]]
        self.C = numpy.c_[self.C, c]
        return PAv


class DeflatedMinres(_DeflationMixin, linsys.Minres):
    '''Deflated preconditioned MINRES method.

    See :py:class:`_DeflationMixin` and :py:class:`~krypy.linsys.Minres` for
    the documentation of the available parameters.
    '''
    pass


class DeflatedGmres(_DeflationMixin, linsys.Gmres):
    '''Deflated preconditioned GMRES method.

    See :py:class:`_DeflationMixin` and :py:class:`~krypy.linsys.Gmres` for
    the documentation of the available parameters.
    '''
    pass


class Arnoldifyer(object):
    def __init__(self, deflated_solver):
        r'''Obtain Arnoldi relations for approximate deflated Krylov subspaces.

        :param deflated_solver: an instance of a deflated solver.
        '''
        self._deflated_solver = deflated_solver
        H = deflated_solver.H
        B_ = deflated_solver.B_
        C = deflated_solver.C
        E = deflated_solver.E

        V = deflated_solver.V
        U = deflated_solver.projection.U
        AU = deflated_solver.projection.AU

        # get dimensions
        n_, n = self.n_, self.n = H.shape
        d = self.d = deflated_solver.projection.U.shape[1]

        # store a few matrices for later use
        EinvC = numpy.linalg.solve(E, C) if d > 0 else numpy.zeros((0, n))
        self.L = numpy.bmat([[H, numpy.zeros((n_, d))],
                             [EinvC, numpy.eye(d)]])
        self.J = numpy.bmat([[numpy.eye(n, n_), B_[:n, :]],
                             [numpy.zeros((d, n_)), E]])
        self.M = numpy.bmat([[H[:n, :n]
                              + B_[:n, :].dot(EinvC),
                              B_[:n, :]],
                             [C, E]])
        self.A_norm = numpy.linalg.norm(self.M, 2)

        if d > 0:
            # rank-revealing QR decomp of projected AU
            Q, R, P = scipy.linalg.qr(AU - U.dot(E) - V.dot(B_),
                                      mode='economic', pivoting=True)
            P_inv = numpy.argsort(P)

            # rank of R
            l = (numpy.abs(numpy.diag(R)) > 1e-14*self.A_norm).sum()
            Q1 = Q[:, :l]
            self.R12 = R[:l, P_inv]
            # residual helper matrix
            self.N = numpy.c_[numpy.eye(l+n_-n, n_-n),
                              numpy.r_[B_[n:, :],
                                       self.R12]
                              ].dot(
                numpy.bmat([[numpy.zeros((d+n_-n, n)),
                             numpy.eye(d+n_-n)]]))
        else:
            Q1 = numpy.zeros((U.shape[0], 0))
            self.R12 = numpy.zeros((0, 0))
            self.N = numpy.bmat([[numpy.zeros((n_-n, n)),
                                  numpy.eye(n_-n, n_-n)]])

        # residual basis
        self.Z = numpy.c_[V[:, n:], Q1]

    def get(self, Wt, full=False):
        r'''Get Arnoldi relation for a deflation subspace choice.

        :param Wt: the coefficients :math:`\tilde{W}` of the deflation vectors
          in the basis :math:`[V_n,U]` with ``Wt.shape == (n+d, k)``, i.e., the
          deflation vectors are :math:`W=[V_n,U]\tilde{W}`. Must fulfill
          :math:`\tilde{W}^*\tilde{W}=I_k`.
        :param full: (optional) should the full Arnoldi
          basis and the full perturbation be returned? Defaults to ``False``.

        :return:

          * ``Hh``: the Hessenberg matrix with ``Hh.shape == (n+d-k, n+d-k)``.
          * ``Rh``: the perturbation core matrix with
            ``Rh.shape == (l, n+d-k)``.
          * ``q_norm``: norm :math:`\|q\|_2`.
          * ``vdiff_norm``: the norm of the difference of the initial vectors
            :math:`\tilde{v}-\hat{v}`.
          * ``PWAW_norm``: norm of the projection
            :math:`P_{\mathcal{W}^\perp,A\mathcal{W}}`.
          * ``Vh``: (if ``full == True``) the Arnoldi basis with ``Vh.shape ==
            (N, n+d-k)``.
          * ``F``: (if ``full == True``) the perturbation matrix
            :math:`F=-Z\hat{R}\hat{V}_n^* - \hat{V}_n\hat{R}^*Z^*`.
        '''
        n = self.n
        n_ = self.n_
        d = self.d
        k = Wt.shape[1]

        # get orthonormal basis of Wt^\perp
        if k > 0:
            Wto, _ = scipy.linalg.qr(Wt)
            Wto = Wto[:, k:]
        else:
            Wto = numpy.eye(Wt.shape[0])

        deflated_solver = self._deflated_solver

        Pt = utils.Projection(self.L.dot(Wt), self.J.T.conj().dot(Wt)) \
            .operator_complement()
        if d > 0:
            qt = Pt*(numpy.r_[[[deflated_solver.MMlr0_norm]],
                              numpy.zeros((self.n_-1, 1)),
                              numpy.linalg.solve(deflated_solver.E,
                                                 deflated_solver.UMlr)])
        else:
            qt = Pt*(numpy.r_[[[deflated_solver.MMlr0_norm]],
                              numpy.zeros((self.n_-1, 1))])
        q = Wto.T.conj().dot(self.J.dot(qt))

        # TODO: q seems to suffer from round-off errors and thus the first
        #       vector in the computed basis may differ from the exact
        #       projected one more than on the level of unit round-off.

        # rotate closest vector in [V_n,U] to first column
        Q = utils.House(q)
        q_norm = Q.xnorm

        # Arnoldify
        WtoQ = Q.apply(Wto.T.conj()).T.conj()
        Hh, T = utils.hessenberg(
            Q.apply(Wto.T.conj().dot(self.J).dot(Pt*(self.L.dot(WtoQ)))))

        QT = Q.apply(T)

        # construct residual
        Rh = self.N.dot(Pt*self.L.dot(Wto.dot(QT)))

        # norm of difference between initial vectors
        vdiff = self.N.dot(qt)
        vdiff_norm = 0 if vdiff.size == 0 else numpy.linalg.norm(vdiff, 2)

        # compute norm of projection P_{W^\perp,AW}
        if k > 0:
            # compute coefficients of orthonormalized AW in the basis [V,Z]
            Y = numpy.bmat([[numpy.eye(n_), deflated_solver.B_],
                            [numpy.zeros((d, n_)), deflated_solver.E],
                            [numpy.zeros((self.R12.shape[0], n_)), self.R12]])
            YL_Q, _ = scipy.linalg.qr(Y.dot(self.L.dot(Wt)), mode='economic')

            # compute <W,X> where X is an orthonormal basis of AW
            WX = Wt.T.conj().dot(numpy.r_[YL_Q[:n, :],
                                          YL_Q[n_:n_+d, :]])
            PWAW_norm = 1./numpy.min(scipy.linalg.svdvals(WX))
        else:
            PWAW_norm = 1.

        if full:
            Vh = numpy.c_[deflated_solver.V[:, :n],
                          deflated_solver.projection.U].dot(Wto.dot(QT))
            F = - self.Z.dot(Rh.dot(Vh.T.conj()))
            F = F + F.T.conj()
            return Hh, Rh, q_norm, vdiff_norm, PWAW_norm, Vh, F
        return Hh, Rh, q_norm, vdiff_norm, PWAW_norm


def bound_pseudo(arnoldifyer, Wt, b_norm,
                 g_norm=0.,
                 G_norm=0.,
                 GW_norm=0.,
                 WGW_norm=0.,
                 tol=1e-6,
                 method='gmres',
                 pseudo_type='auto',
                 delta_n=20,
                 delta_exp=5
                 ):
    r'''Bound residual norms of next deflated system.

    :param arnoldifyer: an instance of
      :py:class:`~krypy.deflation.Arnoldifyer`.
    :param Wt: coefficients :math:`\tilde{W}\in\mathbb{C}^{n+d,k}` of the
      considered deflation vectors :math:`W` for the basis :math:`[V,U]`
      where ``V=last_solver.V`` and ``U=last_P.U``, i.e.,
      :math:`W=[V,U]\tilde{W}` and
      :math:`\mathcal{W}=\operatorname{colspan}(W)`. Must fulfill
      :math:`\tilde{W}^*\tilde{W}=I_k`.
    :param b_norm: norm :math:`\|b\|` of right hand side.
    :param g_norm: norm :math:`\|g\|` of difference :math:`g=c-b` of
      right hand sides.
    :param G_norm: norm :math:`\|G\|` of difference
      :math:`G=B-A` of operators.
    :param GW_norm: Norm :math:`\|G|_{\mathcal{W}}\|` of difference
      :math:`G=B-A` of operators restricted to :math:`\mathcal{W}`.
    :param WGW_norm: Norm :math:`\|\langle W,GW\rangle\|_2`.
    :param method: type of method that is used to construct the residuals and
      residual/error polynomial from :math:`\hat{H}`:

      * ``'cg'``: for the CG method
      * ``'minres'``: for the MINRES method
      * ``'gmres'``: for the GMRES method
    :param pseudo_type: One of

      * ``'auto'``: determines if :math:`\hat{H}` is non-normal, normal or
        Hermitian and uses the corresponding mode (see other options below).
      * ``'nonnormal'``: the pseudospectrum of the Hessenberg matrix
        :math:`\hat{H}` is used (involves one computation of a pseudospectrum)
      * ``'normal'``: the pseudospectrum of :math:`\hat{H}` is computed
        efficiently by the union of circles around the eigenvalues.
      * ``'hermitian'``: the pseudospectrum of :math:`\hat{H}` is computed
        efficiently by the union of intervals around the eigenvalues.
      * ``'contain'``: the pseudospectrum of the extended Hessenberg matrix
        :math:`\begin{bmatrix}\hat{H}\\S_i\end{bmatrix}` is used
        (pseudospectrum has to be re computed for each iteration).
    '''
    Hh, Rh, q_norm, vdiff_norm, PWAW_norm = arnoldifyer.get(Wt)

    # smallest singular value of W^*AW
    WAW = Wt.T.conj().dot(arnoldifyer.J.dot(arnoldifyer.L.dot(Wt)))
    sigma_min = numpy.min(scipy.linalg.svdvals(WAW))

    if sigma_min <= WGW_norm:
        raise utils.AssumptionError(
            'sigma_min(W^*AW) > ||W^*GW|| not satisfied.')
    eta = GW_norm/(sigma_min - WGW_norm)
    beta = PWAW_norm*(eta*(b_norm + g_norm) + g_norm) + vdiff_norm

    # spectrum of Hh
    evals, evecs = scipy.linalg.eig(Hh)
    # norm of Hh
    Hh_norm = numpy.linalg.norm(Hh, 2)

    def _auto():
        '''determine pseudo automatically'''
        # is Hh Hermitian?
        if numpy.linalg.norm(Hh-Hh.T.conj(), 2) < 1e-14*Hh_norm:
            return 'hermitian'

        # is Hh normal?
        if numpy.linalg.cond(evecs, 2) < 1+1e-14:
            return 'normal'

        return 'nonnormal'

    if pseudo_type == 'auto':
        pseudo_type = _auto()

    # compute residual norms of Hh*z=e_1*b_norm
    if method == 'cg':
        Solver = linsys.Cg
    elif method == 'minres':
        Solver = linsys.Minres
    elif method == 'gmres':
        Solver = linsys.Gmres
    else:
        raise ValueError('unknown method: {0}'.format(method))
    solver = Solver(Hh, numpy.eye(Hh.shape[0], 1)*b_norm,
                    tol=tol, maxiter=Hh.shape[0])

    import pseudopy
    if pseudo_type == 'nonnormal':
        # todo: construct bounding box!
        raise NotImplementedError('nonnormal not yet implemented')
    elif pseudo_type == 'normal':
        pseudo = pseudopy.NormalEvals(evals)
    elif pseudo_type == 'hermitian':
        raise NotImplementedError('hermitian not yet implemented')

    bounds = []
    for i in range(1, len(solver.resnorms)):
        res = solver.resnorms[i]*b_norm
        epsilon = PWAW_norm*(eta*(Hh_norm + G_norm) + G_norm) \
            + numpy.linalg.norm(Rh[:, :i], 2)
        if pseudo_type == 'contain':
            raise NotImplementedError('contain not yet implemented')
        pseudo_terms = []
        # TODO: compute polynomial from (harmonic) Ritz values
        for delta in epsilon*numpy.logspace(0, delta_exp, delta_n)[1:]:
            # get pseudospectrum paths
            pseudo_path = pseudo.contour_paths(delta)

            # compute roots of polynomial
            if method == 'cg':
                roots = scipy.linalg.eigvalsh(Hh[:i, :i])
            else:
                # TODO: more stable way of computing the roots of the MINRES
                #       poly with exploitation of symmetry?
                HhQ, HhR = scipy.linalg.qr(Hh[:i+1, :i], mode='economic')
                roots_inv = scipy.linalg.eigvals(HhQ[:i, :].T.conj(), HhR)
                roots = 1./roots_inv[numpy.abs(roots_inv) > 1e-14]

            # compute polynomial
            from numpy.polynomial import Polynomial as P
            p = P.fromroots(roots)

            # evaluate polynomial on points of path
            polymax = numpy.max(numpy.abs(p(pseudo_path.vertices())))

            # compute THE bound
            pseudo_terms.append(
                pseudo_path.length()/(2*numpy.pi*delta)
                * (epsilon/(delta-epsilon)*(q_norm + beta) + beta)
                * polymax
                )
        # append minimal vound value
        bounds.append(res + numpy.min(pseudo_terms))
    return numpy.array(bounds)


class Ritz(object):
    def __init__(self, deflated_solver, mode='ritz'):
        '''Compute Ritz pairs from a deflated Krylov subspace method.

        :param deflated_solver: an instance of a deflated solver.
        :param mode: (optional)

          * ``ritz`` (default): compute Ritz pairs.
          * ``harmonic``: compute harmonic Ritz pairs.
        '''
        self._deflated_solver = deflated_solver
        linear_system = deflated_solver.linear_system

        self.values = None
        '''Ritz values.'''

        self.coeffs = None
        '''Coefficients for Ritz vectors in the basis :math:`[V_n,U]`.'''

        H_ = deflated_solver.H
        (n_, n) = H_.shape
        H = H_[:n, :n]
        projection = deflated_solver.projection
        m = projection.U.shape[1]
        I = numpy.eye
        O = numpy.zeros

        if isinstance(projection, ObliqueProjection):
            E = deflated_solver.E
            C = deflated_solver.C
            if m > 0:
                EinvC = numpy.linalg.solve(E, C)
            else:
                EinvC = C
            B_ = deflated_solver.B_
            B = B_[:n, :]

            # build block matrices
            M = numpy.bmat([[H + B.dot(EinvC), B],
                            [C, E]])
            F = utils.inner(projection.AU, projection.AU,
                            ip_B=linear_system.ip_B)
            S = numpy.bmat([[I(n_), B_, O((n_, m))],
                            [B_.T.conj(), F, E],
                            [O((m, n_)), E.T.conj(), I(m)]])

            # use eigh for self-adjoint problems
            eig = scipy.linalg.eigh if linear_system.self_adjoint \
                else scipy.linalg.eig

            # solve eigenvalue problem
            if mode == 'ritz':
                self.values, self.coeffs = eig(M)
            elif mode == 'harmonic':
                L = numpy.bmat([[H_, O((n_, m))],
                                [EinvC, I(m)]])
                K = numpy.bmat([[I(n_), B_],
                                [B_.T.conj(), F]])
                sigmas, self.coeffs = eig(M.T.conj(),
                                          L.T.conj().dot(K.dot(L)))
                self.values = numpy.zeros(m+n, dtype=sigmas.dtype)
                zero = numpy.abs(sigmas) < numpy.finfo(float).eps
                self.values[~zero] = 1./sigmas[~zero]
                self.values[zero] = numpy.Inf
            else:
                raise ValueError('mode {0} not known'.format(mode))

            self.resnorms = numpy.zeros(m+n)
            '''Residual norms of Ritz pairs.'''

            for i in range(n+m):
                mu = self.values[i]
                y = self.coeffs[:, [i]]
                G = numpy.bmat([[H_ - mu*I(n_, n), O((n_, m))],
                                [EinvC, I(m)],
                                [O((m, n)), -mu*I(m)]])
                Gy = G.dot(y)
                resnorm2 = Gy.T.conj().dot(S.dot(Gy))
                self.resnorms[i] = numpy.sqrt(numpy.abs(resnorm2))

        #elif isinstance(P, TODO):
            #TODO
        else:
            raise ValueError('projection is invalid')

    def get_vectors(self, indices=None):
        '''Compute Ritz vectors.'''
        coeffs = self.coeffs if indices is None else self.coeffs[:, indices]
        return numpy.c_[self._deflated_solver.V[:, :-1],
                        self._deflated_solver.projection.U].dot(coeffs)

    def get_explicit_residual(self, indices=None):
        '''Explicitly computes the Ritz residual.'''
        ritz_vecs = self.get_vectors(indices)
        return self._deflated_solver.linear_system.MlAMr * ritz_vecs \
            - ritz_vecs * self.values

    def get_explicit_resnorms(self, indices=None):
        '''Explicitly computes the Ritz residual norms.'''
        res = self.get_explicit_residual(indices)

        # apply preconditioner
        linear_system = self._deflated_solver.linear_system
        Mres = linear_system.M * res

        # compute norms
        resnorms = numpy.zeros(res.shape[1])
        for i in range(resnorms.shape[0]):
            resnorms[i] = utils.norm(res[:, [i]], Mres[:, [i]],
                                     ip_B=linear_system.ip_B)
        return resnorms
