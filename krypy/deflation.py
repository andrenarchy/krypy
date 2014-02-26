# -*- coding: utf8 -*-
import numpy
import scipy.linalg
from . import utils, linsys


class ObliqueProjection(utils.Projection):
    def __init__(self, A, U, ip_B=None,
                 **kwargs):
        '''Oblique projection for (right) deflation.

        :param A: the linear operator (has to be compatible with
          :py:meth:`~krypy.utils.get_linearoperator`).
        :param U: basis of the deflation space with ``U.shape == (N, d)``.

        All parameters of :py:class:`~krypy.utils.Projection` are valid except
        ``X`` and ``Y``.
        '''
        # check and store input
        (N, d) = U.shape
        self.A = utils.get_linearoperator((N, N), A)
        U, _ = utils.qr(U, ip_B=ip_B)

        self.U = U
        '''Orthonormal basis of deflation space.

        An orthonormalized basis of ``U`` with respect to provided inner
        product.'''

        # apply adjoint operator to U
        self.AH_U = A.adj*U

        # call Projection constructor
        super(ObliqueProjection, self).__init__(U, self.AH_U, **kwargs)

    def _correction(self, z):
        c = utils.inner(self.U, z, ip_B=self.ip_B)
        c = scipy.linalg.solve_triangular(self.WR.T.conj(), c, lower=True)
        if self.Q is not None and self.R is not None:
            c = scipy.linalg.solve_triangular(self.R, self.Q.T.conj().dot(c))
        return self.V.dot(c)

    def get_x0(self, pre):
        '''Get corrected initial guess for deflation.

        :param b: the right hand side. If a left preconditioner ``Ml`` is used,
          then it has to be applied to ``b`` before passing it to this
          function. This does not apply to the left preconditioner ``M`` due
          to the implicitly changed inner product.
        :param x0: (optional) the initial guess. Defaults to ``None`` which
          is treated as the zero initial guess.
        '''
        return pre.x0 + pre.Mr * self._correction(
            pre.Ml*(pre.b - pre.A*pre.x0))


class Arnoldifyer(object):
    def __init__(self, V, U, AU, H, B_, C, E, Pv_norm, U_v):
        r'''Obtain Arnoldi relations for approximate deflated Krylov subspaces.

        :param H: Hessenberg matrix :math:`\underline{H}_n`
          (``H.shape==(n+1,n)``) or :math:`H_n` (``H.shape==(n,n)``; invariant
          case) of already generated Krylov subspace.
        :param B_: :math:`\langle V_{n+1},AU\rangle` (``B_.shape==(n+1,d)``) or
          :math:`\langle V_n,AU\rangle` (``B_.shape==(n,d)``; invariant case).
        :param C: :math:`\langle U,AV_n\rangle` with ``C.shape==(d,n)``.
        :param E: :math:`\langle U,AU\rangle` with ``E.shape==(d,d)``.
        :param V: basis :math:`V_{n+1}` (``V.shape==(N,n+1)``) or
          :math:`V_n` (``V.shape==(N,n)``; invariant case).
        :param U: basis :math:`U` with ``U.shape==(N,d)``.
        '''
        # get dimensions
        n_, n = self.n_, self.n = H.shape
        d = self.d = U.shape[1]

        # store arguments
        self.V = V
        self.U = U
        self.AU = AU
        self.H = H
        self.B_ = B_
        self.C = C
        self.E = E
        self.Pv_norm = Pv_norm
        self.U_v = U_v

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
            Q1 = numpy.zeros((self.U.shape[0], 0))
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

        Pt = utils.Projection(self.L.dot(Wt), self.J.T.conj().dot(Wt)) \
            .operator_complement()
        if d > 0:
            qt = Pt*(numpy.r_[[[self.Pv_norm]],
                              numpy.zeros((self.n_-1, 1)),
                              numpy.linalg.solve(self.E, self.U_v)])
        else:
            qt = Pt*(numpy.r_[[[self.Pv_norm]],
                              numpy.zeros((self.n_-1, 1))])
        q = Wto.T.conj().dot(self.J.dot(qt))

        # TODO: q seems to suffer from round-off errors and thus the first
        #       vector in the computed basis may differ from the exact
        #       projected one more than on the level of unit round-off.

        # rotate closest vector in [V_n,U] to first column
        Q = utils.House(q)

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
            Y = numpy.bmat([[numpy.eye(n_), self.B_],
                            [numpy.zeros((d, n_)), self.E],
                            [numpy.zeros((self.R12.shape[0], n_)), self.R12]])
            YL_Q, _ = scipy.linalg.qr(Y.dot(self.L.dot(Wt)), mode='economic')

            # compute <W,X> where X is an orthonormal basis of AW
            WX = Wt.T.conj().dot(numpy.r_[YL_Q[:n, :],
                                          YL_Q[n_:n_+d, :]])
            PWAW_norm = 1./numpy.min(scipy.linalg.svdvals(WX))
        else:
            PWAW_norm = 1.

        if full:
            Vh = numpy.c_[self.V[:, :n], self.U].dot(Wto.dot(QT))
            F = - self.Z.dot(Rh.dot(Vh.T.conj()))
            F = F + F.T.conj()
            return Hh, Rh, vdiff_norm, PWAW_norm, Vh, F
        return Hh, Rh, vdiff_norm, PWAW_norm


def bound_pseudo(self, arnoldifyer, Wt,
                 g_norm=0.,
                 G_norm=0.,
                 GW_norm=0.,
                 WGW_norm=0.,
                 pseudo='nonnormal'):
    r'''Bound residual norms of next deflated system.

    :param arnoldifyer: an instance of
      :py:class:`~krypy.deflation.Arnoldifyer`.
    :param Wt: coefficients :math:`\tilde{W}\in\C^{n+d,k}` of the
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
    :param pseudo: One of

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
    Hh, Rh, vdiff_norm = arnoldifyer.get(Wt)

    # smallest singular value of E=W^*AW
    sigma_min = numpy.min(scipy.linalg.svdvals(arnoldifyer.E))

    if sigma_min <= WGW_norm:
        raise utils.AssumptionError(
            'sigma_min(W^*AW) > ||W^*GW|| not satisfied.')
    eta = GW_norm / (sigma_min - WGW_norm)
    #beta = 

    # spectrum of Hh
    evals, evecs = scipy.linalg.eig(Hh)
    # norm of Hh
    Hh_norm = numpy.linalg.norm(Hh, 2)

    # determine pseudo automatically
    def _auto():
        # is Hh Hermitian?
        if numpy.linalg.norm(Hh-Hh.T.conj(), 2) < 1e-14*Hh_norm:
            return 'hermitian'

        # is Hh normal?
        if numpy.linalg.cond(evecs, 2) < 1+1e-14:
            return 'normal'

        return 'nonnormal'

    if pseudo == 'auto':
        pseudo = _auto()


    pass
