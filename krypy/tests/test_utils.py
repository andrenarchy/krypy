import krypy
import numpy
import itertools
from scipy.sparse import csr_matrix
import scipy.linalg
from numpy.testing import assert_almost_equal, assert_array_almost_equal, \
    assert_array_equal, assert_equal


def get_matrix_spd():
    a = numpy.linspace(1, 2, 10)
    a[-1] = 1e-2
    return numpy.diag(a)


def get_matrix_hpd():
    a = numpy.array(numpy.linspace(1, 2, 10), dtype=numpy.complex)
    a[0] = 5
    a[-1] = 1e-1
    A = numpy.diag(a)
    A[-1, 0] = 1e-1j
    A[0, -1] = -1e-1j
    return A


def get_matrix_symm_indef():
    a = numpy.linspace(1, 2, 10)
    a[-1] = -1
    return numpy.diag(a)


def get_matrix_herm_indef():
    a = numpy.array(numpy.linspace(1, 2, 10), dtype=numpy.complex)
    a[-1] = 1e-3
    A = numpy.diag(a)
    A[-1, 0] = 10j
    A[0, -1] = -10j
    return A


def get_matrix_nonsymm():
    a = numpy.array(range(1, 11), dtype=numpy.float)
    a[-1] = -1e1
    A = numpy.diag(a)
    A[0, -1] = 1e1
    return A


def get_matrix_comp_nonsymm():
    a = numpy.array(range(1, 11), dtype=numpy.complex)
    a[-1] = -1e1
    A = numpy.diag(a)
    A[0, -1] = 1.e1j
    return A


def get_matrices(spd=True,
                 hpd=True,
                 symm_indef=True,
                 herm_indef=True,
                 nonsymm=True,
                 comp_nonsymm=True):
    matrices = []
    if spd:
        matrices.append(get_matrix_spd())
    if hpd:
        matrices.append(get_matrix_hpd())
    if symm_indef:
        matrices.append(get_matrix_symm_indef())
    if herm_indef:
        matrices.append(get_matrix_herm_indef())
    if nonsymm:
        matrices.append(get_matrix_nonsymm())
    if comp_nonsymm:
        matrices.append(get_matrix_comp_nonsymm())
    return matrices


def get_ip_Bs():
    B = numpy.diag(numpy.linspace(1, 5, 10))
    return [None,
            krypy.utils.MatrixLinearOperator(B),
            lambda x, y: numpy.dot(x.T.conj(), numpy.dot(B, y))
            ]


def get_operators(A):
    return [A, krypy.utils.MatrixLinearOperator(A)]


def get_vecs(v):
    return [v, numpy.reshape(v, (v.shape[0],))]


def test_house():
    factors = [0., 1., 1.j, 1.+1.j, 1e8, 1e-8]
    for (a, b, len) in itertools.product(factors, factors, [10, 1]):
        x = numpy.ones((len, 1), dtype=numpy.array([a]).dtype) * b
        x[0] = a
        yield run_house, x


def run_house(x):
    H = krypy.utils.House(x)
    y = H.apply(x)

    I = numpy.eye(len(x))
    # check that H.matrix() equals to H.apply(I)
    HI = H.apply(I)
    Hm = H.matrix()
    assert(numpy.linalg.norm(HI - Hm, 2) <= 1e-14)

    # check that H.matrix() is Hermitian
    assert(numpy.linalg.norm(Hm - Hm.T.conj(), 2) <= 1e-14)
    # check that H.matrix() is unitary/involutory
    assert(numpy.linalg.norm(I - numpy.dot(Hm.T.conj(), Hm), 2) <= 1e-14)
    # check that absolute value of y[0] equals norm(x)
    assert(numpy.abs(numpy.linalg.norm(x, 2) - numpy.abs(y[0]))
           <= 1e-14*numpy.linalg.norm(x, 2))
    # check that abs(alpha)=1
    assert(numpy.abs(1 - numpy.abs(H.alpha)) <= 1e-14)
    # check that y[0] = alpha*norm(x)
    assert(numpy.abs(y[0] - H.alpha*H.xnorm) <= 1e-14*numpy.linalg.norm(x, 2))
    if y.shape[0] > 1:
        # check that all elements of r except the first are zero
        assert(numpy.linalg.norm(y[1:], 2) <= 1e-14*numpy.linalg.norm(x, 2))


def test_givens():
    factors = [0., 1., 1.j, 1.+1.j, 1e8, 1e-8]
    for (a, b) in itertools.product(factors, factors):
        x = numpy.array([[a], [b]])
        yield run_givens, x


def run_givens(x):
    G = krypy.utils.Givens(x)
    y = G.apply(x)

    I = numpy.eye(2)
    # check that G.G is unitary
    assert(numpy.linalg.norm(I - numpy.dot(G.G.T.conj(), G.G), 2) <= 1e-14)
    # check that absolute value of y[0] equals norm(x)
    assert(numpy.abs(numpy.linalg.norm(x, 2) - numpy.abs(y[0]))
           <= 1e-14*numpy.linalg.norm(x, 2))
    # check that y[0] == 0
    assert(numpy.linalg.norm(y[1], 2) <= 1e-14*numpy.linalg.norm(x, 2))


def test_projection():
    Xs = [numpy.eye(10, 1),
          numpy.eye(10, 5),
          numpy.eye(10, 5) + 1e-1*numpy.ones((10, 5)),
          numpy.eye(10),
          numpy.zeros((10, 0))
          ]
    ip_Bs = get_ip_Bs()
    its = [1, 2, 3]
    for (X, ip_B, iterations) in itertools.product(Xs, ip_Bs, its):
        Ys = [None, X, X + numpy.ones((10, X.shape[1]))]
        for Y in Ys:
            yield run_projection, X, Y, ip_B, iterations


def run_projection(X, Y, ip_B, iterations):
    P = krypy.utils.Projection(X, Y, ip_B=ip_B, iterations=iterations)

    (N, k) = X.shape
    I = numpy.eye(N)
    z = numpy.ones((10, 1))
    z /= numpy.linalg.norm(z, 2)

    # check that it's a projection, i.e. P^2=P
    assert_almost_equal(
        numpy.linalg.norm(P.apply(I - P.apply(I)), 2), 0, 14)
    if k > 0:
        # check that the range is X, i.e. the kernel of I-P is X
        assert_almost_equal(numpy.linalg.norm(X - P.apply(X), 2), 0, 14)

        # check that the kernel is Y^perp, i.e. the range of I-P is orthogonal
        # to Y
        assert_almost_equal(
            numpy.linalg.norm(krypy.utils.inner(X if Y is None else Y,
                                                I - P.apply(I), ip_B=ip_B), 2),
            0, 13)
    else:
        assert_equal(numpy.linalg.norm(P.apply(I)), 0)

    # check that the complementary projection is correct
    assert_almost_equal(
        numpy.linalg.norm(I-P.apply(I) - P.apply_complement(I), 2),
        0, 14)

    # check that operator()*z executes apply(z)
    assert(numpy.linalg.norm(P.operator()*z - P.apply(z)) == 0)

    # check that operator_complement()*z executes apply_complement(z)
    assert(numpy.linalg.norm(P.operator_complement()*z - P.apply_complement(z))
           == 0)

    # check that the matrix representation is correct
    assert_almost_equal(numpy.linalg.norm(P.matrix() - P.apply(I), 2),
                        0, 14)

    # check that <Y,a> is returned correctly with return_Ya=True
    a = numpy.ones((N, 1))
    _, Ya = P.apply(a, return_Ya=True)
    assert_array_almost_equal(Ya, krypy.utils.inner(X if Y is None else Y, a,
                                                    ip_B=ip_B))

    # same check for apply_complement
    _, Ya = P.apply_complement(a, return_Ya=True)
    assert_array_almost_equal(Ya, krypy.utils.inner(X if Y is None else Y, a,
                                                    ip_B=ip_B))


def test_qr():
    Xs = [numpy.eye(10, 5), scipy.linalg.hilbert(10)[:, :5]]
    ip_Bs = get_ip_Bs()
    reorthos = [0, 1, 2]
    for X, ip_B, reortho in itertools.product(Xs, ip_Bs, reorthos):
        yield run_qr, X, ip_B, reortho


def run_qr(X, ip_B, reorthos):
    (N, k) = X.shape
    s = scipy.linalg.svd(X, compute_uv=False)
    Q, R = krypy.utils.qr(X, ip_B=ip_B, reorthos=reorthos)

    # check shapes
    assert(Q.shape == (N, k))
    assert(R.shape == (k, k))
    # check residual
    assert(numpy.linalg.norm(numpy.dot(Q, R) - X, 2) <= 1e-14*max(s))
    # check orthogonality
    orthotol = 1e-8 if reorthos < 1 else 1e-14
    assert(numpy.linalg.norm(krypy.utils.inner(Q, Q, ip_B=ip_B)
                             - numpy.eye(k), 2)
           <= orthotol)
    # check if R is upper triangular
    assert(numpy.linalg.norm(numpy.tril(R, -1)) == 0)


def test_angles():
    FGs = [numpy.eye(10, 1),
           1j*numpy.eye(10, 1),
           numpy.eye(10, 4),
           numpy.eye(10)[:, -4:],
           numpy.dot(numpy.eye(10, 4), numpy.diag([1, 1e1, 1e2, 1e3])),
           numpy.eye(10, 4)
           ]
    ip_Bs = get_ip_Bs()
    for F, G, ip_B, compute_vectors in itertools.product(FGs, FGs, ip_Bs,
                                                         [False, True]):
        yield run_angles, F, G, ip_B, compute_vectors


def run_angles(F, G, ip_B, compute_vectors):
    if compute_vectors:
        theta, U, V = krypy.utils.angles(F, G, ip_B=ip_B,
                                         compute_vectors=compute_vectors)
    else:
        theta = krypy.utils.angles(F, G, ip_B=ip_B,
                                   compute_vectors=compute_vectors)

    # check shape of theta
    assert(theta.shape == (max(F.shape[1], G.shape[1]),))
    # check if theta is sorted
    assert(((theta - numpy.sort(theta)) == 0).all())
    # check that 0 <= theta <= pi/2
    assert((theta >= 0).all())
    assert((theta <= numpy.pi/2).all())
    # check pi/2 angles if dimensions differ
    n = abs(F.shape[1] - G.shape[1])
    if n > 0:
        # == 0 is safe here
        assert((numpy.abs(theta[-n:] - numpy.pi/2) == 0).all())
    # check 0 angles if F==G
    if F is G:
        assert(numpy.linalg.norm(theta) <= 1e-15)

    if compute_vectors:
        # check shapes of U and V
        assert(U.shape == F.shape)
        assert(V.shape == G.shape)
        # check that inner_product(U,V) = diag(cos(theta))
        UV = krypy.utils.inner(U, V, ip_B=ip_B)
        assert(numpy.linalg.norm(UV
               - numpy.diag(numpy.cos(theta))[:F.shape[1], :G.shape[1]])
               <= 1e-14)


def test_hegedus():
    matrices = [get_matrix_spd(),
                get_matrix_hpd(),
                get_matrix_symm_indef(),
                get_matrix_herm_indef(),
                get_matrix_nonsymm(),
                get_matrix_comp_nonsymm()
                ]
    xs = [numpy.ones((10, 1)),
          (1.j+1)*numpy.ones((10, 1))
          ]
    x0s = [numpy.zeros((10, 1)),
           numpy.linspace(1, 5, 10).reshape((10, 1)),
           ] + xs
    m = numpy.array(range(1, 11))
    m[-1] = 1.
    Ms = [None, numpy.diag(m)]
    Mls = [None, numpy.diag(m)]
    ip_Bs = get_ip_Bs()

    for matrix, x, x0, M, Ml, ip_B in itertools.product(matrices, xs, x0s, Ms,
                                                        Mls, ip_Bs):
        b = numpy.dot(matrix, x)
        for A in get_operators(matrix):
            yield run_hegedus, A, b, x0, M, Ml, ip_B


def run_hegedus(A, b, x0, M, Ml, ip_B):
    x0new = krypy.utils.hegedus(A, b, x0, M, Ml, ip_B)

    N = len(b)
    shape = (N, N)
    A = krypy.utils.get_linearoperator(shape, A)
    M = krypy.utils.get_linearoperator(shape, M)
    Ml = krypy.utils.get_linearoperator(shape, Ml)

    Mlr0 = Ml*(b - A*x0)
    MMlr0_norm = krypy.utils.norm(Mlr0, M*Mlr0, ip_B=ip_B)

    Mlr0new = Ml*(b - A*x0new)
    MMlr0new_norm = krypy.utils.norm(Mlr0new, M*Mlr0new,
                                     ip_B=ip_B)

    assert(MMlr0new_norm <= MMlr0_norm + 1e-13)


def test_arnoldi():
    #TODO: reactivate the complex tests once travis-ci uses newer
    #      numpy/scipy versions.
    matrices = [get_matrix_spd(),
                #get_matrix_hpd(),
                get_matrix_symm_indef(),
                #get_matrix_herm_indef(),
                get_matrix_nonsymm(),
                #get_matrix_comp_nonsymm()
                ]
    vs = [numpy.ones((10, 1)), numpy.eye(10, 1)]
    maxiters = [1, 5, 9, 10]
    orthos = ['mgs', 'dmgs', 'house']
    B = numpy.diag(numpy.linspace(1, 5, 10))
    Ms = [None, B]
    ip_Bs = [None,
             B,
             krypy.utils.MatrixLinearOperator(B),
             lambda x, y: x.T.conj().dot(B.dot(y))
             ]

    for (matrix, v, maxiter, ortho, M, ip_B) in \
            itertools.product(matrices, vs, maxiters, orthos, Ms, ip_Bs):
        An = numpy.linalg.norm(matrix, 2)
        for A in get_operators(matrix):
            if ortho == 'house' and (ip_B is not None or M is not None):
                continue
            yield run_arnoldi, A, v, maxiter, ortho, M, ip_B, An

    #TODO: reactivate the complex tests once travis-ci uses newer
    #      numpy/scipy versions.
    matrices = [get_matrix_spd(),
                #get_matrix_hpd(),
                get_matrix_symm_indef(),
                #get_matrix_herm_indef()
                ]
    for (matrix, v, maxiter, M, ip_B) in \
            itertools.product(matrices, vs, maxiters, Ms, ip_Bs):
        An = numpy.linalg.norm(matrix, 2)
        for A in get_operators(matrix):
            yield run_arnoldi, A, v, maxiter, 'lanczos', M, ip_B, An


def run_arnoldi(A, v, maxiter, ortho, M, ip_B, An):
    res = krypy.utils.arnoldi(A, v, maxiter=maxiter, ortho=ortho, M=M,
                              ip_B=ip_B)
    if M is not None:
        V, H, P = res
    else:
        V, H = res
        P = None
    assert_arnoldi(A, v, V, H, P, maxiter, ortho, M, ip_B, An=An)


def assert_arnoldi(A, v, V, H, P, maxiter, ortho, M, ip_B,
                   lanczos=False,
                   arnoldi_const=1,
                   ortho_const=1,
                   proj_const=10,
                   An=None
                   ):
    # the checks in this function are based on the following literature:
    # [1] Drkosova, Greenbaum, Rozloznik and Strakos. Numerical Stability of
    #     GMRES. 1995. BIT.
    N = v.shape[0]
    if An is None:
        An = numpy.linalg.norm(A, 2)
    A = krypy.utils.get_linearoperator((N, N), A)
    eps = numpy.finfo(numpy.double).eps

    k = H.shape[1]

    #maxiter respected?
    assert(k <= maxiter)

    invariant = H.shape[0] == k
    # check shapes of V and H
    assert(V.shape[1] == H.shape[0])

    # check that the initial vector is correct
    M = krypy.utils.get_linearoperator((N, N), M)
    v1n = numpy.sqrt(krypy.utils.inner(v, M*v, ip_B=ip_B))
    if P is not None:
        assert(numpy.linalg.norm(P[:, [0]] - v/v1n) <= 1e-14)
    else:
        assert(numpy.linalg.norm(V[:, [0]] - v/v1n) <= 1e-14)

    # check if H is Hessenberg
    assert(numpy.linalg.norm(numpy.tril(H, -2)) == 0)
    if lanczos:
        # check if H is Hermitian
        assert(numpy.linalg.norm(H - H.T.conj()) == 0)
        # check if H is real
        assert(numpy.isreal(H).all())

    # check if subdiagonal-elements are real and non-negative
    d = numpy.diag(H[1:, :])
    assert(numpy.isreal(d).all())
    assert((d >= 0).all())

    # check Arnoldi residual \| A*V_k - V_{k+1} H \|
    if invariant:
        AV = A * V
    else:
        AV = A * V[:, :-1]
    if M is not None:
        MAV = M*AV
    else:
        MAV = AV
    arnoldi_res = MAV - numpy.dot(V, H)
    arnoldi_resn = krypy.utils.norm(arnoldi_res, ip_B=ip_B)
    # inequality (2.3) in [1]
    arnoldi_tol = arnoldi_const * k * (N**1.5) * eps * An
    assert(arnoldi_resn <= arnoldi_tol)

    # check orthogonality by measuring \| I - <V,V> \|_2
    if P is not None:
        ortho_res = numpy.eye(V.shape[1]) - krypy.utils.inner(V, P, ip_B=ip_B)
    else:
        ortho_res = numpy.eye(V.shape[1]) - krypy.utils.inner(V, V, ip_B=ip_B)
    ortho_resn = numpy.linalg.norm(ortho_res, 2)
    if ortho == 'house':
        ortho_tol = ortho_const * (k**1.5) * N * eps  # inequality (2.4) in [1]
    else:
        vAV_singvals = scipy.linalg.svd(numpy.c_[V[:, [0]], (MAV[:, :-1]
                                        if invariant else MAV)],
                                        compute_uv=False)
        if vAV_singvals[-1] == 0:
            ortho_tol = numpy.inf
        else:
            # inequality (2.5) in [1]
            ortho_tol = ortho_const * (k**2) * N * eps \
                * vAV_singvals[0]/vAV_singvals[-1]
    # mgs or lanczos is not able to detect an invariant subspace reliably
    if (ortho != 'mgs' or N != k) and ortho != 'lanczos':
        assert(ortho_resn <= ortho_tol)

    # check projection residual \| <V_k, A*V_k> - H_k \|
    if P is not None:
        proj_res = krypy.utils.inner(P, MAV, ip_B=ip_B) - H
    else:
        proj_res = krypy.utils.inner(V, MAV, ip_B=ip_B) - H
    proj_tol = proj_const*(ortho_resn * An + arnoldi_resn
                           * krypy.utils.norm(V, ip_B=ip_B))
    assert(numpy.linalg.norm(proj_res, 2) <= proj_tol)


def test_ritz():
    # Hermitian matrices
    matrices_herm = [get_matrix_spd(),
                     #get_matrix_hpd(),
                     #get_matrix_symm_indef(),
                     #get_matrix_herm_indef(),
                     ]
    matrices_nonherm = [get_matrix_nonsymm(),
                        get_matrix_comp_nonsymm()
                        ]
    vs = [numpy.ones((10, 1)), numpy.eye(10, 1)]
    maxiters = [1, 5, 9, 10]
    ip_Bs = get_ip_Bs()
    types = ['ritz', 'harmonic', 'harmonic_improved']
    for (matrix, v, maxiter, ip_B, with_V, type) in \
            itertools.product(matrices_herm+matrices_nonherm, vs, maxiters,
                              ip_Bs, [True, False], types):
        hermitian = any(matrix is x for x in matrices_herm)
        eig = scipy.linalg.eigh if hermitian else scipy.linalg.eig
        Aevals, _ = eig(matrix)
        An = numpy.linalg.norm(matrix, 2)
        for A in get_operators(matrix):
            yield run_ritz, A, v, maxiter, ip_B, Aevals, An, with_V, \
                hermitian, type


def run_ritz(A, v, maxiter, ip_B, Aevals, An, with_V, hermitian, type):
    ortho = 'house' if ip_B is None else 'dmgs'
    V, H = krypy.utils.arnoldi(A, v, maxiter=maxiter, ortho=ortho, ip_B=ip_B)
    N = v.shape[0]
    n = H.shape[1]
    A = krypy.utils.get_linearoperator((N, N), A)

    Z = None
    if with_V:
        theta, U, resnorm, Z = krypy.utils.ritz(H, V=V, hermitian=hermitian,
                                                type=type)
    else:
        theta, U, resnorm = krypy.utils.ritz(H, hermitian=hermitian, type=type)
    # check Z
    if Z is not None:
        assert(numpy.linalg.norm(numpy.dot(V[:, :n], U) - Z, 2) <= 1e-14)
    else:
        Z = numpy.dot(V[:, :n], U)

    # check shapes
    assert(theta.shape == (n,))
    assert(U.shape == (n, n))
    assert(resnorm.shape == (n,))
    assert(Z.shape == (N, n))
    # check norm of Ritz vectors
    for i in range(n):
        assert(numpy.abs(numpy.linalg.norm(U[:, i], 2) - 1) <= 1e-14)
    # check residuals
    R = A*Z - numpy.dot(Z, numpy.diag(theta))
    for i in range(n):
        rnorm = krypy.utils.norm(R[:, [i]], ip_B=ip_B)
        assert(numpy.abs(rnorm - resnorm[i]) <= 1e-14*An)
    # check Ritz projection property
    if type == 'ritz':
        assert(numpy.linalg.norm(krypy.utils.inner(V[:, :n], R, ip_B=ip_B), 2)
               <= 1e-14*An)
    elif type == 'harmonic':
        AVortho = scipy.linalg.orth(A*V[:, :n])
        assert(numpy.linalg.norm(krypy.utils.inner(AVortho, R, ip_B=ip_B), 2)
               <= 1e-12*An)
    else:
        pass

    # compare Ritz values with eigenvalues if n==N
    if n == N:
        Aevals_sort = numpy.argsort(numpy.abs(Aevals))
        theta_sort = numpy.argsort(numpy.abs(theta))
        assert((numpy.abs(Aevals[Aevals_sort] - theta[theta_sort])
                <= 5e-14*An).all())


def test_gap():
    assert_almost_equal(krypy.utils.gap([1, 2], [-4, 3]), 1)
    assert_almost_equal(krypy.utils.gap(5, -5), 10)
    assert_almost_equal(krypy.utils.gap([-5, 5], -5), 0)
    assert_almost_equal(krypy.utils.gap(5, -5, mode='interval'), 10)
    assert_almost_equal(krypy.utils.gap(5, [-5, 6], mode='interval'), 1)
    assert_almost_equal(krypy.utils.gap(-5, [-5, 6], mode='interval'), 0)
    assert(krypy.utils.gap([-5, 5], [0], mode='interval') is None)


def test_Interval():
    Interval = krypy.utils.Interval
    Intervals = krypy.utils.Intervals

    I = Interval(-2, -1)
    J = Interval(1, 2)
    K = Interval(-10, 1.5)
    L = Interval(5)
    M = Interval(-100, -50)
    N = Interval(50, 100)

    assert((I & J) is None)
    assert((I | J) is None)
    assert((J & K).left == 1)
    assert((J & K).right == 1.5)
    assert((J | K).left == -10)
    assert((J | K).right == 2)

    ints = Intervals([I, M])
    assert(ints.max() == -1)
    assert(ints.min() == -100)
    assert(ints.max_neg() == -1)
    assert(ints.min_pos() is None)

    ints = Intervals([I, J, M, N])
    assert(ints.max() == 100)
    assert(ints.min_pos() == 1)
    assert(ints.max_neg() == -1)
    assert(ints.min() == -100)
    assert(ints.contains(0) is False)

    ints = Intervals([I, J, K, L])
    assert(ints.max() == 5)
    assert(ints.min_pos() is None)
    assert(ints.max_neg() is None)
    assert(ints.min() == -10)
    assert(ints.max_abs() == 10)
    assert(ints.contains(0) is True)
    assert_array_equal(ints.get_endpoints(), [-10, 2, 5])


def test_BoundCG():
    from math import ceil
    b = krypy.utils.BoundCG([1, 2])
    assert_almost_equal(b.eval_step(8), 1.5018239652065932e-06)
    assert(ceil(b.get_step(1e-6)) == 9)

    b = krypy.utils.BoundCG(krypy.utils.Intervals(
        [krypy.utils.Interval(1, 1.2), krypy.utils.Interval(2)]))
    assert_almost_equal(b.eval_step(8), 1.5018239652065932e-06)
    assert(ceil(b.get_step(1e-6)) == 9)


def test_BoundMinres():
    from math import ceil
    b = krypy.utils.BoundMinres([-1, 1, 2])
    assert_almost_equal(b.eval_step(8), 0.0017331035544401801)
    assert(ceil(b.get_step(2e-3)) == 8)

    b = krypy.utils.BoundMinres(krypy.utils.Intervals(
        [krypy.utils.Interval(-2, -1), krypy.utils.Interval(2)]))
    assert_almost_equal(b.eval_step(8), 0.0017331035544401801)
    assert(ceil(b.get_step(2e-3)) == 8)


def test_NormalizedRootsPolynomial():
    rs = [[1, 2],
          [1, 1j],
          [1, 2, 1e8],
          [1, 2, 1e8, 1e8+1e-3]
          ]
    for roots in rs:
        yield run_NormalizedRootsPolynomial, roots


def run_NormalizedRootsPolynomial(roots):
    p = krypy.utils.NormalizedRootsPolynomial(roots)

    # check if roots are exactly (!) zero
    assert_array_equal(p(roots), numpy.zeros((len(roots),)))

    # check if polynomial is normalized at origin
    assert_equal(p(0), 1)

    if numpy.isrealobj(roots):
        interval = numpy.linspace(roots[0], roots[1], 100)
        candidates = p.minmax_candidates()
        c = [roots[0], roots[1]]
        for candidate in candidates:
            if roots[0] <= candidate <= roots[1]:
                c.append(candidate)
        assert_almost_equal(numpy.max(numpy.abs(p(interval))),
                            numpy.max(numpy.abs(p(c))), decimal=4)


if __name__ == '__main__':
    import nose
    nose.main()
