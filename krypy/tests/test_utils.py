import krypy
import numpy
import itertools
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
import scipy.linalg

def get_matrix_spd():
    a = numpy.array(range(1,11))
    a[-1] = 1.e2
    return numpy.diag(a)

def get_matrix_hpd():
    a = numpy.array(range(1,11), dtype=numpy.complex)
    a[0] = 1.01e2
    a[-1] = 1.e2
    A = numpy.diag(a)
    A[-1,0] = 100.j
    A[0,-1] = -100.j
    return A

def get_matrix_symm_indef():
    a = numpy.array(range(1,11))
    a[-1] = -1.e2
    return numpy.diag(a)

def get_matrix_herm_indef():
    a = numpy.array(range(1,11), dtype=numpy.complex)
    a[-1] = -1.e2
    A = numpy.diag(a)
    A[-1,0] = 100.j
    A[0,-1] = -100.j
    return A

def get_matrix_nonsymm():
    a = numpy.array(range(1,11), dtype=numpy.float)
    a[-1] = -1e2
    A = numpy.diag(a)
    A[0,-1] = 1e2
    return A

def get_matrix_comp_nonsymm():
    a = numpy.array(range(1,11), dtype=numpy.complex)
    a[-1] = -1e2
    A = numpy.diag(a)
    A[0,-1] = 1.e2j
    return A

def get_inner_products():
    B = numpy.diag(numpy.linspace(1,5,10))
    return [
            krypy.utils.ip_euclid, 
            lambda x,y: numpy.dot(x.T.conj(), numpy.dot(B, y))
            ]

def get_operators(A):
    return [ A,
                LinearOperator(A.shape, lambda x: numpy.dot(A,x), 
                    dtype=A.dtype),
                csr_matrix(A)
                ]

def get_vecs(v):
    return [ v, numpy.reshape(v, (v.shape[0],)) ]

def test_house():
    factors = [0., 1., 1.j, 1.+1.j, 1e8, 1e-8]
    for (a, b, len) in itertools.product(factors, factors, [10,1]):
        x = numpy.ones((len,1), dtype=numpy.array([a]).dtype) * b
        x[0] = a
        yield run_house, x

def run_house(x):
    H = krypy.utils.House(x)
    y = H.apply(x)

    I = numpy.eye(len(x))
    # check that H.matrix() equals to H.apply(I)
    HI = H.apply(I)
    Hm = H.matrix()
    assert( numpy.linalg.norm( HI - Hm, 2) <= 1e-14 )

    # check that H.matrix() is Hermitian
    assert( numpy.linalg.norm( Hm - Hm.T.conj(), 2) <= 1e-14 )
    # check that H.matrix() is unitary/involutory
    assert( numpy.linalg.norm( I - numpy.dot(Hm.T.conj(),Hm),2) <= 1e-14 )
    # check that absolute value of y[0] equals norm(x)
    assert( numpy.abs( numpy.linalg.norm(x,2) - numpy.abs(y[0])) <= 1e-14*numpy.linalg.norm(x,2) )
    # check that abs(alpha)=1
    assert( numpy.abs( 1 - numpy.abs( H.alpha ) ) <= 1e-14 )
    # check that y[0] = alpha*norm(x)
    assert( numpy.abs( y[0] - H.alpha*H.xnorm ) <= 1e-14*numpy.linalg.norm(x,2) )
    if y.shape[0]>1:
        # check that all elements of r except the first are zero
        assert( numpy.linalg.norm(y[1:],2) <= 1e-14*numpy.linalg.norm(x,2) )

def test_givens():
    factors = [0., 1., 1.j, 1.+1.j, 1e8, 1e-8]
    for (a, b) in itertools.product(factors, factors):
        x = numpy.array([[a],[b]])
        yield run_givens, x

def run_givens(x):
    G = krypy.utils.Givens(x)
    y = G.apply(x)

    I = numpy.eye(2)
    # check that G.G is unitary
    assert( numpy.linalg.norm( I - numpy.dot(G.G.T.conj(), G.G), 2) <= 1e-14 )
    # check that absolute value of y[0] equals norm(x)
    assert( numpy.abs( numpy.linalg.norm(x,2) - numpy.abs(y[0])) <= 1e-14*numpy.linalg.norm(x,2) )
    # check that y[0] == 0
    assert( numpy.linalg.norm(y[1],2) <= 1e-14*numpy.linalg.norm(x,2) )

def test_projection():
    Xs = [ numpy.eye(10,1), numpy.eye(10,5), numpy.eye(10) ]
    B = numpy.diag(numpy.linspace(1,5,10))
    inner_products = get_inner_products()
    for (X, inner_product) in itertools.product(Xs, inner_products):
        Ys = [ None, X, X + numpy.ones((10,X.shape[1])) ]
        for Y in Ys:
            ipYXs = [None]
            ipYXs.append( inner_product(X if Y is None else Y, X) )
            for ipYX in ipYXs:
                ipYXinvs = ['explicit', 'ondemand' ]
                if ipYX is not None:
                    ipYXinvs.append( numpy.linalg.inv(ipYX) )
                for ipYXinv in ipYXinvs:
                    yield run_projection, X, Y, inner_product, ipYX, ipYXinv

def run_projection(X, Y, inner_product, ipYX, ipYXinv):
    P = krypy.utils.Projection(X, Y, inner_product, ipYX, ipYXinv)

    (N,k) = X.shape
    I = numpy.eye(N)
    z = numpy.ones((10,1))
    z/= numpy.linalg.norm(z,2)

    # check that it's a projection, i.e. P^2=P
    assert( numpy.linalg.norm( P.apply(I - P.apply(I) ), 2) <= 1e-14)
    # check that the range is X, i.e. the kernel of I-P is X
    assert( numpy.linalg.norm(X - P.apply(X),2) <= 1e-14)
    # check that the kernel is Y^perp, i.e. the range of I-P is orthogonal to Y
    assert( numpy.linalg.norm( inner_product( X if Y is None else Y, I - P.apply(I) ), 2) <= 5e-14)
    # check that the complementary projection is correct
    assert( numpy.linalg.norm( I-P.apply(I) -P.apply_complement(I), 2) <= 1e-14 )
    # check that the matrix representation is correct
    assert( numpy.linalg.norm( P.matrix() - P.apply(I), 2) <= 1e-14 )

def test_qr():
    Xs = [ numpy.eye(10,5), scipy.linalg.hilbert(10)[:,:5] ]
    B = numpy.diag(numpy.linspace(1,5,10))
    inner_products = get_inner_products()
    reorthos = [0, 1, 2]
    for X, inner_product, reortho in itertools.product(Xs, inner_products, reorthos):
        yield run_qr, X, inner_product, reortho

def run_qr(X, inner_product, reorthos):
    (N,k) = X.shape
    s = scipy.linalg.svd(X, compute_uv=False)
    Q, R = krypy.utils.qr(X, inner_product=inner_product, reorthos=reorthos)

    # check shapes
    assert( Q.shape==(N,k) )
    assert( R.shape==(k,k) )
    # check residual
    assert( numpy.linalg.norm( numpy.dot(Q,R) - X, 2) <= 1e-14*max(s) )
    # check orthogonality
    orthotol = 1e-8 if reorthos<1 else 1e-14
    assert( numpy.linalg.norm( inner_product(Q,Q) - numpy.eye(k), 2) <= orthotol )
    # check if R is upper triangular
    assert( numpy.linalg.norm( numpy.tril(R, -1) ) == 0 )

def test_angles():
    FGs = [
            numpy.eye(10,1),
            1j*numpy.eye(10,1),
            numpy.eye(10,4),
            numpy.eye(10)[:,-4:],
            numpy.dot(numpy.eye(10,4), numpy.diag([1,1e1,1e2,1e3])),
            numpy.eye(10,4)
            ]
    inner_products = get_inner_products()
    for F, G, inner_product, compute_vectors in itertools.product(FGs, FGs, inner_products, [False, True]):
        yield run_angles, F, G, inner_product, compute_vectors

def run_angles(F, G, inner_product, compute_vectors):
    if compute_vectors:
        theta, U, V = krypy.utils.angles(F, G, inner_product=inner_product, compute_vectors=compute_vectors)
    else:
        theta = krypy.utils.angles(F, G, inner_product=inner_product, compute_vectors=compute_vectors)

    # check shape of theta
    assert(theta.shape == (max(F.shape[1],G.shape[1]),))
    # check if theta is sorted
    assert( ( (theta - numpy.sort(theta))==0).all() )
    # check that 0 <= theta <= pi/2
    assert( (theta >= 0).all() )
    assert( (theta <= numpy.pi/2).all() )
    # check pi/2 angles if dimensions differ
    n = abs(F.shape[1] - G.shape[1])
    if n>0:
        assert( (numpy.abs( theta[-n:] - numpy.pi/2 ) == 0).all() ) # == 0 is safe here
    # check 0 angles if F==G
    if F is G:
        assert( numpy.linalg.norm(theta) <= 1e-15 )

    if compute_vectors:
        # check shapes of U and V
        assert( U.shape==F.shape )
        assert( V.shape==G.shape )
        # check that inner_product(U,V) = diag(cos(theta))
        m = min(F.shape[1], G.shape[1])
        UV = inner_product(U,V)
        assert( numpy.linalg.norm( UV - numpy.diag(numpy.cos(theta))[:F.shape[1],:G.shape[1]] ) <= 1e-14 )

def test_hegedus():
    matrices = [
            get_matrix_spd(),
            get_matrix_hpd(),
            get_matrix_symm_indef(),
            get_matrix_herm_indef(),
            get_matrix_nonsymm(),
            get_matrix_comp_nonsymm()
            ]
    xs = [
            numpy.ones((10,1)),
            (1.j+1)*numpy.ones((10,1))
            ]
    x0s = [
            numpy.zeros((10,1)),
            numpy.linspace(1,5,10).reshape((10,1)),
            ] + xs
    m = numpy.array(range(1,11))
    m[-1] = 1.
    Ms = [None, numpy.diag(m)]
    Mls = [None, numpy.diag(m)]
    inner_products = get_inner_products()

    for matrix, x, x0, M, Ml, inner_product in itertools.product(matrices,
            xs, x0s, Ms, Mls, inner_products):
        b = numpy.dot(matrix, x)
        for A in get_operators(matrix):
            yield run_hegedus, A, b, x0, M, Ml, inner_product

def run_hegedus(A, b, x0, M, Ml, inner_product):
    x0new = krypy.utils.hegedus(A, b, x0, M, Ml, inner_product)

    N = len(b)
    shape = (N,N)
    A = krypy.utils.get_linearoperator(shape, A)
    M = krypy.utils.get_linearoperator(shape, M)
    Ml = krypy.utils.get_linearoperator(shape, Ml)

    Mlr0 = Ml*(b - A*x0)
    MMlr0_norm = krypy.utils.norm( Mlr0, M*Mlr0, inner_product=inner_product)

    Mlr0new = Ml*(b - A*x0new)
    MMlr0new_norm = krypy.utils.norm( Mlr0new, M*Mlr0new, inner_product=inner_product)

    print(MMlr0new_norm)
    print(MMlr0_norm)
    assert(  MMlr0new_norm <= MMlr0_norm  + 1e-13 )

def test_arnoldi():
    matrices = [
            get_matrix_spd(),
            get_matrix_hpd(),
            get_matrix_symm_indef(),
            get_matrix_herm_indef(),
            get_matrix_nonsymm(),
            get_matrix_comp_nonsymm()
            ]
    vs = [numpy.ones((10,1)), numpy.eye(10,1)]
    maxiters = [1, 5, 9, 10]
    orthos = ['mgs', 'dmgs', 'house']
    B = numpy.diag(numpy.linspace(1,5,10))
    inner_products = get_inner_products()

    for (matrix, v, maxiter, ortho, inner_product) in itertools.product(matrices, vs, maxiters, orthos, inner_products):
        An = numpy.linalg.norm(matrix, 2)
        for A in get_operators(matrix):
            if ortho=='house' and inner_product!=krypy.utils.ip_euclid:
                continue
            yield run_arnoldi, A, v, maxiter, ortho, inner_product, An

def run_arnoldi(A, v, maxiter, ortho, inner_product, An):
    V, H = krypy.utils.arnoldi(A, v, maxiter=maxiter, ortho=ortho, inner_product=inner_product)
    assert_arnoldi(A, v, V, H, maxiter, ortho, inner_product, An=An)


def assert_arnoldi(A, v, V, H, maxiter, ortho, inner_product=krypy.utils.ip_euclid, lanczos=False, arnoldi_const=1, ortho_const=1, proj_const=2, An=None):
    # the checks in this function are based on the following literature:
    # [1] Drkosova, Greenbaum, Rozloznik and Strakos. Numerical Stability of GMRES. 1995. BIT.
    N = v.shape[0]
    if An is None:
        An = numpy.linalg.norm(A, 2)
    A = krypy.utils.get_linearoperator((N,N), A)
    eps = numpy.finfo(numpy.double).eps

    k = H.shape[1]

    #maxiter respected?
    assert( k <= maxiter )

    invariant = H.shape[0]==k
    # check shapes of V and H
    assert(V.shape[1]==H.shape[0])

    # check that the initial vector is correct
    v1 = v / krypy.utils.norm(v, inner_product=inner_product)
    assert( numpy.linalg.norm(V[:,[0]] - v1, 2) <= 1e-14 )

    # check if H is Hessenberg
    assert( numpy.linalg.norm( numpy.tril(H, -2) ) == 0 )
    if lanczos:
        # check if H is Hermitian
        assert( numpy.linalg.norm( H - H.T.conj() ) == 0 )
        # check if H is real
        assert( numpy.isreal(H).all() )

    # check if subdiagonal-elements are real and non-negative
    d = numpy.diag(H[1:,:])
    assert( numpy.isreal(d).all() )
    assert( (d>=0).all() )

    # check Arnoldi residual \| A*V_k - V_{k+1} H \|
    if invariant:
        AV = A * V
    else:
        AV = A * V[:,:-1]
    arnoldi_res = AV - numpy.dot(V, H)
    arnoldi_resn = krypy.utils.norm( arnoldi_res, inner_product=inner_product )
    arnoldi_tol = arnoldi_const * k * (N**1.5) * eps * An  # inequality (2.3) in [1]
    assert( arnoldi_resn <= arnoldi_tol )

    # check orthogonality by measuring \| I - <V,V> \|_2
    ortho_res = numpy.eye(V.shape[1]) - inner_product(V, V)
    ortho_resn = numpy.linalg.norm( ortho_res, 2)
    if ortho=='house':
        ortho_tol = ortho_const * (k**1.5) * N * eps  # inequality (2.4) in [1]
    else:
        vAV_singvals = numpy.linalg.svd( numpy.c_[ V[:,[0]], (AV[:,:-1] if invariant else AV) ], compute_uv=False )
        if vAV_singvals[-1]==0:
            ortho_tol = numpy.inf
        else:
            ortho_tol = ortho_const * (k**2) * N * eps * vAV_singvals[0]/vAV_singvals[-1]  # inequality (2.5) in [1]
    if ortho!='mgs' or N!=k: # mgs is not able to detect an invariant subspace reliably
        assert( ortho_resn <= ortho_tol )

    # check projection residual \| <V_k, A*V_k> - H_k \|
    proj_res = inner_product(V, AV) - H
    proj_tol = proj_const*(ortho_resn * An + arnoldi_resn * krypy.utils.norm(V, inner_product=inner_product))
    assert( numpy.linalg.norm( proj_res, 2) <= proj_tol )

def test_ritz():
    # Hermitian matrices
    matrices_herm = [
            get_matrix_spd(),
            #get_matrix_hpd(),
            #get_matrix_symm_indef(),
            #get_matrix_herm_indef(),
            ]
    matrices_nonherm = [
            get_matrix_nonsymm(),
            get_matrix_comp_nonsymm()
            ]
    vs = [numpy.ones((10,1)), numpy.eye(10,1)]
    maxiters = [1, 5, 9, 10]
    B = numpy.diag(numpy.linspace(1,5,10))
    inner_products = get_inner_products()
    types = ['ritz', 'harmonic', 'harmonic_improved']
    for (matrix, v, maxiter, inner_product, with_V, type) in itertools.product(matrices_herm+matrices_nonherm, vs, maxiters, inner_products, [True, False], types):
        hermitian = any(matrix is x for x in matrices_herm)
        eig = scipy.linalg.eigh if hermitian else scipy.linalg.eig
        Aevals, _ = eig(matrix)
        An = numpy.linalg.norm(matrix, 2)
        for A in get_operators(matrix):
            yield run_ritz, A, v, maxiter, inner_product, Aevals, An, with_V, hermitian, type

def run_ritz(A, v, maxiter, inner_product, Aevals, An, with_V, hermitian, type):
    ortho = 'house' if inner_product==krypy.utils.ip_euclid else 'dmgs'
    V, H = krypy.utils.arnoldi(A, v, maxiter=maxiter, ortho=ortho, inner_product=inner_product)
    N = v.shape[0]
    n = H.shape[1]
    A = krypy.utils.get_linearoperator((N,N), A)

    Z = None
    if with_V:
        theta, U, resnorm, Z = krypy.utils.ritz(H, V=V, hermitian=hermitian, type=type)
    else:
        theta, U, resnorm = krypy.utils.ritz(H, hermitian=hermitian, type=type)
    # check Z
    if Z is not None:
        assert( numpy.linalg.norm( numpy.dot(V[:,:n], U) - Z, 2) <= 1e-14 )
    else:
        Z = numpy.dot(V[:,:n], U)

    # check shapes
    assert( theta.shape==(n,) )
    assert( U.shape==(n,n) )
    assert( resnorm.shape==(n,) )
    assert( Z.shape==(N,n) )
    # check norm of Ritz vectors
    for i in range(n):
        assert( numpy.abs( numpy.linalg.norm(U[:,i],2) - 1) <= 1e-14 )
    # check residuals
    R = A*Z - numpy.dot(Z, numpy.diag(theta))
    for i in range(n):
        rnorm = krypy.utils.norm(R[:,[i]], inner_product=inner_product)
        assert( numpy.abs(rnorm - resnorm[i]) <= 1e-14*An )
    # check Ritz projection property
    if type=='ritz':
        assert( numpy.linalg.norm( inner_product(V[:,:n], R), 2) <= 1e-14*An )
    elif type=='harmonic':
        AVortho = scipy.linalg.orth( A*V[:,:n] )
        assert( numpy.linalg.norm( inner_product(AVortho, R), 2) <= 5e-14*An )
    else:
        pass

    # compare Ritz values with eigenvalues if n==N
    if n==N:
        Aevals_sort = numpy.argsort(numpy.abs(Aevals))
        theta_sort = numpy.argsort(numpy.abs(theta))
        assert( (numpy.abs( Aevals[Aevals_sort] - theta[theta_sort] ) <= 5e-14*An).all() )

if __name__ == '__main__':
    import nose
    nose.main()
