import krypy
import numpy
import itertools
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix

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
    a = numpy.array(range(1,11))
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
    maxiters = [1, 5, 9]
    orthos = ['mgs', 'dmgs', 'house']
    B = numpy.diag(numpy.linspace(1,5,10))
    inner_products = [
            krypy.utils.ip_euclid,
            lambda x,y: numpy.dot(x.T.conj(), numpy.dot(B, y))
            ]

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
    assert( ortho_resn <= ortho_tol )

    # check projection residual \| <V_k, A*V_k> - H_k \|
    proj_res = inner_product(V, AV) - H
    proj_tol = proj_const*(ortho_resn * An + arnoldi_resn * krypy.utils.norm(V, inner_product=inner_product))
    assert( numpy.linalg.norm( proj_res, 2) <= proj_tol )


if __name__ == '__main__':
    import nose
    nose.main()
