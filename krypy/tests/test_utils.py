import krypy
import numpy
import itertools

def test_house():
    factors = [0., 1., 1.j, 1.+1.j, 1e8, 1e-8]
    for (a, b) in itertools.product(factors, factors):
        x = numpy.ones((10,1), dtype=numpy.array([a]).dtype) * b
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

def assert_arnoldi(A, v, V, H, lanczos=False, inner_product=krypy.utils.ip_euclid, arnoldi_tol=1e-13, proj_tol=1e-13, ortho_tol=1e-14):
    N = v.shape[0]
    A = krypy.utils.get_linearoperator((N,N), A)

    # check shapes of V and H
    k = H.shape[1]
    assert( V.shape[1] == k+1 )
    assert( H.shape == (k+1,k) )

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
    AV = A * V[:,:-1]
    arnoldi_res = AV - numpy.dot(V, H)
    assert( krypy.utils.norm( arnoldi_res, inner_product=inner_product )<= arnoldi_tol )

    # check projection residual \| <V_k, A*V_k> - H_k \|
    proj_res = inner_product(V, AV) - H
    assert( numpy.linalg.norm( proj_res, 2) <= proj_tol )

    # check orthogonality by measuring \| I - <V,V> \|_2
    ortho_res = numpy.eye(V.shape[1]) - inner_product(V, V)
    assert( numpy.linalg.norm( ortho_res, 2) <= ortho_tol )

if __name__ == '__main__':
    import nose
    nose.main()
