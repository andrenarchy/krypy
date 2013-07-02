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

if __name__ == '__main__':
    import nose
    nose.main()
