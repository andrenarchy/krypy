import krypy
import numpy
import itertools
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix

def get_spd_matrix():
    a = numpy.array(range(1,11))
    a[-1] = 1e2
    return numpy.diag(a), numpy.diag(1/a)

def get_spd_precon():
    a = numpy.array(range(1,11))
    a[-1] = 1
    return numpy.diag(a), numpy.diag(1/a)

def test_linsys_cg():
    Aorig, Ainv = get_spd_matrix()
    x = numpy.ones((10,1))
    b = numpy.dot(Aorig,x)
    Morig, Minv = get_spd_precon()
    As = [Aorig, LinearOperator(Aorig.shape, lambda x: numpy.dot(Aorig,x), dtype=numpy.double), csr_matrix(Aorig)]
    x0s = [None, numpy.zeros((10,1)), numpy.ones((10,1)), x]
    tols = [1e-14, 1e-5, 1e-2]
    maxiters = [15]
    exact_solutions = [None, x]
    for A, x0, tol, maxiter, exact_solution in itertools.product(As, x0s, tols, maxiters, exact_solutions):
        yield linsys_cg, {'A':A, 'b':b, 'x0':x0, 'tol':tol, 'maxiter': maxiter, 'exact_solution':exact_solution}

def linsys_cg(params):
    ret = krypy.linsys.cg(**params)
    assert( len(ret['relresvec'])-1 <= params['maxiter'] )
    if params['tol']>1e-15:
        assert( ret['relresvec'][-1] <= params['tol'] )

if __name__ == '__main__':
    import nose
    nose.main()
