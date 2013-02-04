import krypy
import numpy
import itertools

def get_spd_matrix():
    a = numpy.array(range(1,11))
    a[-1] = 1e2
    return numpy.diag(a)

def test_linsys_cg():
    A = get_spd_matrix()
    x = numpy.ones((10,1))
    b = numpy.dot(A,x)
    x0s = [None, numpy.zeros((10,1)), numpy.ones((10,1)), x]
    tols = [1e-14, 1e-5, 1e-2]
    maxiters = [15]
    exact_solutions = [None, x]
    for x0, tol, maxiter, exact_solution in itertools.product(x0s, tols, maxiters, exact_solutions):
        yield linsys_cg, {'A':A, 'b':b, 'x0':x0, 'tol':tol, 'maxiter': maxiter, 'exact_solution':exact_solution}

def linsys_cg(params):
    ret = krypy.linsys.cg(**params)
    assert( ret['relresvec'][-1] <= params['tol'] )
    assert( len(ret['relresvec'])-1 <= params['maxiter'] )

if __name__ == '__main__':
    import nose
    nose.main()
