import krypy
import numpy
import itertools
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix


def dictproduct(d):
    '''enhance itertools product to process values of dicts

    example: 
        d = {'a':[1,2],'b':[3,4]}
        then list(dictproduct(d)) == 
        [{'a':1,'b':3}, {'a':1,'b':4}, {'a':2,'b':3}, {'a':2,'b':4}]
    '''
    for p in itertools.product(*d.values()):
        yield dict(zip(d.keys(), p))

def get_spd_matrix():
    a = numpy.array(range(1,11))
    a[-1] = 1e2
    return numpy.diag(a), numpy.diag(1/a)

def get_spd_precon():
    a = numpy.array(range(1,11))
    a[-1] = 1
    return numpy.diag(a), numpy.diag(1/a)

def test_linsys_spd():
    A, Ainv = get_spd_matrix()
    x = numpy.ones((10,1))
    b = numpy.dot(A,x)
    M, Minv = get_spd_precon()
    params = {
        'A': [ A, 
                LinearOperator(A.shape, lambda x: numpy.dot(A,x), 
                    dtype=numpy.double), 
                csr_matrix(A)
                ],
        'b': [ b, 
                numpy.reshape(b, (b.shape[0],))],
        'x0': [None, 
                numpy.zeros((10,1)), 
                numpy.ones((10,1)), 
                x],
        'tol': [1e-14, 1e-5, 1e-2],
        'maxiter': [15],
        'M': [ None,
                Minv,
                LinearOperator(Minv.shape, lambda x: numpy.dot(Minv,x), 
                    dtype=numpy.double),
                csr_matrix(Minv)
                ],
        'exact_solution': [None, x]
        }
    for param in dictproduct(params):
        yield linsys_cg, param

def linsys_cg(params):
    ret = krypy.linsys.cg(**params)
    assert( len(ret['relresvec'])-1 <= params['maxiter'] )
    if params['tol']>1e-15:
        assert( ret['relresvec'][-1] <= params['tol'] )

if __name__ == '__main__':
    import nose
    nose.main()
