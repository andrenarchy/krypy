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

def get_operators(A):
    return [ A,
                LinearOperator(A.shape, lambda x: numpy.dot(A,x), 
                    dtype=numpy.double), 
                csr_matrix(A)
                ]
def get_vecs(v):
    return [ v, numpy.reshape(v, (v.shape[0],)) ]

def test_linsys_spd():
    # build spd diag matrix
    a = numpy.array(range(1,11))
    a[-1] = 1.e2
    A, Ainv = numpy.diag(a), numpy.diag(1./a)

    # solution
    x = numpy.ones((10,1))

    # preconditioner
    m = numpy.array(range(1,11))
    m[-1] = 1.
    M, Minv = numpy.diag(m), numpy.diag(1./m)
    params_adds = [
            { 'M': [ None, Ainv ] + get_operators(Minv) },
            { 'Ml': [ None, Ainv ] + get_operators(Minv) },
            { 'Mr': [ None, Ainv ] + get_operators(Minv) }
            ]
    solvers = [ krypy.linsys.cg, krypy.linsys.minres, krypy.linsys.gmres ]
    for case in produce_cases(A, x, params_adds, solvers):
        yield case

def test_linsys_symm_indef():
    # build symm indef diag matrix
    a = numpy.array(range(1,11))
    a[-1] = -1e2
    A, Ainv = numpy.diag(a), numpy.diag(1./a)

    # solution
    x = numpy.ones((10,1))

    # preconditioner
    m = numpy.array(range(1,11))
    m[-1] = 1.
    M, Minv = numpy.diag(m), numpy.diag(1./m)
    params_adds = [
            { 'M': [ None ] + get_operators(Minv) },
            { 'Ml': [ None, Ainv ] + get_operators(Minv) },
            { 'Mr': [ None, Ainv ] + get_operators(Minv) }
            ]
    solvers = [ krypy.linsys.minres, krypy.linsys.gmres ]
    for case in produce_cases(A, x, params_adds, solvers):
        yield case

def test_linsys_nonsymm():
    # build nonsymm matrix
    a = numpy.array(range(1,11))
    a[-1] = -1e2
    A = numpy.diag(a)
    A[0,-1] = 1e2
    Ainv = numpy.linalg.inv(A)

    # solution
    x = numpy.ones((10,1))

    # preconditioner
    m = numpy.array(range(1,11))
    m[-1] = 1.
    M, Minv = numpy.diag(m), numpy.diag(1./m)
    params_adds = [
            { 'M': [ None ] + get_operators(Minv) },
            { 'Ml': [ None, Ainv ] + get_operators(Minv) },
            { 'Mr': [ None, Ainv ] + get_operators(Minv) }
            ]
    solvers = [ krypy.linsys.gmres ]
    for case in produce_cases(A, x, params_adds, solvers):
        yield case

def produce_cases(A, x, params_adds, solvers):
    b = numpy.dot(A, x)
    params_base = {
        'A': get_operators(A),
        'b': get_vecs(b),
        'x0': [ None, numpy.zeros(b.shape), x ] + get_vecs(numpy.ones(b.shape)),
        'tol': [ 1e-14, 1e-5, 1e-2 ],
        'maxiter': [ 15 ],
        'M': [ None ],
        'Ml': [ None ],
        'Mr': [ None ],
        'inner_product': [ krypy.utils.ip ],
        'exact_solution': [ None ] + get_vecs(x)
        }

    for params_add in params_adds:
        params = dict( list(params_base.items()) + list(params_add.items()))
        for solver, param in itertools.product(solvers, dictproduct(params)):
            yield run_case, solver, param

def run_case(solver, params):
    ret = solver(**params)

    # pick out the interesting data
    A = params['A']
    b = krypy.utils.shape_vec(params['b'])
    xk = krypy.utils.shape_vec(ret['xk'])

    # maxiter respected?
    assert( len(ret['relresvec'])-1 <= params['maxiter'] )

    # tolerance reached (if not near machine eps)?
    if params['tol']>1e-15:
        assert( ret['relresvec'][-1] <= params['tol'] )

    # final residual norm correct?
    # relresvec[-1] == ||M*Ml*(b-A*xk))||_{M^{-1}} / ||M*Ml*b||_{M^{-1}}
    # compute residual norm
    rk = b - krypy.utils.apply( A, xk)
    Mlrk = krypy.utils.apply( params['Ml'], rk )
    MMlrk = krypy.utils.apply( params['M'], Mlrk )
    norm_MMlrk = krypy.utils.norm( Mlrk, MMlrk, inner_product=params['inner_product'] )
    # compute rhs norm
    Mlb = krypy.utils.apply( params['Ml'], b )
    MMlb = krypy.utils.apply( params['M'], Mlb )
    norm_MMlb = krypy.utils.norm( Mlb, MMlb, inner_product=params['inner_product'] )
    # finally: the assertion
    assert( abs(ret['relresvec'][-1] - norm_MMlrk/norm_MMlb) <= 1e-15 )

    # final error norm correct?
    # (if exact_solution was provided)
    if params['exact_solution'] is not None:
        assert( abs(ret['errvec'][-1] - 
            krypy.utils.norm( krypy.utils.shape_vec(params['exact_solution'])
                - krypy.utils.shape_vec(ret['xk']))) <= 1e-15 )

    # if the preconditioner is the inverse, then check if convergence
    # occured after the first iteration
    if isinstance(A, numpy.ndarray) and \
            isinstance(params['M'], numpy.ndarray) and \
            numpy.linalg.norm( numpy.eye(*A.shape)- numpy.dot(A, params['M']) ) < 1e-15:
        assert( len(ret['relresvec'])<=2 )

    # 0 iterations if initial guess was good enough?
    if params['x0'] is not None:
        r0 = b - krypy.utils.apply( A, krypy.utils.shape_vec(params['x0']))
        Mlr0 = krypy.utils.apply( params['Ml'], r0 )
        MMlr0 = krypy.utils.apply( params['M'], Mlr0 )
        norm_MMlr0 = krypy.utils.norm( Mlr0, MMlr0, inner_product=params['inner_product'] )
        if norm_MMlr0/norm_MMlb < params['tol']:
            assert( len(ret['relresvec'])==1 )

    # has gmres found the solution after max N iterations?
    # (cg or minres may take longer because of roundoff errors)
    if solver==krypy.linsys.gmres:
        assert ( len(ret['relresvec'])-1 <= params['b'].shape[0] )

if __name__ == '__main__':
    import nose
    nose.main()
