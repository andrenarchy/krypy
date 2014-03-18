import krypy
import krypy.tests.test_utils as test_utils
import krypy.tests.test_linsys as test_linsys
import numpy
import itertools


def test_RitzFactorySimple():
    N = 100
    d = numpy.linspace(1, 2, N)
    d[:5] = [1e-8, 1e-4, 1e-2, 2e-2, 3e-2]
    ls = krypy.linsys.LinearSystem(numpy.diag(d), numpy.ones((N, 1)),
                                   normal=True, self_adjoint=True,
                                   positive_definite=True)

    Solvers = [krypy.recycling.RecyclingCg,
               krypy.recycling.RecyclingMinres,
               krypy.recycling.RecyclingGmres]
    whichs = ['lm', 'sm', 'lr', 'sr', 'li', 'si', 'smallest_res']
    for Solver, which in itertools.product(Solvers, whichs):
        yield run_RitzFactorySimple, Solver, ls, which


def run_RitzFactorySimple(Solver, ls, which):
    vector_factory = krypy.recycling.factories.RitzFactorySimple(
        n_vectors=3, which=which)
    recycling_solver = Solver()
    sols = []
    params = {'maxiter': 50, 'tol': 1e-5, 'x0': None}
    for i in range(3):
        sols.append(recycling_solver.solve(ls, vector_factory=vector_factory,
                                           **params))
        test_linsys.check_solver(sols[-1], Solver, ls, params)
        if i > 0:
            assert(len(sols[-1].resnorms) <= len(sols[0].resnorms))


#def test_SmallRitzGenerator():
#
#    matrices = [test_utils.get_matrix_spd(),
#                test_utils.get_matrix_hpd(),
#                test_utils.get_symm_indef(),
#                test_utils.get_herm_indef()]
#    max_vectorss = [numpy.Inf, 2]
#    for matrix, max_vectors in itertools.product(matrices, max_vectorss):
#        yield run_SmallRitzGenerator, matrix, True, max_vectors
#
#
#def _get_ritz(A, self_adjoint):
#    linear_system = krypy.linsys.LinearSystem(A, numpy.ones((10, 1)),
#                                              self_adjoint=True)
#    deflated_solver = krypy.deflation.DeflatedGmres(linear_system,
#                                                    store_arnoldi=True)
#    return krypy.deflation.Ritz(deflated_solver)
#
#def run_SmallRitzGenerator(A, self_adjoint, max_vectors):
#    ritz = _get_ritz(A, self_adjoint)
#    small = krypy.recycling.generators.SmallRitz(max_vectors=max_vectors)
#    pass



if __name__ == '__main__':
    import nose
    nose.main()
