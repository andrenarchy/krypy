from .linsys import LinearSystem, Cg, Minres, Gmres
from .deflation import DeflatedCg, DeflatedMinres, DeflatedGmres
import numpy


# The simplest inner product, `numpy.dot`, should work as an input.
# krypy assumes that the inner product _always_ returns a 2D matrix which is why we need
# to wrap.
def wrap_inner_product(inner):
    def _wrap(a, b):
        if a.shape[1] == 0:
            return numpy.array([[]])
        return numpy.array([[inner(a[:, 0], b[:, 0])]])
    return _wrap


def cg(
    A,
    b,
    M=None,
    Minv=None,
    Ml=None,
    Mr=None,
    inner_product=None,
    exact_solution=None,
    x0=None,
    U=None,
    tol=1e-5,
    maxiter=None,
    use_explicit_residual=False,
    store_arnoldi=False,
):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    if inner_product:
        inner_product = wrap_inner_product(inner_product)

    # Make sure that the input vectors have two dimensions
    if U is not None:
        U = U.reshape(U.shape[0], -1)
    if x0 is not None:
        x0 = x0.reshape(x0.shape[0], -1)

    linear_system = LinearSystem(
        A=A,
        b=b,
        M=M,
        Minv=Minv,
        Ml=Ml,
        ip_B=inner_product,
        # Setting those to `True` simply avoids a warning.
        self_adjoint=True,
        positive_definite=True,
        exact_solution=exact_solution,
    )
    if U is None:
        out = Cg(
            linear_system,
            x0=x0,
            tol=tol,
            maxiter=maxiter,
            explicit_residual=use_explicit_residual,
            store_arnoldi=store_arnoldi,
        )
    else:
        out = DeflatedCg(
            linear_system,
            x0=x0,
            U=U,
            tol=tol,
            maxiter=maxiter,
            explicit_residual=use_explicit_residual,
            store_arnoldi=store_arnoldi,
        )
    return out.xk.reshape(b.shape) if out.resnorms[-1] < out.tol else None, out


def minres(
    A,
    b,
    M=None,
    Minv=None,
    Ml=None,
    Mr=None,
    inner_product=None,
    exact_solution=None,
    ortho="mgs",
    x0=None,
    U=None,
    tol=1e-5,
    maxiter=None,
    use_explicit_residual=False,
    store_arnoldi=False,
):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    if inner_product:
        inner_product = wrap_inner_product(inner_product)

    # Make sure that the input vectors have two dimensions
    if U is not None:
        U = U.reshape(U.shape[0], -1)
    if x0 is not None:
        x0 = x0.reshape(x0.shape[0], -1)

    linear_system = LinearSystem(
        A=A,
        b=b,
        M=M,
        Minv=Minv,
        Ml=Ml,
        ip_B=inner_product,
        # setting self_adjoin=True avoids a warning
        self_adjoint=True,
        exact_solution=exact_solution,
    )
    if U is None:
        out = Minres(
            linear_system,
            ortho=ortho,
            x0=x0,
            tol=tol,
            maxiter=maxiter,
            explicit_residual=use_explicit_residual,
            store_arnoldi=store_arnoldi,
        )
    else:
        out = DeflatedMinres(
            linear_system,
            ortho=ortho,
            x0=x0,
            U=U,
            tol=tol,
            maxiter=maxiter,
            explicit_residual=use_explicit_residual,
            store_arnoldi=store_arnoldi,
        )
    return out.xk.reshape(b.shape) if out.resnorms[-1] < out.tol else None, out


def gmres(
    A,
    b,
    M=None,
    Minv=None,
    Ml=None,
    Mr=None,
    inner_product=None,
    exact_solution=None,
    ortho="mgs",
    x0=None,
    U=None,
    tol=1e-5,
    maxiter=None,
    use_explicit_residual=False,
    store_arnoldi=False,
):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    if inner_product:
        inner_product = wrap_inner_product(inner_product)

    # Make sure that the input vectors have two dimensions
    if U is not None:
        U = U.reshape(U.shape[0], -1)
    if x0 is not None:
        x0 = x0.reshape(x0.shape[0], -1)

    linear_system = LinearSystem(
        A=A,
        b=b,
        M=M,
        Minv=Minv,
        Ml=Ml,
        ip_B=inner_product,
        exact_solution=exact_solution,
    )
    if U is None:
        out = Gmres(
            linear_system,
            ortho=ortho,
            x0=x0,
            tol=tol,
            maxiter=maxiter,
            explicit_residual=use_explicit_residual,
            store_arnoldi=store_arnoldi,
        )
    else:
        out = DeflatedGmres(
            linear_system,
            ortho=ortho,
            x0=x0,
            U=U,
            tol=tol,
            maxiter=maxiter,
            explicit_residual=use_explicit_residual,
            store_arnoldi=store_arnoldi,
        )
    return out.xk.reshape(b.shape) if out.resnorms[-1] < out.tol else None, out
