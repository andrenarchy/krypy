# -*- coding: utf8 -*-
import numpy
from . import utils

# ===================================================================
def get_deflation_correction(A, b, Ml, Mr, U, V,
        inner_product=utils.ip_euclid, VtU=None):
    """Build correction function for solutions in deflated methods.
    """
    pass

# ===================================================================
def get_deflation_data(W, AW, b, x0=None, type='left_ortho', 
        inner_product=utils.ip_euclid):
    """Get projection and appropriate initial guess for use in deflated methods.

    Generates a projection and an initial guess for use in
    :py:meth:`~krypy.linsys.gmres`,
    :py:meth:`~krypy.linsys.minres`, or
    :py:meth:`~krypy.linsys.cg`.

    :param W:  the basis vectors used for deflation with ``W.shape==(N,k)``.
    :param AW: A*W with ``AW.shape==(N,k)``, where A is the operator of the
      linear algebraic system to be deflated. A has to be self-adjoint w.r.t.
      inner_product. Do not include the positive-definite preconditioner
      (argument M in MINRES) here. Let N be the dimension of the vector space
      the operator is defined on.
    :param b:  the right hand side of the linear system with ``b.shape==(N,1)``.
    :param x0: (optional) the initial guess with ``x0.shape==(N,1)``. Default is
      zero initial guess.
    :param type: (optional) the type of projection. One of:
      * ``'left_ortho'`` (default): to be used as left preconditioner ``Ml``,
      * ``'left_oblique'`` to be used as left preconditioner ``Ml``,
      * ``'right_oblique'`` to be used as right preconditioner ``Mr``.
    :param inner_product: (optional) the inner product also used for the
      deflated iterative method, default is :py:meth:`~krypy.utils.ip_euclid`.

    :return: P, x0new

      * P: the projection to be used as _right_ preconditioner (e.g. Mr=P in
        MINRES). The preconditioned operator A*P is self-adjoint w.r.t.
        inner_product.
      * x0new: an adapted initial guess s.t. the deflated iterative solver
        does not break down (in exact arithmetics).

    For nW = W.shape[1] = AW.shape[1] the computational cost is
    cost(get_projection): 2*cost(Pfun) + (nW^2)*IP
    cost(Pfun): nW*IP + (2/3)*nW^3 + nW*AXPY
    """
    pass
