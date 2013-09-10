:mod:`krypy.deflation` - Deflated Krylov subspace methods
=========================================================

The deflation module provides methods for the solution of deflated linear
systems. With a deflated method one typically wants to improve the convergence
of the method by modifying a part of the operator's spectrum.

The starting point is a linear system of the form

.. math::

    Ax=b,

where :math:`A` is a linear operator on :math:`\mathbb{C}^N` and
:math:`x,b\in\mathbb{C}^N`. The inner product used in the Krylov subspace
methods is denoted by :math:`\langle\cdot,\cdot\rangle`.

In practice, a preconditioned linear algebraic system has to be solved:

.. math::

    M M_l A M_r y = M M_l b,\quad \text{with}~x=M_r y,

where the inner product is changed to :math:`\langle\cdot,\cdot\rangle_{M^{-1}}`
defined by :math:`\langle x,y\rangle_{M^{-1}}=\langle M^{-1}x,y\rangle`.
The semantics of the involved preconditioners :math:`M, M_l` and :math:`M_r`
is explained in detail in :py:meth:`~krypy.linsys.cg`,
:py:meth:`~krypy.linsys.minres` and :py:meth:`~krypy.linsys.gmres`. Deflation
then should address the preconditioned operator :math:`M M_l A M_r`. In order
to reduce the notational overhead we define

.. math::

    B := M_l A M_r.

Let two matrices :math:`U,V\in\mathbb{C}^{N,d}` such that
:math:`\langle V,B U\rangle\in\mathbb{C}^{d,d}` is non-singular. 
Then a projection :math:`P_M` can be defined by

.. math::
    
    P_M x = x - MBU\langle V,MBU\rangle_{M^{-1}}^{-1}\langle V,x\rangle_{M^{-1}}.

..
    TODO: explain what happens if U spans an invariant subspace.

Furthermore, the projections :math:`P` and :math:`\tilde{P}` can be
defined by

.. math::

    Px &= x - B U \langle V,B U\rangle^{-1}\langle V,x\rangle\quad\text{and}\\
    \tilde{P} x &= x - U \langle V,B U\rangle^{-1}\langle V, B x\rangle.

The Krylov subspace methods in :doc:`krypy.linsys <krypy.linsys>` can then be
applied to the deflated linear system

.. math::

    P_M M M_l A M_r y = P_M M M_l b.

However, in order to guarantee convergence (in exact arithmetic), some
restrictions have to be made concerning the choice of :math:`U` and :math:`V`:

* for :py:meth:`~krypy.linsys.cg`: the projected operator :math:`PB` has to be
  self-adjoint and positive definite with respect to
  :math:`\langle\cdot,\cdot\rangle`. This can be assured by choosing
  :math:`V=U`.
* for :py:meth:`~krypy.linsys.gmres`: the condition 
  :math:`\operatorname{Im}(U)\cap\operatorname{Im}(V)^{\perp_{M^{-1}}}=\{0\}`
  has to be fulfilled, see [GauGLN13]_. This can be achieved by either :math:`V=U` or
  :math:`V=BU`.
* for :py:meth:`~krypy.linsys.minres`: the same as for 
  :py:meth:`~krypy.linsys.gmres` and additionally the operator
  :math:`PB` or the operator :math:`PBP` has to be self-adjoint with respect to
  :math:`\langle\cdot,\cdot\rangle`. This can be fulfilled by choosing
  :math:`V=U` or :math:`V=BU`.

.. automodule:: krypy.deflation
    :members:
    :undoc-members:
    :show-inheritance:
