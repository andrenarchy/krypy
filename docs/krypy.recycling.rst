:mod:`krypy.recycling` - Recycling Linear Systems Solvers
===========================================================

The ``recycling`` module provides functions for the solution of sequences of
linear algebraic systems. Once a linear system has been solved, the
generated data is examined and a deflation space is determined
automatically for the solution of the next linear system. Several selection
strategys are available.

.. toctree::

    krypy.recycling.factories
    krypy.recycling.generators
    krypy.recycling.evaluators

.. automodule:: krypy.recycling
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: krypy.recycling.linsys._RecyclingSolver
    :members:
