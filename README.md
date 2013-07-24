# KryPy

KryPy is a Python (versions 2 and 3) module for Krylov subspace methods for the solution of linear algebraic systems. This includes enhanced versions of CG, MINRES and GMRES as well as methods for the efficient solution of sequences of linear systems.

# Features

KryPy gives you an easy-to-use yet flexible interface to Krylov subspace methods for linear algebraic systems. Compared to the implementations in [SciPy](http://docs.scipy.org/doc/scipy/reference/sparse.linalg.html) (or MATLAB), KryPy allows you to supply additional arguments that may help you to tune the solver for the specific problem you want to solve. The additional arguments may also be of interest if you are doing research on Krylov subspace methods. 

Some features of KryPy are:

*  **User-defined inner products** - useful when solving a linear algebraic system whose operator is self-adjoined in a non-Euclidean inner-product. This way, CG or MINRES can be applied to self-adjoined (but non-symmetric/non-Hermitian) operators easily.
*  **Full control of preconditioners** - the order of applying preconditioners matters. This is why you can supply two left preconditioners (one of whom implicitly changes the inner product and thus has to be positive definite) and one right preconditioner. Take a look at the arguments ```M```, ```Ml``` and ```Mr```.
*  **Get the Arnoldi/Lanczos basis and Hessenberg matrix** - you want to extract further information from the generated vectors (e.g. recycling)? Just pass the optional argument ```return_basis=True```.
*  **Explicitly computed residuals on demand** - if you do research on Krylov subspace methods or preconditioners, then you sometimes want to know the explicitly computed residual in each iteration (in contrast to an updated residual which can be obtained implicitly). Then you should pass the optional argument ```explicit_residual=True```.
*  **Compute errors** - if you have (for research purposes) the exact solution at hand and want to monitor the error in each iteration instead of the residual, you can supply the optional argument ```exact_solution=x_exact```.

# Usage

### Documentation
Here is the [documentation](http://andrenarchy.github.io/krypy/).

### Example
```python
from numpy import ones
from scipy.sparse import spdiags
from krypy.linsys import gmres

N = 10
A = spdiags(range(1,N+1), [0], N, N)
b = ones((N,1))

sol = gmres(A, b)
print (sol['relresvec'])
```

Of course, this is just a toy example. KryPy can handle arbitrary large matrices - as long as they fit into your memory. ;)

### Help

Help can be optained via Python's builtin help system. For example, you can use the ```?``` in ```ipython```:
```ipython
from krypy.linsys import gmres
?gmres
```

# Installing
### Ubuntu
There's an [Ubuntu PPA](https://launchpad.net/~andrenarchy/+archive/python) with packages for Python 2 and Python 3.

### Installing from source
KryPy has the following dependencies:
* NumPy
* SciPy

# Development

KryPy is currently maintained by [André Gaul](http://www.math.tu-berlin.de/~gaul/). Feel free to contact André. Please submit feature requests and bugs as github issues.

KryPy is developed with continuous integration. Each commit is tested with ~60000 automated unittests. Current status: [![Build Status](https://travis-ci.org/andrenarchy/krypy.png?branch=master)](https://travis-ci.org/andrenarchy/krypy)


# License

KryPy is free software licensed under the [MIT License](http://opensource.org/licenses/mit-license.php).

# References

KryPy evolved from the [PyNosh](https://bitbucket.org/nschloe/pynosh) package (Python framework for nonlinear Schrödinger equations; joint work with [Nico Schlömer](https://github.com/nschloe)) which was used for experiments in the following publication:
* [Modified Recycling MINRES with application to nonlinear Schrödinger problems, A. Gaul and N. Schlömer, arxiv: 1208.0264, 2012](http://arxiv.org/abs/1208.0264)
