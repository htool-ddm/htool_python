# Python interface of Htool [![Build Status](https://travis-ci.com/htool-ddm/htool_python.svg?branch=master)](https://travis-ci.com/htool-ddm/htool_python)

## What is Htool?

Htool is an implementation of hierarchical matrices (cf. this [reference](http://www.springer.com/gp/book/9783662473238) or this [one](http://www.springer.com/gp/book/9783540771463)), it was written to test Domain Decomposition Methods (DDM) applied to Boundary Element Method (BEM). It provides:

* routines to build hierarchical matrix structures (cluster trees, block trees, low-rank matrices and block matrices),
* parallel matrix-vector and matrix-matrix product using MPI and OpenMP,
* preconditioning techniques using domain decomposition methods,
* the possibility to use Htool with any generator of coefficients (e.g., your own BEM library),
* an interface with [HPDDM](https://github.com/hpddm/hpddm) for iterative solvers,
* and several utility functions to display information about matrix structures and timing.

It is now used in [FreeFEM](https://freefem.org) starting from version 4.5.

## How to use Htool in Python?

### Dependencies

Htool is a header-only library written in C++11 with MPI and OpenMP, but it can be used without the latter if needed. Then, to use Htool, you need to have:

* BLAS, to perform algebraic operations (dense matrix-matrix or matrix-vector operations).
* LAPACK, to perform SVD compression and to be used in DDM solvers with HPDDM.

### Installing

In the folder of this repository, do:

```bash
pip install .
```

### Embedding Htool in your code

We mostly refer to `smallest_example.py` in the `examples` folder to see how to use Htool.

A function that generates the coefficients must be provided to Htool. To do so, a class inheriting from `IMatrix` or `ComplexIMatrix` must be defined with a method `get_coef(i, j)`, where `i` and `j` are integers. This method will return the coefficient (i,j) of the considered problem. A method `get_submatrix` can also be defined to provide a more efficient way to build a sub-block of the matrix. This new class and the geometry will be used to define an object `HMatrix`.

### Difference with C++

**TL;DR** Due to Htool design and Python limitations, there is no shared parallelism for building hierarchical matrices. But shared parallelism is still used in the hierarchical matrix vector/matrix product.

**Details** One feature of Htool is to take a function from the user to generate a hierarchical matrix. In the case of the Python interface, a Python function. Then, we loop over the blocks to build calling the user's function. In C++, we can use threads to accelerate this loop, but we cannot in Python because of the *Global Interpreter Lock*, which prevents several threads to call the user's function. One way to leverage this issue in terms of scaling is to use threading inside the function provided to Htool.

## Who is behind Htool?

If you need help or have questions regarding Htool, feel free to contact [Pierre Marchand](https://www.ljll.math.upmc.fr/marchandp/) and Pierre-Henri Tournier.

## Acknowledgements

[ANR NonlocalDD](https://www.ljll.math.upmc.fr/~claeys/nonlocaldd/index.html), (grant ANR-15-CE23-0017-01), France  
[Inria](http://www.inria.fr/en/) Paris, France  
[Laboratoire Jacques-Louis Lions](https://www.ljll.math.upmc.fr/en/) Paris, France  

## Collaborators/contributors

[Matthieu Ancellin](https://ancell.in)  
[Xavier Claeys](https://www.ljll.math.upmc.fr/~claeys/)  
[Pierre Jolivet](http://jolivet.perso.enseeiht.fr/)  
[Frédéric Nataf](https://www.ljll.math.upmc.fr/nataf/)

![ANR NonlocalDD](figures/anr_nonlocaldd.png)
