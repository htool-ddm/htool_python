# Python interface of Htool-DDM [![CI](https://github.com/htool-ddm/htool_python/actions/workflows/CI.yml/badge.svg)](https://github.com/htool-ddm/htool_python/actions/workflows/CI.yml) [![codecov](https://codecov.io/gh/htool-ddm/htool_python/branch/main/graph/badge.svg?token=P3FQNL8E64)](https://codecov.io/gh/htool-ddm/htool_python) [![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](http://htool-ddm.pages.math.cnrs.fr/)

**Htool-DDM** is a lightweight header-only C++14 library that provides an easy-to-use interface for parallel iterative solvers and a default matrix compression via in-house hierarchical matrix implementation. Its goal is to provide modern iterative solvers for dense/compressed linear systems.

It is also an extensible framework which contains several customization points. For example, one can provide its own compression algorithm, or customize the default hierarchical compression. Via its interface with [HPDDM](https://github.com/hpddm/hpddm), it is also a flexible tool to test various iterative solvers and preconditioners.

See [documentation](http://htool-ddm.pages.math.cnrs.fr/) for more information.
