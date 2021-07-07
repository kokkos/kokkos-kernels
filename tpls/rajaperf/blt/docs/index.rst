.. # Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level COPYRIGHT file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

BLT
===

**Build, Link, and Test**

BLT is a composition of CMake macros and several widely used open source tools 
assembled to simplify HPC software development. 

BLT was released by Lawrence Livermore National Laboratory (LLNL) under a BSD-style open source license. 
It is developed on github under LLNL's github organization: https://github.com/llnl/blt

.. note::
  BLT officially supports CMake 3.8 and above.  However we only print a warning if you
  are below this version.  Some features in earlier versions may or may not work. Use at your own risk.


BLT at a Glance
~~~~~~~~~~~~~~~~~~

* Simplifies setting up a CMake-based build system

  * CMake macros for:

    * Creating libraries and executables
    * Managing compiler flags
    * Managing external dependencies

  * Multi-platform support (HPC Platforms, OSX, Windows)

* Batteries included

  * Built-in support for HPC Basics: MPI, OpenMP, and CUDA
  * Built-in support for unit testing in C/C++ and Fortran
  
* Streamlines development processes

  * Support for documentation generation
  * Support for code health tools:

    * Runtime and static analysis, benchmarking

Developers
~~~~~~~~~~

 * Chris White (white238@llnl.gov)
 * Cyrus Harrison (harrison37@llnl.gov)
 * George Zagaris (zagaris2@llnl.gov)
 * Kenneth Weiss (kweiss@llnl.gov)
 * Lee Taylor (taylor16@llnl.gov)
 * Aaron Black (black27@llnl.gov)
 * David A. Beckingsale (beckingsale1@llnl.gov)
 * Richard Hornung (hornung1@llnl.gov)
 * Randolph Settgast (settgast1@llnl.gov)
 * Peter Robinson (robinson96@llnl.gov)


Documentation
~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    User Tutorial <tutorial/index>
    API Documentation <api/index>
