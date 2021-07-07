.. # Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level COPYRIGHT file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

User Tutorial
=============

This tutorial provides instructions for:

    * Adding BLT to a CMake project
    * Setting up *host-config* files to handle multiple platform configurations
    * Building, linking, and installing libraries and executables
    * Setting up unit tests with GTest
    * Using external project dependencies
    * Creating documentation with Sphinx and Doxygen

The tutorial provides several examples that calculate the value of :math:`\pi` 
by approximating the integral :math:`f(x) = \int_0^14/(1+x^2)` using numerical
integration. The code is adapted from:
https://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples-usingmpi/simplempi/cpi_c.html.

The tutorial requires a C++ compiler and CMake, we recommend using CMake 3.8.0 or newer. 
Parts of the tutorial also require MPI, CUDA, Sphinx and Doxygen.


.. toctree::
    :maxdepth: 3
    :caption: Tutorial Contents

    setup_blt
    creating_execs_and_libs
    using_flags
    unit_testing
    external_dependencies
    exporting_blt_targets
    creating_documentation
    recommendations
