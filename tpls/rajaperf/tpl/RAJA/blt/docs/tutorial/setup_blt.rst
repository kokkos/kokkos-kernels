.. # Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level COPYRIGHT file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

Setup BLT in your CMake Project
=================================

BLT is easy to include in your CMake project whether it is an existing project or
you are starting from scratch. You simply pull it into your project using a CMake
``include()`` command.

.. code-block:: cmake

    include(path/to/blt/SetupBLT.cmake)

You can include the BLT source in your repository or pass the location 
of BLT at CMake configure time through the optional ``BLT_SOURCE_DIR``
CMake variable. 

There are two standard choices for including the BLT source in your repository:

1. Add BLT as a git submodule
2. Copy BLT into a subdirectory in your repository


BLT as a Git Submodule
--------------------------

This example adds BLT as a submodule, commits, and pushes the changes to your repository.

.. code-block:: bash

    cd <your repository>
    git submodule add https://github.com/LLNL/blt.git blt
    git commit -m "Adding BLT"
    git push

Copy BLT into your repository
-----------------------------

This example will clone BLT into your repository and remove the unneeded 
git files from the clone. It then commits and pushes the changes to your
repository.

.. code-block:: bash

    cd <your repository>
    git clone https://github.com/LLNL/blt.git
    rm -rf blt/.git
    git commit -m "Adding BLT"
    git push


Include BLT in your project
---------------------------

In most projects, including BLT is as simple as including the following CMake
line in your base ``CMakeLists.txt`` after your ``project()`` call.

.. code-block:: cmake

    include(blt/SetupBLT.cmake)

This enables all of BLT's features in your project. 

However if your project is likely to be used by other projects.  The following
is recommended:

.. literalinclude:: blank_project/CMakeLists.txt
   :start-after: _blt_tutorial_include_blt_start
   :end-before:  _blt_tutorial_include_blt_end
   :language: cmake

This is a robust way of setting up BLT and supports an optional external BLT source
directory. This allows the use of a common BLT across large projects. There are some
helpful error messages if the BLT submodule is missing as well as the commands to solve
it. 

.. note::
  Throughout this tutorial, we pass the path to BLT using ``BLT_SOURCE_DIR`` since 
  our tutorial is part of the blt repository and we want this project to be 
  automatically tested using a single clone of our repository.



Running CMake
-------------

To configure a project with CMake, first create a build directory and ``cd`` into it.  
Then run cmake with the path to your project.  

.. code-block:: bash

    cd <your project>
    mkdir build
    cd build
    cmake ..

If you are using BLT outside of your project pass the location of BLT as follows:

.. code-block:: bash

    cd <your project>
    mkdir build
    cd build
    cmake -DBLT_SOURCE_DIR="path/to/blt" ..

Example: blank_project
----------------------

The ``blank_project`` example is provided to show you some of BLT's built-in
features. It demonstrates the bare minimum required for testing purposes.

Here is the entire CMakeLists.txt file for ``blank_project``:

.. literalinclude:: blank_project/CMakeLists.txt
   :language: cmake

BLT also enforces some best practices for building, such as not allowing
in-source builds.  This means that BLT prevents you from generating a
project configuration directly in your project. 

For example if you run the following commands:

.. code-block:: bash

    cd <BLT repository>/docs/tutorial/blank_project
    cmake -DBLT_SOURCE_DIR=../..

you will get the following error:

.. code-block:: bash

    CMake Error at blt/SetupBLT.cmake:59 (message):
      In-source builds are not supported.  Please remove CMakeCache.txt from the
      'src' dir and configure an out-of-source build in another directory.
    Call Stack (most recent call first):
      CMakeLists.txt:55 (include)


    -- Configuring incomplete, errors occurred!

To correctly run cmake, create a build directory and run cmake from there:

.. code-block:: bash

    cd <BLT repository>/docs/blank_project
    mkdir build
    cd build
    cmake -DBLT_SOURCE_DIR=../../.. ..

This will generate a configured ``Makefile`` in your build directory to build
``blank_project``.  The generated makefile includes gtest and several built-in
BLT *smoke* tests, depending on the features that you have enabled in your build.  

To build the project, use the following command:

.. code-block:: bash

    make

As with any other ``make``-based project, you can utilize parallel job tasks 
to speed up the build with the following command:

.. code-block:: bash

    make -j8

Next, run all tests in this project with the following command:

.. code-block:: bash

    make test

If everything went correctly, you should have the following output:

.. code-block:: bash

    Running tests...
    Test project blt/docs/tutorial/blank_project/build
        Start 1: blt_gtest_smoke
    1/1 Test #1: blt_gtest_smoke ..................   Passed    0.01 sec

    100% tests passed, 0 tests failed out of 1

    Total Test time (real) =   0.10 sec

Note that the default options for ``blank_project`` only include a single test
``blt_gtest_smoke``. As we will see later on, BLT includes additional smoke
tests that are activated when BLT is configured with other options enabled,
like Fortran, MPI, OpenMP, and Cuda. 

Host-configs
------------

To capture (and revision control) build options, third party library paths, etc.,
we recommend using CMake's initial-cache file mechanism. This feature allows you
to pass a file to CMake that provides variables to bootstrap the configuration
process. 

You can pass initial-cache files to cmake via the ``-C`` command line option.

.. code-block:: bash

    cmake -C config_file.cmake


We call these initial-cache files ``host-config`` files since we typically create
a file for each platform or for specific hosts, if necessary. 

These files use standard CMake commands. CMake ``set()`` commands need to specify
``CACHE`` as follows:

.. code-block:: cmake

    set(CMAKE_VARIABLE_NAME {VALUE} CACHE PATH "")

Here is a snippet from a host-config file that specifies compiler details for
using gcc 4.9.3 on LLNL's surface cluster. 

.. literalinclude:: ../../host-configs/other/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake
   :start-after: _blt_tutorial_surface_compiler_config_start
   :end-before:  _blt_tutorial_surface_compiler_config_end
   :language: cmake


