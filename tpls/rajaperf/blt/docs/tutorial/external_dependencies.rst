.. # Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level COPYRIGHT file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

External Dependencies
=====================

One key goal for BLT is to simplify the use of external dependencies when building your libraries and executables. 

To accomplish this BLT provides a ``DEPENDS_ON`` option for the
``blt_add_library()`` and ``blt_add_executable()`` macros that supports both CMake targets 
and external dependencies registered using the ``blt_register_library()`` macro.

The ``blt_import_library()`` macro allows you to reuse all information needed
for an external dependency under a single name.  This includes any include
directories, libraries, compile flags, link flags, defines, etc.  You can also
hide any warnings created by their headers by setting the
``TREAT_INCLUDES_AS_SYSTEM`` argument.

For example, to find and add the external dependency *axom* as a CMake target, you can simply use:

.. code-block:: cmake

    # FindAxom.cmake takes in AXOM_DIR, which is a installed Axom build and
    # sets variables AXOM_INCLUDES, AXOM_LIBRARIES
    include(FindAxom.cmake)
    blt_import_library(NAME      axom
                       TREAT_INCLUDES_AS_SYSTEM ON
                       DEFINES    HAVE_AXOM=1
                       INCLUDES   ${AXOM_INCLUDES}
                       LIBRARIES  ${AXOM_LIBRARIES}
                       EXPORTABLE ON)

Then *axom* is available to be used in the DEPENDS_ON list in the following
``blt_add_executable()`` or ``blt_add_library()`` calls, or in any CMake command that accepts a target.

CMake targets created by ``blt_import_library()`` are ``INTERFACE`` libraries that can be installed
and exported if the ``EXPORTABLE`` option is enabled.  For example, if the ``calc_pi`` project depends on
Axom, it could export its ``axom`` target.  To avoid introducing target name conflicts for users of the
``calc_pi`` project who might also create a target called ``axom``, ``axom`` should be exported as
``calc_pi::axom``.

This is especially helpful for "converting" external libraries that are not built with CMake
into CMake-friendly imported targets.

BLT also supports using ``blt_register_library()`` to provide additional options for existing CMake targets. 
The implementation doesn't modify the properties of the existing targets, 
it just exposes these options via BLT's support for  ``DEPENDS_ON``.

This first-class importing functionality provided by ``blt_import_library()`` should be preferred, but ``blt_register_library()`` is
still available for compatibility.  ``blt_import_library()`` is generally usable as a drop-in replacement, 
though it does not support the creation of targets with the same name as a target that already exists.  

.. note::
    Because CMake targets are only accessible from within the directory they were defined (including
    subdirectories), the ``include()`` command should be preferred to the ``add_subdirectory()`` command for adding CMake files
    that create imported library targets needed in other directories. The GLOBAL option to ``blt_import_library()``
    can also be used to manage visibility.

BLT's ``mpi``, ``cuda``, ``cuda_runtime``, and ``openmp`` targets are all defined via ``blt_import_library()``. 
You can see how in ``blt/thirdparty_builtin/CMakelists.txt``.  If your project exports targets and you would like
BLT's provided third-party targets to also be exported (for example, if a project that imports your project does not
use BLT), you can set the ``BLT_EXPORT_THIRDPARTY`` option to ``ON``.  As with other EXPORTABLE targets created by
``blt_import_library()``, these targets should be prefixed with the name of the project.  Either the ``EXPORT_NAME``
target property or the ``NAMESPACE`` option to CMake's ``install`` command can be used to modify the name of an
installed target.  See the "Exporting BLT Targets" page for more info.


.. admonition:: blt_register_library
   :class: hint

   A macro to register external libraries and dependencies with BLT.
   The named target can be added to the ``DEPENDS_ON`` argument of other BLT macros, 
   like ``blt_add_library()`` and ``blt_add_executable()``.  


You have already seen one use of ``DEPENDS_ON`` for a BLT
registered dependency in test_1:  ``gtest``

.. literalinclude:: calc_pi/CMakeLists.txt 
   :start-after: _blt_tutorial_calcpi_test1_executable_start
   :end-before:  _blt_tutorial_calcpi_test1_executable_end
   :language: cmake


``gtest`` is the name for the Google Test dependency in BLT registered via 
``blt_register_library()``. Even though Google Test is built-in and uses CMake,
``blt_register_library()`` allows us to easily set defines needed by all dependent
targets.


MPI Example
~~~~~~~~~~~~~~~~~~~~~

Our next example, ``test_2``, builds and tests the ``calc_pi_mpi`` library,
which uses MPI to parallelize the calculation over the integration intervals.


To enable MPI, we set ``ENABLE_MPI``, ``MPI_C_COMPILER``, and ``MPI_CXX_COMPILER`` in our host config file. Here is a snippet with these settings for LLNL's Surface Cluster:

.. literalinclude:: ../../host-configs/other/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake
   :start-after: _blt_tutorial_surface_mpi_config_start
   :end-before:  _blt_tutorial_surface_mpi_config_end
   :language: cmake


Here, you can see how ``calc_pi_mpi`` and ``test_2`` use ``DEPENDS_ON``:

.. literalinclude:: calc_pi/CMakeLists.txt 
   :start-after: _blt_tutorial_calcpi_test2_executable_start
   :end-before:  _blt_tutorial_calcpi_test2_executable_end
   :language: cmake


For MPI unit tests, you also need to specify the number of MPI Tasks
to launch. We use the ``NUM_MPI_TASKS`` argument to ``blt_add_test()`` macro.

.. literalinclude:: calc_pi/CMakeLists.txt 
   :start-after: _blt_tutorial_calcpi_test2_test_start
   :end-before:  _blt_tutorial_calcpi_test2_test_end
   :language: cmake


As mentioned in :ref:`UnitTesting`, google test provides a default ``main()``
driver that will execute all unit tests defined in the source. To test MPI code,
we need to create a main that initializes and finalizes MPI in addition to Google
Test. ``test_2.cpp`` provides an example driver for MPI with Google Test.

.. literalinclude:: calc_pi/test_2.cpp
   :start-after: _blt_tutorial_calcpi_test2_main_start
   :end-before:  _blt_tutorial_calcpi_test2_main_end
   :language: cpp

.. note::
  While we have tried to ensure that BLT chooses the correct setup information
  for MPI, there are several niche cases where the default behavior is
  insufficient. We have provided several available override variables:
  
  * ``BLT_MPI_COMPILE_FLAGS``
  * ``BLT_MPI_INCLUDES``
  * ``BLT_MPI_LIBRARIES``
  * ``BLT_MPI_LINK_FLAGS``
  
  BLT also has the variable ``ENABLE_FIND_MPI`` which turns off all CMake's ``FindMPI``
  logic and then uses the MPI wrapper directly when you provide them as the default
  compilers.


CUDA Example
~~~~~~~~~~~~~~~~~~~~~

Finally, ``test_3`` builds and tests the ``calc_pi_cuda`` library,
which uses CUDA to parallelize the calculation over the integration intervals.

To enable CUDA, we set ``ENABLE_CUDA``, ``CMAKE_CUDA_COMPILER``, and
``CUDA_TOOLKIT_ROOT_DIR`` in our host config file.  Also before enabling the
CUDA language in CMake, you need to set ``CMAKE_CUDA_HOST_COMPILER`` in CMake 3.9+
or ``CUDA_HOST_COMPILER`` in previous versions.  If you do not call 
``enable_language(CUDA)``, BLT will set the appropriate host compiler variable
for you and enable the CUDA language.

Here is a snippet with these settings for LLNL's Surface Cluster:

.. literalinclude:: ../../host-configs/other/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake
   :start-after: _blt_tutorial_surface_cuda_config_start
   :end-before:  _blt_tutorial_surface_cuda_config_end
   :language: cmake

Here, you can see how ``calc_pi_cuda`` and ``test_3`` use ``DEPENDS_ON``:

.. literalinclude:: calc_pi/CMakeLists.txt 
   :start-after: _blt_tutorial_calcpi_cuda_start
   :end-before:  _blt_tutorial_calcpi_cuda_end
   :language: cmake

The ``cuda`` dependency for ``calc_pi_cuda``  is a little special, 
along with adding the normal CUDA library and headers to your library or executable,
it also tells BLT that this target's C/CXX/CUDA source files need to be compiled via
``nvcc`` or ``cuda-clang``. If this is not a requirement, you can use the dependency
``cuda_runtime`` which also adds the CUDA runtime library and headers but will not
compile each source file with ``nvcc``.

Some other useful CUDA flags are:

.. code-block:: cmake

    # Enable separable compilation of all CUDA files for given target or all following targets
    set(CUDA_SEPARABLE_COMPILIATION ON CACHE BOOL “”)
    set(CUDA_ARCH “sm_60” CACHE STRING “”)
    set(CMAKE_CUDA_FLAGS “-restrict –arch ${CUDA_ARCH} –std=c++11” CACHE STRING “”)
    set(CMAKE_CUDA_LINK_FLAGS “-Xlinker –rpath –Xlinker /path/to/mpi” CACHE STRING “”)
    # Needed when you have CUDA decorations exposed in libraries
    set(CUDA_LINK_WITH_NVCC ON CACHE BOOL “”)


OpenMP
~~~~~~

To enable OpenMP, set ``ENABLE_OPENMP`` in your host-config file or before loading
``SetupBLT.cmake``.  Once OpenMP is enabled, simply add ``openmp`` to your library
executable's ``DEPENDS_ON`` list.

Here is an example of how to add an OpenMP enabled executable:

   .. literalinclude:: ../../tests/smoke/CMakeLists.txt
     :start-after: _blt_tutorial_openmp_executable_start
     :end-before:  _blt_tutorial_openmp_executable_end
     :language: cmake

.. note::
  While we have tried to ensure that BLT chooses the correct compile and link flags for
  OpenMP, there are several niche cases where the default options are insufficient.
  For example, linking with NVCC requires to link in the OpenMP libraries directly instead
  of relying on the compile and link flags returned by CMake's FindOpenMP package.  An
  example of this is in ``host-configs/llnl/blueos_3_ppc64le_ib_p9/clang@upstream_link_with_nvcc.cmake``. 
  We provide two variables to override BLT's OpenMP flag logic: 
  
  * ``BLT_OPENMP_COMPILE_FLAGS``
  * ``BLT_OPENMP_LINK_FLAGS``

Here is an example of how to add an OpenMP enabled test that sets the amount of threads used:

   .. literalinclude:: ../../tests/smoke/CMakeLists.txt
     :start-after: _blt_tutorial_openmp_test_start
     :end-before:  _blt_tutorial_openmp_test_end
     :language: cmake


Example Host-configs
~~~~~~~~~~~~~~~~~~~~

Here are the full example host-config files that use gcc 4.9.3 for LLNL's Surface, Ray and Quartz Clusters.

:download:`llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake <../../host-configs/other/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake>`

:download:`llnl/blueos_3_ppc64le_ib_p9/clang@upstream_nvcc_xlf <../../host-configs/llnl/blueos_3_ppc64le_ib_p9/clang@upstream_nvcc_xlf.cmake>`

:download:`llnl/toss_3_x86_64_ib/gcc@4.9.3.cmake <../../host-configs/llnl/toss_3_x86_64_ib/gcc@4.9.3.cmake>`

.. note::  Quartz does not have GPUs, so CUDA is not enabled in the Quartz host-config.

Here is a full example host-config file for an OSX laptop, using a set of dependencies built with spack.

:download:`darwin/elcapitan-x86_64/naples-clang@7.3.0.cmake  <../../host-configs/darwin/elcapitan-x86_64/naples-clang@7.3.0.cmake>`


Building and testing on Surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is how you can use the host-config file to configure a build of the ``calc_pi``  project with MPI and CUDA enabled on Surface:

.. code-block:: bash
    
    # load new cmake b/c default on surface is too old
    ml cmake/3.9.2
    # create build dir
    mkdir build
    cd build
    # configure using host-config
    cmake -C ../../host-configs/other/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake  \
          -DBLT_SOURCE_DIR=../../../../blt  ..

After building (``make``), you can run ``make test`` on a batch node (where the GPUs reside) 
to run the unit tests that are using MPI and CUDA:

.. code-block:: console

  bash-4.1$ salloc -A <valid bank>
  bash-4.1$ make   
  bash-4.1$ make test
  
  Running tests...
  Test project blt/docs/tutorial/calc_pi/build
      Start 1: test_1
  1/8 Test #1: test_1 ...........................   Passed    0.01 sec
      Start 2: test_2
  2/8 Test #2: test_2 ...........................   Passed    2.79 sec
      Start 3: test_3
  3/8 Test #3: test_3 ...........................   Passed    0.54 sec
      Start 4: blt_gtest_smoke
  4/8 Test #4: blt_gtest_smoke ..................   Passed    0.01 sec
      Start 5: blt_fruit_smoke
  5/8 Test #5: blt_fruit_smoke ..................   Passed    0.01 sec
      Start 6: blt_mpi_smoke
  6/8 Test #6: blt_mpi_smoke ....................   Passed    2.82 sec
      Start 7: blt_cuda_smoke
  7/8 Test #7: blt_cuda_smoke ...................   Passed    0.48 sec
      Start 8: blt_cuda_runtime_smoke
  8/8 Test #8: blt_cuda_runtime_smoke ...........   Passed    0.11 sec

  100% tests passed, 0 tests failed out of 8

  Total Test time (real) =   6.80 sec


Building and testing on Ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is how you can use the host-config file to configure a build of the ``calc_pi``  project with MPI and CUDA 
enabled on the blue_os Ray cluster:

.. code-block:: bash
    
    # load new cmake b/c default on ray is too old
    ml cmake
    # create build dir
    mkdir build
    cd build
    # configure using host-config
    cmake -C ../../host-configs/llnl/blueos_3_ppc64le_ib_p9/clang@upstream_nvcc_xlf.cmake \
          -DBLT_SOURCE_DIR=../../../../blt  ..

And here is how to build and test the code on Ray:

.. code-block:: console

  bash-4.2$ lalloc 1 -G <valid group>
  bash-4.2$ make
  bash-4.2$ make test
  
  Running tests...
  Test project projects/blt/docs/tutorial/calc_pi/build
      Start 1: test_1
  1/7 Test #1: test_1 ...........................   Passed    0.01 sec
      Start 2: test_2
  2/7 Test #2: test_2 ...........................   Passed    1.24 sec
      Start 3: test_3
  3/7 Test #3: test_3 ...........................   Passed    0.17 sec
      Start 4: blt_gtest_smoke
  4/7 Test #4: blt_gtest_smoke ..................   Passed    0.01 sec
      Start 5: blt_mpi_smoke
  5/7 Test #5: blt_mpi_smoke ....................   Passed    0.82 sec
      Start 6: blt_cuda_smoke
  6/7 Test #6: blt_cuda_smoke ...................   Passed    0.15 sec
      Start 7: blt_cuda_runtime_smoke
  7/7 Test #7: blt_cuda_runtime_smoke ...........   Passed    0.04 sec
  
  100% tests passed, 0 tests failed out of 7
  
  Total Test time (real) =   2.47 sec  
