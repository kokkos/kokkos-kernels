.. # Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level COPYRIGHT file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

.. _UnitTesting:

Unit Testing
============

BLT has a built-in copy of the 
`Google Test framework (gtest) <https://github.com/google/googletest>`_ for C
and C++ unit tests and the 
`Fortran Unit Test Framework (FRUIT) <https://sourceforge.net/projects/fortranxunit/>`_
for Fortran unit tests.

Each Google Test or FRUIT file may contain multiple tests and is compiled into
its own executable that can be run directly or as a ``make`` target. 

In this section, we give a brief overview of GTest and discuss how to add unit
tests using the ``blt_add_test()`` macro.


Configuring tests within BLT
----------------------------

Unit testing in BLT is controlled by the ``ENABLE_TESTS`` cmake option and is on
by default. 

For additional configuration granularity, BLT provides configuration options 
for the individual built-in unit testing libraries.  The following additional
options are available when ``ENABLE_TESTS`` is on:

``ENABLE_GTEST``
  Option to enable gtest (default: ``ON``).
``ENABLE_GMOCK``
  Option to control gmock (default: ``OFF``).
  Since gmock requires gtest, gtest is also enabled whenever ``ENABLE_GMOCK`` is true, 
  regardless of the value of ``ENABLE_GTEST``. 
``ENABLE_FRUIT``
  Option to control FRUIT (Default ``ON``). It is only active when ``ENABLE_FORTRAN`` is enabled.


Google Test (C++/C Tests)
--------------------------

The contents of a typical Google Test file look like this:

.. code:: cpp

  #include "gtest/gtest.h"

  #include ...    // include headers needed to compile tests in file

  // ...

  TEST(<test_case_name>, <test_name_1>) 
  {
     // Test 1 code here...
     // ASSERT_EQ(...);
  }

  TEST(<test_case_name>, <test_name_2>) 
  {
     // Test 2 code here...
     // EXPECT_TRUE(...);
  }

  // Etc.

Each unit test is defined by the Google Test ``TEST()`` macro which accepts a 
*test case name* identifier, such as the name of the C++ class being tested, 
and a *test name*, which indicates the functionality being verified by the 
test.  Within a test, failure of logical assertions (macros prefixed by ``ASSERT_``)
will cause the test to fail immediately, while failures of expected values 
(macros prefixed by ``EXPECT_``) will cause the test to fail, but will 
continue running code within the test.

Note that the Google Test framework will generate a ``main()`` routine for 
each test file if it is not explicitly provided. However, sometimes it is 
necessary to provide a ``main()`` routine that contains operation to run 
before or after the unit tests in a file; e.g., initialization code or 
pre-/post-processing operations. A ``main()`` routine provided in a test 
file should be placed at the end of the file in which it resides.


Note that Google Test is initialized before ``MPI_Init()`` is called. 

Other Google Test features, such as *fixtures* and *mock* objects (gmock) may
be used as well. 

See the `Google Test Primer <https://github.com/google/googletest/blob/master/googletest/docs/Primer.md>`_ 
for a discussion of Google Test concepts, how to use them, and a listing of 
available assertion macros, etc.

FRUIT (Fortran Tests)
--------------------------

Fortran unit tests using the FRUIT framework are similar in structure to 
the Google Test tests for C and C++ described above.

The contents of a typical FRUIT test file look like this::

  module <test_case_name>
    use iso_c_binding
    use fruit
    use <your_code_module_name>
    implicit none

  contains

  subroutine test_name_1
  !  Test 1 code here...
  !  call assert_equals(...)
  end subroutine test_name_1

  subroutine test_name_2
  !  Test 2 code here...
  !  call assert_true(...)
  end subroutine test_name_2

  ! Etc.

The tests in a FRUIT test file are placed in a Fortran *module* named for
the *test case name*, such as the name of the C++ class whose Fortran interface
is being tested. Each unit test is in its own Fortran subroutine named
for the *test name*, which indicates the functionality being verified by the
unit test. Within each unit test, logical assertions are defined using
FRUIT methods. Failure of expected values will cause the test
to fail, but other tests will continue to run.

Note that each FRUIT test file defines an executable Fortran program. The
program is defined at the end of the test file and is organized as follows::

    program fortran_test
      use fruit
      use <your_component_unit_name>
      implicit none
      logical ok
      
      ! initialize fruit
      call init_fruit
      
      ! run tests
      call test_name_1
      call test_name_2
      
      ! compile summary and finalize fruit
      call fruit_summary
      call fruit_finalize
      
      call is_all_successful(ok)
      if (.not. ok) then
        call exit(1)
      endif
    end program fortran_test


Please refer to the `FRUIT documentation <https://sourceforge.net/projects/fortranxunit/>`_ for more information.


Adding a BLT unit test 
----------------------

After writing a unit test, we add it to the project's build system 
by first generating an executable for the test using the
``blt_add_executable()`` macro. We then register the test using the
``blt_add_test()`` macro.

.. admonition:: blt_add_test
   :class: hint

   This macro generates a named unit test from an existing executable
   and allows users to pass in command line arguments.


Returning to our running example (see  :ref:`AddTarget`), 
let's add a simple test for the ``calc_pi`` library, 
which has a single function with signature:

  .. code-block:: cpp

   double calc_pi(int num_intervals);

We add a simple unit test that invokes ``calc_pi()`` 
and compares the result to :math:`\pi`, within a given tolerance (``1e-6``). 
Here is the test code:

.. literalinclude:: calc_pi/test_1.cpp
   :start-after: _blt_tutorial_calpi_test1_start
   :end-before:  _blt_tutorial_calpi_test1_end
   :language: cpp

To add this test to the build system, we first generate a test executable:

.. literalinclude:: calc_pi/CMakeLists.txt
   :start-after: _blt_tutorial_calcpi_test1_executable_start
   :end-before:  _blt_tutorial_calcpi_test1_executable_end
   :language: cmake

Note that this test executable depends on two targets: ``calc_pi`` and ``gtest``.

We then register this executable as a test:

.. literalinclude:: calc_pi/CMakeLists.txt
   :start-after: _blt_tutorial_calcpi_test1_test_start
   :end-before:  _blt_tutorial_calcpi_test1_test_end
   :language: cmake

.. Hide these for now until we bring into an example
.. .. note::
..    The ``COMMAND`` parameter can be used to pass arguments into a test executable.
..    This feature is not exercised in this example.
..
.. .. note::
..    The ``NAME`` of the test can be different from the test executable,
..    which is passed in through the ``COMMAND`` parameter.
..    This feature is not exercised in this example.


Running tests and examples
--------------------------

To run the tests, type the following command in the build directory:

.. code-block:: bash

  $ make test 

This will run all tests through cmake's ``ctest`` tool 
and report a summary of passes and failures. 
Detailed output on individual tests is suppressed.

If a test fails, you can invoke its executable directly to see the detailed
output of which checks passed or failed. This is especially useful when 
you are modifying or adding code and need to understand how unit test details
are working, for example.

.. note:: 
    You can pass arguments to ctest via the ``TEST_ARGS`` parameter, e.g.
    ``make test TEST_ARGS="..."``
    Useful arguments include:
    
    -R
      Regular expression filtering of tests.  
      E.g. ``-R foo`` only runs tests whose names contain ``foo``
    -j
      Run tests in parallel, E.g. ``-j 16`` will run tests using 16 processors
    -VV
      (Very verbose) Dump test output to stdout

