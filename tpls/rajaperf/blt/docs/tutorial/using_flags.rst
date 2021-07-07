.. # Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level COPYRIGHT file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

Portable compiler flags
=========================

To simplify the development of code that is portable across different architectures
and compilers, BLT provides the ``blt_append_custom_compiler_flag()`` macro,
which allows users to easily place a compiler dependent flag into a CMake variable.

.. admonition:: blt_append_custom_compiler_flag
   :class: hint

   To use this macro, supply a cmake variable in which to append a flag (``FLAGS_VAR``), 
   and the appropriate flag for each of our supported compilers. 

   This macro currently supports the following compilers:

   * GNU
   * CLANG
   * XL (IBM compiler)
   * INTEL (Intel compiler)
   * MSVC (Microsoft Visual Studio)
   * MSVC_INTEL (Intel toolchain in Microsoft Visual Studio)
   * PGI
   * HCC (AMD GPU)

Here is an example for setting the appropriate flag to treat warnings as errors:

.. code-block:: cmake

    blt_append_custom_compiler_flag(
      FLAGS_VAR BLT_WARNINGS_AS_ERRORS_FLAG
      DEFAULT  "-Werror"
      MSVC     "/WX"
      XL       "qhalt=w"
      )

Since values for ``GNU``, ``CLANG`` and ``INTEL`` are not supplied, 
they will get the default value (``-Werror``)
which is supplied by the macro's ``DEFAULT`` argument.

BLT also provides a simple macro to add compiler flags to a target.  
You can append the above compiler flag to an already defined executable, 
such as ``example_1`` with the following line:

.. code-block:: cmake

    blt_add_target_compile_flags(TO example_1
                                 FLAGS BLT_WARNINGS_AS_ERRORS_FLAG )

Here is another example to disable warnings about unknown OpenMP pragmas in the code:

.. code-block:: cmake

    # Flag for disabling warnings about omp pragmas in the code
    blt_append_custom_compiler_flag(
        FLAGS_VAR DISABLE_OMP_PRAGMA_WARNINGS_FLAG
        DEFAULT "-Wno-unknown-pragmas"
        XL      "-qignprag=omp"
        INTEL   "-diag-disable 3180"
        MSVC    "/wd4068"
        )

Note that GNU does not have a way to only disable warnings about openmp pragmas, 
so one must disable warnings about all unknown pragmas on this compiler.

