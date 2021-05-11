# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
#
# SPDX-License-Identifier: (BSD-3-Clause)

###############################################################################
#
# Tests to make sure that definitions from one target do not propagate to another
#
###############################################################################

blt_add_executable(NAME       blt_hip_defines_test_foo
                   SOURCES    foo.cpp
                   DEFINES    FOO_ONLY
                   DEPENDS_ON hip)

blt_add_executable(NAME       blt_hip_defines_test_bar
                   SOURCES    bar.cpp
                   DEFINES    BAR_ONLY
                   DEPENDS_ON hip)

blt_add_test( NAME blt_test_hip_defines_first
              COMMAND blt_hip_defines_test_foo
            )

blt_add_test( NAME blt_test_hip_defines_second
              COMMAND blt_hip_defines_test_bar
            )
