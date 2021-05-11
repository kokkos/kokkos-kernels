###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../basic)

blt_add_library(
  NAME basic-kokkos
  SOURCES
          ATOMIC_PI-Kokkos.cpp
          DAXPY-Kokkos.cpp
          IF_QUAD-Kokkos.cpp
          INIT3-Kokkos.cpp
          INIT_VIEW1D-Kokkos.cpp
          INIT_VIEW1D_OFFSET-Kokkos.cpp
          MULADDSUB-Kokkos.cpp
          NESTED_INIT-Kokkos.cpp
          REDUCE3_INT-Kokkos.cpp
          TRAP_INT-Kokkos.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )

# Diagnostics
message (STATUS "${RAJA_PERFSUITE_DEPENDS}")
#blt_print_target_properties(TARGET basic-kokkos)
#blt_print_target_properties(TARGET kokkos)

#kokkoscore;kokkoscontainers;kokkosalgorithms
#blt_print_target_properties(TARGET kokkoscore)
#blt_print_target_properties(TARGET kokkoscontainers)
#blt_print_target_properties(TARGET kokkosalgorithms)

blt_print_target_properties(TARGET RAJA)

get_source_file_property(blah ATOMIC_PI-Kokkos.cpp HIP_SOURCE_PROPERTY_FORMAT)
message (STATUS "DOGS1 - ${blah}")
