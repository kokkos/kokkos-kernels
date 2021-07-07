###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME kokkos-mechanics
  SOURCES ViewAllocate.cpp
          ViewAllocate-Stubs.cpp
          ViewAllocate-KokkosSeq.cpp
          ViewAllocate-KokkosCuda.cpp
  	  ViewStreamAdd.cpp
          ViewStreamAdd-Stubs.cpp
          ViewStreamAdd-KokkosSeq.cpp
          ViewStreamAdd-KokkosCuda.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
