###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME algorithm
  SOURCES SORT.cpp
          SORT-Seq.cpp
          SORT-Hip.cpp
          SORT-Cuda.cpp
          SORT-OMP.cpp
          SORTPAIRS.cpp
          SORTPAIRS-Seq.cpp
          SORTPAIRS-Hip.cpp
          SORTPAIRS-Cuda.cpp
          SORTPAIRS-OMP.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
