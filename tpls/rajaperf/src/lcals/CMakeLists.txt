###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME lcals
  SOURCES DIFF_PREDICT.cpp
          DIFF_PREDICT-Seq.cpp
          DIFF_PREDICT-Hip.cpp
          DIFF_PREDICT-Cuda.cpp
          DIFF_PREDICT-OMP.cpp
          DIFF_PREDICT-OMPTarget.cpp
          EOS.cpp
          EOS-Seq.cpp
          EOS-Hip.cpp
          EOS-Cuda.cpp
          EOS-OMP.cpp
          EOS-OMPTarget.cpp
          FIRST_DIFF.cpp
          FIRST_DIFF-Seq.cpp
          FIRST_DIFF-Hip.cpp
          FIRST_DIFF-Cuda.cpp
          FIRST_DIFF-OMP.cpp
          FIRST_DIFF-OMPTarget.cpp
          FIRST_MIN.cpp
          FIRST_MIN-Seq.cpp
          FIRST_MIN-Hip.cpp
          FIRST_MIN-Cuda.cpp
          FIRST_MIN-OMP.cpp
          FIRST_MIN-OMPTarget.cpp
          FIRST_SUM.cpp
          FIRST_SUM-Seq.cpp
          FIRST_SUM-Hip.cpp
          FIRST_SUM-Cuda.cpp
          FIRST_SUM-OMP.cpp
          FIRST_SUM-OMPTarget.cpp
          GEN_LIN_RECUR.cpp
          GEN_LIN_RECUR-Seq.cpp
          GEN_LIN_RECUR-Hip.cpp
          GEN_LIN_RECUR-Cuda.cpp
          GEN_LIN_RECUR-OMP.cpp
          GEN_LIN_RECUR-OMPTarget.cpp
          HYDRO_1D.cpp
          HYDRO_1D-Seq.cpp
          HYDRO_1D-Hip.cpp
          HYDRO_1D-Cuda.cpp
          HYDRO_1D-OMP.cpp
          HYDRO_1D-OMPTarget.cpp
          HYDRO_2D.cpp
          HYDRO_2D-Seq.cpp
          HYDRO_2D-Hip.cpp
          HYDRO_2D-Cuda.cpp
          HYDRO_2D-OMP.cpp
          HYDRO_2D-OMPTarget.cpp
          INT_PREDICT.cpp
          INT_PREDICT-Seq.cpp
          INT_PREDICT-Hip.cpp
          INT_PREDICT-Cuda.cpp
          INT_PREDICT-OMP.cpp
          INT_PREDICT-OMPTarget.cpp
          PLANCKIAN.cpp
          PLANCKIAN-Seq.cpp
          PLANCKIAN-Hip.cpp
          PLANCKIAN-Cuda.cpp
          PLANCKIAN-OMP.cpp
          PLANCKIAN-OMPTarget.cpp
          TRIDIAG_ELIM.cpp
          TRIDIAG_ELIM-Seq.cpp
          TRIDIAG_ELIM-Hip.cpp
          TRIDIAG_ELIM-Cuda.cpp
          TRIDIAG_ELIM-OMP.cpp
          TRIDIAG_ELIM-OMPTarget.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
