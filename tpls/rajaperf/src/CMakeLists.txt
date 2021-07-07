###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include_directories(.)

add_subdirectory(common)
set(INFRASTRUCTURE_ONLY TRUE)
if(NOT INFRASTRUCTURE_ONLY)
#add_subdirectory(apps)
add_subdirectory(basic)
add_subdirectory(basic-kokkos)
#add_subdirectory(kokkos-mechanics)
#add_subdirectory(lcals)
#add_subdirectory(polybench)
#add_subdirectory(stream)
#add_subdirectory(algorithm)
endif()
set(RAJA_PERFSUITE_EXECUTABLE_DEPENDS common)

if(NOT INFRASTRUCTURE_ONLY)
list(APPEND RAJA_PERFSUITE_EXECUTABLE_DEPENDS
    basic
    basic-kokkos
    )
endif()
list(APPEND RAJA_PERFSUITE_EXECUTABLE_DEPENDS ${RAJA_PERFSUITE_DEPENDS}) 

if(ENABLE_TARGET_OPENMP)
  remove_definitions(-DRUN_RAJA_SEQ -DRUN_OPENMP )

blt_add_executable(
  NAME raja-perf-omptarget.exe
  SOURCES RAJAPerfSuiteDriver.cpp
  #apps/AppsData.cpp
  #apps/DEL_DOT_VEC_2D.cpp
  #apps/DEL_DOT_VEC_2D-Seq.cpp
  #apps/DEL_DOT_VEC_2D-OMPTarget.cpp
  #apps/ENERGY.cpp
  #apps/ENERGY-Seq.cpp
  #apps/ENERGY-OMPTarget.cpp
  #apps/FIR.cpp
  #apps/FIR-Seq.cpp
  #apps/FIR-OMPTarget.cpp
  #apps/PRESSURE.cpp
  #apps/PRESSURE-Seq.cpp
  #apps/PRESSURE-OMPTarget.cpp
  #apps/LTIMES.cpp
  #apps/LTIMES-Seq.cpp
  #apps/LTIMES-OMPTarget.cpp
  #apps/LTIMES_NOVIEW.cpp
  #apps/LTIMES_NOVIEW-Seq.cpp
  #apps/LTIMES_NOVIEW-OMPTarget.cpp
  #apps/VOL3D.cpp
  #apps/VOL3D-Seq.cpp
  #apps/VOL3D-OMPTarget.cpp
  #apps/WIP-COUPLE.cpp
  basic/ATOMIC_PI.cpp
  basic/ATOMIC_PI-Seq.cpp
  basic/ATOMIC_PI-OMPTarget.cpp
  basic/DAXPY.cpp
  basic/DAXPY-Seq.cpp
  basic/DAXPY-OMPTarget.cpp
  basic/IF_QUAD.cpp
  basic/IF_QUAD-Seq.cpp
  basic/IF_QUAD-OMPTarget.cpp
  basic/INIT3.cpp
  basic/INIT3-Seq.cpp
  basic/INIT3-OMPTarget.cpp
  basic/INIT_VIEW1D.cpp
  basic/INIT_VIEW1D-Seq.cpp
  basic/INIT_VIEW1D-OMPTarget.cpp
  basic/INIT_VIEW1D_OFFSET.cpp
  basic/INIT_VIEW1D_OFFSET-Seq.cpp
  basic/INIT_VIEW1D_OFFSET-OMPTarget.cpp
  basic/MULADDSUB.cpp
  basic/MULADDSUB-Seq.cpp
  basic/MULADDSUB-OMPTarget.cpp
  basic/NESTED_INIT.cpp
  basic/NESTED_INIT-Seq.cpp
  basic/NESTED_INIT-OMPTarget.cpp
  basic/REDUCE3_INT.cpp
  basic/REDUCE3_INT-Seq.cpp
  basic/REDUCE3_INT-OMPTarget.cpp
  basic/TRAP_INT.cpp
  basic/TRAP_INT-Seq.cpp
  basic/TRAP_INT-OMPTarget.cpp
  basic-kokkos/ATOMIC_PI-Kokkos.cpp
  basic-kokkos/DAXPY-Kokkos.cpp
  basic-kokkos/IF_QUAD-Kokkos.cpp
  basic-kokkos/INIT3-Kokkos.cpp
  basic-kokkos/INIT_VIEW1D-Kokkos.cpp
  basic-kokkos/INIT_VIEW1D_OFFSET-Kokkos.cpp
  basic-kokkos/MULADDSUB-Kokkos.cpp
  basic-kokkos/NESTED_INIT-Kokkos.cpp
  basic-kokkos/REDUCE3_INT-Kokkos.cpp
  basic-kokkos/TRAP_INT-Kokkos.cpp
  #lcals/DIFF_PREDICT.cpp
  #lcals/DIFF_PREDICT-Seq.cpp
  #lcals/DIFF_PREDICT-OMPTarget.cpp
  #lcals/EOS.cpp
  #lcals/EOS-Seq.cpp
  #lcals/EOS-OMPTarget.cpp
  #lcals/FIRST_DIFF.cpp
  #lcals/FIRST_DIFF-Seq.cpp
  #lcals/FIRST_DIFF-OMPTarget.cpp
  #lcals/FIRST_MIN.cpp
  #lcals/FIRST_MIN-Seq.cpp
  #lcals/FIRST_MIN-OMPTarget.cpp
  #lcals/FIRST_SUM.cpp
  #lcals/FIRST_SUM-Seq.cpp
  #lcals/FIRST_SUM-OMPTarget.cpp
  #lcals/GEN_LIN_RECUR.cpp
  #lcals/GEN_LIN_RECUR-Seq.cpp
  #lcals/GEN_LIN_RECUR-OMPTarget.cpp
  #lcals/HYDRO_1D.cpp
  #lcals/HYDRO_1D-Seq.cpp
  #lcals/HYDRO_1D-OMPTarget.cpp
  #lcals/HYDRO_2D.cpp
  #lcals/HYDRO_2D-Seq.cpp
  #lcals/HYDRO_2D-OMPTarget.cpp
  #lcals/INT_PREDICT.cpp
  #lcals/INT_PREDICT-Seq.cpp
  #lcals/INT_PREDICT-OMPTarget.cpp
  #lcals/PLANCKIAN.cpp
  #lcals/PLANCKIAN-Seq.cpp
  #lcals/PLANCKIAN-OMPTarget.cpp
  #lcals/TRIDIAG_ELIM.cpp
  #lcals/TRIDIAG_ELIM-Seq.cpp
  #lcals/TRIDIAG_ELIM-OMPTarget.cpp
  #polybench/POLYBENCH_2MM.cpp
  #polybench/POLYBENCH_2MM-Seq.cpp
  #polybench/POLYBENCH_2MM-OMPTarget.cpp
  #polybench/POLYBENCH_3MM.cpp
  #polybench/POLYBENCH_3MM-Seq.cpp
  #polybench/POLYBENCH_3MM-OMPTarget.cpp
  #polybench/POLYBENCH_ADI.cpp
  #polybench/POLYBENCH_ADI-Seq.cpp
  #polybench/POLYBENCH_ADI-OMPTarget.cpp
  #polybench/POLYBENCH_ATAX.cpp
  #polybench/POLYBENCH_ATAX-Seq.cpp
  #polybench/POLYBENCH_ATAX-OMPTarget.cpp
  #polybench/POLYBENCH_FDTD_2D.cpp
  #polybench/POLYBENCH_FDTD_2D-Seq.cpp
  #polybench/POLYBENCH_FDTD_2D-OMPTarget.cpp
  #polybench/POLYBENCH_FLOYD_WARSHALL.cpp
  #polybench/POLYBENCH_FLOYD_WARSHALL-Seq.cpp
  #polybench/POLYBENCH_FLOYD_WARSHALL-OMPTarget.cpp
  #polybench/POLYBENCH_GEMM.cpp
  #polybench/POLYBENCH_GEMM-Seq.cpp
  #polybench/POLYBENCH_GEMM-OMPTarget.cpp
  #polybench/POLYBENCH_GEMVER.cpp
  #polybench/POLYBENCH_GEMVER-Seq.cpp
  #polybench/POLYBENCH_GEMVER-OMPTarget.cpp
  #polybench/POLYBENCH_GESUMMV.cpp
  #polybench/POLYBENCH_GESUMMV-Seq.cpp
  #polybench/POLYBENCH_GESUMMV-OMPTarget.cpp
  #polybench/POLYBENCH_HEAT_3D.cpp
  #polybench/POLYBENCH_HEAT_3D-Seq.cpp
  #polybench/POLYBENCH_HEAT_3D-OMPTarget.cpp
  #polybench/POLYBENCH_JACOBI_1D.cpp
  #polybench/POLYBENCH_JACOBI_1D-Seq.cpp
  #polybench/POLYBENCH_JACOBI_1D-OMPTarget.cpp
  #polybench/POLYBENCH_JACOBI_2D.cpp
  #polybench/POLYBENCH_JACOBI_2D-Seq.cpp
  #polybench/POLYBENCH_JACOBI_2D-OMPTarget.cpp
  #polybench/POLYBENCH_MVT.cpp
  #polybench/POLYBENCH_MVT-Seq.cpp
  #polybench/POLYBENCH_MVT-OMPTarget.cpp
  #stream/ADD.cpp
  #stream/ADD-Seq.cpp
  #stream/ADD-OMPTarget.cpp
  #stream/COPY.cpp
  #stream/COPY-Seq.cpp
  #stream/COPY-OMPTarget.cpp
  #stream/DOT.cpp
  #stream/DOT-Seq.cpp
  #stream/DOT-OMPTarget.cpp
  #stream/MUL.cpp
  #stream/MUL-Seq.cpp
  #stream/MUL-OMPTarget.cpp
  #stream/TRIAD.cpp
  #stream/TRIAD-Seq.cpp
  #stream/TRIAD-OMPTarget.cpp
  common/DataUtils.cpp
  common/Executor.cpp
  common/KernelBase.cpp
  common/OutputUtils.cpp
  common/RAJAPerfSuite.cpp
  common/RPTypes.hpp
  common/RunParams.cpp
  #algorithm/SORT.cpp
  #algorithm/SORT-Seq.cpp
  #algorithm/SORTPAIRS.cpp
  #algorithm/SORTPAIRS-Seq.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_DEPENDS}
)

else()
  if(NOT INFRASTRUCTURE_ONLY)
blt_add_executable(
  NAME raja-perf.exe
  SOURCES RAJAPerfSuiteDriver.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_EXECUTABLE_DEPENDS}
  )
    endif()
endif()
