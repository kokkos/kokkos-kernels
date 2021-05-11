//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

#include "POLYBENCH_JACOBI_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_JACOBI_2D_DATA_SETUP_CUDA \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(A, m_Ainit, m_N*m_N, did, hid); \
  allocAndInitOpenMPDeviceData(B, m_Binit, m_N*m_N, did, hid);


#define POLYBENCH_JACOBI_2D_TEARDOWN_CUDA \
  getOpenMPDeviceData(m_A, A, m_N*m_N, hid, did); \
  getOpenMPDeviceData(m_B, B, m_N*m_N, hid, did); \
  deallocOpenMPDeviceData(A, did); \
  deallocOpenMPDeviceData(B, did);


void POLYBENCH_JACOBI_2D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    POLYBENCH_JACOBI_2D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        #pragma omp target is_device_ptr(A,B) device( did )
        #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
        for (Index_type i = 1; i < N-1; ++i ) {
          for (Index_type j = 1; j < N-1; ++j ) {
            POLYBENCH_JACOBI_2D_BODY1;
          }
        }
    
        #pragma omp target is_device_ptr(A,B) device( did )
        #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
        for (Index_type i = 1; i < N-1; ++i ) {
          for (Index_type j = 1; j < N-1; ++j ) {
            POLYBENCH_JACOBI_2D_BODY2;
          }
        }  

      }

    }
    stopTimer();

    POLYBENCH_JACOBI_2D_TEARDOWN_CUDA;

  } else if (vid == RAJA_OpenMPTarget) {

    POLYBENCH_JACOBI_2D_DATA_SETUP_CUDA;

    POLYBENCH_JACOBI_2D_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>, 
          RAJA::statement::Lambda<0>
        >,
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>, 
          RAJA::statement::Lambda<1>
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                 RAJA::RangeSegment{1, N-1}),
          [=] (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY1_RAJA;
          },
          [=] (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY2_RAJA;
          }
        );

      }

    }
    stopTimer();

    POLYBENCH_JACOBI_2D_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_JACOBI_2D : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
  
