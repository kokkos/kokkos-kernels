//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 

#include "POLYBENCH_GEMM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_GEMM_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(A, m_A, ni*nk, did, hid); \
  allocAndInitOpenMPDeviceData(B, m_B, nk*nj, did, hid); \
  allocAndInitOpenMPDeviceData(C, m_C, ni*nj, did, hid);


#define POLYBENCH_GEMM_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_C, C, ni*nj, hid, did); \
  deallocOpenMPDeviceData(A, did); \
  deallocOpenMPDeviceData(B, did); \
  deallocOpenMPDeviceData(C, did);


void POLYBENCH_GEMM::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_GEMM_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    POLYBENCH_GEMM_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(A,B,C) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      for (Index_type i = 0; i < ni; ++i ) {
        for (Index_type j = 0; j < nj; ++j ) {
          POLYBENCH_GEMM_BODY1;
          POLYBENCH_GEMM_BODY2;
          for (Index_type k = 0; k < nk; ++k ) {
             POLYBENCH_GEMM_BODY3;
          }
          POLYBENCH_GEMM_BODY4;
        }
      }

    }
    stopTimer();

    POLYBENCH_GEMM_TEARDOWN_OMP_TARGET;

  } else if (vid == RAJA_OpenMPTarget) {

    POLYBENCH_GEMM_DATA_SETUP_OMP_TARGET;

    POLYBENCH_GEMM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>,
          RAJA::statement::Lambda<0, RAJA::Params<0>>,
          RAJA::statement::Lambda<1, RAJA::Segs<0,1>>,
          RAJA::statement::For<2, RAJA::seq_exec,
            RAJA::statement::Lambda<2, RAJA::Segs<0,1,2>, RAJA::Params<0>>
          >,
          RAJA::statement::Lambda<3, RAJA::Segs<0,1>, RAJA::Params<0>>
        >
      >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(

          RAJA::make_tuple( RAJA::RangeSegment{0, ni},
                            RAJA::RangeSegment{0, nj},
                            RAJA::RangeSegment{0, nk} ),
          RAJA::tuple<Real_type>{0.0},  // variable for dot

          [=] (Real_type& dot) {
            POLYBENCH_GEMM_BODY1_RAJA;
          },
          [=] (Index_type i, Index_type j) {
            POLYBENCH_GEMM_BODY2_RAJA;
          },
          [=] (Index_type i, Index_type j, Index_type k, 
               Real_type& dot) {
            POLYBENCH_GEMM_BODY3_RAJA;
          },
          [=] (Index_type i, Index_type j, 
               Real_type& dot) {
            POLYBENCH_GEMM_BODY4_RAJA;
          }
        );

      }
      stopTimer();

    POLYBENCH_GEMM_TEARDOWN_OMP_TARGET;

  } else {
      std::cout << "\n  POLYBENCH_GEMM : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
  
