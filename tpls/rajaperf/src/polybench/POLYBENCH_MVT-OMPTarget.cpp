//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

#include "POLYBENCH_MVT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define POLYBENCH_MVT_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(x1, m_x1, N, did, hid); \
  allocAndInitOpenMPDeviceData(x2, m_x2, N, did, hid); \
  allocAndInitOpenMPDeviceData(y1, m_y1, N, did, hid); \
  allocAndInitOpenMPDeviceData(y2, m_y2, N, did, hid); \
  allocAndInitOpenMPDeviceData(A, m_A, N * N, did, hid);


#define POLYBENCH_MVT_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_x1, x1, N, hid, did); \
  getOpenMPDeviceData(m_x2, x2, N, hid, did); \
  deallocOpenMPDeviceData(x1, did); \
  deallocOpenMPDeviceData(x2, did); \
  deallocOpenMPDeviceData(y1, did); \
  deallocOpenMPDeviceData(y2, did); \
  deallocOpenMPDeviceData(A, did);


void POLYBENCH_MVT::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_MVT_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    POLYBENCH_MVT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(x1,A,y1) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = 0; i < N; ++i ) {
        POLYBENCH_MVT_BODY1;
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_MVT_BODY2;
        }
        POLYBENCH_MVT_BODY3;
      }

      #pragma omp target is_device_ptr(x2,A,y2) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = 0; i < N; ++i ) {
        POLYBENCH_MVT_BODY4;
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_MVT_BODY5;
        }
        POLYBENCH_MVT_BODY6;
      }

    }
    stopTimer();

    POLYBENCH_MVT_TEARDOWN_OMP_TARGET;

  } else if (vid == RAJA_OpenMPTarget) {

    POLYBENCH_MVT_DATA_SETUP_OMP_TARGET;

    POLYBENCH_MVT_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<threads_per_team>,
          RAJA::statement::Lambda<0, RAJA::Params<0>>,
          RAJA::statement::For<1, RAJA::seq_exec,
            RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
          >,
          RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, N},
                           RAJA::RangeSegment{0, N}),
          RAJA::tuple<Real_type>{0.0},

          [=] (Real_type &dot) {
            POLYBENCH_MVT_BODY1_RAJA;
          },
          [=] (Index_type i, Index_type j, Real_type &dot) {
            POLYBENCH_MVT_BODY2_RAJA;
          },
          [=] (Index_type i, Real_type &dot) {
            POLYBENCH_MVT_BODY3_RAJA;
          }

        );

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, N},
                           RAJA::RangeSegment{0, N}),
          RAJA::tuple<Real_type>{0.0},

          [=] (Real_type &dot) {
            POLYBENCH_MVT_BODY4_RAJA;
          },
          [=] (Index_type i, Index_type j, Real_type &dot) {
            POLYBENCH_MVT_BODY5_RAJA;
          },
          [=] (Index_type i, Real_type &dot) {
            POLYBENCH_MVT_BODY6_RAJA;
          }

        );

      }); // end sequential region (for single-source code)

    }
    stopTimer();

    POLYBENCH_MVT_TEARDOWN_OMP_TARGET;

  } else {
      std::cout << "\n  POLYBENCH_MVT : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
  
