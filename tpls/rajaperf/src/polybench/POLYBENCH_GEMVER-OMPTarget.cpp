//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMVER.hpp"

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

#define POLYBENCH_GEMVER_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(A, m_A, m_n * m_n, did, hid); \
  allocAndInitOpenMPDeviceData(u1, m_u1, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(v1, m_v1, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(u2, m_u2, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(v2, m_v2, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(w, m_w, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(x, m_x, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(y, m_y, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(z, m_z, m_n, did, hid); 

#define POLYBENCH_GEMVER_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_w, w, m_n, hid, did); \
  deallocOpenMPDeviceData(A, did); \
  deallocOpenMPDeviceData(u1, did); \
  deallocOpenMPDeviceData(v1, did); \
  deallocOpenMPDeviceData(u2, did); \
  deallocOpenMPDeviceData(v2, did); \
  deallocOpenMPDeviceData(w, did); \
  deallocOpenMPDeviceData(x, did); \
  deallocOpenMPDeviceData(y, did); \
  deallocOpenMPDeviceData(z, did); 

  

void POLYBENCH_GEMVER::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_GEMVER_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    POLYBENCH_GEMVER_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(A,u1,v1,u2,v2) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      for (Index_type i = 0; i < n; i++) {
        for(Index_type j = 0; j < n; j++) {
          POLYBENCH_GEMVER_BODY1;
        }
      }

      #pragma omp target is_device_ptr(A,x,y) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = 0; i < n; i++) { 
        POLYBENCH_GEMVER_BODY2;
        for (Index_type j = 0; j < n; j++) {
          POLYBENCH_GEMVER_BODY3;
        }
        POLYBENCH_GEMVER_BODY4;
      }

      #pragma omp target is_device_ptr(x,z) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) 
      for (Index_type i = 0; i < n; i++) {
        POLYBENCH_GEMVER_BODY5;
      }

      #pragma omp target is_device_ptr(A,w,x) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = 0; i < n; i++) {
        POLYBENCH_GEMVER_BODY6;
        for (Index_type j = 0; j < n; j++) {
          POLYBENCH_GEMVER_BODY7;
        }
        POLYBENCH_GEMVER_BODY8;
      }

    } // end run_reps
    stopTimer(); 

    POLYBENCH_GEMVER_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    POLYBENCH_GEMVER_DATA_SETUP_OMP_TARGET;

    POLYBENCH_GEMVER_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>,
          RAJA::statement::Lambda<0>
        >
      >;

    using EXEC_POL2 =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<threads_per_team>,
          RAJA::statement::Lambda<0, RAJA::Params<0>>,
          RAJA::statement::For<1, RAJA::seq_exec,
            RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
          >,
          RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
        >
      >;

    using EXEC_POL4 =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<threads_per_team>,
          RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
          RAJA::statement::For<1, RAJA::seq_exec,
            RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
          >,
          RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
        >
      >;
  
    using EXEC_POL3 = RAJA::omp_target_parallel_for_exec<threads_per_team>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL1>( RAJA::make_tuple(RAJA::RangeSegment{0, n},
                                                RAJA::RangeSegment{0, n}),
        [=] (Index_type i, Index_type j) {
          POLYBENCH_GEMVER_BODY1_RAJA;
        }
      );

      RAJA::kernel_param<EXEC_POL2>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        RAJA::tuple<Real_type>{0.0},

        [=] (Real_type &dot) {
          POLYBENCH_GEMVER_BODY2_RAJA;
        },
        [=] (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_GEMVER_BODY3_RAJA;
        },
        [=] (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY4_RAJA;
        }
      );

      RAJA::forall<EXEC_POL3> (RAJA::RangeSegment{0, n},
        [=] (Index_type i) {
          POLYBENCH_GEMVER_BODY5_RAJA;
        }
      );

      RAJA::kernel_param<EXEC_POL4>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        RAJA::tuple<Real_type>{0.0},

        [=] (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY6_RAJA;
        },
        [=] (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_GEMVER_BODY7_RAJA;
        },
        [=] (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY8_RAJA;
        }
      );

    }
    stopTimer();

    POLYBENCH_GEMVER_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  POLYBENCH_GEMVER : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

