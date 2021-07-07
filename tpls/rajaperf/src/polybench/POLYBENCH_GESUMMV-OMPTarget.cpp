//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 

#include "POLYBENCH_GESUMMV.hpp"

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

#define POLYBENCH_GESUMMV_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(x, m_x, N, did, hid); \
  allocAndInitOpenMPDeviceData(y, m_y, N, did, hid); \
  allocAndInitOpenMPDeviceData(A, m_A, N*N, did, hid); \
  allocAndInitOpenMPDeviceData(B, m_B, N*N, did, hid);


#define POLYBENCH_GESUMMV_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_y, y, N, hid, did); \
  deallocOpenMPDeviceData(x, did); \
  deallocOpenMPDeviceData(y, did); \
  deallocOpenMPDeviceData(A, did); \
  deallocOpenMPDeviceData(B, did);


void POLYBENCH_GESUMMV::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_GESUMMV_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    POLYBENCH_GESUMMV_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(x, y, A, B) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = 0; i < N; ++i ) {
        POLYBENCH_GESUMMV_BODY1;
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_GESUMMV_BODY2;
        }
        POLYBENCH_GESUMMV_BODY3;
      }

    }
    stopTimer();

    POLYBENCH_GESUMMV_TEARDOWN_OMP_TARGET;

  } else if (vid == RAJA_OpenMPTarget) {

    POLYBENCH_GESUMMV_DATA_SETUP_OMP_TARGET;

    POLYBENCH_GESUMMV_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<threads_per_team>,
          RAJA::statement::Lambda<0, RAJA::Params<0,1>>,
          RAJA::statement::For<1, RAJA::seq_exec,
            RAJA::statement::Lambda<1,  RAJA::Segs<0,1>, RAJA::Params<0,1>>
          >,
          RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0,1>>
        >
      >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple( RAJA::RangeSegment{0, N},
                            RAJA::RangeSegment{0, N} ),
          RAJA::make_tuple(static_cast<Real_type>(0.0),
                           static_cast<Real_type>(0.0)),

          [=] (Real_type& tmpdot,
               Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY1_RAJA;
          },
          [=] (Index_type i, Index_type j, Real_type& tmpdot,
                                           Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY2_RAJA;
          },
          [=] (Index_type i, Real_type& tmpdot,
                             Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY3_RAJA;
          }
        );

      }
      stopTimer();

    POLYBENCH_GESUMMV_TEARDOWN_OMP_TARGET;

  } else {
      std::cout << "\n  POLYBENCH_GESUMMV : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
  
