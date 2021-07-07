//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(pin, m_pin, m_N * m_N, did, hid); \
  allocAndInitOpenMPDeviceData(pout, m_pout, m_N * m_N, did, hid);


#define POLYBENCH_FLOYD_WARSHALL_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_pout, pout, m_N * m_N, hid, did); \
  deallocOpenMPDeviceData(pin, did); \
  deallocOpenMPDeviceData(pout, did);


void POLYBENCH_FLOYD_WARSHALL::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_FLOYD_WARSHALL_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type k = 0; k < N; ++k) {

        #pragma omp target is_device_ptr(pout,pin) device( did )
        #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
        for (Index_type i = 0; i < N; ++i) {
          for (Index_type j = 0; j < N; ++j) {
            POLYBENCH_FLOYD_WARSHALL_BODY;
          }
        }

      }

    }
    stopTimer();

    POLYBENCH_FLOYD_WARSHALL_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_OMP_TARGET;

    POLYBENCH_FLOYD_WARSHALL_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                    RAJA::ArgList<1, 2>,
            RAJA::statement::Lambda<0>
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                               RAJA::RangeSegment{0, N},
                                               RAJA::RangeSegment{0, N}),
        [=] (Index_type k, Index_type i, Index_type j) {
          POLYBENCH_FLOYD_WARSHALL_BODY_RAJA;
        }
      );

    }
    stopTimer();

    POLYBENCH_FLOYD_WARSHALL_TEARDOWN_OMP_TARGET;

  } else {
      std::cout << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
  
