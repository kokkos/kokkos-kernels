//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_DIFF.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define FIRST_DIFF_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(x, m_x, m_N, did, hid); \
  allocAndInitOpenMPDeviceData(y, m_y, m_N, did, hid);

#define FIRST_DIFF_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_x, x, iend, hid, did); \
  deallocOpenMPDeviceData(x, did); \
  deallocOpenMPDeviceData(y, did);


void FIRST_DIFF::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  FIRST_DIFF_DATA_SETUP;
 
  if ( vid == Base_OpenMPTarget ) {

    FIRST_DIFF_DATA_SETUP_OMP_TARGET;
                       
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(x, y) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) 
        
      for (Index_type i = ibegin; i < iend; ++i ) {
        FIRST_DIFF_BODY;
      }

    }
    stopTimer();

    FIRST_DIFF_DATA_TEARDOWN_OMP_TARGET;
                       
  } else if ( vid == RAJA_OpenMPTarget ) {

    FIRST_DIFF_DATA_SETUP_OMP_TARGET;
                       
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        FIRST_DIFF_BODY;
      });

    }
    stopTimer();

    FIRST_DIFF_DATA_TEARDOWN_OMP_TARGET;
                       
  } else {                          
     std::cout << "\n  FIRST_DIFF : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
