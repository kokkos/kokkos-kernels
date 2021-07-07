//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INT_PREDICT.hpp"

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

#define INT_PREDICT_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(px, m_px, m_array_length, did, hid);

#define INT_PREDICT_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_px, px, m_array_length, hid, did); \
  deallocOpenMPDeviceData(px, did);


void INT_PREDICT::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  INT_PREDICT_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    INT_PREDICT_DATA_SETUP_OMP_TARGET;
                              
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(px) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) 
      for (Index_type i = ibegin; i < iend; ++i ) {
        INT_PREDICT_BODY;
      }

    }
    stopTimer();

    INT_PREDICT_DATA_TEARDOWN_OMP_TARGET;
                              
  } else if ( vid == RAJA_OpenMPTarget ) {

    INT_PREDICT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) { 

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        INT_PREDICT_BODY;
      });

    }
    stopTimer();

    INT_PREDICT_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  INT_PREDICT : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
