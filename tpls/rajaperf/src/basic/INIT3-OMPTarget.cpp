//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT3.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define INIT3_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(out1, m_out1, iend, did, hid); \
  allocAndInitOpenMPDeviceData(out2, m_out2, iend, did, hid); \
  allocAndInitOpenMPDeviceData(out3, m_out3, iend, did, hid); \
  allocAndInitOpenMPDeviceData(in1, m_in1, iend, did, hid); \
  allocAndInitOpenMPDeviceData(in2, m_in2, iend, did, hid);

#define INIT3_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_out1, out1, iend, hid, did); \
  getOpenMPDeviceData(m_out2, out2, iend, hid, did); \
  getOpenMPDeviceData(m_out3, out3, iend, hid, did); \
  deallocOpenMPDeviceData(out1, did); \
  deallocOpenMPDeviceData(out2, did); \
  deallocOpenMPDeviceData(out3, did); \
  deallocOpenMPDeviceData(in1, did); \
  deallocOpenMPDeviceData(in2, did);


void INIT3::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  INIT3_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    INIT3_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(out1, out2, out3, in1, in2) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        INIT3_BODY;
      }

    }
    stopTimer();

    INIT3_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    INIT3_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      //RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
      //  RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
      //  INIT3_BODY;
      //});

    }
    stopTimer();

    INIT3_DATA_TEARDOWN_OMP_TARGET;
  
  } else {
     std::cout << "\n  INIT3 : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
