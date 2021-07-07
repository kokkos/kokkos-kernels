//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "IF_QUAD.hpp"

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

#define IF_QUAD_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(a, m_a, iend, did, hid); \
  allocAndInitOpenMPDeviceData(b, m_b, iend, did, hid); \
  allocAndInitOpenMPDeviceData(c, m_c, iend, did, hid); \
  allocAndInitOpenMPDeviceData(x1, m_x1, iend, did, hid); \
  allocAndInitOpenMPDeviceData(x2, m_x2, iend, did, hid);

#define IF_QUAD_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_x1, x1, iend, hid, did); \
  getOpenMPDeviceData(m_x2, x2, iend, hid, did); \
  deallocOpenMPDeviceData(a, did); \
  deallocOpenMPDeviceData(b, did); \
  deallocOpenMPDeviceData(c, did); \
  deallocOpenMPDeviceData(x1, did); \
  deallocOpenMPDeviceData(x2, did);

void IF_QUAD::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  IF_QUAD_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    IF_QUAD_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(a, b, c, x1, x2) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) 
      for (Index_type i = ibegin; i < iend; ++i ) {
        IF_QUAD_BODY;
      }

    }
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    IF_QUAD_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      //RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
      //    RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
      //  IF_QUAD_BODY;
      //});

    }
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  IF_QUAD : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
