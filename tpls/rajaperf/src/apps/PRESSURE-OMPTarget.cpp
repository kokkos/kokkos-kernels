//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define PRESSURE_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(compression, m_compression, iend, did, hid); \
  allocAndInitOpenMPDeviceData(bvc, m_bvc, iend, did, hid); \
  allocAndInitOpenMPDeviceData(p_new, m_p_new, iend, did, hid); \
  allocAndInitOpenMPDeviceData(e_old, m_e_old, iend, did, hid); \
  allocAndInitOpenMPDeviceData(vnewc, m_vnewc, iend, did, hid);

#define PRESSURE_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_p_new, p_new, iend, hid, did); \
  deallocOpenMPDeviceData(compression, did); \
  deallocOpenMPDeviceData(bvc, did); \
  deallocOpenMPDeviceData(p_new, did); \
  deallocOpenMPDeviceData(e_old, did); \
  deallocOpenMPDeviceData(vnewc, did);


void PRESSURE::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();
  
  PRESSURE_DATA_SETUP;
  
  if ( vid == Base_OpenMPTarget ) {

    PRESSURE_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(compression, bvc) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) 
      for (Index_type i = ibegin; i < iend; ++i ) {
        PRESSURE_BODY1;
      }

      #pragma omp target is_device_ptr(bvc, p_new, e_old, vnewc) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) 
      for (Index_type i = ibegin; i < iend; ++i ) {
        PRESSURE_BODY2;
      }

    }
    stopTimer();

    PRESSURE_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    PRESSURE_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          PRESSURE_BODY1;
        });

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          PRESSURE_BODY2;
        });

      }); // end sequential region (for single-source code)

    }
    stopTimer();

    PRESSURE_DATA_TEARDOWN_OMP_TARGET;

  } else {
    std::cout << "\n  PRESSURE : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
