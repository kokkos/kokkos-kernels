//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAG_ELIM.hpp"

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

#define TRIDIAG_ELIM_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(xout, m_xout, m_N, did, hid); \
  allocAndInitOpenMPDeviceData(xin, m_xin, m_N, did, hid); \
  allocAndInitOpenMPDeviceData(y, m_y, m_N, did, hid); \
  allocAndInitOpenMPDeviceData(z, m_z, m_N, did, hid);

#define TRIDIAG_ELIM_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_xout, xout, m_N, hid, did); \
  deallocOpenMPDeviceData(xout, did); \
  deallocOpenMPDeviceData(xin, did); \
  deallocOpenMPDeviceData(y, did); \
  deallocOpenMPDeviceData(z, did);


void TRIDIAG_ELIM::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = m_N;

  TRIDIAG_ELIM_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    TRIDIAG_ELIM_DATA_SETUP_OMP_TARGET

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(xout, xin, y, z) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) 
      for (Index_type i = ibegin; i < iend; ++i ) {
        TRIDIAG_ELIM_BODY;
      }

    }
    stopTimer();

    TRIDIAG_ELIM_DATA_TEARDOWN_OMP_TARGET

  } else if ( vid == RAJA_OpenMPTarget ) {

    TRIDIAG_ELIM_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        TRIDIAG_ELIM_BODY;
      });  

    }
    stopTimer();

    TRIDIAG_ELIM_DATA_TEARDOWN_OMP_TARGET

  } else { 
     std::cout << "\n  TRIDIAG_ELIM : Unknown OMP Tagretvariant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
