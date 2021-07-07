//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf 
{
namespace apps
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define FIR_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr coeff; \
\
  allocAndInitOpenMPDeviceData(in, m_in, getRunSize(), did, hid); \
  allocAndInitOpenMPDeviceData(out, m_out, getRunSize(), did, hid); \
  Real_ptr tcoeff = &coeff_array[0]; \
  allocAndInitOpenMPDeviceData(coeff, tcoeff, FIR_COEFFLEN, did, hid);


#define FIR_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_out, out, getRunSize(), hid, did); \
  deallocOpenMPDeviceData(in, did); \
  deallocOpenMPDeviceData(out, did); \
  deallocOpenMPDeviceData(coeff, did);


void FIR::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize() - m_coefflen;

  FIR_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    FIR_COEFF;

    FIR_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(in, out, coeff) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
         FIR_BODY;
      }

    }
    stopTimer();

    FIR_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    FIR_COEFF;

    FIR_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        FIR_BODY;
      });

    }
    stopTimer();

    FIR_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  FIR : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
