//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ATOMIC_PI.hpp"

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

#define ATOMIC_PI_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(pi, m_pi, 1, did, hid);

#define ATOMIC_PI_DATA_TEARDOWN_OMP_TARGET \
  deallocOpenMPDeviceData(pi, did);


void ATOMIC_PI::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ATOMIC_PI_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    ATOMIC_PI_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initOpenMPDeviceData(pi, &m_pi_init, 1, did, hid);
      
      #pragma omp target is_device_ptr(pi) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        double x = (double(i) + 0.5) * dx;
        #pragma omp atomic
        *pi += dx / (1.0 + x * x);
      }

      getOpenMPDeviceData(m_pi, pi, 1, hid, did);
      *m_pi *= 4.0;

    }
    stopTimer();

    ATOMIC_PI_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    ATOMIC_PI_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initOpenMPDeviceData(pi, &m_pi_init, 1, did, hid);

      //RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
      //  RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
      //    double x = (double(i) + 0.5) * dx;
      //    RAJA::atomicAdd<RAJA::omp_atomic>(pi, dx / (1.0 + x * x));
      //});

      getOpenMPDeviceData(m_pi, pi, 1, hid, did); 
      *m_pi *= 4.0;

    }
    stopTimer();

    ATOMIC_PI_DATA_TEARDOWN_OMP_TARGET;
  
  } else {
     std::cout << "\n  ATOMIC_PI : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
