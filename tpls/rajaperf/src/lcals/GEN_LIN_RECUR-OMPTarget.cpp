//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

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

#define GEN_LIN_RECUR_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(b5, m_b5, m_N, did, hid); \
  allocAndInitOpenMPDeviceData(stb5, m_stb5, m_N, did, hid); \
  allocAndInitOpenMPDeviceData(sa, m_sa, m_N, did, hid); \
  allocAndInitOpenMPDeviceData(sb, m_sb, m_N, did, hid);

#define GEN_LIN_RECUR_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_b5, b5, m_N, hid, did); \
  deallocOpenMPDeviceData(b5, did); \
  deallocOpenMPDeviceData(stb5, did); \
  deallocOpenMPDeviceData(sa, did); \
  deallocOpenMPDeviceData(sb, did);


void GEN_LIN_RECUR::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  GEN_LIN_RECUR_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    GEN_LIN_RECUR_DATA_SETUP_OMP_TARGET

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(b5, stb5, sa, sb) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) 
      for (Index_type k = 0; k < N; ++k ) {
        GEN_LIN_RECUR_BODY1;
      }

      #pragma omp target is_device_ptr(b5, stb5, sa, sb) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = 1; i < N+1; ++i ) {
        GEN_LIN_RECUR_BODY2;
      }

    }
    stopTimer();

    GEN_LIN_RECUR_DATA_TEARDOWN_OMP_TARGET

  } else if ( vid == RAJA_OpenMPTarget ) {

    GEN_LIN_RECUR_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
        RAJA::RangeSegment(0, N), [=] (Index_type k) {
        GEN_LIN_RECUR_BODY1;
      });

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
        RAJA::RangeSegment(1, N+1), [=] (Index_type i) {
        GEN_LIN_RECUR_BODY2;
      });

    }
    stopTimer();

    GEN_LIN_RECUR_DATA_TEARDOWN_OMP_TARGET

  } else { 
     std::cout << "\n  GEN_LIN_RECUR : Unknown OMP Tagretvariant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
