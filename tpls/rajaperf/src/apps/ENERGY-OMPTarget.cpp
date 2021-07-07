//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ENERGY.hpp"

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

#define ENERGY_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(e_new, m_e_new, iend, did, hid); \
  allocAndInitOpenMPDeviceData(e_old, m_e_old, iend, did, hid); \
  allocAndInitOpenMPDeviceData(delvc, m_delvc, iend, did, hid); \
  allocAndInitOpenMPDeviceData(p_new, m_p_new, iend, did, hid); \
  allocAndInitOpenMPDeviceData(p_old, m_p_old, iend, did, hid); \
  allocAndInitOpenMPDeviceData(q_new, m_q_new, iend, did, hid); \
  allocAndInitOpenMPDeviceData(q_old, m_q_old, iend, did, hid); \
  allocAndInitOpenMPDeviceData(work, m_work, iend, did, hid); \
  allocAndInitOpenMPDeviceData(compHalfStep, m_compHalfStep, iend, did, hid); \
  allocAndInitOpenMPDeviceData(pHalfStep, m_pHalfStep, iend, did, hid); \
  allocAndInitOpenMPDeviceData(bvc, m_bvc, iend, did, hid); \
  allocAndInitOpenMPDeviceData(pbvc, m_pbvc, iend, did, hid); \
  allocAndInitOpenMPDeviceData(ql_old, m_ql_old, iend, did, hid); \
  allocAndInitOpenMPDeviceData(qq_old, m_qq_old, iend, did, hid); \
  allocAndInitOpenMPDeviceData(vnewc, m_vnewc, iend, did, hid);

#define ENERGY_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_e_new, e_new, iend, hid, did); \
  getOpenMPDeviceData(m_q_new, q_new, iend, hid, did); \
  deallocOpenMPDeviceData(e_new, did); \
  deallocOpenMPDeviceData(e_old, did); \
  deallocOpenMPDeviceData(delvc, did); \
  deallocOpenMPDeviceData(p_new, did); \
  deallocOpenMPDeviceData(p_old, did); \
  deallocOpenMPDeviceData(q_new, did); \
  deallocOpenMPDeviceData(q_old, did); \
  deallocOpenMPDeviceData(work, did); \
  deallocOpenMPDeviceData(compHalfStep, did); \
  deallocOpenMPDeviceData(pHalfStep, did); \
  deallocOpenMPDeviceData(bvc, did); \
  deallocOpenMPDeviceData(pbvc, did); \
  deallocOpenMPDeviceData(ql_old, did); \
  deallocOpenMPDeviceData(qq_old, did); \
  deallocOpenMPDeviceData(vnewc, did);

void ENERGY::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ENERGY_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    ENERGY_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(e_new, e_old, delvc, \
                                       p_old, q_old, work) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ENERGY_BODY1;
      }

      #pragma omp target is_device_ptr(delvc, q_new, compHalfStep, \
                                       pHalfStep, e_new, bvc, pbvc, \
                                       ql_old, qq_old) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ENERGY_BODY2;
      }

      #pragma omp target is_device_ptr(e_new, delvc, p_old, \
                                       q_old, pHalfStep, q_new) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ENERGY_BODY3;
      }

      #pragma omp target is_device_ptr(e_new, work) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ENERGY_BODY4;
      }

      #pragma omp target is_device_ptr(delvc, pbvc, e_new, vnewc, \
                                       bvc, p_new, ql_old, qq_old, \
                                       p_old, q_old, pHalfStep, q_new) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ENERGY_BODY5;
      }

      #pragma omp target is_device_ptr(delvc, pbvc, e_new, vnewc, \
                                       bvc, p_new, q_new, ql_old, qq_old) \
                                       device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ENERGY_BODY6;
      }
        
    }
    stopTimer();

    ENERGY_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    ENERGY_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ENERGY_BODY1;
        });

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ENERGY_BODY2;
        });

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ENERGY_BODY3;
        });

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ENERGY_BODY4;
        });
  
        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ENERGY_BODY5;
        });

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ENERGY_BODY6;
        });

      }); // end sequential region (for single-source code)

    }
    stopTimer();

    ENERGY_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  ENERGY : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
