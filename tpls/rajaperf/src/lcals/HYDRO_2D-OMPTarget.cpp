//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

#define HYDRO_2D_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(zadat, m_za, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(zbdat, m_zb, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(zmdat, m_zm, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(zpdat, m_zp, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(zqdat, m_zq, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(zrdat, m_zr, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(zudat, m_zu, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(zvdat, m_zv, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(zzdat, m_zz, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(zroutdat, m_zrout, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(zzoutdat, m_zzout, m_array_length, did, hid);

#define HYDRO_2D_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_zrout, zroutdat, m_array_length, hid, did); \
  getOpenMPDeviceData(m_zzout, zzoutdat, m_array_length, hid, did); \
  deallocOpenMPDeviceData(zadat, did); \
  deallocOpenMPDeviceData(zbdat, did); \
  deallocOpenMPDeviceData(zmdat, did); \
  deallocOpenMPDeviceData(zpdat, did); \
  deallocOpenMPDeviceData(zqdat, did); \
  deallocOpenMPDeviceData(zrdat, did); \
  deallocOpenMPDeviceData(zudat, did); \
  deallocOpenMPDeviceData(zvdat, did); \
  deallocOpenMPDeviceData(zzdat, did); \
  deallocOpenMPDeviceData(zroutdat, did); \
  deallocOpenMPDeviceData(zzoutdat, did);



void HYDRO_2D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  HYDRO_2D_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    HYDRO_2D_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(zadat, zbdat, zpdat, \
                                       zqdat, zrdat, zmdat) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(2) 
      for (Index_type k = kbeg; k < kend; ++k ) {
        for (Index_type j = jbeg; j < jend; ++j ) {
          HYDRO_2D_BODY1;
        }
      }

      #pragma omp target is_device_ptr(zudat, zvdat, zadat, \
                                       zbdat, zzdat, zrdat) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(2) 
      for (Index_type k = kbeg; k < kend; ++k ) {
        for (Index_type j = jbeg; j < jend; ++j ) {
          HYDRO_2D_BODY2;
        }
      }

      #pragma omp target is_device_ptr(zroutdat, zzoutdat, \
                                       zrdat, zudat, zzdat, zvdat) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(2) 
      for (Index_type k = kbeg; k < kend; ++k ) {
        for (Index_type j = jbeg; j < jend; ++j ) {
          HYDRO_2D_BODY3;
        }
      }

    }
    stopTimer();

    HYDRO_2D_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    HYDRO_2D_DATA_SETUP_OMP_TARGET;

    HYDRO_2D_VIEWS_RAJA;

    using EXECPOL =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>,
          RAJA::statement::Lambda<0>
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY1_RAJA;
      });

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY2_RAJA;
      });

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY3_RAJA;
      });

    }
    stopTimer();

    HYDRO_2D_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  HYDRO_2D : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
