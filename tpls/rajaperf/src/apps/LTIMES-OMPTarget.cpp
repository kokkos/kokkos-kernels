//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define LTIMES_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(phidat, m_phidat, m_philen, did, hid); \
  allocAndInitOpenMPDeviceData(elldat, m_elldat, m_elllen, did, hid); \
  allocAndInitOpenMPDeviceData(psidat, m_psidat, m_psilen, did, hid);

#define LTIMES_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_phidat, phidat, m_philen, hid, did); \
  deallocOpenMPDeviceData(phidat, did); \
  deallocOpenMPDeviceData(elldat, did); \
  deallocOpenMPDeviceData(psidat, did);


void LTIMES::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  LTIMES_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    LTIMES_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(phidat, elldat, psidat) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(3)
      for (Index_type z = 0; z < num_z; ++z ) {
        for (Index_type g = 0; g < num_g; ++g ) {
          for (Index_type m = 0; m < num_m; ++m ) {
            for (Index_type d = 0; d < num_d; ++d ) {
              LTIMES_BODY;
            }
          }
        }
      }

    }
    stopTimer();

    LTIMES_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    LTIMES_DATA_SETUP_OMP_TARGET;

    LTIMES_VIEWS_RANGES_RAJA;

    using EXEC_POL = 
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<1, 2, 3>, // z, g, m
          RAJA::statement::For<0, RAJA::seq_exec,         // d
            RAJA::statement::Lambda<0>
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(IDRange(0, num_d),
                                               IZRange(0, num_z),
                                               IGRange(0, num_g),
                                               IMRange(0, num_m)),
        [=] (ID d, IZ z, IG g, IM m) {
        LTIMES_BODY_RAJA;
      }); 

    }
    stopTimer();

    LTIMES_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n LTIMES : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
