//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define NESTED_INIT_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(array, m_array, m_array_length, did, hid);

#define NESTED_INIT_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_array, array, m_array_length, hid, did); \
  deallocOpenMPDeviceData(array, did);


void NESTED_INIT::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    NESTED_INIT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(array) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(3) 
      for (Index_type k = 0; k < nk; ++k ) {
        for (Index_type j = 0; j < nj; ++j ) {
          for (Index_type i = 0; i < ni; ++i ) {
            NESTED_INIT_BODY;
          }
        }
      }  

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    NESTED_INIT_DATA_SETUP_OMP_TARGET;

    //using EXEC_POL = 
    //  RAJA::KernelPolicy<
    //    RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
    //                              RAJA::ArgList<2, 1, 0>, // k, j, i
    //      RAJA::statement::Lambda<0>
    //    >
    //  >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

     // RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
     //                                          RAJA::RangeSegment(0, nj),
     //                                          RAJA::RangeSegment(0, nk)),
     //      [=](Index_type i, Index_type j, Index_type k) {
     //      NESTED_INIT_BODY;
     // });

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_OMP_TARGET;

  } else { 
     std::cout << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
