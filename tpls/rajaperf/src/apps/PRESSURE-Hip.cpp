//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define PRESSURE_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(compression, m_compression, iend); \
  allocAndInitHipDeviceData(bvc, m_bvc, iend); \
  allocAndInitHipDeviceData(p_new, m_p_new, iend); \
  allocAndInitHipDeviceData(e_old, m_e_old, iend); \
  allocAndInitHipDeviceData(vnewc, m_vnewc, iend);

#define PRESSURE_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_p_new, p_new, iend); \
  deallocHipDeviceData(compression); \
  deallocHipDeviceData(bvc); \
  deallocHipDeviceData(p_new); \
  deallocHipDeviceData(e_old); \
  deallocHipDeviceData(vnewc);

__global__ void pressurecalc1(Real_ptr bvc, Real_ptr compression,
                              const Real_type cls,
                              Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     PRESSURE_BODY1;
   }
}

__global__ void pressurecalc2(Real_ptr p_new, Real_ptr bvc, Real_ptr e_old,
                              Real_ptr vnewc,
                              const Real_type p_cut, const Real_type eosvmax,
                              const Real_type pmin,
                              Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     PRESSURE_BODY2;
   }
}


void PRESSURE::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  PRESSURE_DATA_SETUP;

  if ( vid == Base_HIP ) {

    PRESSURE_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

       hipLaunchKernelGGL((pressurecalc1), dim3(grid_size), dim3(block_size), 0, 0,  bvc, compression,
                                                 cls,
                                                 iend );

       hipLaunchKernelGGL((pressurecalc2), dim3(grid_size), dim3(block_size), 0, 0,  p_new, bvc, e_old,
                                                 vnewc,
                                                 p_cut, eosvmax, pmin,
                                                 iend );

    }
    stopTimer();

    PRESSURE_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    PRESSURE_DATA_SETUP_HIP;

    const bool async = true;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RAJA::forall< RAJA::hip_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          PRESSURE_BODY1;
        });
        RAJA::forall< RAJA::hip_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          PRESSURE_BODY2;
        });

      });  // end sequential region (for single-source code)

    }
    stopTimer();

    PRESSURE_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  PRESSURE : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
