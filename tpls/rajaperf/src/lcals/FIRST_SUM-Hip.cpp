//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_SUM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define FIRST_SUM_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, m_N); \
  allocAndInitHipDeviceData(y, m_y, m_N);

#define FIRST_SUM_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_x, x, m_N); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(y);

__global__ void first_sum(Real_ptr x, Real_ptr y,
                          Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i > 0 && i < iend) {
     FIRST_SUM_BODY;
   }
}


void FIRST_SUM::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getRunSize();

  FIRST_SUM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    FIRST_SUM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       hipLaunchKernelGGL(first_sum,grid_size, block_size, 0, 0, x, y,
                                              iend );

    }
    stopTimer();

    FIRST_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    FIRST_SUM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         FIRST_SUM_BODY;
       });

    }
    stopTimer();

    FIRST_SUM_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  FIRST_SUM : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
