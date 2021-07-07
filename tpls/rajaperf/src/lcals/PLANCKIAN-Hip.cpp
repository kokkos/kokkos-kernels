//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PLANCKIAN.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>
#include <cmath>

namespace rajaperf
{
namespace lcals
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define PLANCKIAN_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, iend); \
  allocAndInitHipDeviceData(y, m_y, iend); \
  allocAndInitHipDeviceData(u, m_u, iend); \
  allocAndInitHipDeviceData(v, m_v, iend); \
  allocAndInitHipDeviceData(w, m_w, iend);

#define PLANCKIAN_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_w, w, iend); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(y); \
  deallocHipDeviceData(u); \
  deallocHipDeviceData(v); \
  deallocHipDeviceData(w);

__global__ void planckian(Real_ptr x, Real_ptr y,
                          Real_ptr u, Real_ptr v, Real_ptr w,
                          Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     PLANCKIAN_BODY;
   }
}


void PLANCKIAN::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  PLANCKIAN_DATA_SETUP;

  if ( vid == Base_HIP ) {

    PLANCKIAN_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       hipLaunchKernelGGL((planckian), dim3(grid_size), dim3(block_size), 0, 0,  x, y,
                                             u, v, w,
                                             iend );

    }
    stopTimer();

    PLANCKIAN_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    PLANCKIAN_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         PLANCKIAN_BODY;
       });

    }
    stopTimer();

    PLANCKIAN_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  PLANCKIAN : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
