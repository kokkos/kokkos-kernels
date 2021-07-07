//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ADD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define ADD_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(a, m_a, iend); \
  allocAndInitHipDeviceData(b, m_b, iend); \
  allocAndInitHipDeviceData(c, m_c, iend);

#define ADD_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_c, c, iend); \
  deallocHipDeviceData(a); \
  deallocHipDeviceData(b); \
  deallocHipDeviceData(c);

__global__ void add(Real_ptr c, Real_ptr a, Real_ptr b,
                     Index_type iend)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < iend) {
    ADD_BODY;
  }
}


void ADD::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ADD_DATA_SETUP;

  if ( vid == Base_HIP ) {

    ADD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((add), dim3(grid_size), dim3(block_size), 0, 0,  c, a, b,
                                      iend );

    }
    stopTimer();

    ADD_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    ADD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto add_lambda = [=] __device__ (Index_type i) {
        ADD_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL(lambda_hip_forall<decltype(add_lambda)>,
        grid_size, block_size, 0, 0, ibegin, iend, add_lambda);

    }
    stopTimer();

    ADD_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    ADD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        ADD_BODY;
      });

    }
    stopTimer();

    ADD_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  ADD : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

