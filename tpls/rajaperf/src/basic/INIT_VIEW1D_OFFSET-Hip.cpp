//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D_OFFSET.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define INIT_VIEW1D_OFFSET_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(a, m_a, getRunSize());

#define INIT_VIEW1D_OFFSET_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_a, a, getRunSize()); \
  deallocHipDeviceData(a);

__global__ void initview1d_offset(Real_ptr a,
                                  Real_type v,
                                  const Index_type ibegin,
                                  const Index_type iend)
{
  Index_type i = ibegin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i < iend) {
    INIT_VIEW1D_OFFSET_BODY;
  }
}


void INIT_VIEW1D_OFFSET::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getRunSize()+1;

  INIT_VIEW1D_OFFSET_DATA_SETUP;

  if ( vid == Base_HIP ) {

    INIT_VIEW1D_OFFSET_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
      hipLaunchKernelGGL((initview1d_offset), dim3(grid_size), dim3(block_size), 0, 0,
          a, v, ibegin, iend );

    }
    stopTimer();

    INIT_VIEW1D_OFFSET_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    INIT_VIEW1D_OFFSET_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto initview1d_offset_lambda = [=] __device__ (Index_type i) {
        INIT_VIEW1D_OFFSET_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
      hipLaunchKernelGGL(lambda_hip_forall<decltype(initview1d_offset_lambda)>,
        grid_size, block_size, 0, 0, ibegin, iend, initview1d_offset_lambda);

    }
    stopTimer();

    INIT_VIEW1D_OFFSET_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    INIT_VIEW1D_OFFSET_DATA_SETUP_HIP;

    INIT_VIEW1D_OFFSET_VIEW_RAJA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        INIT_VIEW1D_OFFSET_BODY_RAJA;
      });

    }
    stopTimer();

    INIT_VIEW1D_OFFSET_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  INIT_VIEW1D_OFFSET : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
