//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MULADDSUB.hpp"

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


#define MULADDSUB_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(out1, m_out1, iend); \
  allocAndInitHipDeviceData(out2, m_out2, iend); \
  allocAndInitHipDeviceData(out3, m_out3, iend); \
  allocAndInitHipDeviceData(in1, m_in1, iend); \
  allocAndInitHipDeviceData(in2, m_in2, iend);

#define MULADDSUB_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_out1, out1, iend); \
  getHipDeviceData(m_out2, out2, iend); \
  getHipDeviceData(m_out3, out3, iend); \
  deallocHipDeviceData(out1); \
  deallocHipDeviceData(out2); \
  deallocHipDeviceData(out3); \
  deallocHipDeviceData(in1); \
  deallocHipDeviceData(in2);

__global__ void muladdsub(Real_ptr out1, Real_ptr out2, Real_ptr out3,
                          Real_ptr in1, Real_ptr in2,
                          Index_type iend)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < iend) {
    MULADDSUB_BODY;
  }
}


void MULADDSUB::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  MULADDSUB_DATA_SETUP;

  if ( vid == Base_HIP ) {

    MULADDSUB_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((muladdsub), dim3(grid_size), dim3(block_size), 0, 0,
          out1, out2, out3, in1, in2, iend );

    }
    stopTimer();

    MULADDSUB_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    MULADDSUB_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto muladdsub_lambda = [=] __device__ (Index_type i) {
        MULADDSUB_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL(lambda_hip_forall<decltype(muladdsub_lambda)>,
        grid_size, block_size, 0, 0, ibegin, iend, muladdsub_lambda );

    }
    stopTimer();

    MULADDSUB_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    MULADDSUB_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        MULADDSUB_BODY;
      });

    }
    stopTimer();

    MULADDSUB_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  MULADDSUB : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
