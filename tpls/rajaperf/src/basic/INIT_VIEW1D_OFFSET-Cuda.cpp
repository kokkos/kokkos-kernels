//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D_OFFSET.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define INIT_VIEW1D_OFFSET_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(a, m_a, getRunSize());

#define INIT_VIEW1D_OFFSET_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_a, a, getRunSize()); \
  deallocCudaDeviceData(a);

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


void INIT_VIEW1D_OFFSET::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getRunSize()+1;

  INIT_VIEW1D_OFFSET_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    INIT_VIEW1D_OFFSET_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
      initview1d_offset<<<grid_size, block_size>>>( a, v,
                                                    ibegin,
                                                    iend );

    }
    stopTimer();

    INIT_VIEW1D_OFFSET_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    INIT_VIEW1D_OFFSET_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
      lambda_cuda_forall<<<grid_size, block_size>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
        INIT_VIEW1D_OFFSET_BODY;
      });

    }
    stopTimer();

    INIT_VIEW1D_OFFSET_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    INIT_VIEW1D_OFFSET_DATA_SETUP_CUDA;

    INIT_VIEW1D_OFFSET_VIEW_RAJA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        INIT_VIEW1D_OFFSET_BODY_RAJA;
      });

    }
    stopTimer();

    INIT_VIEW1D_OFFSET_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  INIT_VIEW1D_OFFSET : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
