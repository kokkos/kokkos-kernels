//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "COPY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define COPY_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(a, m_a, iend); \
  allocAndInitCudaDeviceData(c, m_c, iend);

#define COPY_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_c, c, iend); \
  deallocCudaDeviceData(a); \
  deallocCudaDeviceData(c);

__global__ void copy(Real_ptr c, Real_ptr a,
                     Index_type iend)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < iend) {
    COPY_BODY;
  }
}


void COPY::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  COPY_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    COPY_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      copy<<<grid_size, block_size>>>( c, a,
                                       iend );

    }
    stopTimer();

    COPY_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    COPY_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<<<grid_size, block_size>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
        COPY_BODY;
      });

    }
    stopTimer();

    COPY_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    COPY_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        COPY_BODY;
      });

    }
    stopTimer();

    COPY_DATA_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  COPY : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
