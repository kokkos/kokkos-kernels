//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define NESTED_INIT_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(array, m_array, m_array_length);

#define NESTED_INIT_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_array, array, m_array_length); \
  deallocCudaDeviceData(array);

__global__ void nested_init(Real_ptr array,
                            Index_type ni, Index_type nj)
{
  Index_type i = threadIdx.x;
  Index_type j = blockIdx.y;
  Index_type k = blockIdx.z;

  NESTED_INIT_BODY;
}


void NESTED_INIT::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    NESTED_INIT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nthreads_per_block(ni, 1, 1);
      dim3 nblocks(1, nj, nk);

      nested_init<<<nblocks, nthreads_per_block>>>(array,
                                                   ni, nj);

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    NESTED_INIT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nthreads_per_block(ni, 1, 1);
      dim3 nblocks(1, nj, nk);

      lambda_cuda_kernel<RAJA::cuda_thread_x_direct, RAJA::cuda_block_y_direct, RAJA::cuda_block_z_direct><<<nblocks, nthreads_per_block>>>(
        0, ni, 0, nj, 0, nk,
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
        NESTED_INIT_BODY;
      });

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    NESTED_INIT_DATA_SETUP_CUDA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::For<2, RAJA::cuda_block_z_direct,      // k
            RAJA::statement::For<1, RAJA::cuda_block_y_direct,    // j
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct, // i
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                               RAJA::RangeSegment(0, nj),
                                               RAJA::RangeSegment(0, nk)),
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
        NESTED_INIT_BODY;
      });

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  NESTED_INIT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
