//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define NESTED_INIT_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(array, m_array, m_array_length);

#define NESTED_INIT_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_array, array, m_array_length); \
  deallocHipDeviceData(array);

__global__ void nested_init(Real_ptr array,
                            Index_type ni, Index_type nj)
{
  Index_type i = threadIdx.x;
  Index_type j = blockIdx.y;
  Index_type k = blockIdx.z;

  NESTED_INIT_BODY;
}


void NESTED_INIT::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    NESTED_INIT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nthreads_per_block(ni, 1, 1);
      dim3 nblocks(1, nj, nk);

      hipLaunchKernelGGL((nested_init), dim3(nblocks), dim3(nthreads_per_block), 0, 0, array,
                                                   ni, nj);

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    NESTED_INIT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto nested_init_lambda = [=] __device__ (Index_type i, Index_type j, Index_type k) {
        NESTED_INIT_BODY;
      };

      dim3 nthreads_per_block(ni, 1, 1);
      dim3 nblocks(1, nj, nk);

      auto kernel = lambda_hip_kernel<RAJA::hip_thread_x_direct, RAJA::hip_block_y_direct, RAJA::hip_block_z_direct, decltype(nested_init_lambda)>;
      hipLaunchKernelGGL(kernel,
        nblocks, nthreads_per_block, 0, 0,
        0, ni, 0, nj, 0, nk, nested_init_lambda);

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    NESTED_INIT_DATA_SETUP_HIP;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelAsync<
          RAJA::statement::For<2, RAJA::hip_block_z_direct,      // k
            RAJA::statement::For<1, RAJA::hip_block_y_direct,    // j
              RAJA::statement::For<0, RAJA::hip_thread_x_direct, // i
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

    NESTED_INIT_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  NESTED_INIT : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
