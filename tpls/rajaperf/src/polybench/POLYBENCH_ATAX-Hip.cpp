//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ATAX.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;

#define POLYBENCH_ATAX_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(tmp, m_tmp, N); \
  allocAndInitHipDeviceData(y, m_y, N); \
  allocAndInitHipDeviceData(x, m_x, N); \
  allocAndInitHipDeviceData(A, m_A, N * N);


#define POLYBENCH_ATAX_TEARDOWN_HIP \
  getHipDeviceData(m_y, y, N); \
  deallocHipDeviceData(tmp); \
  deallocHipDeviceData(y); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(A);


__global__ void poly_atax_1(Real_ptr A, Real_ptr x, Real_ptr y, Real_ptr tmp,
                            Index_type N)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < N) {
     POLYBENCH_ATAX_BODY1;
     for (Index_type j = 0; j < N; ++j ) {
       POLYBENCH_ATAX_BODY2;
     }
     POLYBENCH_ATAX_BODY3;
   }
}

__global__ void poly_atax_2(Real_ptr A, Real_ptr tmp, Real_ptr y,
                            Index_type N)
{
   Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

   if (j < N) {
     POLYBENCH_ATAX_BODY4;
     for (Index_type i = 0; i < N; ++i ) {
       POLYBENCH_ATAX_BODY5;
     }
     POLYBENCH_ATAX_BODY6;
   }
}


void POLYBENCH_ATAX::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_ATAX_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_ATAX_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);

      hipLaunchKernelGGL((poly_atax_1), dim3(grid_size), dim3(block_size), 0, 0,
                                      A, x, y, tmp, N);

      hipLaunchKernelGGL((poly_atax_2), dim3(grid_size), dim3(block_size), 0, 0,
                                      A, tmp, y, N);

    }
    stopTimer();

    POLYBENCH_ATAX_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    POLYBENCH_ATAX_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);

      auto poly_atax_1_lambda = [=] __device__ (Index_type i) {

        POLYBENCH_ATAX_BODY1;
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_ATAX_BODY2;
        }
        POLYBENCH_ATAX_BODY3;
      };

      hipLaunchKernelGGL(lambda_hip_forall<decltype(poly_atax_1_lambda)>,
        grid_size, block_size, 0, 0,
        0, N, poly_atax_1_lambda);

      auto poly_atax_2_lambda = [=] __device__ (Index_type j) {

        POLYBENCH_ATAX_BODY4;
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_ATAX_BODY5;
        }
        POLYBENCH_ATAX_BODY6;
      };

      hipLaunchKernelGGL(lambda_hip_forall<decltype(poly_atax_2_lambda)>,
        grid_size, block_size, 0, 0,
        0, N, poly_atax_2_lambda);

    }
    stopTimer();

    POLYBENCH_ATAX_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_ATAX_DATA_SETUP_HIP;

    POLYBENCH_ATAX_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelAsync<
          RAJA::statement::Tile<0, RAJA::tile_fixed<block_size>,
                                   RAJA::hip_block_x_direct,
            RAJA::statement::For<0, RAJA::hip_thread_x_direct,
              RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
              RAJA::statement::For<1, RAJA::seq_exec,
                RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
            >
          >
        >
      >;

    using EXEC_POL2 =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelAsync<
          RAJA::statement::Tile<1, RAJA::tile_fixed<block_size>,
                                   RAJA::hip_block_x_direct,
            RAJA::statement::For<1, RAJA::hip_thread_x_direct,
              RAJA::statement::Lambda<0, RAJA::Segs<1>, RAJA::Params<0>>,
              RAJA::statement::For<0, RAJA::seq_exec,
                RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<1>, RAJA::Params<0>>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel_param<EXEC_POL1>(
        RAJA::make_tuple(RAJA::RangeSegment{0, N},
                         RAJA::RangeSegment{0, N}),
        RAJA::tuple<Real_type>{0.0},

        [=] __device__ (Index_type i, Real_type &dot) {
          POLYBENCH_ATAX_BODY1_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_ATAX_BODY2_RAJA;
        },
        [=] __device__ (Index_type i, Real_type &dot) {
          POLYBENCH_ATAX_BODY3_RAJA;
        }

      );

      RAJA::kernel_param<EXEC_POL2>(
        RAJA::make_tuple(RAJA::RangeSegment{0, N},
                         RAJA::RangeSegment{0, N}),
        RAJA::tuple<Real_type>{0.0},

        [=] __device__ (Index_type j, Real_type &dot) {
          POLYBENCH_ATAX_BODY4_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j , Real_type &dot) {
          POLYBENCH_ATAX_BODY5_RAJA;
        },
        [=] __device__ (Index_type j, Real_type &dot) {
          POLYBENCH_ATAX_BODY6_RAJA;
        }

     );

    }
    stopTimer();

    POLYBENCH_ATAX_TEARDOWN_HIP;

  } else {
      std::cout << "\n  POLYBENCH_ATAX : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

