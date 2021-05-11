//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ATAX.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;

#define POLYBENCH_ATAX_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(tmp, m_tmp, N); \
  allocAndInitCudaDeviceData(y, m_y, N); \
  allocAndInitCudaDeviceData(x, m_x, N); \
  allocAndInitCudaDeviceData(A, m_A, N * N);


#define POLYBENCH_ATAX_TEARDOWN_CUDA \
  getCudaDeviceData(m_y, y, N); \
  deallocCudaDeviceData(tmp); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(A);


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


void POLYBENCH_ATAX::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_ATAX_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    POLYBENCH_ATAX_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);

      poly_atax_1<<<grid_size, block_size>>>(A, x, y, tmp, N);

      poly_atax_2<<<grid_size, block_size>>>(A, tmp, y, N);

    }
    stopTimer();

    POLYBENCH_ATAX_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    POLYBENCH_ATAX_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);

      lambda_cuda_forall<<<grid_size, block_size>>>(
        0, N,
        [=] __device__ (Index_type i) {

        POLYBENCH_ATAX_BODY1;
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_ATAX_BODY2;
        }
        POLYBENCH_ATAX_BODY3;
      });

      lambda_cuda_forall<<<grid_size, block_size>>>(
        0, N,
        [=] __device__ (Index_type j) {

        POLYBENCH_ATAX_BODY4;
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_ATAX_BODY5;
        }
        POLYBENCH_ATAX_BODY6;
      });

    }
    stopTimer();

    POLYBENCH_ATAX_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_ATAX_DATA_SETUP_CUDA;

    POLYBENCH_ATAX_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::Tile<0, RAJA::tile_fixed<block_size>,
                                   RAJA::cuda_block_x_direct,
            RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
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
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::Tile<1, RAJA::tile_fixed<block_size>,
                                   RAJA::cuda_block_x_direct,
            RAJA::statement::For<1, RAJA::cuda_thread_x_direct,
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

    POLYBENCH_ATAX_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_ATAX : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
