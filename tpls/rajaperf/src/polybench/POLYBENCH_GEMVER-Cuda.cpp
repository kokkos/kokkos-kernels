//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMVER.hpp"

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

#define POLYBENCH_GEMVER_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(A, m_A, m_n * m_n); \
  allocAndInitCudaDeviceData(u1, m_u1, m_n); \
  allocAndInitCudaDeviceData(v1, m_v1, m_n); \
  allocAndInitCudaDeviceData(u2, m_u2, m_n); \
  allocAndInitCudaDeviceData(v2, m_v2, m_n); \
  allocAndInitCudaDeviceData(w, m_w, m_n); \
  allocAndInitCudaDeviceData(x, m_x, m_n); \
  allocAndInitCudaDeviceData(y, m_y, m_n); \
  allocAndInitCudaDeviceData(z, m_z, m_n);


#define POLYBENCH_GEMVER_TEARDOWN_CUDA \
  getCudaDeviceData(m_w, w, m_n); \
  deallocCudaDeviceData(A); \
  deallocCudaDeviceData(u1); \
  deallocCudaDeviceData(v1); \
  deallocCudaDeviceData(u2); \
  deallocCudaDeviceData(v2); \
  deallocCudaDeviceData(w); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(z);

__global__ void poly_gemmver_1(Real_ptr A,
                               Real_ptr u1, Real_ptr v1,
                               Real_ptr u2, Real_ptr v2,
                               Index_type n)
{
  Index_type i = blockIdx.y;
  Index_type j = threadIdx.x;

  POLYBENCH_GEMVER_BODY1;
}

__global__ void poly_gemmver_2(Real_ptr A,
                               Real_ptr x, Real_ptr y,
                               Real_type beta,
                               Index_type n)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    POLYBENCH_GEMVER_BODY2;
    for (Index_type j = 0; j < n; ++j) {
      POLYBENCH_GEMVER_BODY3;
    }
    POLYBENCH_GEMVER_BODY4;
  }
}

__global__ void poly_gemmver_3(Real_ptr x, Real_ptr z,
                               Index_type n)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    POLYBENCH_GEMVER_BODY5;
  }
}

__global__ void poly_gemmver_4(Real_ptr A,
                               Real_ptr x, Real_ptr w,
                               Real_type alpha,
                               Index_type n)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    POLYBENCH_GEMVER_BODY6;
    for (Index_type j = 0; j < n; ++j) {
      POLYBENCH_GEMVER_BODY7;
    }
    POLYBENCH_GEMVER_BODY8;
  }
}


void POLYBENCH_GEMVER::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_GEMVER_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    POLYBENCH_GEMVER_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nblocks(1, n, 1);
      dim3 nthreads_per_block(n, 1, 1);
      poly_gemmver_1<<<nblocks, nthreads_per_block>>>(A, u1, v1, u2, v2,
                                                      n);

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_n, block_size);

      poly_gemmver_2<<<grid_size, block_size>>>(A, x, y,
                                                beta,
                                                n);

      poly_gemmver_3<<<grid_size, block_size>>>(x, z,
                                                n);

      poly_gemmver_4<<<grid_size, block_size>>>(A, x, w,
                                                alpha,
                                                n);

    }
    stopTimer();

    POLYBENCH_GEMVER_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    POLYBENCH_GEMVER_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nblocks(1, n, 1);
      dim3 nthreads_per_block(n, 1, 1);
      lambda_cuda_kernel<RAJA::cuda_block_y_direct, RAJA::cuda_thread_x_direct>
                        <<<nblocks, nthreads_per_block>>>(
        0, n, 0, n,
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_GEMVER_BODY1;
      });

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_n, block_size);

      lambda_cuda_forall<<<grid_size, block_size>>>(
        0, n,
        [=] __device__ (Index_type i) {
          POLYBENCH_GEMVER_BODY2;
          for (Index_type j = 0; j < n; ++j) {
            POLYBENCH_GEMVER_BODY3;
          }
          POLYBENCH_GEMVER_BODY4;
      });

      lambda_cuda_forall<<<grid_size, block_size>>>(
        0, n,
        [=] __device__ (Index_type i) {
          POLYBENCH_GEMVER_BODY5;
      });

      lambda_cuda_forall<<<grid_size, block_size>>>(
        0, n,
        [=] __device__ (Index_type i) {
          POLYBENCH_GEMVER_BODY6;
          for (Index_type j = 0; j < n; ++j) {
            POLYBENCH_GEMVER_BODY7;
          }
          POLYBENCH_GEMVER_BODY8;
      });

    }
    stopTimer();

    POLYBENCH_GEMVER_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_GEMVER_DATA_SETUP_CUDA;

    POLYBENCH_GEMVER_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::For<0, RAJA::cuda_block_y_direct,
            RAJA::statement::For<1, RAJA::cuda_thread_x_direct,
              RAJA::statement::Lambda<0>,
            >
          >
        >
      >;

    using EXEC_POL2 =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::Tile<0, RAJA::tile_fixed<block_size>,
                                   RAJA::cuda_block_x_direct,
            RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
              RAJA::statement::Lambda<0, RAJA::Params<0>>,
              RAJA::statement::For<1, RAJA::seq_exec,
                RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>,
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
            >
          >
        >
      >;

    using EXEC_POL3 = RAJA::cuda_exec<block_size, true /*async*/>;

    using EXEC_POL4 =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::Tile<0, RAJA::tile_fixed<block_size>,
                                   RAJA::cuda_block_x_direct,
            RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
              RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
              RAJA::statement::For<1, RAJA::seq_exec,
                RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>,
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL1>( RAJA::make_tuple(RAJA::RangeSegment{0, n},
                                                RAJA::RangeSegment{0, n}),
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_GEMVER_BODY1_RAJA;
        }
      );

      RAJA::kernel_param<EXEC_POL2>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        RAJA::tuple<Real_type>{0.0},

        [=] __device__ (Real_type &dot) {
          POLYBENCH_GEMVER_BODY2_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_GEMVER_BODY3_RAJA;
        },
        [=] __device__ (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY4_RAJA;
        }
      );

      RAJA::forall<EXEC_POL3> (RAJA::RangeSegment{0, n},
        [=] __device__ (Index_type i) {
          POLYBENCH_GEMVER_BODY5_RAJA;
        }
      );

      RAJA::kernel_param<EXEC_POL4>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        RAJA::tuple<Real_type>{0.0},

        [=] __device__ (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY6_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_GEMVER_BODY7_RAJA;
        },
        [=] __device__ (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY8_RAJA;
        }
      );

    }
    stopTimer();

    POLYBENCH_GEMVER_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_GEMVER : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA

