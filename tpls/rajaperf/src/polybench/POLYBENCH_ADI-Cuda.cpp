//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ADI.hpp"

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

#define POLYBENCH_ADI_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(U, m_U, m_n * m_n); \
  allocAndInitCudaDeviceData(V, m_V, m_n * m_n); \
  allocAndInitCudaDeviceData(P, m_P, m_n * m_n); \
  allocAndInitCudaDeviceData(Q, m_Q, m_n * m_n);

#define POLYBENCH_ADI_TEARDOWN_CUDA \
  getCudaDeviceData(m_U, U, m_n * m_n); \
  deallocCudaDeviceData(U); \
  deallocCudaDeviceData(V); \
  deallocCudaDeviceData(P); \
  deallocCudaDeviceData(Q);


__global__ void adi1(const Index_type n,
                     const Real_type a, const Real_type b, const Real_type c,
                     const Real_type d, const Real_type f,
                     Real_ptr P, Real_ptr Q, Real_ptr U, Real_ptr V)
{
  Index_type i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n-1) {
    POLYBENCH_ADI_BODY2;
    for (Index_type j = 1; j < n-1; ++j) {
       POLYBENCH_ADI_BODY3;
    }
    POLYBENCH_ADI_BODY4;
    for (Index_type k = n-2; k >= 1; --k) {
       POLYBENCH_ADI_BODY5;
    }
  }
}

__global__ void adi2(const Index_type n,
                     const Real_type a, const Real_type c, const Real_type d,
                     const Real_type e, const Real_type f,
                     Real_ptr P, Real_ptr Q, Real_ptr U, Real_ptr V)
{
  Index_type i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n-1) {
    POLYBENCH_ADI_BODY6;
    for (Index_type j = 1; j < n-1; ++j) {
      POLYBENCH_ADI_BODY7;
    }
    POLYBENCH_ADI_BODY8;
    for (Index_type k = n-2; k >= 1; --k) {
      POLYBENCH_ADI_BODY9;
    }
  }
}


void POLYBENCH_ADI::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_ADI_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    POLYBENCH_ADI_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 1; t <= tsteps; ++t) {

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(n-2, block_size);

        adi1<<<grid_size, block_size>>>(n,
                                        a, b, c, d, f,
                                        P, Q, U, V);

        adi2<<<grid_size, block_size>>>(n,
                                        a, c, d, e, f,
                                        P, Q, U, V);

      }  // tstep loop

    }
    stopTimer();

    POLYBENCH_ADI_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    POLYBENCH_ADI_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 1; t <= tsteps; ++t) {

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(n-2, block_size);

        lambda_cuda_forall<<<grid_size, block_size>>>(
          1, n-1,
          [=] __device__ (Index_type i) {

          POLYBENCH_ADI_BODY2;
          for (Index_type j = 1; j < n-1; ++j) {
             POLYBENCH_ADI_BODY3;
          }
          POLYBENCH_ADI_BODY4;
          for (Index_type k = n-2; k >= 1; --k) {
             POLYBENCH_ADI_BODY5;
          }
        });

        lambda_cuda_forall<<<grid_size, block_size>>>(
          1, n-1,
          [=] __device__ (Index_type i) {

          POLYBENCH_ADI_BODY6;
          for (Index_type j = 1; j < n-1; ++j) {
            POLYBENCH_ADI_BODY7;
          }
          POLYBENCH_ADI_BODY8;
          for (Index_type k = n-2; k >= 1; --k) {
            POLYBENCH_ADI_BODY9;
          }
        });
      }  // tstep loop

    }
    stopTimer();

    POLYBENCH_ADI_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_ADI_DATA_SETUP_CUDA;

    POLYBENCH_ADI_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::Tile<0, RAJA::tile_fixed<block_size>,
                                   RAJA::cuda_block_x_direct,
            RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
              RAJA::statement::Lambda<0, RAJA::Segs<0>>,
              RAJA::statement::For<1, RAJA::seq_exec,
                RAJA::statement::Lambda<1, RAJA::Segs<0,1>>
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0>>,
              RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<3, RAJA::Segs<0,2>>
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 1; t <= tsteps; ++t) {

        RAJA::kernel<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                           RAJA::RangeSegment{1, n-1},
                           RAJA::RangeStrideSegment{n-2, 0, -1}),

          [=] __device__ (Index_type i) {
            POLYBENCH_ADI_BODY2_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_ADI_BODY3_RAJA;
          },
          [=] __device__ (Index_type i) {
            POLYBENCH_ADI_BODY4_RAJA;
          },
          [=] __device__ (Index_type i, Index_type k) {
            POLYBENCH_ADI_BODY5_RAJA;
          }
        );

        RAJA::kernel<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                           RAJA::RangeSegment{1, n-1},
                           RAJA::RangeStrideSegment{n-2, 0, -1}),

          [=] __device__ (Index_type i) {
            POLYBENCH_ADI_BODY6_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_ADI_BODY7_RAJA;
          },
          [=] __device__ (Index_type i) {
            POLYBENCH_ADI_BODY8_RAJA;
          },
          [=] __device__ (Index_type i, Index_type k) {
            POLYBENCH_ADI_BODY9_RAJA;
          }
        );

      }  // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_ADI_TEARDOWN_CUDA

  } else {
      std::cout << "\n  POLYBENCH_ADI : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA

