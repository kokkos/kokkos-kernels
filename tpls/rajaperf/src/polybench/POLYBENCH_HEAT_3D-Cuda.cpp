//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_HEAT_3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

#define POLYBENCH_HEAT_3D_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(A, m_Ainit, m_N*m_N*m_N); \
  allocAndInitCudaDeviceData(B, m_Binit, m_N*m_N*m_N);


#define POLYBENCH_HEAT_3D_TEARDOWN_CUDA \
  getCudaDeviceData(m_A, A, m_N*m_N*m_N); \
  getCudaDeviceData(m_B, B, m_N*m_N*m_N); \
  deallocCudaDeviceData(A); \
  deallocCudaDeviceData(B);


__global__ void poly_heat_3D_1(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = 1 + blockIdx.y;
   Index_type j = 1 + blockIdx.z;
   Index_type k = 1 + threadIdx.x;

   if (i < N-1 && j < N-1 && k < N-1) {
     POLYBENCH_HEAT_3D_BODY1;
   }
}

__global__ void poly_heat_3D_2(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = 1 + blockIdx.y;
   Index_type j = 1 + blockIdx.z;
   Index_type k = 1 + threadIdx.x;

   if (i < N-1 && j < N-1 && k < N-1) {
     POLYBENCH_HEAT_3D_BODY2;
   }
}


void POLYBENCH_HEAT_3D::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_HEAT_3D_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    POLYBENCH_HEAT_3D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        dim3 nblocks(1, N-2, N-2);
        dim3 nthreads_per_block(N-2, 1, 1);

        poly_heat_3D_1<<<nblocks, nthreads_per_block>>>(A, B, N);

        poly_heat_3D_2<<<nblocks, nthreads_per_block>>>(A, B, N);

      }

    }
    stopTimer();

    POLYBENCH_HEAT_3D_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    POLYBENCH_HEAT_3D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        dim3 nblocks(1, N-2, N-2);
        dim3 nthreads_per_block(N-2, 1, 1);

        lambda_cuda_kernel<RAJA::cuda_block_y_direct, RAJA::cuda_block_z_direct, RAJA::cuda_thread_x_direct>
                         <<<nblocks, nthreads_per_block>>>(
          1, N-1, 1, N-1, 1, N-1,
          [=] __device__ (Index_type i, Index_type j, Index_type k) {

          POLYBENCH_HEAT_3D_BODY1;
        });

        lambda_cuda_kernel<RAJA::cuda_block_y_direct, RAJA::cuda_block_z_direct, RAJA::cuda_thread_x_direct>
                         <<<nblocks, nthreads_per_block>>>(
          1, N-1, 1, N-1, 1, N-1,
          [=] __device__ (Index_type i, Index_type j, Index_type k) {

          POLYBENCH_HEAT_3D_BODY2;
        });

      }

    }
    stopTimer();

    POLYBENCH_HEAT_3D_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_HEAT_3D_DATA_SETUP_CUDA;

    POLYBENCH_HEAT_3D_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::For<0, RAJA::cuda_block_z_direct,
            RAJA::statement::For<1, RAJA::cuda_block_y_direct,
              RAJA::statement::For<2, RAJA::cuda_thread_x_direct,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >,
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::For<0, RAJA::cuda_block_z_direct,
            RAJA::statement::For<1, RAJA::cuda_block_y_direct,
              RAJA::statement::For<2, RAJA::cuda_thread_x_direct,
                RAJA::statement::Lambda<1>
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                 RAJA::RangeSegment{1, N-1},
                                                 RAJA::RangeSegment{1, N-1}),
          [=] __device__ (Index_type i, Index_type j, Index_type k) {
            POLYBENCH_HEAT_3D_BODY1_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j, Index_type k) {
            POLYBENCH_HEAT_3D_BODY2_RAJA;
          }
        );

      }

    }
    stopTimer();

    POLYBENCH_HEAT_3D_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_HEAT_3D : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
