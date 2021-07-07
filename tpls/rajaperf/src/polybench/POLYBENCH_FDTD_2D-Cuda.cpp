//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FDTD_2D.hpp"

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

#define POLYBENCH_FDTD_2D_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(hz, m_hz, m_nx * m_ny); \
  allocAndInitCudaDeviceData(ex, m_ex, m_nx * m_ny); \
  allocAndInitCudaDeviceData(ey, m_ey, m_nx * m_ny); \
  allocAndInitCudaDeviceData(fict, m_fict, m_tsteps);


#define POLYBENCH_FDTD_2D_TEARDOWN_CUDA \
  getCudaDeviceData(m_hz, hz, m_nx * m_ny); \
  deallocCudaDeviceData(ex); \
  deallocCudaDeviceData(ey); \
  deallocCudaDeviceData(fict);


__global__ void poly_fdtd2d_1(Real_ptr ey, Real_ptr fict,
                              Index_type ny, Index_type t)
{
  Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < ny) {
    POLYBENCH_FDTD_2D_BODY1;
  }
}

__global__ void poly_fdtd2d_2(Real_ptr ey, Real_ptr hz, Index_type ny)
{
  Index_type i = blockIdx.y;
  Index_type j = threadIdx.x;

  if (i > 0) {
    POLYBENCH_FDTD_2D_BODY2;
  }
}

__global__ void poly_fdtd2d_3(Real_ptr ex, Real_ptr hz, Index_type ny)
{
  Index_type i = blockIdx.y;
  Index_type j = threadIdx.x;

  if (j > 0) {
    POLYBENCH_FDTD_2D_BODY3;
  }
}

__global__ void poly_fdtd2d_4(Real_ptr hz, Real_ptr ex, Real_ptr ey,
                              Index_type nx, Index_type ny)
{
  Index_type i = blockIdx.y;
  Index_type j = threadIdx.x;

  if (i < nx-1 && j < ny-1) {
    POLYBENCH_FDTD_2D_BODY4;
  }
}



void POLYBENCH_FDTD_2D::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_FDTD_2D_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    POLYBENCH_FDTD_2D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (t = 0; t < tsteps; ++t) {

        const size_t grid_size1 = RAJA_DIVIDE_CEILING_INT(ny, block_size);
        poly_fdtd2d_1<<<grid_size1, block_size>>>(ey, fict, ny, t);

        dim3 nblocks234(1, nx, 1);
        dim3 nthreads_per_block234(ny, 1, 1);
        poly_fdtd2d_2<<<nblocks234, nthreads_per_block234>>>(ey, hz, ny);

        poly_fdtd2d_3<<<nblocks234, nthreads_per_block234>>>(ex, hz, ny);

        poly_fdtd2d_4<<<nblocks234, nthreads_per_block234>>>(hz, ex, ey, nx, ny);

      } // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_FDTD_2D_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    POLYBENCH_FDTD_2D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (t = 0; t < tsteps; ++t) {

        const size_t grid_size1 = RAJA_DIVIDE_CEILING_INT(ny, block_size);
        lambda_cuda_forall<<<grid_size1, block_size>>>(
          0, ny,
          [=] __device__ (Index_type j) {
            POLYBENCH_FDTD_2D_BODY1;
        });

        dim3 nblocks234(1, nx, 1);
        dim3 nthreads_per_block234(ny, 1, 1);
        lambda_cuda_kernel<RAJA::cuda_block_y_direct, RAJA::cuda_thread_x_direct>
                          <<<nblocks234, nthreads_per_block234>>>(
          1, nx, 0, ny,
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY2;
        });

        lambda_cuda_kernel<RAJA::cuda_block_y_direct, RAJA::cuda_thread_x_direct>
                          <<<nblocks234, nthreads_per_block234>>>(
          0, nx, 1, ny,
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY3;
        });

        lambda_cuda_kernel<RAJA::cuda_block_y_direct, RAJA::cuda_thread_x_direct>
                          <<<nblocks234, nthreads_per_block234>>>(
          0, nx-1, 0, ny-1,
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY4;
        });

      } // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_FDTD_2D_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_FDTD_2D_DATA_SETUP_CUDA;

    POLYBENCH_FDTD_2D_VIEWS_RAJA;

    using EXEC_POL1 = RAJA::cuda_exec<block_size, true /*async*/>;

    using EXEC_POL234 =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::For<0, RAJA::cuda_block_y_direct,
            RAJA::statement::For<1, RAJA::cuda_thread_x_direct,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (t = 0; t < tsteps; ++t) {

        RAJA::forall<EXEC_POL1>( RAJA::RangeSegment(0, ny),
        [=] __device__ (Index_type j) {
          POLYBENCH_FDTD_2D_BODY1_RAJA;
        });

        RAJA::kernel<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{1, nx},
                           RAJA::RangeSegment{0, ny}),
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY2_RAJA;
          }
        );

        RAJA::kernel<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nx},
                           RAJA::RangeSegment{1, ny}),
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY3_RAJA;
          }
        );

        RAJA::kernel<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nx-1},
                           RAJA::RangeSegment{0, ny-1}),
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY4_RAJA;
          }
        );

      }  // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_FDTD_2D_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_FDTD_2D : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA

