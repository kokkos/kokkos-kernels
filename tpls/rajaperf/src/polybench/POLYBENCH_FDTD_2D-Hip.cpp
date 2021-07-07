//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FDTD_2D.hpp"

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

#define POLYBENCH_FDTD_2D_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(hz, m_hz, m_nx * m_ny); \
  allocAndInitHipDeviceData(ex, m_ex, m_nx * m_ny); \
  allocAndInitHipDeviceData(ey, m_ey, m_nx * m_ny); \
  allocAndInitHipDeviceData(fict, m_fict, m_tsteps);


#define POLYBENCH_FDTD_2D_TEARDOWN_HIP \
  getHipDeviceData(m_hz, hz, m_nx * m_ny); \
  deallocHipDeviceData(ex); \
  deallocHipDeviceData(ey); \
  deallocHipDeviceData(fict);


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



void POLYBENCH_FDTD_2D::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_FDTD_2D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_FDTD_2D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (t = 0; t < tsteps; ++t) {

        const size_t grid_size1 = RAJA_DIVIDE_CEILING_INT(ny, block_size);
        hipLaunchKernelGGL((poly_fdtd2d_1), dim3(grid_size1), dim3(block_size), 0, 0, ey, fict, ny, t);

        dim3 nblocks234(1, nx, 1);
        dim3 nthreads_per_block234(ny, 1, 1);
        hipLaunchKernelGGL((poly_fdtd2d_2), dim3(nblocks234), dim3(nthreads_per_block234),
                                    0, 0, ey, hz, ny);

        hipLaunchKernelGGL((poly_fdtd2d_3), dim3(nblocks234), dim3(nthreads_per_block234),
                                    0, 0, ex, hz, ny);

        hipLaunchKernelGGL((poly_fdtd2d_4), dim3(nblocks234), dim3(nthreads_per_block234),
                                    0, 0, hz, ex, ey, nx, ny);

      } // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_FDTD_2D_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    POLYBENCH_FDTD_2D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (t = 0; t < tsteps; ++t) {

        auto poly_fdtd2d_1_lambda = [=] __device__ (Index_type j) {
            POLYBENCH_FDTD_2D_BODY1;
        };

        const size_t grid_size1 = RAJA_DIVIDE_CEILING_INT(ny, block_size);
        hipLaunchKernelGGL(lambda_hip_forall<decltype(poly_fdtd2d_1_lambda)>,
          grid_size1, block_size, 0, 0,
          0, ny, poly_fdtd2d_1_lambda);

        auto poly_fdtd2d_2_lambda = [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY2;
        };

        dim3 nblocks234(1, nx, 1);
        dim3 nthreads_per_block234(ny, 1, 1);
        auto kernel2 = lambda_hip_kernel<RAJA::hip_block_y_direct, RAJA::hip_thread_x_direct, decltype(poly_fdtd2d_2_lambda)>;
        hipLaunchKernelGGL(kernel2,
          nblocks234, nthreads_per_block234, 0, 0,
          1, nx, 0, ny, poly_fdtd2d_2_lambda);

        auto poly_fdtd2d_3_lambda = [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY3;
        };

        auto kernel3 = lambda_hip_kernel<RAJA::hip_block_y_direct, RAJA::hip_thread_x_direct, decltype(poly_fdtd2d_3_lambda)>;
        hipLaunchKernelGGL(kernel3,
          nblocks234, nthreads_per_block234, 0, 0,
          0, nx, 1, ny, poly_fdtd2d_3_lambda);

        auto poly_fdtd2d_4_lambda = [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY4;
        };

        auto kernel4 = lambda_hip_kernel<RAJA::hip_block_y_direct, RAJA::hip_thread_x_direct, decltype(poly_fdtd2d_4_lambda)>;
        hipLaunchKernelGGL(kernel4,
          nblocks234, nthreads_per_block234, 0, 0,
          0, nx-1, 0, ny-1, poly_fdtd2d_4_lambda);

      } // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_FDTD_2D_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_FDTD_2D_DATA_SETUP_HIP;

    POLYBENCH_FDTD_2D_VIEWS_RAJA;

    using EXEC_POL1 = RAJA::hip_exec<block_size, true /*async*/>;

    using EXEC_POL234 =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelAsync<
          RAJA::statement::For<0, RAJA::hip_block_y_direct,
            RAJA::statement::For<1, RAJA::hip_thread_x_direct,
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

    POLYBENCH_FDTD_2D_TEARDOWN_HIP;

  } else {
      std::cout << "\n  POLYBENCH_FDTD_2D : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

