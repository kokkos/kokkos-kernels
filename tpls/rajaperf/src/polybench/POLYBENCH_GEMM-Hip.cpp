//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

#define POLYBENCH_GEMM_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(A, m_A, ni*nk); \
  allocAndInitHipDeviceData(B, m_B, nk*nj); \
  allocAndInitHipDeviceData(C, m_C, ni*nj);


#define POLYBENCH_GEMM_TEARDOWN_HIP \
  getHipDeviceData(m_C, C, ni*nj); \
  deallocHipDeviceData(A); \
  deallocHipDeviceData(B); \
  deallocHipDeviceData(C);


__global__ void poly_gemm(Real_ptr C, Real_ptr A, Real_ptr B,
                          Real_type alpha, Real_type beta,
                          Index_type nj, Index_type nk)
{
   Index_type i = blockIdx.y;
   Index_type j = threadIdx.x;

   POLYBENCH_GEMM_BODY1;
   POLYBENCH_GEMM_BODY2;
   for (Index_type k = 0; k < nk; ++k ) {
     POLYBENCH_GEMM_BODY3;
   }
   POLYBENCH_GEMM_BODY4;
}


void POLYBENCH_GEMM::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_GEMM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_GEMM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nblocks(1, ni, 1);
      dim3 nthreads_per_block(nj, 1, 1);

      hipLaunchKernelGGL((poly_gemm), dim3(nblocks), dim3(nthreads_per_block), 0,0,C, A, B,
                                                 alpha, beta,
                                                 nj, nk);

    }
    stopTimer();

    POLYBENCH_GEMM_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    POLYBENCH_GEMM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto poly_gemm_lambda = [=] __device__ (Index_type i, Index_type j) {

        POLYBENCH_GEMM_BODY1;
        POLYBENCH_GEMM_BODY2;
        for (Index_type k = 0; k < nk; ++k ) {
          POLYBENCH_GEMM_BODY3;
        }
        POLYBENCH_GEMM_BODY4;
      };

      dim3 nblocks(1, ni, 1);
      dim3 nthreads_per_block(nj, 1, 1);

      auto kernel = lambda_hip_kernel<RAJA::hip_block_y_direct, RAJA::hip_thread_x_direct, decltype(poly_gemm_lambda)>;
      hipLaunchKernelGGL(kernel,
        nblocks, nthreads_per_block, 0, 0,
        0, ni, 0, nj, poly_gemm_lambda);

    }
    stopTimer();

    POLYBENCH_GEMM_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_GEMM_DATA_SETUP_HIP;

    POLYBENCH_GEMM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelAsync<
          RAJA::statement::For<0, RAJA::hip_block_y_direct,
            RAJA::statement::For<1, RAJA::hip_thread_x_direct,
              RAJA::statement::Lambda<0, RAJA::Params<0>>,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>>,
              RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<2, RAJA::Segs<0,1,2>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<3, RAJA::Segs<0,1>, RAJA::Params<0>>
            >
          >
        >
      >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(

          RAJA::make_tuple( RAJA::RangeSegment{0, ni},
                            RAJA::RangeSegment{0, nj},
                            RAJA::RangeSegment{0, nk} ),
          RAJA::tuple<Real_type>{0.0},  // variable for dot

          [=] __device__ (Real_type& dot) {
            POLYBENCH_GEMM_BODY1_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_GEMM_BODY2_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j, Index_type k,
                          Real_type& dot) {
            POLYBENCH_GEMM_BODY3_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j,
                          Real_type& dot) {
            POLYBENCH_GEMM_BODY4_RAJA;
          }
        );

      }
      stopTimer();

    POLYBENCH_GEMM_TEARDOWN_HIP;

  } else {
      std::cout << "\n  POLYBENCH_GEMM : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

