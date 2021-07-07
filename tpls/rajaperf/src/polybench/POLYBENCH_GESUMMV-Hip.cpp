//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GESUMMV.hpp"

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

#define POLYBENCH_GESUMMV_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, N); \
  allocAndInitHipDeviceData(y, m_y, N); \
  allocAndInitHipDeviceData(A, m_A, N*N); \
  allocAndInitHipDeviceData(B, m_B, N*N);


#define POLYBENCH_GESUMMV_TEARDOWN_HIP \
  getHipDeviceData(m_y, y, N); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(y); \
  deallocHipDeviceData(A); \
  deallocHipDeviceData(B);


__global__ void poly_gesummv(Real_ptr x, Real_ptr y,
                             Real_ptr A, Real_ptr B,
                             Real_type alpha, Real_type beta,
                             Index_type N)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < N) {
     POLYBENCH_GESUMMV_BODY1;
     for (Index_type j = 0; j < N; ++j ) {
       POLYBENCH_GESUMMV_BODY2;
     }
     POLYBENCH_GESUMMV_BODY3;
   }
}


void POLYBENCH_GESUMMV::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_GESUMMV_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_GESUMMV_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);

      hipLaunchKernelGGL((poly_gesummv), dim3(grid_size), dim3(block_size),0,0,x, y,
                                              A, B,
                                              alpha, beta,
                                              N);

    }
    stopTimer();

    POLYBENCH_GESUMMV_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_GESUMMV_DATA_SETUP_HIP;

    POLYBENCH_GESUMMV_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelAsync<
          RAJA::statement::Tile<0, RAJA::tile_fixed<block_size>,
                                   RAJA::hip_block_x_direct,
            RAJA::statement::For<0, RAJA::hip_thread_x_direct,
              RAJA::statement::Lambda<0, RAJA::Params<0,1>>,
              RAJA::statement::For<1, RAJA::seq_exec,
                RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0,1>>
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0,1>>
            >
          >
        >
      >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple( RAJA::RangeSegment{0, N},
                            RAJA::RangeSegment{0, N} ),
          RAJA::make_tuple(static_cast<Real_type>(0.0),
                           static_cast<Real_type>(0.0)),

          [=] __device__ (Real_type& tmpdot,
                          Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY1_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j, Real_type& tmpdot,
                                                      Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY2_RAJA;
          },
          [=] __device__ (Index_type i, Real_type& tmpdot,
                                        Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY3_RAJA;
          }
        );

      }
      stopTimer();

    POLYBENCH_GESUMMV_TEARDOWN_HIP;

  } else {
      std::cout << "\n  POLYBENCH_GESUMMV : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

