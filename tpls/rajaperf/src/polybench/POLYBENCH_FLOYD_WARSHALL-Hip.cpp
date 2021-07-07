//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

namespace rajaperf
{
namespace polybench
{

#define POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(pin, m_pin, m_N * m_N); \
  allocAndInitHipDeviceData(pout, m_pout, m_N * m_N);


#define POLYBENCH_FLOYD_WARSHALL_TEARDOWN_HIP \
  getHipDeviceData(m_pout, pout, m_N * m_N); \
  deallocHipDeviceData(pin); \
  deallocHipDeviceData(pout);


__global__ void poly_floyd_warshall(Real_ptr pout, Real_ptr pin,
                                    Index_type k,
                                    Index_type N)
{
   Index_type i = blockIdx.y;
   Index_type j = threadIdx.x;

   POLYBENCH_FLOYD_WARSHALL_BODY;
}


void POLYBENCH_FLOYD_WARSHALL::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_FLOYD_WARSHALL_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type k = 0; k < N; ++k) {

        dim3 nblocks1(1, N, 1);
        dim3 nthreads_per_block1(N, 1, 1);
        hipLaunchKernelGGL((poly_floyd_warshall),dim3(nblocks1), dim3(nthreads_per_block1),0,0,pout, pin,
                                                               k, N);

      }

    }
    stopTimer();

    POLYBENCH_FLOYD_WARSHALL_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type k = 0; k < N; ++k) {

        auto poly_floyd_warshall_lambda = [=] __device__ (Index_type i, Index_type j) {

          POLYBENCH_FLOYD_WARSHALL_BODY;
        };

        dim3 nblocks1(1, N, 1);
        dim3 nthreads_per_block1(N, 1, 1);
        auto kernel = lambda_hip_kernel<RAJA::hip_block_y_direct, RAJA::hip_thread_x_direct, decltype(poly_floyd_warshall_lambda)>;
        hipLaunchKernelGGL(kernel,
          nblocks1, nthreads_per_block1,0,0,
          0, N, 0, N, poly_floyd_warshall_lambda);

      }

    }
    stopTimer();

    POLYBENCH_FLOYD_WARSHALL_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_HIP;

    POLYBENCH_FLOYD_WARSHALL_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::HipKernelAsync<
            RAJA::statement::For<1, RAJA::hip_block_y_direct,
              RAJA::statement::For<2, RAJA::hip_thread_x_direct,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                               RAJA::RangeSegment{0, N},
                                               RAJA::RangeSegment{0, N}),
        [=] __device__ (Index_type k, Index_type i, Index_type j) {
          POLYBENCH_FLOYD_WARSHALL_BODY_RAJA;
        }
      );

    }
    stopTimer();

    POLYBENCH_FLOYD_WARSHALL_TEARDOWN_HIP;

  } else {
      std::cout << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

