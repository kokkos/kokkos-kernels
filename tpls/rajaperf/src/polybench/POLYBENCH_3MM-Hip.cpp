//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_3MM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

#define POLYBENCH_3MM_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(A, m_A, m_ni * m_nk); \
  allocAndInitHipDeviceData(B, m_B, m_nk * m_nj); \
  allocAndInitHipDeviceData(C, m_C, m_nj * m_nm); \
  allocAndInitHipDeviceData(D, m_D, m_nm * m_nl); \
  allocAndInitHipDeviceData(E, m_E, m_ni * m_nj); \
  allocAndInitHipDeviceData(F, m_F, m_nj * m_nl); \
  allocAndInitHipDeviceData(G, m_G, m_ni * m_nl);


#define POLYBENCH_3MM_TEARDOWN_HIP \
  getHipDeviceData(m_G, G, m_ni * m_nl); \
  deallocHipDeviceData(A); \
  deallocHipDeviceData(B); \
  deallocHipDeviceData(C); \
  deallocHipDeviceData(D); \
  deallocHipDeviceData(E); \
  deallocHipDeviceData(F); \
  deallocHipDeviceData(G);

__global__ void poly_3mm_1(Real_ptr E, Real_ptr A, Real_ptr B,
                           Index_type nj, Index_type nk)
{
   Index_type i = blockIdx.y;
   Index_type j = threadIdx.x;

   POLYBENCH_3MM_BODY1;
   for (Index_type k=0; k < nk; ++k) {
     POLYBENCH_3MM_BODY2;
   }
   POLYBENCH_3MM_BODY3;
}

__global__ void poly_3mm_2(Real_ptr F, Real_ptr C, Real_ptr D,
                           Index_type nl, Index_type nm)
{
   Index_type j = blockIdx.y;
   Index_type l = threadIdx.x;

   POLYBENCH_3MM_BODY4;
   for (Index_type m=0; m < nm; ++m) {
     POLYBENCH_3MM_BODY5;
   }
   POLYBENCH_3MM_BODY6;
}

__global__ void poly_3mm_3(Real_ptr G, Real_ptr E, Real_ptr F,
                           Index_type nl, Index_type nj)
{
   Index_type i = blockIdx.y;
   Index_type l = threadIdx.x;

   POLYBENCH_3MM_BODY7;
   for (Index_type j=0; j < nj; ++j) {
     POLYBENCH_3MM_BODY8;
   }
   POLYBENCH_3MM_BODY9;
}


void POLYBENCH_3MM::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_3MM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_3MM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nblocks1(1, ni, 1);
      dim3 nthreads_per_block1(nj, 1, 1);
      hipLaunchKernelGGL((poly_3mm_1), dim3(nblocks1) , dim3(nthreads_per_block1), 0, 0,
                                                    E, A, B,
                                                    nj, nk);

      dim3 nblocks2(1, nj, 1);
      dim3 nthreads_per_block2(nl, 1, 1);
      hipLaunchKernelGGL((poly_3mm_2), dim3(nblocks2), dim3(nthreads_per_block2), 0, 0,
                                                    F, C, D,
                                                    nl, nm);

      dim3 nblocks3(1, ni, 1);
      dim3 nthreads_per_block3(nl, 1, 1);
      hipLaunchKernelGGL((poly_3mm_3), dim3(nblocks3), dim3(nthreads_per_block3), 0, 0,
                                                    G, E, F,
                                                    nl, nj);

    }
    stopTimer();

    POLYBENCH_3MM_TEARDOWN_HIP;

  } else if (vid == Lambda_HIP) {

    POLYBENCH_3MM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto poly_3mm_1_lambda = [=] __device__ (Index_type i, Index_type j) {

        POLYBENCH_3MM_BODY1;
        for (Index_type k=0; k < nk; ++k) {
          POLYBENCH_3MM_BODY2;
        }
        POLYBENCH_3MM_BODY3;
      };

      dim3 nblocks1(1, ni, 1);
      dim3 nthreads_per_block1(nj, 1, 1);
      auto kernel1 = lambda_hip_kernel<RAJA::hip_block_y_direct, RAJA::hip_thread_x_direct, decltype(poly_3mm_1_lambda)>;
      hipLaunchKernelGGL(kernel1,
        nblocks1, nthreads_per_block1, 0, 0,
        0, ni, 0, nj, poly_3mm_1_lambda);

      auto poly_3mm_2_lambda = [=] __device__ (Index_type j, Index_type l) {

        POLYBENCH_3MM_BODY4;
        for (Index_type m=0; m < nm; ++m) {
          POLYBENCH_3MM_BODY5;
        }
        POLYBENCH_3MM_BODY6;
      };

      dim3 nblocks2(1, nj, 1);
      dim3 nthreads_per_block2(nl, 1, 1);
      auto kernel2 = lambda_hip_kernel<RAJA::hip_block_y_direct, RAJA::hip_thread_x_direct, decltype(poly_3mm_2_lambda)>;
      hipLaunchKernelGGL(kernel2,
        nblocks2, nthreads_per_block2, 0, 0,
        0, nj, 0, nl, poly_3mm_2_lambda);

      auto poly_3mm_3_lambda = [=] __device__ (Index_type i, Index_type l) {

        POLYBENCH_3MM_BODY7;
        for (Index_type j=0; j < nj; ++j) {
          POLYBENCH_3MM_BODY8;
        }
        POLYBENCH_3MM_BODY9;
      };

      dim3 nblocks3(1, ni, 1);
      dim3 nthreads_per_block3(nl, 1, 1);
      auto kernel3 = lambda_hip_kernel<RAJA::hip_block_y_direct, RAJA::hip_thread_x_direct, decltype(poly_3mm_3_lambda)>;
      hipLaunchKernelGGL(kernel3,
        nblocks3, nthreads_per_block3, 0, 0,
        0, ni, 0, nl, poly_3mm_3_lambda);

    }
    stopTimer();

    POLYBENCH_3MM_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_3MM_DATA_SETUP_HIP;

    POLYBENCH_3MM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelAsync<
          RAJA::statement::For<0, RAJA::hip_block_x_direct,
            RAJA::statement::For<1, RAJA::hip_thread_y_direct,
              RAJA::statement::Lambda<0, RAJA::Params<0>>,
              RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<1, RAJA::Segs<0,1,2>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0,1>, RAJA::Params<0>>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel_param<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nj},
                         RAJA::RangeSegment{0, nk}),
        RAJA::tuple<Real_type>{0.0},

        [=] __device__ ( Real_type &dot) {
          POLYBENCH_3MM_BODY1_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j, Index_type k,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY2_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY3_RAJA;
        }

      );

      RAJA::kernel_param<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                         RAJA::RangeSegment{0, nl},
                         RAJA::RangeSegment{0, nm}),
        RAJA::tuple<Real_type>{0.0},

        [=] __device__ ( Real_type &dot) {
          POLYBENCH_3MM_BODY4_RAJA;
        },
        [=] __device__ (Index_type j, Index_type l, Index_type m,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY5_RAJA;
        },
        [=] __device__ (Index_type j, Index_type l,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY6_RAJA;
        }

      );

      RAJA::kernel_param<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nl},
                         RAJA::RangeSegment{0, nj}),
        RAJA::tuple<Real_type>{0.0},

        [=] __device__ ( Real_type &dot) {
          POLYBENCH_3MM_BODY7_RAJA;
        },
        [=] __device__ (Index_type i, Index_type l, Index_type j,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY8_RAJA;
        },
        [=] __device__ (Index_type i, Index_type l,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY9_RAJA;
        }

      );

    }
    stopTimer();

    POLYBENCH_3MM_TEARDOWN_HIP;

  } else {
      std::cout << "\n  POLYBENCH_3MM : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

