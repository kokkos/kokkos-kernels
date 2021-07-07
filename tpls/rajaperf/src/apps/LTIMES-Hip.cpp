//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


#define LTIMES_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(phidat, m_phidat, m_philen); \
  allocAndInitHipDeviceData(elldat, m_elldat, m_elllen); \
  allocAndInitHipDeviceData(psidat, m_psidat, m_psilen);

#define LTIMES_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_phidat, phidat, m_philen); \
  deallocHipDeviceData(phidat); \
  deallocHipDeviceData(elldat); \
  deallocHipDeviceData(psidat);

__global__ void ltimes(Real_ptr phidat, Real_ptr elldat, Real_ptr psidat,
                       Index_type num_d, Index_type num_g, Index_type num_m)
{
   Index_type m = threadIdx.x;
   Index_type g = threadIdx.y;
   Index_type z = blockIdx.z;

   for (Index_type d = 0; d < num_d; ++d ) {
     LTIMES_BODY;
   }
}


void LTIMES::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  LTIMES_DATA_SETUP;

  if ( vid == Base_HIP ) {

    LTIMES_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nthreads_per_block(num_m, num_g, 1);
      dim3 nblocks(1, 1, num_z);

      hipLaunchKernelGGL((ltimes), dim3(nblocks), dim3(nthreads_per_block), 0, 0, phidat, elldat, psidat,
                                              num_d, num_g, num_m);

    }
    stopTimer();

    LTIMES_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    LTIMES_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nthreads_per_block(num_m, num_g, 1);
      dim3 nblocks(1, 1, num_z);

      auto ltimes_lambda = [=] __device__ (Index_type z, Index_type g, Index_type m) {

        for (Index_type d = 0; d < num_d; ++d ) {
          LTIMES_BODY;
        }
      };

      auto kernel = lambda_hip_kernel<RAJA::hip_block_z_direct, RAJA::hip_thread_y_direct, RAJA::hip_thread_x_direct, decltype(ltimes_lambda)>;
      hipLaunchKernelGGL(kernel,
        nblocks, nthreads_per_block, 0, 0,
        0, num_z, 0, num_g, 0, num_m, ltimes_lambda);

    }
    stopTimer();

    LTIMES_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    LTIMES_DATA_SETUP_HIP;

    LTIMES_VIEWS_RANGES_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelAsync<
          RAJA::statement::For<1, RAJA::hip_block_z_direct,      //z
            RAJA::statement::For<2, RAJA::hip_thread_y_direct,    //g
              RAJA::statement::For<3, RAJA::hip_thread_x_direct, //m
                RAJA::statement::For<0, RAJA::seq_exec,       //d
                  RAJA::statement::Lambda<0>
                >
              >
            >
          >
        >
      >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(IDRange(0, num_d),
                                                 IZRange(0, num_z),
                                                 IGRange(0, num_g),
                                                 IMRange(0, num_m)),
          [=] __device__ (ID d, IZ z, IG g, IM m) {
          LTIMES_BODY_RAJA;
        });

      }
      stopTimer();

      LTIMES_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n LTIMES : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
