//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ATOMIC_PI.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define ATOMIC_PI_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(pi, m_pi, 1);

#define ATOMIC_PI_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(pi);

__global__ void atomic_pi(Real_ptr pi,
                          Real_type dx,
                          Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     double x = (double(i) + 0.5) * dx;
     RAJA::atomicAdd<RAJA::hip_atomic>(pi, dx / (1.0 + x * x));
   }
}


void ATOMIC_PI::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ATOMIC_PI_DATA_SETUP;

  if ( vid == Base_HIP ) {

    ATOMIC_PI_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(pi, &m_pi_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL(atomic_pi,grid_size, block_size, 0, 0, pi, dx, iend );

      getHipDeviceData(m_pi, pi, 1);
      *m_pi *= 4.0;

    }
    stopTimer();

    ATOMIC_PI_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    ATOMIC_PI_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(pi, &m_pi_init, 1);

      auto atomic_pi_lambda = [=] __device__ (Index_type i) {
          double x = (double(i) + 0.5) * dx;
          RAJA::atomicAdd<RAJA::hip_atomic>(pi, dx / (1.0 + x * x));
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL(lambda_hip_forall<decltype(atomic_pi_lambda)>,
          grid_size, block_size, 0, 0, ibegin, iend, atomic_pi_lambda);

      getHipDeviceData(m_pi, pi, 1);
      *m_pi *= 4.0;

    }
    stopTimer();

    ATOMIC_PI_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    ATOMIC_PI_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(pi, &m_pi_init, 1);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          double x = (double(i) + 0.5) * dx;
          RAJA::atomicAdd<RAJA::hip_atomic>(pi, dx / (1.0 + x * x));
      });

      getHipDeviceData(m_pi, pi, 1);
      *m_pi *= 4.0;

    }
    stopTimer();

    ATOMIC_PI_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  ATOMIC_PI : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
