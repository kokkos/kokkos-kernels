//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAG_ELIM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define TRIDIAG_ELIM_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(xout, m_xout, m_N); \
  allocAndInitHipDeviceData(xin, m_xin, m_N); \
  allocAndInitHipDeviceData(y, m_y, m_N); \
  allocAndInitHipDeviceData(z, m_z, m_N);

#define TRIDIAG_ELIM_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_xout, xout, m_N); \
  deallocHipDeviceData(xout); \
  deallocHipDeviceData(xin); \
  deallocHipDeviceData(y); \
  deallocHipDeviceData(z);

__global__ void eos(Real_ptr xout, Real_ptr xin, Real_ptr y, Real_ptr z,
                    Index_type N)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i > 0 && i < N) {
     TRIDIAG_ELIM_BODY;
   }
}


void TRIDIAG_ELIM::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = m_N;

  TRIDIAG_ELIM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    TRIDIAG_ELIM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       hipLaunchKernelGGL(eos, grid_size, block_size, 0, 0, xout, xin, y, z,
                                       iend );

    }
    stopTimer();

    TRIDIAG_ELIM_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    TRIDIAG_ELIM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         TRIDIAG_ELIM_BODY;
       });

    }
    stopTimer();

    TRIDIAG_ELIM_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  TRIDIAG_ELIM : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
