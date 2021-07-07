//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

//
// Function used in TRAP_INT loop.
//
RAJA_INLINE
RAJA_DEVICE
Real_type trap_int_func(Real_type x,
                        Real_type y,
                        Real_type xp,
                        Real_type yp)
{
   Real_type denom = (x - xp)*(x - xp) + (y - yp)*(y - yp);
   denom = 1.0/sqrt(denom);
   return denom;
}


  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define TRAP_INT_DATA_SETUP_HIP // nothing to do here...

#define TRAP_INT_DATA_TEARDOWN_HIP // nothing to do here...


__global__ void trapint(Real_type x0, Real_type xp,
                        Real_type y, Real_type yp,
                        Real_type h,
                        Real_ptr sumx,
                        Index_type iend)
{
  HIP_DYNAMIC_SHARED( Real_type, psumx)

  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;

  psumx[ threadIdx.x ] = 0.0;
  for ( ; i < iend ; i += gridDim.x * blockDim.x ) {
    Real_type x = x0 + i*h;
    Real_type val = trap_int_func(x, y, xp, yp);
    psumx[ threadIdx.x ] += val;
  }
  __syncthreads();

  for ( i = blockDim.x / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      psumx[ threadIdx.x ] += psumx[ threadIdx.x + i ];
    }
     __syncthreads();
  }

#if 1 // serialized access to shared data;
  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::hip_atomic>( sumx, psumx[ 0 ] );
  }
#else // this doesn't work due to data races
  if ( threadIdx.x == 0 ) {
    *sumx += psumx[ 0 ];
  }
#endif

}


void TRAP_INT::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  TRAP_INT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    TRAP_INT_DATA_SETUP_HIP;

    Real_ptr sumx;
    allocAndInitHipDeviceData(sumx, &m_sumx_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(sumx, &m_sumx_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((trapint), dim3(grid_size), dim3(block_size), sizeof(Real_type)*block_size, 0, x0, xp,
                                                y, yp,
                                                h,
                                                sumx,
                                                iend);

      Real_type lsumx;
      Real_ptr plsumx = &lsumx;
      getHipDeviceData(plsumx, sumx, 1);
      m_sumx += lsumx * h;

    }
    stopTimer();

    deallocHipDeviceData(sumx);

    TRAP_INT_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    TRAP_INT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce, Real_type> sumx(m_sumx_init);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        TRAP_INT_BODY;
      });

      m_sumx += static_cast<Real_type>(sumx.get()) * h;

    }
    stopTimer();

    TRAP_INT_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  TRAP_INT : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
