//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"

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


#define REDUCE3_INT_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(vec, m_vec, iend);

#define REDUCE3_INT_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(vec);


__global__ void reduce3int(Int_ptr vec,
                           Int_ptr vsum, Int_type vsum_init,
                           Int_ptr vmin, Int_type vmin_init,
                           Int_ptr vmax, Int_type vmax_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED( Int_type, psum)
  Int_type* pmin = (Int_type*)&psum[ 1 * blockDim.x ];
  Int_type* pmax = (Int_type*)&psum[ 2 * blockDim.x ];

  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;

  psum[ threadIdx.x ] = vsum_init;
  pmin[ threadIdx.x ] = vmin_init;
  pmax[ threadIdx.x ] = vmax_init;

  for ( ; i < iend ; i += gridDim.x * blockDim.x ) {
    psum[ threadIdx.x ] += vec[ i ];
    pmin[ threadIdx.x ] = RAJA_MIN( pmin[ threadIdx.x ], vec[ i ] );
    pmax[ threadIdx.x ] = RAJA_MAX( pmax[ threadIdx.x ], vec[ i ] );
  }
  __syncthreads();

  for ( i = blockDim.x / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      psum[ threadIdx.x ] += psum[ threadIdx.x + i ];
      pmin[ threadIdx.x ] = RAJA_MIN( pmin[ threadIdx.x ], pmin[ threadIdx.x + i ] );
      pmax[ threadIdx.x ] = RAJA_MAX( pmax[ threadIdx.x ], pmax[ threadIdx.x + i ] );
    }
     __syncthreads();
  }

#if 1 // serialized access to shared data;
  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::hip_atomic>( vsum, psum[ 0 ] );
    RAJA::atomicMin<RAJA::hip_atomic>( vmin, pmin[ 0 ] );
    RAJA::atomicMax<RAJA::hip_atomic>( vmax, pmax[ 0 ] );
  }
#else // this doesn't work due to data races
  if ( threadIdx.x == 0 ) {
    *vsum += psum[ 0 ];
    *vmin = RAJA_MIN( *vmin, pmin[ 0 ] );
    *vmax = RAJA_MAX( *vmax, pmax[ 0 ] );
  }
#endif
}


void REDUCE3_INT::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  REDUCE3_INT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    REDUCE3_INT_DATA_SETUP_HIP;

    Int_ptr vsum;
    allocAndInitHipDeviceData(vsum, &m_vsum_init, 1);
    Int_ptr vmin;
    allocAndInitHipDeviceData(vmin, &m_vmin_init, 1);
    Int_ptr vmax;
    allocAndInitHipDeviceData(vmax, &m_vmax_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(vsum, &m_vsum_init, 1);
      initHipDeviceData(vmin, &m_vmin_init, 1);
      initHipDeviceData(vmax, &m_vmax_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((reduce3int), dim3(grid_size), dim3(block_size), 3*sizeof(Int_type)*block_size, 0, vec,
                                                    vsum, m_vsum_init,
                                                    vmin, m_vmin_init,
                                                    vmax, m_vmax_init,
                                                    iend );

      Int_type lsum;
      Int_ptr plsum = &lsum;
      getHipDeviceData(plsum, vsum, 1);
      m_vsum += lsum;

      Int_type lmin;
      Int_ptr plmin = &lmin;
      getHipDeviceData(plmin, vmin, 1);
      m_vmin = RAJA_MIN(m_vmin, lmin);

      Int_type lmax;
      Int_ptr plmax = &lmax;
      getHipDeviceData(plmax, vmax, 1);
      m_vmax = RAJA_MAX(m_vmax, lmax);

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_HIP;

    deallocHipDeviceData(vsum);
    deallocHipDeviceData(vmin);
    deallocHipDeviceData(vmax);

  } else if ( vid == RAJA_HIP ) {

    REDUCE3_INT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce, Int_type> vsum(m_vsum_init);
      RAJA::ReduceMin<RAJA::hip_reduce, Int_type> vmin(m_vmin_init);
      RAJA::ReduceMax<RAJA::hip_reduce, Int_type> vmax(m_vmax_init);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        REDUCE3_INT_BODY_RAJA;
      });

      m_vsum += static_cast<Int_type>(vsum.get());
      m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
      m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  REDUCE3_INT : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
