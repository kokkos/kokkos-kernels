//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_MIN.hpp"

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


#define FIRST_MIN_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, m_N);

#define FIRST_MIN_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(x);

__global__ void first_min(Real_ptr x,
                          MyMinLoc* dminloc,
                          Index_type iend)
{
  extern __shared__ MyMinLoc minloc[ ];

  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;

  minloc[ threadIdx.x ] = *dminloc;

  for ( ; i < iend ; i += gridDim.x * blockDim.x ) {
    MyMinLoc& mymin = minloc[ threadIdx.x ];
    FIRST_MIN_BODY;
  }
  __syncthreads();

  for ( i = blockDim.x / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      if ( minloc[ threadIdx.x + i].val < minloc[ threadIdx.x ].val ) {
        minloc[ threadIdx.x ] = minloc[ threadIdx.x + i];
      }
    }
     __syncthreads();
  }

  if ( threadIdx.x == 0 ) {
    if ( minloc[ 0 ].val < (*dminloc).val ) {
      *dminloc = minloc[ 0 ];
    }
  }
}


void FIRST_MIN::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  FIRST_MIN_DATA_SETUP;

  if ( vid == Base_HIP ) {

    FIRST_MIN_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       FIRST_MIN_MINLOC_INIT;

       MyMinLoc* dminloc;
       hipErrchk( hipMalloc( (void**)&dminloc, sizeof(MyMinLoc) ) );
       hipErrchk( hipMemcpy( dminloc, &mymin, sizeof(MyMinLoc),
                               hipMemcpyHostToDevice ) );

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       hipLaunchKernelGGL(first_min, grid_size, block_size,
                   sizeof(MyMinLoc)*block_size, 0, x,
                                                   dminloc,
                                                   iend );

       hipErrchk( hipMemcpy( &mymin, dminloc, sizeof(MyMinLoc),
                               hipMemcpyDeviceToHost ) );
       m_minloc = RAJA_MAX(m_minloc, mymin.loc);

       hipErrchk( hipFree( dminloc ) );

    }
    stopTimer();

    FIRST_MIN_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    FIRST_MIN_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::ReduceMinLoc<RAJA::hip_reduce, Real_type, Index_type> loc(
                                                        m_xmin_init, m_initloc);

       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         FIRST_MIN_BODY_RAJA;
       });

       m_minloc = RAJA_MAX(m_minloc, loc.getLoc());

    }
    stopTimer();

    FIRST_MIN_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  FIRST_MIN : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
