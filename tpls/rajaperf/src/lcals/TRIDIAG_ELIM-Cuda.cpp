//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAG_ELIM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define TRIDIAG_ELIM_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(xout, m_xout, m_N); \
  allocAndInitCudaDeviceData(xin, m_xin, m_N); \
  allocAndInitCudaDeviceData(y, m_y, m_N); \
  allocAndInitCudaDeviceData(z, m_z, m_N);

#define TRIDIAG_ELIM_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_xout, xout, m_N); \
  deallocCudaDeviceData(xout); \
  deallocCudaDeviceData(xin); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(z);

__global__ void eos(Real_ptr xout, Real_ptr xin, Real_ptr y, Real_ptr z,
                    Index_type N) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i > 0 && i < N) {
     TRIDIAG_ELIM_BODY; 
   }
}


void TRIDIAG_ELIM::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = m_N;

  TRIDIAG_ELIM_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    TRIDIAG_ELIM_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       eos<<<grid_size, block_size>>>( xout, xin, y, z,
                                       iend ); 

    }
    stopTimer();

    TRIDIAG_ELIM_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    TRIDIAG_ELIM_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         TRIDIAG_ELIM_BODY;
       });

    }
    stopTimer();

    TRIDIAG_ELIM_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  TRIDIAG_ELIM : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
