//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "EOS.hpp"

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


#define EOS_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, m_array_length); \
  allocAndInitCudaDeviceData(y, m_y, m_array_length); \
  allocAndInitCudaDeviceData(z, m_z, m_array_length); \
  allocAndInitCudaDeviceData(u, m_u, m_array_length);

#define EOS_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x, x, m_array_length); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(z); \
  deallocCudaDeviceData(u);

__global__ void eos(Real_ptr x, Real_ptr y, Real_ptr z, Real_ptr u,
                    Real_type q, Real_type r, Real_type t,
                    Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     EOS_BODY; 
   }
}


void EOS::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  EOS_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    EOS_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       eos<<<grid_size, block_size>>>( x, y, z, u, 
                                       q, r, t,
                                       iend ); 

    }
    stopTimer();

    EOS_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    EOS_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         EOS_BODY;
       });

    }
    stopTimer();

    EOS_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  EOS : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
