//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_1D.hpp"

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


#define HYDRO_1D_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, m_array_length); \
  allocAndInitCudaDeviceData(y, m_y, m_array_length); \
  allocAndInitCudaDeviceData(z, m_z, m_array_length);

#define HYDRO_1D_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x, x, m_array_length); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(z); \

__global__ void hydro_1d(Real_ptr x, Real_ptr y, Real_ptr z,
                         Real_type q, Real_type r, Real_type t,
                         Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     HYDRO_1D_BODY; 
   }
}


void HYDRO_1D::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  HYDRO_1D_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    HYDRO_1D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       hydro_1d<<<grid_size, block_size>>>( x, y, z,
                                            q, r, t,
                                            iend ); 

    }
    stopTimer();

    HYDRO_1D_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    HYDRO_1D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         HYDRO_1D_BODY;
       });

    }
    stopTimer();

    HYDRO_1D_DATA_TEARDOWN_CUDA;

  } else { 
     std::cout << "\n  HYDRO_1D : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
