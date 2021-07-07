//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "VOL3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define VOL3D_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, m_array_length); \
  allocAndInitCudaDeviceData(y, m_y, m_array_length); \
  allocAndInitCudaDeviceData(z, m_z, m_array_length); \
  allocAndInitCudaDeviceData(vol, m_vol, m_array_length);

#define VOL3D_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_vol, vol, m_array_length); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(z); \
  deallocCudaDeviceData(vol);

__global__ void vol3d(Real_ptr vol,
                      const Real_ptr x0, const Real_ptr x1,
                      const Real_ptr x2, const Real_ptr x3,
                      const Real_ptr x4, const Real_ptr x5,
                      const Real_ptr x6, const Real_ptr x7,
                      const Real_ptr y0, const Real_ptr y1,
                      const Real_ptr y2, const Real_ptr y3,
                      const Real_ptr y4, const Real_ptr y5,
                      const Real_ptr y6, const Real_ptr y7,
                      const Real_ptr z0, const Real_ptr z1,
                      const Real_ptr z2, const Real_ptr z3,
                      const Real_ptr z4, const Real_ptr z5,
                      const Real_ptr z6, const Real_ptr z7,
                      const Real_type vnormq,
                      Index_type ibegin, Index_type iend)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i = ii + ibegin; 
   if (i < iend) {
     VOL3D_BODY;
   }
}


void VOL3D::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = m_domain->fpz;
  const Index_type iend = m_domain->lpz+1;

  VOL3D_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    VOL3D_DATA_SETUP_CUDA;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
 
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      vol3d<<<grid_size, block_size>>>(vol,
                                       x0, x1, x2, x3, x4, x5, x6, x7,
                                       y0, y1, y2, y3, y4, y5, y6, y7,
                                       z0, z1, z2, z3, z4, z5, z6, z7,
                                       vnormq,
                                       ibegin, iend);
 
    }
    stopTimer();
 
    VOL3D_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    VOL3D_DATA_SETUP_CUDA;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
 
      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        VOL3D_BODY;
      });
 
    }
    stopTimer();
 
    VOL3D_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  VOL3D : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
