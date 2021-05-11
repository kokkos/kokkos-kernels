//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define USE_CUDA_CONSTANT_MEMORY
//#undef USE_CUDA_CONSTANT_MEMORY

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#if defined(USE_CUDA_CONSTANT_MEMORY)

__constant__ Real_type coeff[FIR_COEFFLEN];

#define FIR_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(in, m_in, getRunSize()); \
  allocAndInitCudaDeviceData(out, m_out, getRunSize()); \
  cudaMemcpyToSymbol(coeff, coeff_array, FIR_COEFFLEN * sizeof(Real_type));


#define FIR_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_out, out, getRunSize()); \
  deallocCudaDeviceData(in); \
  deallocCudaDeviceData(out);

__global__ void fir(Real_ptr out, Real_ptr in,
                    const Index_type coefflen,
                    Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     FIR_BODY;
   }
}

#else  // use global memry for coefficients 

#define FIR_DATA_SETUP_CUDA \
  Real_ptr coeff; \
\
  allocAndInitCudaDeviceData(in, m_in, getRunSize()); \
  allocAndInitCudaDeviceData(out, m_out, getRunSize()); \
  Real_ptr tcoeff = &coeff_array[0]; \
  allocAndInitCudaDeviceData(coeff, tcoeff, FIR_COEFFLEN);


#define FIR_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_out, out, getRunSize()); \
  deallocCudaDeviceData(in); \
  deallocCudaDeviceData(out); \
  deallocCudaDeviceData(coeff);

__global__ void fir(Real_ptr out, Real_ptr in,
                    Real_ptr coeff,
                    const Index_type coefflen,
                    Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     FIR_BODY;
   }
}

#endif 


void FIR::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize() - m_coefflen;

  FIR_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    FIR_COEFF;

    FIR_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

#if defined(USE_CUDA_CONSTANT_MEMORY)
       fir<<<grid_size, block_size>>>( out, in,
                                       coefflen,
                                       iend );
#else
       fir<<<grid_size, block_size>>>( out, in,
                                       coeff,
                                       coefflen,
                                       iend );
#endif

    }
    stopTimer();

    FIR_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    FIR_COEFF;

    FIR_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         FIR_BODY;
       });

    }
    stopTimer();

    FIR_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  FIR : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
