//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf
{
namespace apps
{

#define USE_HIP_CONSTANT_MEMORY
// #undef USE_HIP_CONSTANT_MEMORY

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#if defined(USE_HIP_CONSTANT_MEMORY)

__constant__ Real_type coeff[FIR_COEFFLEN];

#define FIR_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(in, m_in, getRunSize()); \
  allocAndInitHipDeviceData(out, m_out, getRunSize()); \
  hipMemcpyToSymbol(HIP_SYMBOL(coeff), coeff_array, FIR_COEFFLEN * sizeof(Real_type), 0, hipMemcpyHostToDevice);


#define FIR_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_out, out, getRunSize()); \
  deallocHipDeviceData(in); \
  deallocHipDeviceData(out);

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

#define FIR_DATA_SETUP_HIP \
  Real_ptr coeff; \
\
  allocAndInitHipDeviceData(in, m_in, getRunSize()); \
  allocAndInitHipDeviceData(out, m_out, getRunSize()); \
  Real_ptr tcoeff = &coeff_array[0]; \
  allocAndInitHipDeviceData(coeff, tcoeff, FIR_COEFFLEN);


#define FIR_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_out, out, getRunSize()); \
  deallocHipDeviceData(in); \
  deallocHipDeviceData(out); \
  deallocHipDeviceData(coeff);

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


void FIR::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize() - m_coefflen;

  FIR_DATA_SETUP;

  if ( vid == Base_HIP ) {

    FIR_COEFF;

    FIR_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

#if defined(USE_HIP_CONSTANT_MEMORY)
       hipLaunchKernelGGL((fir), dim3(grid_size), dim3(block_size), 0, 0,  out, in,
                                       coefflen,
                                       iend );
#else
       hipLaunchKernelGGL((fir), dim3(grid_size), dim3(block_size), 0, 0,  out, in,
                                       coeff,
                                       coefflen,
                                       iend );
#endif

    }
    stopTimer();

    FIR_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    FIR_COEFF;

    FIR_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         FIR_BODY;
       });

    }
    stopTimer();

    FIR_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  FIR : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
