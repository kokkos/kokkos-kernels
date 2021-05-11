//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "IF_QUAD.hpp"

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


#define IF_QUAD_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(a, m_a, iend); \
  allocAndInitHipDeviceData(b, m_b, iend); \
  allocAndInitHipDeviceData(c, m_c, iend); \
  allocAndInitHipDeviceData(x1, m_x1, iend); \
  allocAndInitHipDeviceData(x2, m_x2, iend);

#define IF_QUAD_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_x1, x1, iend); \
  getHipDeviceData(m_x2, x2, iend); \
  deallocHipDeviceData(a); \
  deallocHipDeviceData(b); \
  deallocHipDeviceData(c); \
  deallocHipDeviceData(x1); \
  deallocHipDeviceData(x2);

__global__ void ifquad(Real_ptr x1, Real_ptr x2,
                       Real_ptr a, Real_ptr b, Real_ptr c,
                       Index_type iend)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < iend) {
    IF_QUAD_BODY;
  }
}


void IF_QUAD::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  IF_QUAD_DATA_SETUP;

  if ( vid == Base_HIP ) {

    IF_QUAD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((ifquad), dim3(grid_size), dim3(block_size), 0, 0,  x1, x2, a, b, c,
                                          iend );

    }
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    IF_QUAD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto ifquad_lambda = [=] __device__ (Index_type i) {
        IF_QUAD_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL(lambda_hip_forall<decltype(ifquad_lambda)>,
        grid_size, block_size, 0, 0, ibegin, iend, ifquad_lambda);

    }
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    IF_QUAD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        IF_QUAD_BODY;
      });

    }
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  IF_QUAD : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
