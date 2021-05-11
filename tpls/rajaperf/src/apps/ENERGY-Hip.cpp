//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define ENERGY_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(e_new, m_e_new, iend); \
  allocAndInitHipDeviceData(e_old, m_e_old, iend); \
  allocAndInitHipDeviceData(delvc, m_delvc, iend); \
  allocAndInitHipDeviceData(p_new, m_p_new, iend); \
  allocAndInitHipDeviceData(p_old, m_p_old, iend); \
  allocAndInitHipDeviceData(q_new, m_q_new, iend); \
  allocAndInitHipDeviceData(q_old, m_q_old, iend); \
  allocAndInitHipDeviceData(work, m_work, iend); \
  allocAndInitHipDeviceData(compHalfStep, m_compHalfStep, iend); \
  allocAndInitHipDeviceData(pHalfStep, m_pHalfStep, iend); \
  allocAndInitHipDeviceData(bvc, m_bvc, iend); \
  allocAndInitHipDeviceData(pbvc, m_pbvc, iend); \
  allocAndInitHipDeviceData(ql_old, m_ql_old, iend); \
  allocAndInitHipDeviceData(qq_old, m_qq_old, iend); \
  allocAndInitHipDeviceData(vnewc, m_vnewc, iend);

#define ENERGY_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_e_new, e_new, iend); \
  getHipDeviceData(m_q_new, q_new, iend); \
  deallocHipDeviceData(e_new); \
  deallocHipDeviceData(e_old); \
  deallocHipDeviceData(delvc); \
  deallocHipDeviceData(p_new); \
  deallocHipDeviceData(p_old); \
  deallocHipDeviceData(q_new); \
  deallocHipDeviceData(q_old); \
  deallocHipDeviceData(work); \
  deallocHipDeviceData(compHalfStep); \
  deallocHipDeviceData(pHalfStep); \
  deallocHipDeviceData(bvc); \
  deallocHipDeviceData(pbvc); \
  deallocHipDeviceData(ql_old); \
  deallocHipDeviceData(qq_old); \
  deallocHipDeviceData(vnewc);

__global__ void energycalc1(Real_ptr e_new, Real_ptr e_old, Real_ptr delvc,
                            Real_ptr p_old, Real_ptr q_old, Real_ptr work,
                            Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY1;
   }
}

__global__ void energycalc2(Real_ptr delvc, Real_ptr q_new,
                            Real_ptr compHalfStep, Real_ptr pHalfStep,
                            Real_ptr e_new, Real_ptr bvc, Real_ptr pbvc,
                            Real_ptr ql_old, Real_ptr qq_old,
                            Real_type rho0,
                            Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY2;
   }
}

__global__ void energycalc3(Real_ptr e_new, Real_ptr delvc,
                            Real_ptr p_old, Real_ptr q_old,
                            Real_ptr pHalfStep, Real_ptr q_new,
                            Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY3;
   }
}

__global__ void energycalc4(Real_ptr e_new, Real_ptr work,
                            Real_type e_cut, Real_type emin,
                            Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY4;
   }
}

__global__ void energycalc5(Real_ptr delvc,
                            Real_ptr pbvc, Real_ptr e_new, Real_ptr vnewc,
                            Real_ptr bvc, Real_ptr p_new,
                            Real_ptr ql_old, Real_ptr qq_old,
                            Real_ptr p_old, Real_ptr q_old,
                            Real_ptr pHalfStep, Real_ptr q_new,
                            Real_type rho0, Real_type e_cut, Real_type emin,
                            Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY5;
   }
}

__global__ void energycalc6(Real_ptr delvc,
                            Real_ptr pbvc, Real_ptr e_new, Real_ptr vnewc,
                            Real_ptr bvc, Real_ptr p_new,
                            Real_ptr q_new,
                            Real_ptr ql_old, Real_ptr qq_old,
                            Real_type rho0, Real_type q_cut,
                            Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY6;
   }
}


void ENERGY::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ENERGY_DATA_SETUP;

  if ( vid == Base_HIP ) {

    ENERGY_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

       hipLaunchKernelGGL((energycalc1), dim3(grid_size), dim3(block_size), 0, 0,  e_new, e_old, delvc,
                                               p_old, q_old, work,
                                               iend );

       hipLaunchKernelGGL((energycalc2), dim3(grid_size), dim3(block_size), 0, 0,  delvc, q_new,
                                               compHalfStep, pHalfStep,
                                               e_new, bvc, pbvc,
                                               ql_old, qq_old,
                                               rho0,
                                               iend );

       hipLaunchKernelGGL((energycalc3), dim3(grid_size), dim3(block_size), 0, 0,  e_new, delvc,
                                               p_old, q_old,
                                               pHalfStep, q_new,
                                               iend );

       hipLaunchKernelGGL((energycalc4), dim3(grid_size), dim3(block_size), 0, 0,  e_new, work,
                                               e_cut, emin,
                                               iend );

       hipLaunchKernelGGL((energycalc5), dim3(grid_size), dim3(block_size), 0, 0,  delvc,
                                               pbvc, e_new, vnewc,
                                               bvc, p_new,
                                               ql_old, qq_old,
                                               p_old, q_old,
                                               pHalfStep, q_new,
                                               rho0, e_cut, emin,
                                               iend );

       hipLaunchKernelGGL((energycalc6), dim3(grid_size), dim3(block_size), 0, 0,  delvc,
                                               pbvc, e_new, vnewc,
                                               bvc, p_new,
                                               q_new,
                                               ql_old, qq_old,
                                               rho0, q_cut,
                                               iend );

    }
    stopTimer();

    ENERGY_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    ENERGY_DATA_SETUP_HIP;

    const bool async = true;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RAJA::forall< RAJA::hip_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY1;
        });

        RAJA::forall< RAJA::hip_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY2;
        });

        RAJA::forall< RAJA::hip_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY3;
        });

        RAJA::forall< RAJA::hip_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY4;
        });

        RAJA::forall< RAJA::hip_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY5;
        });

        RAJA::forall< RAJA::hip_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY6;
        });

      });  // end sequential region (for single-source code) 

    }
    stopTimer();

    ENERGY_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  ENERGY : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
