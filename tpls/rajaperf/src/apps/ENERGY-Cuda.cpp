//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define ENERGY_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(e_new, m_e_new, iend); \
  allocAndInitCudaDeviceData(e_old, m_e_old, iend); \
  allocAndInitCudaDeviceData(delvc, m_delvc, iend); \
  allocAndInitCudaDeviceData(p_new, m_p_new, iend); \
  allocAndInitCudaDeviceData(p_old, m_p_old, iend); \
  allocAndInitCudaDeviceData(q_new, m_q_new, iend); \
  allocAndInitCudaDeviceData(q_old, m_q_old, iend); \
  allocAndInitCudaDeviceData(work, m_work, iend); \
  allocAndInitCudaDeviceData(compHalfStep, m_compHalfStep, iend); \
  allocAndInitCudaDeviceData(pHalfStep, m_pHalfStep, iend); \
  allocAndInitCudaDeviceData(bvc, m_bvc, iend); \
  allocAndInitCudaDeviceData(pbvc, m_pbvc, iend); \
  allocAndInitCudaDeviceData(ql_old, m_ql_old, iend); \
  allocAndInitCudaDeviceData(qq_old, m_qq_old, iend); \
  allocAndInitCudaDeviceData(vnewc, m_vnewc, iend);

#define ENERGY_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_e_new, e_new, iend); \
  getCudaDeviceData(m_q_new, q_new, iend); \
  deallocCudaDeviceData(e_new); \
  deallocCudaDeviceData(e_old); \
  deallocCudaDeviceData(delvc); \
  deallocCudaDeviceData(p_new); \
  deallocCudaDeviceData(p_old); \
  deallocCudaDeviceData(q_new); \
  deallocCudaDeviceData(q_old); \
  deallocCudaDeviceData(work); \
  deallocCudaDeviceData(compHalfStep); \
  deallocCudaDeviceData(pHalfStep); \
  deallocCudaDeviceData(bvc); \
  deallocCudaDeviceData(pbvc); \
  deallocCudaDeviceData(ql_old); \
  deallocCudaDeviceData(qq_old); \
  deallocCudaDeviceData(vnewc);

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


void ENERGY::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ENERGY_DATA_SETUP;

  if ( vid == Base_CUDA ) {
    
    ENERGY_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

       energycalc1<<<grid_size, block_size>>>( e_new, e_old, delvc,
                                               p_old, q_old, work,
                                               iend );

       energycalc2<<<grid_size, block_size>>>( delvc, q_new,
                                               compHalfStep, pHalfStep,
                                               e_new, bvc, pbvc,
                                               ql_old, qq_old,
                                               rho0,
                                               iend );

       energycalc3<<<grid_size, block_size>>>( e_new, delvc,
                                               p_old, q_old,
                                               pHalfStep, q_new,
                                               iend );

       energycalc4<<<grid_size, block_size>>>( e_new, work,
                                               e_cut, emin,
                                               iend );

       energycalc5<<<grid_size, block_size>>>( delvc,
                                               pbvc, e_new, vnewc,
                                               bvc, p_new,
                                               ql_old, qq_old,
                                               p_old, q_old,
                                               pHalfStep, q_new,
                                               rho0, e_cut, emin,
                                               iend );

       energycalc6<<<grid_size, block_size>>>( delvc,
                                               pbvc, e_new, vnewc,
                                               bvc, p_new,
                                               q_new,
                                               ql_old, qq_old,
                                               rho0, q_cut,
                                               iend );

    }
    stopTimer();

    ENERGY_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    ENERGY_DATA_SETUP_CUDA;

    const bool async = true;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if CUDART_VERSION >= 9000
// Defining an extended __device__ lambda inside inside another lambda
// was not supported until CUDA 9.x
      RAJA::region<RAJA::seq_region>( [=]() {
#endif

        RAJA::forall< RAJA::cuda_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY1;
        });

        RAJA::forall< RAJA::cuda_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY2;
        });
 
        RAJA::forall< RAJA::cuda_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY3;
        });
 
        RAJA::forall< RAJA::cuda_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY4;
        });
 
        RAJA::forall< RAJA::cuda_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY5;
        });
 
        RAJA::forall< RAJA::cuda_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY6;
        });

#if CUDART_VERSION >= 9000
      }); // end sequential region (for single-source code)
#endif

    }
    stopTimer();

    ENERGY_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  ENERGY : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
