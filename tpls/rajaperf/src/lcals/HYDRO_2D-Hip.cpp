//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

#define HYDRO_2D_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(zadat, m_za, m_array_length); \
  allocAndInitHipDeviceData(zbdat, m_zb, m_array_length); \
  allocAndInitHipDeviceData(zmdat, m_zm, m_array_length); \
  allocAndInitHipDeviceData(zpdat, m_zp, m_array_length); \
  allocAndInitHipDeviceData(zqdat, m_zq, m_array_length); \
  allocAndInitHipDeviceData(zrdat, m_zr, m_array_length); \
  allocAndInitHipDeviceData(zudat, m_zu, m_array_length); \
  allocAndInitHipDeviceData(zvdat, m_zv, m_array_length); \
  allocAndInitHipDeviceData(zzdat, m_zz, m_array_length); \
  allocAndInitHipDeviceData(zroutdat, m_zrout, m_array_length); \
  allocAndInitHipDeviceData(zzoutdat, m_zzout, m_array_length);

#define HYDRO_2D_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_zrout, zroutdat, m_array_length); \
  getHipDeviceData(m_zzout, zzoutdat, m_array_length); \
  deallocHipDeviceData(zadat); \
  deallocHipDeviceData(zbdat); \
  deallocHipDeviceData(zmdat); \
  deallocHipDeviceData(zpdat); \
  deallocHipDeviceData(zqdat); \
  deallocHipDeviceData(zrdat); \
  deallocHipDeviceData(zudat); \
  deallocHipDeviceData(zvdat); \
  deallocHipDeviceData(zzdat); \
  deallocHipDeviceData(zroutdat); \
  deallocHipDeviceData(zzoutdat);

__global__ void hydro_2d1(Real_ptr zadat, Real_ptr zbdat,
                          Real_ptr zpdat, Real_ptr zqdat,
                          Real_ptr zrdat, Real_ptr zmdat,
                          Index_type jn, Index_type kn)
{
   Index_type k = blockIdx.y;
   Index_type j = threadIdx.x;
   if (k > 0 && k < kn-1 && j > 0 && j < jn-1) {
     HYDRO_2D_BODY1;
   }
}

__global__ void hydro_2d2(Real_ptr zudat, Real_ptr zvdat,
                          Real_ptr zadat, Real_ptr zbdat,
                          Real_ptr zzdat, Real_ptr zrdat,
                          Real_type s,
                          Index_type jn, Index_type kn)
{
   Index_type k = blockIdx.y;
   Index_type j = threadIdx.x;
   if (k > 0 && k < kn-1 && j > 0 && j < jn-1) {
     HYDRO_2D_BODY2;
   }
}

__global__ void hydro_2d3(Real_ptr zroutdat, Real_ptr zzoutdat,
                          Real_ptr zrdat, Real_ptr zudat,
                          Real_ptr zzdat, Real_ptr zvdat,
                          Real_type t,
                          Index_type jn, Index_type kn)
{
   Index_type k = blockIdx.y;
   Index_type j = threadIdx.x;
   if (k > 0 && k < kn-1 && j > 0 && j < jn-1) {
     HYDRO_2D_BODY3;
   }
}


void HYDRO_2D::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  HYDRO_2D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    HYDRO_2D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       dim3 nthreads_per_block(jn, 1, 1);
       dim3 nblocks(1, kn, 1);

       hipLaunchKernelGGL((hydro_2d1), dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                                                  zadat, zbdat,
                                                  zpdat, zqdat, zrdat, zmdat,
                                                  jn, kn);

       hipLaunchKernelGGL((hydro_2d2), dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                                                  zudat, zvdat,
                                                  zadat, zbdat, zzdat, zrdat,
                                                  s,
                                                  jn, kn);

       hipLaunchKernelGGL((hydro_2d3), dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                                                  zroutdat, zzoutdat,
                                                  zrdat, zudat, zzdat, zvdat,
                                                  t,
                                                  jn, kn);

    }
    stopTimer();

    HYDRO_2D_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    HYDRO_2D_DATA_SETUP_HIP;

    HYDRO_2D_VIEWS_RAJA;

      using EXECPOL =
        RAJA::KernelPolicy<
          RAJA::statement::HipKernelAsync<
            RAJA::statement::For<0, RAJA::hip_block_y_direct,  // k
              RAJA::statement::For<1, RAJA::hip_thread_x_direct,  // j
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] __device__ (Index_type k, Index_type j) {
        HYDRO_2D_BODY1_RAJA;
      });

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] __device__ (Index_type k, Index_type j) {
        HYDRO_2D_BODY2_RAJA;
      });

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] __device__ (Index_type k, Index_type j) {
        HYDRO_2D_BODY3_RAJA;
      });

    }
    stopTimer();

    HYDRO_2D_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  HYDRO_2D : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
