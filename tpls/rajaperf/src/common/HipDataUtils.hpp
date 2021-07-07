//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Methods for HIP kernel data allocation, initialization, and deallocation.
///


#ifndef RAJAPerf_HipDataUtils_HPP
#define RAJAPerf_HipDataUtils_HPP

#include "RPTypes.hpp"

#if defined(RAJA_ENABLE_HIP)


#include "RAJA/policy/hip/raja_hiperrchk.hpp"


namespace rajaperf
{

/*!
 * \brief Simple forall hip kernel that runs a lambda.
 */
template < typename Lambda >
__global__ void lambda_hip_forall(Index_type ibegin, Index_type iend, Lambda body)
{
   Index_type i = ibegin + blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     body(i);
   }
}

/*!
 * \brief Getters for hip kernel indices.
 */
template < typename Index >
__device__ inline Index_type lambda_hip_get_index();

template < >
__device__ inline Index_type lambda_hip_get_index<RAJA::hip_thread_x_direct>() {
  return threadIdx.x;
}
template < >
__device__ inline Index_type lambda_hip_get_index<RAJA::hip_thread_y_direct>() {
  return threadIdx.y;
}
template < >
__device__ inline Index_type lambda_hip_get_index<RAJA::hip_thread_z_direct>() {
  return threadIdx.z;
}

template < >
__device__ inline Index_type lambda_hip_get_index<RAJA::hip_block_x_direct>() {
  return blockIdx.x;
}
template < >
__device__ inline Index_type lambda_hip_get_index<RAJA::hip_block_y_direct>() {
  return blockIdx.y;
}
template < >
__device__ inline Index_type lambda_hip_get_index<RAJA::hip_block_z_direct>() {
  return blockIdx.z;
}

/*!
 * \brief Simple multi-dimensional hip kernel that runs a lambda.
 */
template < typename I_Index, typename J_Index, typename Lambda >
__global__ void lambda_hip_kernel(Index_type ibegin, Index_type iend,
                                  Index_type jbegin, Index_type jend,
                                  Lambda body)
{
  Index_type i = ibegin + lambda_hip_get_index<I_Index>();
  Index_type j = jbegin + lambda_hip_get_index<J_Index>();
  if (i < iend) {
    if (j < jend) {
      body(i, j);
    }
  }
}
///
template < typename I_Index, typename J_Index, typename K_Index, typename Lambda >
__global__ void lambda_hip_kernel(Index_type ibegin, Index_type iend,
                                  Index_type jbegin, Index_type jend,
                                  Index_type kbegin, Index_type kend,
                                  Lambda body)
{
  Index_type i = ibegin + lambda_hip_get_index<I_Index>();
  Index_type j = jbegin + lambda_hip_get_index<J_Index>();
  Index_type k = kbegin + lambda_hip_get_index<K_Index>();
  if (i < iend) {
    if (j < jend) {
      if (k < kend) {
        body(i, j, k);
      }
    }
  }
}

/*!
 * \brief Copy given hptr (host) data to HIP device (dptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void initHipDeviceData(T& dptr, const T hptr, int len)
{
  hipErrchk( hipMemcpy( dptr, hptr,
                          len * sizeof(typename std::remove_pointer<T>::type),
                          hipMemcpyHostToDevice ) );

  incDataInitCount();
}

/*!
 * \brief Allocate HIP device data array (dptr) and copy given hptr (host)
 * data to device array.
 */
template <typename T>
void allocAndInitHipDeviceData(T& dptr, const T hptr, int len)
{
  hipErrchk( hipMalloc( (void**)&dptr,
              len * sizeof(typename std::remove_pointer<T>::type) ) );

  initHipDeviceData(dptr, hptr, len);
}

/*!
 * \brief Copy given dptr (HIP device) data to host (hptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void getHipDeviceData(T& hptr, const T dptr, int len)
{
  hipErrchk( hipMemcpy( hptr, dptr,
              len * sizeof(typename std::remove_pointer<T>::type),
              hipMemcpyDeviceToHost ) );
}

/*!
 * \brief Free device data array.
 */
template <typename T>
void deallocHipDeviceData(T& dptr)
{
  hipErrchk( hipFree( dptr ) );
  dptr = 0;
}


}  // closing brace for rajaperf namespace

#endif // RAJA_ENABLE_HIP

#endif  // closing endif for header file include guard

