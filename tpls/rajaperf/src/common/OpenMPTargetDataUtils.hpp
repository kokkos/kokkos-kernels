//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Methods for openmp target kernel data allocation, initialization, 
/// and deallocation.
///


#ifndef RAJAPerf_OpenMPTargetDataUtils_HPP
#define RAJAPerf_OpenMPTargetDataUtils_HPP

#include "RPTypes.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)


namespace rajaperf
{

/*!
 * \brief Copy given hptr (host) data to device (dptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void initOpenMPDeviceData(T& dptr, const T hptr, int len, 
                          int did, int hid)
{
  omp_target_memcpy( dptr, hptr, 
                     len * sizeof(typename std::remove_pointer<T>::type),
                     0, 0, did, hid ); 

  incDataInitCount();
}

/*!
 * \brief Allocate device data array (dptr) and copy given hptr (host) 
 * data to device array.
 */
template <typename T>
void allocAndInitOpenMPDeviceData(T& dptr, const T hptr, int len,
                                  int did, int hid)
{
  dptr = static_cast<T>( omp_target_alloc(
                         len * sizeof(typename std::remove_pointer<T>::type), 
                         did) );

  initOpenMPDeviceData(dptr, hptr, len, did, hid);
}

/*!
 * \brief Copy given device ptr (dptr) data to host ptr (hptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void getOpenMPDeviceData(T& hptr, const T dptr, int len, int hid, int did)
{
  omp_target_memcpy( hptr, dptr, 
                     len * sizeof(typename std::remove_pointer<T>::type),
                     0, 0, hid, did );
}

/*!
 * \brief Free device data array.
 */
template <typename T>
void deallocOpenMPDeviceData(T& dptr, int did)
{
  omp_target_free( dptr, did );
  dptr = 0;
}


}  // closing brace for rajaperf namespace

#endif // RAJA_ENABLE_TARGET_OPENMP

#endif  // closing endif for header file include guard
