/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/
#ifndef TFTK_TFTK_UTIL_TFTK_UTIL_TFTK_MACROS_H_
#define TFTK_TFTK_UTIL_TFTK_UTIL_TFTK_MACROS_H_

#include <Kokkos_Macros.hpp>

// clang-format off
#ifdef KOKKOS_ENABLE_CUDA
  #define TFTK_ENABLE_GPU
  #define TFTK_ENABLE_CUDA
  #define TFTK_DISTINCT_DEVICE_SPACE
  #define TFTK_DISTINCT_UVM_SPACE
#elif defined(KOKKOS_ENABLE_HIP)
  #define TFTK_ENABLE_GPU
  #define TFTK_ENABLE_HIP
  #define TFTK_DISTINCT_DEVICE_SPACE
#else
  #define TFTK_DEVICE_SAFE_STD
  #ifndef STK_HAVE_NO_SIMD
    #define TFTK_ENABLE_SIMD
  #endif
#endif

#ifdef KOKKOS_ENABLE_OPENMP
  #define TFTK_ENABLE_OPENMP
#endif
// clang-format on

#endif /* TFTK_TFTK_UTIL_TFTK_UTIL_TFTK_MACROS_H_ */
