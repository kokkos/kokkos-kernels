/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef CAMP_DEFINES_HPP
#define CAMP_DEFINES_HPP

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

namespace camp
{

#if defined(__NVCC__)
#define CAMP_ALLOW_UNUSED_LOCAL(X) sink(X)
#else
#define CAMP_ALLOW_UNUSED_LOCAL(X) (void)(X)
#endif

#if defined(__clang__)
#define CAMP_COMPILER_CLANG
#elif defined(__INTEL_COMPILER)
#define CAMP_COMPILER_INTEL
#elif defined(__xlc__)
#define CAMP_COMPILER_XLC
#elif defined(__PGI)
#define CAMP_COMPILER_PGI
#elif defined(_WIN32)
#define CAMP_COMPILER_MSVC
#elif defined(__GNUC__)
#define RAJA_COMPILER_GNU
#else
#pragma warn("Unknown compiler!")
#endif

// detect empty_bases for MSVC
#ifndef __has_declspec_attribute
#define __has_declspec_attribute(__x) 0
#endif
#if defined(CAMP_COMPILER_MSVC) || __has_declspec_attribute(empty_bases)
#define CAMP_EMPTY_BASES __declspec(empty_bases)
#else
#define CAMP_EMPTY_BASES
#endif

#if defined(__cpp_constexpr) && __cpp_constexpr >= 201304
#define CAMP_HAS_CONSTEXPR14
#define CAMP_CONSTEXPR14 constexpr
#else
#define CAMP_CONSTEXPR14
#endif

// define host device macros
#define CAMP_HIP_HOST_DEVICE

#if defined(__CUDACC__)
#define CAMP_DEVICE __device__
#define CAMP_HOST_DEVICE __host__ __device__
#define CAMP_HAVE_CUDA 1

#if defined(__NVCC__)
#if defined(_WIN32)  // windows is non-compliant, yay
#define CAMP_SUPPRESS_HD_WARN __pragma(nv_exec_check_disable)
#else
#define CAMP_SUPPRESS_HD_WARN _Pragma("nv_exec_check_disable")
#endif
#else // only nvcc supports this pragma
#define CAMP_SUPPRESS_HD_WARN
#endif

#elif defined(__HIPCC__)
#define CAMP_DEVICE __device__
#define CAMP_HOST_DEVICE __host__ __device__
#define CAMP_HAVE_HIP 1
#undef CAMP_HIP_HOST_DEVICE
#define CAMP_HIP_HOST_DEVICE __host__ __device__
#define CAMP_HAVE_HIP 1

#define CAMP_SUPPRESS_HD_WARN

#else
#define CAMP_DEVICE
#define CAMP_HOST_DEVICE
#define CAMP_SUPPRESS_HD_WARN
#endif

#if _OPENMP >= 201511
#define CAMP_HAVE_OMP_OFFLOAD 1
#endif

// This works for:
//   clang
//   nvcc 10 and higher using clang as a host compiler
//   MSVC 1911... and higher, see check below
//   XL C++ at least back to 16.1.0, possibly farther
#define CAMP_USE_MAKE_INTEGER_SEQ 0
#define CAMP_USE_TYPE_PACK_ELEMENT 0
#if defined(_MSC_FULL_VER) && _MSC_FULL_VER >= 191125507
  // __has_builtin exists but does not always expose this
#undef CAMP_USE_MAKE_INTEGER_SEQ
#define CAMP_USE_MAKE_INTEGER_SEQ 1
  // __type_pack_element remains unsupported
#elif defined(__has_builtin)
#if __has_builtin(__make_integer_seq) && ((!defined(__NVCC__) || __CUDACC_VER_MAJOR >= 10))
#undef CAMP_USE_MAKE_INTEGER_SEQ
#define CAMP_USE_MAKE_INTEGER_SEQ 1
#undef CAMP_USE_TYPE_PACK_ELEMENT
#define CAMP_USE_TYPE_PACK_ELEMENT 1
#endif
#endif

// This works for:
//   GCC >= 8
//   intel 19+ in GCC 8 or higher mode
//   nvcc 10+ in GCC 8 or higher mode
//   PGI 19+ in GCC 8 or higher mode
#if __GNUC__ >= 8 && (\
    /* intel compiler in gcc 8+ mode */ \
    ((!defined(__INTEL_COMPILER)) || __INTEL_COMPILER >= 1900) \
    /* nvcc in gcc 8+ mode */ \
  ||((!defined(__NVCC__)) || __CUDACC_VER_MAJOR >= 10) \
  ||((!defined(__PGIC__)) || __PGIC__ >= 19) \
    )
#define CAMP_USE_INTEGER_PACK 1
#else
#define CAMP_USE_INTEGER_PACK 0
#endif

// libstdc++ from GCC below version 5 lacks the type trait
#if defined(__GLIBCXX__) && (__GLIBCXX__ < 20150422 || __GNUC__ < 5)
#define CAMP_HAS_IS_TRIVIALLY_COPY_CONSTRUCTIBLE 0
#else
#define CAMP_HAS_IS_TRIVIALLY_COPY_CONSTRUCTIBLE 1
#endif

// distinguish between the use of 'THE' default stream of a platform
// or 'A' general platform stream created by camp Resource
#ifndef CAMP_USE_PLATFORM_DEFAULT_STREAM
#define CAMP_USE_PLATFORM_DEFAULT_STREAM 0
#endif



// Types
using idx_t = std::ptrdiff_t;
using nullptr_t = decltype(nullptr);

// Helper macros
// TODO: -> CAMP_MAKE_LAMBDA_CONSUMER
#define CAMP_MAKE_L(X)                                             \
  template <typename Lambda, typename... Rest>                     \
  struct X##_l {                                                   \
    using type = typename X<Lambda::template expr, Rest...>::type; \
  }

}  // namespace camp

#endif
