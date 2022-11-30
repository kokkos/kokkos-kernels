/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
 */

#ifndef _KOKKOSKERNELS_ALIGNUTILS_HPP
#define _KOKKOSKERNELS_ALIGNUTILS_HPP

#if defined(KOKKOS_COMPILER_MSVC)
#include <intrin.h>
#endif

namespace KokkosKernels {

namespace Impl {

/* 
    For CUDA device code, __builtin_assume_aligned was added in CUDA 11.2
    When compiled with clang, it works on all supported clang compilers

    On the host, clang and gcc use __builtin_assume_aligned
    Intel's compilers use __assume_aligned
*/


#if defined(__CUDA_ARCH__)

#if ((__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR >= 2)) || defined(__clang__) 
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T *assume_aligned(T *ptr, size_t align) {
  return static_cast<T*>(__builtin_assume_aligned(ptr, align));
}
#else // CUDA < 11.2 and not clang
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T *assume_aligned(T *ptr, size_t /*align*/) {
  return ptr; // no-op
}
#endif

#else // __CUDA_ARCH__

/*! \brief Return ptr, and may allow the compiler to assume that the returned pointer is at least align-byte aligned

    \tparam T the type of the pointer
    \param ptr The pointer to return
    \param n the alignment the compiler may assume the returned pointer has

    This is best-effort, and may be a no-op on certain compilers.
*/
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T *assume_aligned(T *ptr, size_t n) {
#if defined(KOKKOS_COMPILER_INTEL)
  __assume_aligned(ptr, n);
  return ptr;
#elif defined(KOKKOS_COMPILER_GNU) || defined(KOKKOS_COMPILER_CLANG)
  return static_cast<T*>(__builtin_assume_aligned(ptr, n));
#elif defined(KOKKOS_COMPILER_MSVC)
  __assume(std::uintptr_t(ptr) % n == 0); // can the compiler see this through a function return?
  return ptr;
#else
#warning KokkosKernels::Impl::assume_aligned is not supported for this compiler. Please report this
#endif
}

#endif // __CUDA_ARCH__

template <typename InPtr, typename T>
KOKKOS_INLINE_FUNCTION T *alignPtr(InPtr p) {
  // ugly but computationally free and the "right" way to do this in C++
  std::uintptr_t ptrVal = reinterpret_cast<std::uintptr_t>(p);
  // ptrVal + (align - 1) lands inside the next valid aligned scalar_t,
  // and the mask produces the start of that scalar_t.
  return reinterpret_cast<T *>((ptrVal + alignof(T) - 1) & (~(alignof(T) - 1)));
}

}  // namespace Impl
}  // namespace KokkosKernels

#endif
