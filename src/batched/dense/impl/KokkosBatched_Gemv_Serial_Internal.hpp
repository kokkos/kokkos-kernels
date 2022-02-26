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
#ifndef __KOKKOSBATCHED_GEMV_SERIAL_INTERNAL_HPP__
#define __KOKKOSBATCHED_GEMV_SERIAL_INTERNAL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

#include "KokkosBatched_Set_Internal.hpp"
#include "KokkosBatched_Scale_Internal.hpp"

#include "KokkosBatched_InnerMultipleDotProduct_Serial_Impl.hpp"
#include "KokkosBatched_InnerMultipleHermitianProduct_Serial_Impl.hpp"

namespace KokkosBatched {

///
/// Serial Internal Impl
/// ====================

template <typename ArgAlgo>
struct SerialGemvInternal {
  template <typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const int m, const int n, const ScalarType alpha,
      const ValueType *KOKKOS_RESTRICT A, const int as0, const int as1,
      const ValueType *KOKKOS_RESTRICT x, const int xs0, const ScalarType beta,
      /**/ ValueType *KOKKOS_RESTRICT y, const int ys0);
};

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int SerialGemvInternal<Algo::Gemv::Unblocked>::invoke(
    const int m, const int n, const ScalarType alpha,
    const ValueType *KOKKOS_RESTRICT A, const int as0, const int as1,
    const ValueType *KOKKOS_RESTRICT x, const int xs0, const ScalarType beta,
    /**/ ValueType *KOKKOS_RESTRICT y, const int ys0) {
  const ScalarType one(1.0), zero(0.0);

  // y = beta y + alpha A x
  // y (m), A(m x n), B(n)

  if (beta == zero)
    SerialSetInternal ::invoke(m, zero, y, ys0);
  else if (beta != one)
    SerialScaleInternal::invoke(m, beta, y, ys0);

  if (alpha != zero) {
    if (m <= 0 || n <= 0) return 0;

    for (int i = 0; i < m; ++i) {
      ValueType t(0);
      const ValueType *KOKKOS_RESTRICT tA = (A + i * as0);

#if defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (int j = 0; j < n; ++j) t += tA[j * as1] * x[j * xs0];
      y[i * ys0] += alpha * t;
    }
  }
  return 0;
}

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int SerialGemvInternal<Algo::Gemv::Blocked>::invoke(
    const int m, const int n, const ScalarType alpha,
    const ValueType *KOKKOS_RESTRICT A, const int as0, const int as1,
    const ValueType *KOKKOS_RESTRICT x, const int xs0, const ScalarType beta,
    /**/ ValueType *KOKKOS_RESTRICT y, const int ys0) {
  const ScalarType one(1.0), zero(0.0);

  // y = beta y + alpha A x
  // y (m), A(m x n), B(n)

  constexpr int mbAlgo = Algo::Gemv::Blocked::mb();

  if (beta == zero)
    SerialSetInternal ::invoke(m, zero, y, ys0);
  else if (beta != one)
    SerialScaleInternal::invoke(m, beta, y, ys0);

  if (alpha != zero) {
    if (m <= 0 || n <= 0) return 0;

    InnerMultipleDotProduct<mbAlgo> inner(as0, as1, xs0, ys0);
    const int mb = mbAlgo;
    for (int i = 0; i < m; i += mb)
      inner.serial_invoke(alpha, A + i * as0, x, (i + mb) > m ? (m - i) : mb, n,
                          y + i * ys0);
  }
  return 0;
}

//
// Version with conj(A)
//

template <typename ArgAlgo>
struct SerialConjGemvInternal {
  template <typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const int m, const int n, const ScalarType alpha,
      const ValueType *KOKKOS_RESTRICT A, const int as0, const int as1,
      const ValueType *KOKKOS_RESTRICT x, const int xs0, const ScalarType beta,
      /**/ ValueType *KOKKOS_RESTRICT y, const int ys0);
};

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int
SerialConjGemvInternal<Algo::Gemv::Unblocked>::invoke(
    const int m, const int n, const ScalarType alpha,
    const ValueType *KOKKOS_RESTRICT A, const int as0, const int as1,
    const ValueType *KOKKOS_RESTRICT x, const int xs0, const ScalarType beta,
    /**/ ValueType *KOKKOS_RESTRICT y, const int ys0) {
  const ScalarType one(1.0), zero(0.0);

  // y = beta y + alpha conj(A) x
  // y (m), A(m x n), B(n)

  if (beta == zero)
    SerialSetInternal ::invoke(m, zero, y, ys0);
  else if (beta != one)
    SerialScaleInternal::invoke(m, beta, y, ys0);

  if (alpha != zero) {
    if (m <= 0 || n <= 0) return 0;

    for (int i = 0; i < m; ++i) {
      ValueType t(0);
      const ValueType *KOKKOS_RESTRICT tA = (A + i * as0);

#if defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (int j = 0; j < n; ++j)
        t += Kokkos::Details::ArithTraits<ValueType>::conj(tA[j * as1]) *
             x[j * xs0];
      y[i * ys0] += alpha * t;
    }
  }
  return 0;
}

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int SerialConjGemvInternal<Algo::Gemv::Blocked>::invoke(
    const int m, const int n, const ScalarType alpha,
    const ValueType *KOKKOS_RESTRICT A, const int as0, const int as1,
    const ValueType *KOKKOS_RESTRICT x, const int xs0, const ScalarType beta,
    /**/ ValueType *KOKKOS_RESTRICT y, const int ys0) {
  const ScalarType one(1.0), zero(0.0);

  // y = beta y + alpha A x
  // y (m), A(m x n), B(n)

  constexpr int mbAlgo = Algo::Gemv::Blocked::mb();

  if (beta == zero)
    SerialSetInternal ::invoke(m, zero, y, ys0);
  else if (beta != one)
    SerialScaleInternal::invoke(m, beta, y, ys0);

  if (alpha != zero) {
    if (m <= 0 || n <= 0) return 0;

    InnerMultipleHermitianProduct<mbAlgo> inner(as0, as1, xs0, ys0);
    const int mb = mbAlgo;
    for (int i = 0; i < m; i += mb)
      inner.serial_invoke(alpha, A + i * as0, x, (i + mb) > m ? (m - i) : mb, n,
                          y + i * ys0);
  }
  return 0;
}

}  // namespace KokkosBatched

#endif
