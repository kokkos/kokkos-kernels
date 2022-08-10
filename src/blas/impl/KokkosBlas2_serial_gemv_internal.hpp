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
#ifndef __KOKKOSBLAS_GEMV_SERIAL_INTERNAL_HPP__
#define __KOKKOSBLAS_GEMV_SERIAL_INTERNAL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBlas_util.hpp"
#include "KokkosBlas1_set_impl.hpp"
#include "KokkosBlas1_serial_scal_impl.hpp"
#include "KokkosBlas2_serial_gemv_inner_multiple_dot.hpp"

namespace KokkosBlas {
namespace Impl {
///
/// Serial Internal Impl
/// ====================

template <typename ArgAlgo>
struct SerialGemvInternal {
  template <typename ExecSpace, typename OpA, typename ScalarType,
            typename ValueAType, typename ValueXType, typename ValueYType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      ExecSpace /* ex */, OpA op, const int m, const int n,
      const ScalarType alpha, const ValueAType *KOKKOS_RESTRICT A,
      const int as0, const int as1, const ValueXType *KOKKOS_RESTRICT x,
      const int xs0, const ScalarType beta,
      /**/ ValueYType *KOKKOS_RESTRICT y, const int ys0);

  // default OpA = OpID
  template <typename ExecSpace, typename ScalarType, typename ValueAType,
            typename ValueXType, typename ValueYType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      ExecSpace ex, const int m, const int n, const ScalarType alpha,
      const ValueAType *KOKKOS_RESTRICT A, const int as0, const int as1,
      const ValueXType *KOKKOS_RESTRICT x, const int xs0, const ScalarType beta,
      /**/ ValueYType *KOKKOS_RESTRICT y, const int ys0) {
    return invoke(ex, OpID(), m, n, alpha, A, as0, as1, x, xs0, beta, y, ys0);
  }
};

template <>
template <typename ExecSpace, typename OpA, typename ScalarType,
          typename ValueAType, typename ValueXType, typename ValueYType>
KOKKOS_INLINE_FUNCTION int SerialGemvInternal<Algo::Gemv::Unblocked>::invoke(
    ExecSpace /* ex */, OpA op, const int m, const int n,
    const ScalarType alpha, const ValueAType *KOKKOS_RESTRICT A, const int as0,
    const int as1, const ValueXType *KOKKOS_RESTRICT x, const int xs0,
    const ScalarType beta,
    /**/ ValueYType *KOKKOS_RESTRICT y, const int ys0) {
  const ScalarType one(1.0), zero(0.0);

  // y = beta y + alpha A x
  // y (m), A(m x n), B(n)

  if (beta == zero) {
    ValueYType val_zero{};  // can be Vector<SIMD> so avoid assigning explicit 0
    KokkosBlas::Impl::SerialSetInternal::invoke(m, val_zero, y, ys0);
  } else if (beta != one)
    KokkosBlas::Impl::SerialScaleInternal::invoke(m, beta, y, ys0);

  if (alpha != zero) {
    if (m <= 0 || n <= 0) return 0;

    for (int i = 0; i < m; ++i) {
      ValueYType t(0);
      const ValueAType *KOKKOS_RESTRICT tA = A + i * as0;

#if defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (int j = 0; j < n; ++j) t += op(tA[j * as1]) * x[j * xs0];
      y[i * ys0] += alpha * t;
    }
  }
  return 0;
}

template <>
template <typename ExecSpace, typename OpA, typename ScalarType,
          typename ValueAType, typename ValueXType, typename ValueYType>
KOKKOS_INLINE_FUNCTION int SerialGemvInternal<Algo::Gemv::Blocked>::invoke(
    ExecSpace /* ex */, OpA /* op */, const int m, const int n,
    const ScalarType alpha, const ValueAType *KOKKOS_RESTRICT A, const int as0,
    const int as1, const ValueXType *KOKKOS_RESTRICT x, const int xs0,
    const ScalarType beta,
    /**/ ValueYType *KOKKOS_RESTRICT y, const int ys0) {
  const ScalarType one(1.0), zero(0.0);

  // y = beta y + alpha A x
  // y (m), A(m x n), B(n)

  constexpr int mbAlgo = Algo::Gemv::mb<ExecSpace>();

  if (beta == zero)
    Impl::SerialSetInternal::invoke(m, zero, y, ys0);
  else if (beta != one)
    Impl::SerialScaleInternal::invoke(m, beta, y, ys0);

  if (alpha != zero) {
    if (m <= 0 || n <= 0) return 0;

    Impl::InnerMultipleDotProduct<mbAlgo> inner(as0, as1, xs0, ys0);
    for (int i = 0; i < m; i += mbAlgo)
      inner.template serial_invoke<OpA>(alpha, A + i * as0, x,
                                        (i + mbAlgo) > m ? (m - i) : mbAlgo, n,
                                        y + i * ys0);
  }
  return 0;
}
}  // namespace Impl
}  // namespace KokkosBlas

#endif
