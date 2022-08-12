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

#ifndef KOKKOSBLAS2_TEAM_GEMV_IMPL_HPP_
#define KOKKOSBLAS2_TEAM_GEMV_IMPL_HPP_

#include "KokkosBlas1_set_impl.hpp"
#include "KokkosBlas1_team_scal_impl.hpp"
#include "KokkosBlas2_serial_gemv_inner_multiple_dot.hpp"

namespace KokkosBlas {
namespace Impl {

template <typename ArgAlgo>
struct TeamGemvInternal {
  template <typename MemberType, typename ScalarType, typename ValueAType,
            typename ValueXType, typename ValueYType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType &member, const int m, const int n,
      const ScalarType alpha, const ValueAType *KOKKOS_RESTRICT A,
      const int as0, const int as1, const ValueXType *KOKKOS_RESTRICT x,
      const int xs0, const ScalarType beta,
      /**/ ValueYType *KOKKOS_RESTRICT y, const int ys0);
};

template <typename ArgAlgo>
struct TeamVectorGemvInternal {
  template <typename MemberType, typename ScalarType, typename ValueAType,
            typename ValueXType, typename ValueYType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType &member, const int m, const int n,
      const ScalarType alpha, const ValueAType *KOKKOS_RESTRICT A,
      const int as0, const int as1, const ValueXType *KOKKOS_RESTRICT x,
      const int xs0, const ScalarType beta,
      /**/ ValueYType *KOKKOS_RESTRICT y, const int ys0);
};

///
/// Team Internal Impl
/// ====================

template <>
template <typename MemberType, typename ScalarType, typename ValueAType,
          typename ValueXType, typename ValueYType>
KOKKOS_INLINE_FUNCTION int TeamGemvInternal<Algo::Gemv::Unblocked>::invoke(
    const MemberType &member, const int m, const int n, const ScalarType alpha,
    const ValueAType *KOKKOS_RESTRICT A, const int as0, const int as1,
    const ValueXType *KOKKOS_RESTRICT x, const int xs0, const ScalarType beta,
    /**/ ValueYType *KOKKOS_RESTRICT y, const int ys0) {
  const ScalarType one(1.0), zero(0.0);

  // y = beta y + alpha A x
  // y (m), A(m x n), B(n)

  if (beta == zero)
    KokkosBlas::Impl::TeamSetInternal::invoke(member, m, zero, y, ys0);
  else if (beta != one)
    KokkosBlas::Impl::TeamScaleInternal::invoke(member, m, beta, y, ys0);

  if (alpha != zero) {
    if (m <= 0 || n <= 0) return 0;

    if (beta != one) member.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, m),
                         [&](const int &i) {
                           ValueYType t(0);
                           const ValueAType *KOKKOS_RESTRICT tA = (A + i * as0);
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
                           for (int j = 0; j < n; ++j)
                             t += tA[j * as1] * x[j * xs0];
                           y[i * ys0] += alpha * t;
                         });
  }
  return 0;
}

template <>
template <typename MemberType, typename ScalarType, typename ValueAType,
          typename ValueXType, typename ValueYType>
KOKKOS_INLINE_FUNCTION int TeamGemvInternal<Algo::Gemv::Blocked>::invoke(
    const MemberType &member, const int m, const int n, const ScalarType alpha,
    const ValueAType *KOKKOS_RESTRICT A, const int as0, const int as1,
    const ValueXType *KOKKOS_RESTRICT x, const int xs0, const ScalarType beta,
    /**/ ValueYType *KOKKOS_RESTRICT y, const int ys0) {
  const ScalarType one(1.0), zero(0.0);

  // y = beta y + alpha A x
  // y (m), A(m x n), B(n)

  constexpr int mbAlgo = Algo::Gemv::Blocked::mb();

  if (beta == zero)
    KokkosBlas::Impl::TeamSetInternal::invoke(member, m, zero, y, ys0);
  else if (beta != one)
    KokkosBlas::Impl::TeamScaleInternal::invoke(member, m, beta, y, ys0);

  if (alpha != zero) {
    if (m <= 0 || n <= 0) return 0;

    if (beta != one) member.team_barrier();

    KokkosBlas::Impl::InnerMultipleDotProduct<mbAlgo> inner(as0, as1, xs0, ys0);
    const int tsize = member.team_size();
    const int mb_a = m / tsize + (m % tsize > 0), mb_b = mbAlgo;
    // Made this non-const in order to WORKAROUND issue #349
    int mb = mb_a < mb_b ? mb_a : mb_b, mp = m % mb;

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, (m / mb) + (mp > 0)),
                         [&](const int &ii) {
                           const int i = ii * mb;
                           inner.serial_invoke<KokkosBlas::Impl::OpID>(
                               alpha, A + i * as0, x,
                               (i + mb) > m ? (m - i) : mb, n, y + i * ys0);
                         });
    member.team_barrier();
  }

  return 0;
}

///
/// TeamVector Internal Impl
/// ====================

template <>
template <typename MemberType, typename ScalarType, typename ValueAType,
          typename ValueXType, typename ValueYType>
KOKKOS_INLINE_FUNCTION int
TeamVectorGemvInternal<Algo::Gemv::Unblocked>::invoke(
    const MemberType &member, const int m, const int n, const ScalarType alpha,
    const ValueAType *KOKKOS_RESTRICT A, const int as0, const int as1,
    const ValueXType *KOKKOS_RESTRICT x, const int xs0, const ScalarType beta,
    /**/ ValueYType *KOKKOS_RESTRICT y, const int ys0) {
  const ScalarType one(1.0), zero(0.0);

  // y = beta y + alpha A x
  // y (m), A(m x n), B(n)

  if (beta == zero)
    KokkosBlas::Impl::TeamVectorSetInternal::invoke(member, m, zero, y, ys0);
  else if (beta != one)
    KokkosBlas::Impl::TeamVectorScaleInternal::invoke(member, m, beta, y, ys0);

  if (alpha != zero) {
    if (m <= 0 || n <= 0) return 0;

    if (beta != one) member.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, m), [&](const int &i) {
      ValueYType t(0);
      const ValueAType *KOKKOS_RESTRICT tA = (A + i * as0);
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(member, n),
          [&](const int &j, ValueYType &update) {
            update += tA[j * as1] * x[j * xs0];
          },
          t);
      Kokkos::single(Kokkos::PerThread(member),
                     [&]() { y[i * ys0] += alpha * t; });
    });
  }
  return 0;
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif
