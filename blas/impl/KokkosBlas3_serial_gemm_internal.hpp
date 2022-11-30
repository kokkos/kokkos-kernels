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

#ifndef KOKKOSBLAS3_GEMM_SERIAL_INTERNAL_HPP
#define KOKKOSBLAS3_GEMM_SERIAL_INTERNAL_HPP

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBlas_util.hpp"
#include "KokkosBlas1_set_impl.hpp"
#include "KokkosBlas1_serial_scal_impl.hpp"
#include "KokkosBlas3_gemm_inner_fix.hpp"

namespace KokkosBlas {
namespace Impl {

///
/// Serial Internal Impl
/// ====================

template <typename ArgAlgo>
struct SerialGemmInternal {
  template <typename OpA, typename OpB, typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      OpA opA, OpB opB, const int m, const int n, const int k,
      const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A, const int as0,
      const int as1, const ValueType *KOKKOS_RESTRICT B, const int bs0,
      const int bs1, const ScalarType beta,
      /**/ ValueType *KOKKOS_RESTRICT C, const int cs0, const int cs1);

  // default OpA=OpB=Impl::OpID
  template <typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const int m, const int n, const int k, const ScalarType alpha,
      const ValueType *KOKKOS_RESTRICT A, const int as0, const int as1,
      const ValueType *KOKKOS_RESTRICT B, const int bs0, const int bs1,
      const ScalarType beta,
      /**/ ValueType *KOKKOS_RESTRICT C, const int cs0, const int cs1) {
    return invoke(OpID{}, OpID{}, m, n, k, alpha, A, as0, as1, B, bs0, bs1,
                  beta, C, cs0, cs1);
  }
};

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int SerialGemmInternal<Algo::Gemm::Unblocked>::invoke(
    OpA opA, OpB opB, const int m, const int n, const int k,
    const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A, const int as0,
    const int as1, const ValueType *KOKKOS_RESTRICT B, const int bs0,
    const int bs1, const ScalarType beta,
    /**/ ValueType *KOKKOS_RESTRICT C, const int cs0, const int cs1) {
  // C = beta C + alpha opA(A) opB(B)
  // C (m x n), A(m x k), B(k x n)

  const ScalarType one(1.0), zero(0.0);

  if (beta == zero)
    KokkosBlas::Impl::SerialSetInternal::invoke(m, n, zero, C, cs0, cs1);
  else if (beta != one)
    KokkosBlas::Impl::SerialScaleInternal::invoke(m, n, beta, C, cs0, cs1);

  if (alpha != zero) {
    if (m <= 0 || n <= 0 || k <= 0) return 0;

    ValueType *KOKKOS_RESTRICT pC = C;
    for (int p = 0; p < k; ++p) {
      const ValueType *KOKKOS_RESTRICT pA = A + p * as1;
      const ValueType *KOKKOS_RESTRICT pB = B + p * bs0;
      for (int i = 0; i < m; ++i) {
        const ValueType tA(alpha * opA(pA[i * as0]));
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int j = 0; j < n; ++j)
          pC[i * cs0 + j * cs1] += tA * opB(pB[j * bs1]);
      }
    }
  }
  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int SerialGemmInternal<Algo::Gemm::Blocked>::invoke(
    OpA opA, OpB opB, const int m, const int n, const int k,
    const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A, const int as0,
    const int as1, const ValueType *KOKKOS_RESTRICT B, const int bs0,
    const int bs1, const ScalarType beta,
    /**/ ValueType *KOKKOS_RESTRICT C, const int cs0, const int cs1) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)

  constexpr int mbAlgo = Algo::Gemm::Blocked::mb();
  constexpr int nbAlgo = Algo::Gemm::Blocked::mb();

  const ScalarType one(1.0), zero(0.0);

  if (beta == zero)
    KokkosBlas::Impl::SerialSetInternal::invoke(m, n, zero, C, cs0, cs1);
  else if (beta != one)
    KokkosBlas::Impl::SerialScaleInternal::invoke(m, n, beta, C, cs0, cs1);

  if (alpha != zero) {
    if (m <= 0 || n <= 0 || k <= 0) return 0;
    const ValueType alpha_value(alpha);

    KokkosBlas::InnerGemmFixC<mbAlgo, nbAlgo> inner(as0, as1, bs0, bs1, cs0,
                                                    cs1);
    auto gemm = [&](const int ib, const int jb, const int pb,
                    const ValueType *KOKKOS_RESTRICT AA,
                    const ValueType *KOKKOS_RESTRICT BB,
                    /**/ ValueType *KOKKOS_RESTRICT CC) {
      const int mb = mbAlgo, nb = nbAlgo;
      for (int i = 0; i < ib; i += mb)
        for (int j = 0; j < jb; j += nb)
          inner.serial_invoke(opA, opB, alpha_value, AA + i * as0, BB + j * bs1,
                              (i + mb) > ib ? (ib - i) : mb,
                              (j + nb) > jb ? (jb - j) : nb, pb,
                              CC + i * cs0 + j * cs1);
    };

    const bool is_small = true;  //(m*n*k <= 64*64*64);
    if (is_small) {
      gemm(m, n, k, A, B, C);
    } else {
      // // cache blocking
      // const int
      //   nc = nb*10, kc = mb*4, mc = mb*4;

      // for (int jj=0;jj<n;jj+=nc) {
      //   const int tj = n-jj, jb = (tj < nc ? tj : nc);
      //   for (int pp=0;pp<k;pp+=kc) {
      //     const int tp = k-pp, pb = (tp < kc ? tp : kc);
      //     //const int pb = k, pp = 0;
      //     for (int ii=0;ii<m;ii+=mc) {
      //       const int ti = m-ii, ib = (ti < mc ? ti : mc);

      //       const ValueType *KOKKOS_RESTRICT AA = A+ii*as0+pp*as1;
      //       const ValueType *KOKKOS_RESTRICT BB = B+pp*bs0+jj*bs1;
      //       /**/  ValueType *KOKKOS_RESTRICT CC = C+ii*cs0+jj*cs1;

      //       gemm(ib, jb, pb, AA, BB, CC);
      //     } // for ii
      //   } // for pp
      // } // for jj
    }
  }
  return 0;
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif
