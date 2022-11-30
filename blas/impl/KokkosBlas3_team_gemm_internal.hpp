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

#ifndef KOKKOSBLAS3_TEAM_GEMM_INTERNAL_HPP_
#define KOKKOSBLAS3_TEAM_GEMM_INTERNAL_HPP_

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBlas1_set_impl.hpp"
#include "KokkosBlas1_team_scal_impl.hpp"
#include "KokkosBlas3_serial_gemm_inner_fixc_impl.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"

namespace KokkosBlas {
namespace Impl {

///
/// Team Internal Impl
/// ====================

template <typename ArgAlgo>
struct TeamGemmInternal {
  template <typename OpA, typename OpB, typename MemberType,
            typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      OpA opA, OpB opB, const MemberType &member, const int m, const int n,
      const int k, const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
      const int as0, const int as1, const ValueType *KOKKOS_RESTRICT B,
      const int bs0, const int bs1, const ScalarType beta,
      /**/ ValueType *KOKKOS_RESTRICT C, const int cs0, const int cs1);

  // default OpA=OpB=Impl::OpID
  template <typename MemberType, typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType &member, const int m, const int n, const int k,
      const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A, const int as0,
      const int as1, const ValueType *KOKKOS_RESTRICT B, const int bs0,
      const int bs1, const ScalarType beta,
      /**/ ValueType *KOKKOS_RESTRICT C, const int cs0, const int cs1) {
    return invoke(OpID{}, OpID{}, member, m, n, k, alpha, A, as0, as1, B, bs0,
                  bs1, beta, C, cs0, cs1);
  }
};

template <>
template <typename OpA, typename OpB, typename MemberType, typename ScalarType,
          typename ValueType>
KOKKOS_INLINE_FUNCTION int TeamGemmInternal<Algo::Gemm::Unblocked>::invoke(
    OpA opA, OpB opB, const MemberType &member, const int m, const int n,
    const int k, const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
    const int as0, const int as1, const ValueType *KOKKOS_RESTRICT B,
    const int bs0, const int bs1, const ScalarType beta,
    /**/ ValueType *KOKKOS_RESTRICT C, const int cs0, const int cs1) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)

  const ScalarType one(1.0), zero(0.0);

  if (beta == zero)
    KokkosBlas::Impl::TeamSetInternal::invoke(member, m, n, zero, C, cs0, cs1);
  else if (beta != one)
    KokkosBlas::Impl::TeamScaleInternal::invoke(member, m, n, beta, C, cs0,
                                                cs1);

  if (alpha != ScalarType(0.0)) {
    if (m <= 0 || n <= 0 || k <= 0) return 0;

    if (beta != one) member.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, m * n),
                         [&](const int &ij) {
                           // assume layout right for batched computation
                           const int i = ij / n, j = ij % n;
                           const ValueType *KOKKOS_RESTRICT pA = A + i * as0;
                           const ValueType *KOKKOS_RESTRICT pB = B + j * bs1;

                           ValueType c = ValueType(0);
                           for (int p = 0; p < k; ++p)
                             c += opA(pA[p * as1]) * opB(pB[p * bs0]);
                           C[i * cs0 + j * cs1] += alpha * c;
                         });
  }
  return 0;
}

template <>
template <typename OpA, typename OpB, typename MemberType, typename ScalarType,
          typename ValueType>
KOKKOS_INLINE_FUNCTION int TeamGemmInternal<Algo::Gemm::Blocked>::invoke(
    OpA opA, OpB opB, const MemberType &member, const int m, const int n,
    const int k, const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
    const int as0, const int as1, const ValueType *KOKKOS_RESTRICT B,
    const int bs0, const int bs1, const ScalarType beta,
    /**/ ValueType *KOKKOS_RESTRICT C, const int cs0, const int cs1) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)

  constexpr int mbAlgo = Algo::Gemm::Blocked::mb();
  constexpr int nbAlgo = Algo::Gemm::Blocked::mb();

  const ScalarType one(1.0), zero(0.0);

  if (beta == zero)
    KokkosBlas::Impl::TeamSetInternal::invoke(member, m, n, zero, C, cs0, cs1);
  else if (beta != one)
    KokkosBlas::Impl::TeamScaleInternal::invoke(member, m, n, beta, C, cs0,
                                                cs1);

  if (alpha != ScalarType(0.0)) {
    if (m <= 0 || n <= 0 || k <= 0) return 0;

    if (beta != one) member.team_barrier();

    ///
    /// GPU case: team size is large and blocksize (mb,nb) is small
    KokkosBlas::InnerGemmFixC<mbAlgo, nbAlgo> inner(as0, as1, bs0, bs1, cs0,
                                                    cs1);
    auto gemm = [&](const int ib, const int jb, const int pb,
                    const ValueType *KOKKOS_RESTRICT AA,
                    const ValueType *KOKKOS_RESTRICT BB,
                    /**/ ValueType *KOKKOS_RESTRICT CC) {
      // Made this non-const in order to WORKAROUND issue #349
      int mb = mbAlgo, mp = (ib % mb), mq = (ib / mb) + (mp > 0), nb = nbAlgo,
          np = (jb % nb), nq = (jb / nb) + (np > 0);

      // square tiling
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, mq * nq), [&](const int &ij) {
            int i, j;
            // note: the condition is constexpr
            if (KokkosKernels::Impl::kk_is_gpu_exec_space<
                    typename MemberType::execution_space>()) {
              i = ij % mq * mb;
              j = ij / mq * nb;
            } else {
              i = ij / nq * mb;
              j = ij % nq * nb;
            }
            inner.serial_invoke(opA, opB, alpha, AA + i * as0, BB + j * bs1,
                                (i + mb) > ib ? mp : mb,
                                (j + nb) > jb ? np : nb, pb,
                                CC + i * cs0 + j * cs1);
          });
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

///
/// TeamVector Internal Impl
/// ====================

template <typename ArgAlgo>
struct TeamVectorGemmInternal {
  template <typename OpA, typename OpB, typename MemberType,
            typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      OpA opA, OpB opB, const MemberType &member, const int m, const int n,
      const int k, const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
      const int as0, const int as1, const ValueType *KOKKOS_RESTRICT B,
      const int bs0, const int bs1, const ScalarType beta,
      /**/ ValueType *KOKKOS_RESTRICT C, const int cs0, const int cs1);

  // default OpA=OpB=Impl::OpID
  template <typename MemberType, typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType &member, const int m, const int n, const int k,
      const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A, const int as0,
      const int as1, const ValueType *KOKKOS_RESTRICT B, const int bs0,
      const int bs1, const ScalarType beta,
      /**/ ValueType *KOKKOS_RESTRICT C, const int cs0, const int cs1) {
    return invoke(OpID{}, OpID{}, member, m, n, k, alpha, A, as0, as1, B, bs0,
                  bs1, beta, C, cs0, cs1);
  }
};

template <>
template <typename OpA, typename OpB, typename MemberType, typename ScalarType,
          typename ValueType>
KOKKOS_INLINE_FUNCTION int
TeamVectorGemmInternal<Algo::Gemm::Unblocked>::invoke(
    OpA opA, OpB opB, const MemberType &member, const int m, const int n,
    const int k, const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
    const int as0, const int as1, const ValueType *KOKKOS_RESTRICT B,
    const int bs0, const int bs1, const ScalarType beta,
    /**/ ValueType *KOKKOS_RESTRICT C, const int cs0, const int cs1) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)

  const ScalarType one(1.0), zero(0.0);

  if (beta == zero)
    KokkosBlas::Impl::TeamVectorSetInternal::invoke(member, m, n, zero, C, cs0,
                                                    cs1);
  else if (beta != one)
    KokkosBlas::Impl::TeamVectorScaleInternal::invoke(member, m, n, beta, C,
                                                      cs0, cs1);

  if (alpha != ScalarType(0.0)) {
    if (m <= 0 || n <= 0 || k <= 0) return 0;

    if (beta != one) member.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, m), [&](const int &i) {
      const ValueType *KOKKOS_RESTRICT pA = A + i * as0;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n),
                           [&](const int &j) {
                             const ValueType *KOKKOS_RESTRICT pB = B + j * bs1;

                             ValueType c = ValueType(0);
                             for (int p = 0; p < k; ++p)
                               c += opA(pA[p * as1]) * opB(pB[p * bs0]);
                             C[i * cs0 + j * cs1] += alpha * c;
                           });
    });
  }
  return 0;
}

///
/// ThreadVector Internal Impl
/// ====================

template <typename ArgAlgo>
struct ThreadVectorGemmInternal {
  template <typename MemberType, typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType &member, const int m, const int n, const int k,
      const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A, const int as0,
      const int as1, const ValueType *KOKKOS_RESTRICT B, const int bs0,
      const int bs1, const ScalarType beta,
      /**/ ValueType *KOKKOS_RESTRICT C, const int cs0, const int cs1);
};

template <>
template <typename MemberType, typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int
ThreadVectorGemmInternal<Algo::Gemm::Unblocked>::invoke(
    const MemberType &member, const int m, const int n, const int k,
    const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A, const int as0,
    const int as1, const ValueType *KOKKOS_RESTRICT B, const int bs0,
    const int bs1, const ScalarType beta,
    /**/ ValueType *KOKKOS_RESTRICT C, const int cs0, const int cs1) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)

  const ScalarType one(1.0), zero(0.0);

  if (beta == zero)
    KokkosBlas::Impl::ThreadVectorSetInternal::invoke(member, m, n, zero, C,
                                                      cs0, cs1);
  else if (beta != one)
    KokkosBlas::Impl::ThreadVectorScaleInternal::invoke(member, m, n, beta, C,
                                                        cs0, cs1);

  if (alpha != ScalarType(0.0)) {
    if (m <= 0 || n <= 0 || k <= 0) return 0;

    if (beta != one) member.team_barrier();

    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(member, 0, m * n), [&](const int &ij) {
          // assume layout right for batched computation
          const int i = ij / n, j = ij % n;
          const ValueType *KOKKOS_RESTRICT pA                  = A + i * as0,
                                           *KOKKOS_RESTRICT pB = B + j * bs1;

          ValueType c = ValueType(0);
          for (int p = 0; p < k; ++p) c += pA[p * as1] * pB[p * bs0];
          C[i * cs0 + j * cs1] += alpha * c;
        });
  }
  return 0;
}

template <>
template <typename MemberType, typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int
ThreadVectorGemmInternal<Algo::Gemm::Blocked>::invoke(
    const MemberType &member, const int m, const int n, const int k,
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
    KokkosBlas::Impl::ThreadVectorSetInternal::invoke(member, m, n, zero, C,
                                                      cs0, cs1);
  else if (beta != one)
    KokkosBlas::Impl::ThreadVectorScaleInternal::invoke(member, m, n, beta, C,
                                                        cs0, cs1);

  if (alpha != ScalarType(0.0)) {
    if (m <= 0 || n <= 0 || k <= 0) return 0;

    if (beta != one) member.team_barrier();

    ///
    /// GPU case: team size is large and blocksize (mb,nb) is small
    KokkosBlas::InnerGemmFixC<mbAlgo, nbAlgo> inner(as0, as1, bs0, bs1, cs0,
                                                    cs1);
    auto gemm = [&](const int ib, const int jb, const int pb,
                    const ValueType *KOKKOS_RESTRICT AA,
                    const ValueType *KOKKOS_RESTRICT BB,
                    /**/ ValueType *KOKKOS_RESTRICT CC) {
      // Made this non-const in order to WORKAROUND issue #349
      int mb = mbAlgo, mp = (ib % mb), mq = (ib / mb) + (mp > 0), nb = nbAlgo,
          np = (jb % nb), nq = (jb / nb) + (np > 0);

      // square tiling
      Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(member, mq * nq), [&](const int &ij) {
            int i, j;
            // note: the condition is constexpr
            if (KokkosKernels::Impl::kk_is_gpu_exec_space<
                    typename MemberType::execution_space>()) {
              i = ij % mq * mb;
              j = ij / mq * nb;
            } else {
              i = ij / nq * mb;
              j = ij % nq * nb;
            }
            inner.serial_invoke(
                alpha, AA + i * as0, BB + j * bs1, (i + mb) > ib ? mp : mb,
                (j + nb) > jb ? np : nb, pb, CC + i * cs0 + j * cs1);
          });
    };

    gemm(m, n, k, A, B, C);
  }
  return 0;
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif
