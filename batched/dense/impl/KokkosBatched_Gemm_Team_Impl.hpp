//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#ifndef KOKKOSBATCHED_GEMM_TEAM_IMPL_HPP
#define KOKKOSBATCHED_GEMM_TEAM_IMPL_HPP

/// \author Kyungjoo Kim (kyukim@sandia.gov)
/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

#include "KokkosBlas_util.hpp"
#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Gemm_Common_Impl.hpp"
#include "KokkosBatched_Gemm_Team_Internal.hpp"

namespace KokkosBatched {

///
/// Team Impl
/// =========

///
/// NT/NT
///

template <typename MemberType>
struct TeamGemm<MemberType, Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Unblocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(1);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::NoTranspose, Trans::NoTranspose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A B
    // C (m x n), A(m x k), B(k x n)
    return Impl::TeamGemmInternal<Algo::Gemm::Unblocked>::invoke(
        member, KokkosBlas::Impl::OpID(), KokkosBlas::Impl::OpID(), C.extent(0), C.extent(1), A.extent(1), alpha,
        A.data(), A.stride_0(), A.stride_1(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

template <typename MemberType>
struct TeamGemm<MemberType, Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Blocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(1);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::NoTranspose, Trans::NoTranspose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A B
    // C (m x n), A(m x k), B(k x n)
    return Impl::TeamGemmInternal<Algo::Gemm::Blocked>::invoke(
        member, KokkosBlas::Impl::OpID(), KokkosBlas::Impl::OpID(), C.extent(0), C.extent(1), A.extent(1), alpha,
        A.data(), A.stride_0(), A.stride_1(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

///
/// T/NT
///

template <typename MemberType>
struct TeamGemm<MemberType, Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Unblocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(0);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::Transpose, Trans::NoTranspose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A^T B
    // C (m x n), A(k x m), B(k x n)
    return Impl::TeamGemmInternal<Algo::Gemm::Unblocked>::invoke(
        member, KokkosBlas::Impl::OpID(), KokkosBlas::Impl::OpID(), C.extent(0), C.extent(1), A.extent(0), alpha,
        A.data(), A.stride_1(), A.stride_0(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

template <typename MemberType>
struct TeamGemm<MemberType, Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Blocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(0);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::Transpose, Trans::NoTranspose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A^T B
    // C (m x n), A(k x m), B(k x n)
    return Impl::TeamGemmInternal<Algo::Gemm::Blocked>::invoke(
        member, KokkosBlas::Impl::OpID(), KokkosBlas::Impl::OpID(), C.extent(0), C.extent(1), A.extent(0), alpha,
        A.data(), A.stride_1(), A.stride_0(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

///
/// C/NT
///

template <typename MemberType>
struct TeamGemm<MemberType, Trans::ConjTranspose, Trans::NoTranspose, Algo::Gemm::Unblocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(0);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::ConjTranspose, Trans::NoTranspose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A^H B
    // C (m x n), A(k x m), B(k x n)
    return Impl::TeamGemmInternal<Algo::Gemm::Unblocked>::invoke(
        member, KokkosBlas::Impl::OpConj(), KokkosBlas::Impl::OpID(), C.extent(0), C.extent(1), A.extent(0), alpha,
        A.data(), A.stride_1(), A.stride_0(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

template <typename MemberType>
struct TeamGemm<MemberType, Trans::ConjTranspose, Trans::NoTranspose, Algo::Gemm::Blocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(0);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::ConjTranspose, Trans::NoTranspose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A^H B
    // C (m x n), A(k x m), B(k x n)
    return Impl::TeamGemmInternal<Algo::Gemm::Blocked>::invoke(
        member, KokkosBlas::Impl::OpConj(), KokkosBlas::Impl::OpID(), C.extent(0), C.extent(1), A.extent(0), alpha,
        A.data(), A.stride_1(), A.stride_0(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

///
/// NT/T
///

template <typename MemberType>
struct TeamGemm<MemberType, Trans::NoTranspose, Trans::Transpose, Algo::Gemm::Unblocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(1);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::NoTranspose, Trans::Transpose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A B^T
    // C (m x n), A(m x k), B(n x k)
    return Impl::TeamGemmInternal<Algo::Gemm::Unblocked>::invoke(
        member, KokkosBlas::Impl::OpID(), KokkosBlas::Impl::OpID(), C.extent(0), C.extent(1), A.extent(1), alpha,
        A.data(), A.stride_0(), A.stride_1(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

template <typename MemberType>
struct TeamGemm<MemberType, Trans::NoTranspose, Trans::Transpose, Algo::Gemm::Blocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(1);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::NoTranspose, Trans::Transpose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A B^T
    // C (m x n), A(m x k), B(n x k)
    return Impl::TeamGemmInternal<Algo::Gemm::Blocked>::invoke(
        member, KokkosBlas::Impl::OpID(), KokkosBlas::Impl::OpID(), C.extent(0), C.extent(1), A.extent(1), alpha,
        A.data(), A.stride_0(), A.stride_1(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

///
/// T/T
///

template <typename MemberType>
struct TeamGemm<MemberType, Trans::Transpose, Trans::Transpose, Algo::Gemm::Unblocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(0);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::Transpose, Trans::Transpose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A^T B^T
    // C (m x n), A(k x m), B(n x k)
    return Impl::TeamGemmInternal<Algo::Gemm::Unblocked>::invoke(
        member, KokkosBlas::Impl::OpID(), KokkosBlas::Impl::OpID(), C.extent(0), C.extent(1), A.extent(0), alpha,
        A.data(), A.stride_1(), A.stride_0(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

template <typename MemberType>
struct TeamGemm<MemberType, Trans::Transpose, Trans::Transpose, Algo::Gemm::Blocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(0);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::Transpose, Trans::Transpose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A^T B^T
    // C (m x n), A(k x m), B(n x k)
    return Impl::TeamGemmInternal<Algo::Gemm::Blocked>::invoke(
        member, KokkosBlas::Impl::OpID(), KokkosBlas::Impl::OpID(), C.extent(0), C.extent(1), A.extent(0), alpha,
        A.data(), A.stride_1(), A.stride_0(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

///
/// C/T
///

template <typename MemberType>
struct TeamGemm<MemberType, Trans::ConjTranspose, Trans::Transpose, Algo::Gemm::Unblocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(0);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::ConjTranspose, Trans::Transpose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A^H B^T
    // C (m x n), A(k x m), B(n x k)
    return Impl::TeamGemmInternal<Algo::Gemm::Unblocked>::invoke(
        member, KokkosBlas::Impl::OpConj(), KokkosBlas::Impl::OpID(), C.extent(0), C.extent(1), A.extent(0), alpha,
        A.data(), A.stride_1(), A.stride_0(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

template <typename MemberType>
struct TeamGemm<MemberType, Trans::ConjTranspose, Trans::Transpose, Algo::Gemm::Blocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(0);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::ConjTranspose, Trans::Transpose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A^H B^T
    // C (m x n), A(k x m), B(n x k)
    return Impl::TeamGemmInternal<Algo::Gemm::Blocked>::invoke(
        member, KokkosBlas::Impl::OpConj(), KokkosBlas::Impl::OpID(), C.extent(0), C.extent(1), A.extent(0), alpha,
        A.data(), A.stride_1(), A.stride_0(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

///
/// NT/C
///

template <typename MemberType>
struct TeamGemm<MemberType, Trans::NoTranspose, Trans::ConjTranspose, Algo::Gemm::Unblocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(1);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::NoTranspose, Trans::ConjTranspose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A B^H
    // C (m x n), A(m x k), B(n x k)
    return Impl::TeamGemmInternal<Algo::Gemm::Unblocked>::invoke(
        member, KokkosBlas::Impl::OpID(), KokkosBlas::Impl::OpConj(), C.extent(0), C.extent(1), A.extent(1), alpha,
        A.data(), A.stride_0(), A.stride_1(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

template <typename MemberType>
struct TeamGemm<MemberType, Trans::NoTranspose, Trans::ConjTranspose, Algo::Gemm::Blocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(1);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::NoTranspose, Trans::ConjTranspose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A B^H
    // C (m x n), A(m x k), B(n x k)
    return Impl::TeamGemmInternal<Algo::Gemm::Blocked>::invoke(
        member, KokkosBlas::Impl::OpID(), KokkosBlas::Impl::OpConj(), C.extent(0), C.extent(1), A.extent(1), alpha,
        A.data(), A.stride_0(), A.stride_1(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

///
/// T/C
///

template <typename MemberType>
struct TeamGemm<MemberType, Trans::Transpose, Trans::ConjTranspose, Algo::Gemm::Unblocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(0);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::Transpose, Trans::ConjTranspose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A^T B^H
    // C (m x n), A(k x m), B(n x k)
    return Impl::TeamGemmInternal<Algo::Gemm::Unblocked>::invoke(
        member, KokkosBlas::Impl::OpID(), KokkosBlas::Impl::OpConj(), C.extent(0), C.extent(1), A.extent(0), alpha,
        A.data(), A.stride_1(), A.stride_0(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

template <typename MemberType>
struct TeamGemm<MemberType, Trans::Transpose, Trans::ConjTranspose, Algo::Gemm::Blocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(0);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::Transpose, Trans::Transpose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A^T B^H
    // C (m x n), A(k x m), B(n x k)
    return Impl::TeamGemmInternal<Algo::Gemm::Blocked>::invoke(
        member, KokkosBlas::Impl::OpID(), KokkosBlas::Impl::OpConj(), C.extent(0), C.extent(1), A.extent(0), alpha,
        A.data(), A.stride_1(), A.stride_0(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

///
/// C/C
///

template <typename MemberType>
struct TeamGemm<MemberType, Trans::ConjTranspose, Trans::ConjTranspose, Algo::Gemm::Unblocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(0);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::ConjTranspose, Trans::ConjTranspose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A^H B^H
    // C (m x n), A(k x m), B(n x k)
    return Impl::TeamGemmInternal<Algo::Gemm::Unblocked>::invoke(
        member, KokkosBlas::Impl::OpConj(), KokkosBlas::Impl::OpConj(), C.extent(0), C.extent(1), A.extent(0), alpha,
        A.data(), A.stride_1(), A.stride_0(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

template <typename MemberType>
struct TeamGemm<MemberType, Trans::ConjTranspose, Trans::ConjTranspose, Algo::Gemm::Blocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // Quick return if possible
    const int m = C.extent_int(0), n = C.extent_int(1), k = A.extent_int(0);
    if (m == 0 || n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

    auto info = Impl::checkGemmInput<Trans::ConjTranspose, Trans::ConjTranspose>(A, B, C);
    if (info) return info;

    // C = beta C + alpha A^H B^H
    // C (m x n), A(k x m), B(n x k)
    return Impl::TeamGemmInternal<Algo::Gemm::Blocked>::invoke(
        member, KokkosBlas::Impl::OpConj(), KokkosBlas::Impl::OpConj(), C.extent(0), C.extent(1), A.extent(0), alpha,
        A.data(), A.stride_1(), A.stride_0(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(), C.stride_0(),
        C.stride_1());
  }
};

}  // namespace KokkosBatched

#endif
