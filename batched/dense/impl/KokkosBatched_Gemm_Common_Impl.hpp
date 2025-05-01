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
#ifndef KOKKOSBATCHED_GEMM_COMMON_IMPL_HPP
#define KOKKOSBATCHED_GEMM_COMMON_IMPL_HPP

#include "KokkosBlas_util.hpp"
#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {
namespace Impl {
template <typename ArgTransA, typename ArgTransB, typename AViewType, typename BViewType, typename CViewType>
KOKKOS_INLINE_FUNCTION static int checkGemmInput([[maybe_unused]] const AViewType &A,
                                                 [[maybe_unused]] const BViewType &B,
                                                 [[maybe_unused]] const CViewType &C) {
  static_assert(Kokkos::is_view_v<AViewType>, "KokkosBatched::gemm: AViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view_v<BViewType>, "KokkosBatched::gemm: BViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view_v<CViewType>, "KokkosBatched::gemm: CViewType is not a Kokkos::View.");

  constexpr std::size_t A_rank = AViewType::rank();
  constexpr std::size_t B_rank = BViewType::rank();
  constexpr std::size_t C_rank = CViewType::rank();
  static_assert(A_rank <= 2, "KokkosBatched::gemm: AViewType must have rank 0, 1 or 2.");
  static_assert(B_rank <= 2, "KokkosBatched::gemm: BViewType must have rank 0, 1 or 2.");
  static_assert(C_rank <= 2, "KokkosBatched::gemm: CViewType must have rank 0, 1 or 2.");

  static_assert(std::is_same_v<typename CViewType::value_type, typename CViewType::non_const_value_type>,
                "KokkosBatched::gemm: CViewType must have non-const value type.");

#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  const int A_extent_0 = get_extent_int(A, 0);
  const int A_extent_1 = get_extent_int(A, 1);
  const int B_extent_0 = get_extent_int(B, 0);
  const int B_extent_1 = get_extent_int(B, 1);
  const int C_extent_0 = get_extent_int(C, 0);
  const int C_extent_1 = get_extent_int(C, 1);

  const int m = C_extent_0, n = C_extent_1;
  const int lda = A_extent_0;
  const int ldb = B_extent_0;

  const int ka = std::is_same_v<ArgTransA, Trans::NoTranspose> ? A_extent_1 : A_extent_0;
  const int kb = std::is_same_v<ArgTransB, Trans::NoTranspose> ? B_extent_0 : B_extent_1;

  if (ka != kb) {
    Kokkos::printf(
        "KokkosBatched::gemm: Dimensions of A and B do not match: A: %d x %d, "
        "B: %d x %d\n",
        A_extent_0, A_extent_1, B_extent_0, B_extent_1);
    return 1;
  }

  const int nrowa = std::is_same_v<ArgTransA, Trans::NoTranspose> ? m : ka;
  const int nrowb = std::is_same_v<ArgTransB, Trans::NoTranspose> ? kb : n;

  if (lda < Kokkos::max(1, nrowa)) {
    Kokkos::printf(
        "KokkosBatched::gemm: leading dimension of A must not be smaller than "
        "max(1, nrowa): "
        "lda = %d, nrowa = %d\n",
        lda, nrowa);
    return 1;
  }
  if (ldb < Kokkos::max(1, nrowb)) {
    Kokkos::printf(
        "KokkosBatched::gemm: leading dimension of B must not be smaller than "
        "max(1, nrowb): "
        "ldb = %d, nrowb = %d\n",
        ldb, nrowb);
    return 1;
  }

#endif

  return 0;
}
}  // namespace Impl
}  // namespace KokkosBatched

#endif
