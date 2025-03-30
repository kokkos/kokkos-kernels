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

  static_assert(AViewType::rank <= 2, "KokkosBatched::gemm: AViewType must have rank 0, 1 or 2.");
  static_assert(BViewType::rank <= 2, "KokkosBatched::gemm: BViewType must have rank 0, 1 or 2.");
  static_assert(CViewType::rank <= 2, "KokkosBatched::gemm: CViewType must have rank 0, 1 or 2.");

#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  const int m = C.extent(0), n = C.extent(1);
  const int lda = A.extent(0);
  const int ldb = B.extent(0);

  const int ka = std::is_same_v<ArgTransA, Trans::NoTranspose> ? A.extent(1) : A.extent(0);
  const int kb = std::is_same_v<ArgTransB, Trans::NoTranspose> ? B.extent(0) : B.extent(1);

  if (ka != kb) {
    Kokkos::printf(
        "KokkosBatched::gemm: Dimensions of A and B do not match: A: %d x %d, "
        "B: %d x %d\n",
        A.extent(0), A.extent(1), B.extent(0), B.extent(1));
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
