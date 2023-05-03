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
#ifndef __KOKKOSBATCHED_GEMM_DECL_HPP__
#define __KOKKOSBATCHED_GEMM_DECL_HPP__

#include "KokkosBatched_Vector.hpp"

/********************* BEGIN non-functor-level routines *********************/
namespace KokkosBatched {
/********************* BEGIN functor-level routines *********************/
///
/// Serial Gemm
///

template <typename ArgTransA, typename ArgTransB, typename ArgAlgo>
struct SerialGemm {
  template <typename ScalarType, typename AViewType, typename BViewType,
            typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const ScalarType alpha,
                                           const AViewType &A,
                                           const BViewType &B,
                                           const ScalarType beta,
                                           const CViewType &C);
};

///
/// Team Gemm
///

template <typename MemberType, typename ArgTransA, typename ArgTransB,
          typename ArgAlgo>
struct TeamGemm {
  template <typename ScalarType, typename AViewType, typename BViewType,
            typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType &member, const ScalarType alpha, const AViewType &A,
      const BViewType &B, const ScalarType beta, const CViewType &C);
};

///
/// TeamVector Gemm
///

template <typename MemberType, typename ArgTransA, typename ArgTransB,
          typename ArgAlgo>
struct TeamVectorGemm {
  template <typename ScalarType, typename AViewType, typename BViewType,
            typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType &member, const ScalarType alpha, const AViewType &A,
      const BViewType &B, const ScalarType beta, const CViewType &C);
};

///
/// Selective Interface
///
template <typename MemberType, typename ArgTransA, typename ArgTransB,
          typename ArgMode, typename ArgAlgo>
struct Gemm {
  template <typename ScalarType, typename AViewType, typename BViewType,
            typename CViewType>
  KOKKOS_FORCEINLINE_FUNCTION static int invoke(
      const MemberType &member, const ScalarType alpha, const AViewType &A,
      const BViewType &B, const ScalarType beta, const CViewType &C) {
    int r_val = 0;
    if (std::is_same<ArgMode, Mode::Serial>::value) {
      r_val = SerialGemm<ArgTransA, ArgTransB, ArgAlgo>::invoke(alpha, A, B,
                                                                beta, C);
    } else if (std::is_same<ArgMode, Mode::Team>::value) {
      r_val = TeamGemm<MemberType, ArgTransA, ArgTransB, ArgAlgo>::invoke(
          member, alpha, A, B, beta, C);
    }
    return r_val;
  }
};
/********************* END functor-level routines *********************/

namespace Impl {
template <typename ArgTransA, typename ArgTransB, typename ArgBatchSzDim,
          typename BatchedGemmHandleType, typename ScalarType,
          typename AViewType, typename BViewType, typename CViewType, bool>
struct BatchedGemmWrapper;
}  // namespace Impl

// clang-format off
/// \brief Non-blocking solve of general matrix multiply on a batch of
/// uniform matrices.
///
/// Note: If a TPL is selected, this interface follows the blocking
/// behavior (either blocking or non-blocking) of the TPL vendor's API.
///
/// Note: To leverage SIMD instructions, 4-rank views must be selected via the
/// template parameters documented below.
///
///        C = alpha * op(A) * op(B) + beta * C
///
/// \tparam ArgTransA      Specifies what op does to A:
///                        Trans::NoTranspose   for non-transpose
///                        Trans::Transpose     for transpose
///                        Trans::ConjTranspose for conjugate transpose
/// \tparam ArgTransB      Specifies what op does to B:
///                        Trans::NoTranspose   for non-transpose
///                        Trans::Transpose     for transpose
///                        Trans::ConjTranspose for conjugate transpose
/// \tparam ArgBatchSzDim  Specifies where the batch dimension is allocated in
///                        AViewType, BViewType, and CViewType:
///                        BatchLayout::Left  Batch dimension is leftmost
///                        BatchLayout::Right Batch dimension is rightmost
/// \tparam ScalarType     Specifies the scalar type of alpha and beta
/// \tparam AViewType      Input matrix, as either a 3-rank Kokkos::View or a
///                        4-rank Kokkos::View for SIMD operations.
/// \tparam BViewType      Input matrix, as either a 3-rank Kokkos::View or a
///                        4-rank Kokkos::View for SIMD operations.
/// \tparam CViewType      Input(RHS)/Output(LHS) matrix, as either a 3-rank
///                        Kokkos::View or a 4-rank Kokkos::View for SIMD
///                        operations.
///
/// \param handle [in]     A handle which specifies how to invoke the batched
///                        gemm.
///                        See struct BatchedGemmHandle for details.
/// \param alpha [in]      Input coefficient used for multiplication with A
/// \param A [in]          Input matrix, as a 3-rank Kokkos::View
///                        If ArgBatchSzDim == "BatchLayout::Right", matrix A is MxKxB
///                        If ArgBatchSzDim == "BatchLayout::Left",  matrix A is BxMxK
/// \param B [in]          Input matrix, as a 3-rank Kokkos::View
///                        If ArgBatchSzDim == "BatchLayout::Right", matrix B is KxNxB
///                        If ArgBatchSzDim == "BatchLayout::Left",  matrix B is BxKxN
/// \param beta [in]       Input coefficient used for multiplication with C
/// \param C [in/out]      Input/Output matrix, as a 3-rank Kokkos::View
///                        If ArgBatchSzDim == "BatchLayout::Right", matrix C is MxNxB
///                        If ArgBatchSzDim == "BatchLayout::Left",  matrix C is BxMxN
/// \return 0 upon success, non-zero otherwise
///
/// Usage Example:
///   BatchedGemm<ArgTransA, ArgTransB,
///               ArgBatchSzDim>(handle, alpha, A, B, beta, C);
// clang-format on
template <typename ArgTransA, typename ArgTransB, typename ArgBatchSzDim,
          typename BatchedGemmHandleType, typename ScalarType,
          typename AViewType, typename BViewType, typename CViewType>
inline int BatchedGemm(BatchedGemmHandleType *const handle,
                       const ScalarType alpha, const AViewType &A,
                       const BViewType &B, const ScalarType beta,
                       const CViewType &C) {
  // If either this is being processed by a *.cpp.in file or KK ETI_ONLY
  // is defined, use the ETI specialization. Defer till link time
  // for which specilization will be used from
  // KokkosBatched_HostLevel_Gemm_Impl.hpp.
#if defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
  return Impl::BatchedGemmWrapper<ArgTransA, ArgTransB, ArgBatchSzDim,
                                  BatchedGemmHandleType, ScalarType, AViewType,
                                  BViewType, CViewType, true>::run(handle,
                                                                   alpha, A, B,
                                                                   beta, C);
#else
  // Use the non-ETI specialization.
  return Impl::BatchedGemmWrapper<ArgTransA, ArgTransB, ArgBatchSzDim,
                                  BatchedGemmHandleType, ScalarType, AViewType,
                                  BViewType, CViewType, false>::run(handle,
                                                                    alpha, A, B,
                                                                    beta, C);
#endif  // KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
}
}  // namespace KokkosBatched
/********************* END non-functor-level routines *********************/

#include "KokkosBatched_HostLevel_Gemm_Impl.hpp"
#include "KokkosBatched_Gemm_Serial_Impl.hpp"
#include "KokkosBatched_Gemm_Team_Impl.hpp"
#include "KokkosBatched_Gemm_TeamVector_Impl.hpp"

#endif
