// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBATCHED_SYR2K_IMPL_HPP_
#define KOKKOSBATCHED_SYR2K_IMPL_HPP_

#include <KokkosBlas_util.hpp>
#include <KokkosBatched_Util.hpp>
#include "KokkosBatched_Syr2k_Internal.hpp"

namespace KokkosBatched {
namespace Impl {
template <bool is_trans, typename AViewType, typename BViewType, typename CViewType>
KOKKOS_INLINE_FUNCTION static int checkSyr2kInput([[maybe_unused]] const AViewType &A,
                                                  [[maybe_unused]] const BViewType &B,
                                                  [[maybe_unused]] const CViewType &C) {
  static_assert(Kokkos::is_view_v<AViewType>, "KokkosBatched::syr2k: AViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view_v<BViewType>, "KokkosBatched::syr2k: BViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view_v<CViewType>, "KokkosBatched::syr2k: CViewType is not a Kokkos::View.");
  static_assert(std::is_same_v<typename CViewType::value_type, typename CViewType::non_const_value_type>,
                "KokkosBatched::syr2k: CViewType must have non-const value type.");
  static_assert(AViewType::rank == 2, "KokkosBatched::syr2k: AViewType must have rank 2.");
  static_assert(BViewType::rank == 2, "KokkosBatched::syr2k: BViewType must have rank 2.");
  static_assert(CViewType::rank == 2, "KokkosBatched::syr2k: CViewType must have rank 2.");
#ifndef NDEBUG
  const int ldc = C.extent_int(0), n = C.extent_int(1);
  const int lda   = A.extent_int(0);
  const int k     = is_trans ? A.extent_int(0) : A.extent_int(1);
  const int nrowa = is_trans ? k : n;

  if (A.extent_int(0) != B.extent_int(0) || A.extent_int(1) != B.extent_int(1)) {
    Kokkos::printf(
        "KokkosBatched::syr2k: dimensions of A and B must match: "
        "A = (%d, %d), B = (%d, %d)\n",
        A.extent_int(0), A.extent_int(1), B.extent_int(0), B.extent_int(1));
    return 1;
  }

  if (lda < Kokkos::max(1, nrowa)) {
    Kokkos::printf(
        "KokkosBatched::syr2k: leading dimension of A must not be smaller than "
        "max(1, nrowa): "
        "lda = %d, nrowa = %d\n",
        lda, nrowa);
    return 1;
  }

  if (ldc < Kokkos::max(1, n)) {
    Kokkos::printf(
        "KokkosBatched::syr2k: leading dimension of C must not be smaller than "
        "max(1, n): "
        "ldc = %d, n = %d\n",
        ldc, n);
    return 1;
  }
#endif
  return 0;
}
}  // namespace Impl

/// Serial Batched Syr2k:
/// {s,d,c,z}syr2k and {c,z}her2k interface
/// Performs one of the symmetric rank 2k operations
///   C := alpha*A*B^T + alpha*B*A^T + beta*C
///   C := alpha*A*B^H + alpha*B*A^H + beta*C
///   C := alpha*A^T*B + alpha*B^T*A + beta*C
///   C := alpha*A^H*B + alpha*B^H*A + beta*C
template <typename ArgUplo, typename ArgTrans>
template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
KOKKOS_INLINE_FUNCTION int SerialSyr2k<ArgUplo, ArgTrans>::invoke(const ScalarType alpha, const AViewType &A,
                                                                  const BViewType &B, const ScalarType beta,
                                                                  const CViewType &C) {
  // Quick return if possible
  constexpr bool is_trans = std::same_as<ArgTrans, Trans::Transpose> || std::same_as<ArgTrans, Trans::ConjTranspose>;
  const int n             = C.extent_int(1);
  const int k             = is_trans ? A.extent_int(0) : A.extent_int(1);
  if (n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

  auto info = Impl::checkSyr2kInput<is_trans>(A, B, C);
  if (info) return info;

  using op = std::conditional_t<(std::same_as<ArgTrans, Trans::ConjNoTranspose> ||
                                 std::same_as<ArgTrans, Trans::ConjTranspose>),
                                KokkosBlas::Impl::OpConj, KokkosBlas::Impl::OpID>;

  using sym_op = std::conditional_t<(std::same_as<ArgTrans, Trans::ConjNoTranspose> ||
                                     std::same_as<ArgTrans, Trans::ConjTranspose>),
                                    KokkosBlas::Impl::OpReal, KokkosBlas::Impl::OpID>;

  if constexpr (std::same_as<ArgUplo, Uplo::Lower>) {
    return Impl::SerialSyr2kInternalLower::invoke<is_trans>(op(), sym_op(), n, k, alpha, A.data(), A.stride(0),
                                                            A.stride(1), B.data(), B.stride(0), B.stride(1), beta,
                                                            C.data(), C.stride(0), C.stride(1));
  } else {
    return Impl::SerialSyr2kInternalUpper::invoke<is_trans>(op(), sym_op(), n, k, alpha, A.data(), A.stride(0),
                                                            A.stride(1), B.data(), B.stride(0), B.stride(1), beta,
                                                            C.data(), C.stride(0), C.stride(1));
  }
}

/// Team Batched Syr2k:
/// {s,d,c,z}syr2k and {c,z}her2k interface
/// Performs one of the symmetric rank 2k operations
///   C := alpha*A*B^T + alpha*B*A^T + beta*C
///   C := alpha*A*B^H + alpha*B*A^H + beta*C
///   C := alpha*A^T*B + alpha*B^T*A + beta*C
///   C := alpha*A^H*B + alpha*B^H*A + beta*C
template <typename MemberType, typename ArgUplo, typename ArgTrans>
template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
KOKKOS_INLINE_FUNCTION int TeamSyr2k<MemberType, ArgUplo, ArgTrans>::invoke(const MemberType &member,
                                                                            const ScalarType alpha, const AViewType &A,
                                                                            const BViewType &B, const ScalarType beta,
                                                                            const CViewType &C) {
  // Quick return if possible
  constexpr bool is_trans = std::same_as<ArgTrans, Trans::Transpose> || std::same_as<ArgTrans, Trans::ConjTranspose>;
  const int n             = C.extent_int(1);
  const int k             = is_trans ? A.extent_int(0) : A.extent_int(1);
  if (n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

  auto info = Impl::checkSyr2kInput<is_trans>(A, B, C);
  if (info) return info;

  using op = std::conditional_t<(std::same_as<ArgTrans, Trans::ConjNoTranspose> ||
                                 std::same_as<ArgTrans, Trans::ConjTranspose>),
                                KokkosBlas::Impl::OpConj, KokkosBlas::Impl::OpID>;

  using sym_op = std::conditional_t<(std::same_as<ArgTrans, Trans::ConjNoTranspose> ||
                                     std::same_as<ArgTrans, Trans::ConjTranspose>),
                                    KokkosBlas::Impl::OpReal, KokkosBlas::Impl::OpID>;

  if constexpr (std::same_as<ArgUplo, Uplo::Lower>) {
    return Impl::TeamSyr2kInternalLower::invoke<is_trans>(member, op(), sym_op(), n, k, alpha, A.data(), A.stride(0),
                                                          A.stride(1), B.data(), B.stride(0), B.stride(1), beta,
                                                          C.data(), C.stride(0), C.stride(1));
  } else {
    return Impl::TeamSyr2kInternalUpper::invoke<is_trans>(member, op(), sym_op(), n, k, alpha, A.data(), A.stride(0),
                                                          A.stride(1), B.data(), B.stride(0), B.stride(1), beta,
                                                          C.data(), C.stride(0), C.stride(1));
  }
}

/// TeamVector Batched Syr2k:
/// {s,d,c,z}syr2k and {c,z}her2k interface
/// Performs one of the symmetric rank 2k operations
///   C := alpha*A*B^T + alpha*B*A^T + beta*C
///   C := alpha*A*B^H + alpha*B*A^H + beta*C
///   C := alpha*A^T*B + alpha*B^T*A + beta*C
///   C := alpha*A^H*B + alpha*B^H*A + beta*C
template <typename MemberType, typename ArgUplo, typename ArgTrans>
template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
KOKKOS_INLINE_FUNCTION int TeamVectorSyr2k<MemberType, ArgUplo, ArgTrans>::invoke(
    const MemberType &member, const ScalarType alpha, const AViewType &A, const BViewType &B, const ScalarType beta,
    const CViewType &C) {
  // Quick return if possible
  constexpr bool is_trans = std::same_as<ArgTrans, Trans::Transpose> || std::same_as<ArgTrans, Trans::ConjTranspose>;
  const int n             = C.extent_int(1);
  const int k             = is_trans ? A.extent_int(0) : A.extent_int(1);
  if (n == 0 || ((alpha == ScalarType(0) || k == 0) && beta == ScalarType(1))) return 0;

  auto info = Impl::checkSyr2kInput<is_trans>(A, B, C);
  if (info) return info;

  using op = std::conditional_t<(std::same_as<ArgTrans, Trans::ConjNoTranspose> ||
                                 std::same_as<ArgTrans, Trans::ConjTranspose>),
                                KokkosBlas::Impl::OpConj, KokkosBlas::Impl::OpID>;

  using sym_op = std::conditional_t<(std::same_as<ArgTrans, Trans::ConjNoTranspose> ||
                                     std::same_as<ArgTrans, Trans::ConjTranspose>),
                                    KokkosBlas::Impl::OpReal, KokkosBlas::Impl::OpID>;

  if constexpr (std::same_as<ArgUplo, Uplo::Lower>) {
    return Impl::TeamVectorSyr2kInternalLower::invoke<is_trans>(member, op(), sym_op(), n, k, alpha, A.data(),
                                                                A.stride(0), A.stride(1), B.data(), B.stride(0),
                                                                B.stride(1), beta, C.data(), C.stride(0), C.stride(1));
  } else {
    return Impl::TeamVectorSyr2kInternalUpper::invoke<is_trans>(member, op(), sym_op(), n, k, alpha, A.data(),
                                                                A.stride(0), A.stride(1), B.data(), B.stride(0),
                                                                B.stride(1), beta, C.data(), C.stride(0), C.stride(1));
  }
}

}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_SYRK_IMPL_HPP_
