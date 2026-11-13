// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef KOKKOSBATCHED_SYPMV_HPP_
#define KOKKOSBATCHED_SYPMV_HPP_

#include <concepts>
#include <KokkosBatched_Util.hpp>

/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

namespace KokkosBatched {

/// \brief Batched Serial Sypmv:
///
/// performs the matrix-vector operation
///   y := alpha*A*x + beta*y,
///   alpha and beta are scalars, x and y are n element vectors, and A is an n by n symmetric or hermitian matrix,
///   supplied in packed form.
///
/// \tparam ArgUplo: Type indicating whether the A is upper (Uplo::Upper) or lower (Uplo::Lower) triangular.
/// \tparam Conj: Boolean indicating whether the matrix A is symmetric (false) or hermitian (true)
template <typename ArgUplo, bool Conj>
struct SerialSypmv {
  static_assert(is_uplo_v<ArgUplo>, "KokkosBatched::sypmv: ArgUplo must be a KokkosBatched::Uplo.");
  /// \tparam ScalarType: Input type for the scalar alpha
  /// \tparam APViewType: Input type for the matrix A, needs to be a 1D view (packed storage)
  /// \tparam XViewType: Input type for the vector x, needs to be a 1D view
  /// \tparam YViewType: output type for the vector y, needs to be a 1D view
  ///
  /// \param[in] alpha: alpha is a scalar
  /// \param[inout] ap: ap is a n by n matrix, supplied in packed form, a rank 1 view
  /// \param[in] x: x is a length n vector, a rank 1 view
  /// \param[in] beta: beta is a scalar
  /// \param[in,out] y: y is a length n vector, a rank 1 view
  ///
  /// No nested parallel_for is used inside of the function.
  ///
  template <typename ScalarType, typename APViewType, typename XViewType, typename YViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const ScalarType alpha, const APViewType &ap, const XViewType &x,
                                           const ScalarType beta, const YViewType &y);
};

/// \brief Batched Team Sypmv:
///
/// performs the matrix-vector operation
///   y := alpha*A*x + beta*y,
///   alpha and beta are scalars, x and y are n element vectors, and A is an n by n symmetric or hermitian matrix,
///   supplied in packed form.
///
/// \tparam MemberType: Member type of the Kokkos team policy
/// \tparam ArgUplo: Type indicating whether the A is upper (Uplo::Upper) or lower (Uplo::Lower) triangular.
/// \tparam Conj: Boolean indicating whether the matrix A is symmetric (false) or hermitian (true)
template <typename MemberType, typename ArgUplo, bool Conj>
struct TeamSypmv {
  static_assert(is_uplo_v<ArgUplo>, "KokkosBatched::sypmv: ArgUplo must be a KokkosBatched::Uplo.");
  /// \tparam ScalarType: Input type for the scalar alpha
  /// \tparam APViewType: Input type for the matrix A, needs to be a 1D view (packed storage)
  /// \tparam XViewType: Input type for the vector x, needs to be a 1D view
  /// \tparam YViewType: output type for the vector y, needs to be a 1D view
  ///
  /// \param[in] alpha: alpha is a scalar
  /// \param[inout] ap: ap is a n by n matrix, supplied in packed form, a rank 1 view
  /// \param[in] x: x is a length n vector, a rank 1 view
  /// \param[in] beta: beta is a scalar
  /// \param[in,out] y: y is a length n vector, a rank 1 view
  ///
  /// Team thread parallelization is used inside of the function.
  ///
  template <typename ScalarType, typename APViewType, typename XViewType, typename YViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const APViewType &ap,
                                           const XViewType &x, const ScalarType beta, const YViewType &y);
};

/// \brief Batched TeamVector Sypmv:
///
/// performs the matrix-vector operation
///   y := alpha*A*x + beta*y,
///   alpha and beta are scalars, x and y are n element vectors, and A is an n by n symmetric or hermitian matrix,
///   supplied in packed form.
///
/// \tparam MemberType: Member type of the Kokkos team policy
/// \tparam ArgUplo: Type indicating whether the A is upper (Uplo::Upper) or lower (Uplo::Lower) triangular.
/// \tparam Conj: Boolean indicating whether the matrix A is symmetric (false) or hermitian (true)
template <typename MemberType, typename ArgUplo, bool Conj>
struct TeamVectorSypmv {
  static_assert(is_uplo_v<ArgUplo>, "KokkosBatched::sypmv: ArgUplo must be a KokkosBatched::Uplo.");
  /// \tparam ScalarType: Input type for the scalar alpha
  /// \tparam APViewType: Input type for the matrix A, needs to be a 1D view (packed storage)
  /// \tparam XViewType: Input type for the vector x, needs to be a 1D view
  /// \tparam YViewType: output type for the vector y, needs to be a 1D view
  ///
  /// \param[in] alpha: alpha is a scalar
  /// \param[inout] ap: ap is a n by n matrix, supplied in packed form, a rank 1 view
  /// \param[in] x: x is a length n vector, a rank 1 view
  /// \param[in] beta: beta is a scalar
  /// \param[in,out] y: y is a length n vector, a rank 1 view
  ///
  /// Team vector parallelization is used inside of the function.
  ///
  template <typename ScalarType, typename APViewType, typename XViewType, typename YViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const APViewType &ap,
                                           const XViewType &x, const ScalarType beta, const YViewType &y);
};

}  // namespace KokkosBatched

#include "KokkosBatched_Sypmv_Impl.hpp"

#endif  // KOKKOSBATCHED_SYPMV_HPP_
