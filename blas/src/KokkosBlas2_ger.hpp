/*
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
*/

#ifndef KOKKOSBLAS2_GER_HPP_
#define KOKKOSBLAS2_GER_HPP_

#include <KokkosBlas2_ger_spec.hpp>

namespace KokkosBlas {

/// \brief Rank-1 update of a general matrix: A = A + alpha * x * y^t.
///
/// \tparam AViewType Input matrix, as a 2-D Kokkos::View
/// \tparam XViewType Input vector, as a 1-D Kokkos::View
/// \tparam YViewType Output vector, as a nonconst 1-D Kokkos::View
/// \tparam AlphaCoeffType Type of input coefficient alpha
///
/// \param space [in] execution space instance on which to run the
///   kernel. This may contain information about which stream to
///   run on.
/// \param alpha [in] Input coefficient of A*x
/// \param A [in] Input matrix, as a 2-D Kokkos::View
/// \param x [in] Input vector, as a 1-D Kokkos::View
/// \param y [in/out] Output vector, as a nonconst 1-D Kokkos::View
template <class AViewType, class XViewType, class YViewType>
void ger(const typename AViewType::execution_space& space,
         typename AViewType::const_value_type& alpha, const AViewType& A,
         const XViewType& x,
         const YViewType& y) {
  static_assert(Kokkos::is_view<AViewType>::value,
                "AViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<XViewType>::value,
                "XViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<YViewType>::value,
                "YViewType must be a Kokkos::View.");
  static_assert(static_cast<int>(AViewType::rank) == 2,
                "AViewType must have rank 2.");
  static_assert(static_cast<int>(XViewType::rank) == 1,
                "XViewType must have rank 1.");
  static_assert(static_cast<int>(YViewType::rank) == 1,
                "YViewType must have rank 1.");

  // EEP
}

/// \brief Rank-1 update of a general matrix: A = A + alpha * x * y^t.
///
/// \tparam AViewType Input matrix, as a 2-D Kokkos::View
/// \tparam XViewType Input vector, as a 1-D Kokkos::View
/// \tparam YViewType Output vector, as a nonconst 1-D Kokkos::View
/// \tparam AlphaCoeffType Type of input coefficient alpha
///
/// \param alpha [in] Input coefficient of A*x
/// \param A [in] Input matrix, as a 2-D Kokkos::View
/// \param x [in] Input vector, as a 1-D Kokkos::View
/// \param y [in/out] Output vector, as a nonconst 1-D Kokkos::View
template <class AViewType, class XViewType, class YViewType>
void ger(typename AViewType::const_value_type& alpha,
         const AViewType& A, const XViewType& x,
         const YViewType& y) {
  const typename AViewType::execution_space space =
      typename AViewType::execution_space();
  ger(space, trans, alpha, A, x, y);
}

}  // namespace KokkosBlas

#endif  // KOKKOSBLAS2_GER_HPP_
