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
#ifndef KOKKOSBATCHED_GETRF_HPP_
#define KOKKOSBATCHED_GETRF_HPP_

#include <KokkosBatched_Util.hpp>

/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

namespace KokkosBatched {

/// \brief Serial Batched Getrf:
/// Compute a LU factorization of a general m-by-n matrix A using partial
///   pivoting with row interchanges.
///   The factorization has the form
///     A = P * L * U
///   where P is a permutation matrix, L is lower triangular with unit
///   diagonal elements (lower trapezoidal if m > n), and U is upper
///   triangular (upper trapezoidal if m < n).
/// \tparam AViewType: Input type for the matrix, needs to be a 2D view
/// \tparam PivViewType: Input type for the pivot indices, needs to be a 1D view
///
/// \param A [inout]: A is a m by n general matrix, a rank 2 view
/// \param piv [out]: On exit, the pivot indices, a rank 1 view
///
/// No nested parallel_for is used inside of the function.
///

template <typename ArgAlgo>
struct SerialGetrf {
  template <typename AViewType, typename PivViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A, const PivViewType &piv);
};
}  // namespace KokkosBatched

#include "KokkosBatched_Getrf_Serial_Impl.hpp"

#endif  // KOKKOSBATCHED_GETRF_HPP_
