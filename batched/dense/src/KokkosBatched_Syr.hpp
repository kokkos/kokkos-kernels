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
#ifndef KOKKOSBATCHED_SYR_HPP_
#define KOKKOSBATCHED_SYR_HPP_

#include <KokkosBatched_Util.hpp>

/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

namespace KokkosBatched {

/// \brief Serial Batched Syr:
/// Performs the symmetric rank 1 operation
///   A := alpha*x*x**T + A or A := alpha*x*x**H + A
///    where alpha is a scalar, x is an n element vector, and A is a n by n symmetric or Hermitian matrix.
///
/// \tparam ScalarType: Input type for the scalar alpha
/// \tparam XViewType: Input type for the vector x, needs to be a 1D view
/// \tparam AViewType: Input/output type for the matrix A, needs to be a 2D view
///
/// \param alpha [in]: alpha is a scalar
/// \param x [in]: x is a length n vector, a rank 1 view
/// \param A [inout]: A is a n by n matrix, a rank 2 view
///
/// No nested parallel_for is used inside of the function.
///
template <typename ArgUplo, typename ArgTrans>
struct SerialSyr {
  template <typename ScalarType, typename XViewType, typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const ScalarType alpha, const XViewType &x, const AViewType &a);
};
}  // namespace KokkosBatched

#include "KokkosBatched_Syr_Serial_Impl.hpp"

#endif  // KOKKOSBATCHED_SYR_HPP_
