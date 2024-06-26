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
#ifndef KOKKOSBATCHED_PTTRF_HPP_
#define KOKKOSBATCHED_PTTRF_HPP_

#include <KokkosBatched_Util.hpp>

/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

namespace KokkosBatched {

/// \brief Serial Batched Pttrf:
///
/// Solve a tridiagonal system of the form Ab_l x_l = b_l for all l = 0, ..., N
///   using the factorization A = U**H *D*U or A = L*D*L**H computed by pttrf.
///   D is a diagonal matrix specified in the vector D, U (or L) is a unit
///   bidiagonal matrix whose superdiagonal (subdiagonal) is specified in the
///   vector E, and X and B are stored in the vector B.
///
/// \tparam DViewType: Input type for the a diagonal matrix, needs to be a 1D
/// view \tparam EViewType: Input type for the a superdiagonal matrix, needs to
/// be a 1D view \tparam BViewType: Input type for the right-hand side and the
/// solution, needs to be a 1D view
///
/// \param d [in]: n diagonal elements of the diagonal matrix D
/// \param e [in]: n diagonal elements of the diagonal matrix E
///
/// No nested parallel_for is used inside of the function.
///

template <typename ArgAlgo>
struct SerialPttrf {
  template <typename DViewType, typename EViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const DViewType &d,
                                           const EViewType &e);
};

}  // namespace KokkosBatched

#include "KokkosBatched_Pttrf_Serial_Impl.hpp"

#endif  // KOKKOSBATCHED_TBSV_HPP_
