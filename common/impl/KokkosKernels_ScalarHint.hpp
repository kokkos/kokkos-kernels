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

#ifndef KOKKOSKERNELS_SCALARHINT_HPP
#define KOKKOSKERNELS_SCALARHINT_HPP

namespace KokkosKernels::Impl {

/*! An enum that can be used as a template param to optimize an implementation
*/
enum class ScalarHint : int {
  none,
  zero,
  pos_one,
  neg_one
};

} // namespace KokkosKernels::Impl

#endif // KOKKOSKERNELS_SCALARHINT_HPP