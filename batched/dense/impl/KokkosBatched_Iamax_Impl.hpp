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

#ifndef KOKKOSBATCHED_IAMAX_IMPL_HPP_
#define KOKKOSBATCHED_IAMAX_IMPL_HPP_

#include "KokkosBatched_Util.hpp"
#include <Kokkos_InnerProductSpaceTraits.hpp>

namespace KokkosBatched {

///
/// Serial Internal Impl
/// ========================

struct SerialIamaxInternal {
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int n, const ValueType *KOKKOS_RESTRICT x, const int xs0);
};

template <typename ValueType>
KOKKOS_INLINE_FUNCTION int SerialIamaxInternal::invoke(const int n, const ValueType *KOKKOS_RESTRICT x, const int xs0) {
  using IPT = Kokkos::Details::InnerProductSpaceTraits<ValueType>;

  ValueType amax = IPT::norm(x[0 * xs0]);
  int imax       = 0;

  for (int i = 1; i < n; ++i) {
    const ValueType abs_x_i = IPT::norm(x[i * xs0]);
    if (abs_x_i > amax) {
      amax = abs_x_i;
      imax = i;
    }
  }

  return imax;
};

template <typename XViewType>
KOKKOS_INLINE_FUNCTION int SerialIamax::invoke(const XViewType &x) {
  if (x.extent(0) <= 0) return -1;
  if (x.extent(0) == 1) return 0;
  return SerialIamaxInternal::invoke(x.extent(0), x.data(), x.stride(0));
}

}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_IAMAX_IMPL_HPP_
