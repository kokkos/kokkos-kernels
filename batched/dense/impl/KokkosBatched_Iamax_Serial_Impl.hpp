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

#ifndef KOKKOSBATCHED_IAMAX_SERIAL_IMPL_HPP_
#define KOKKOSBATCHED_IAMAX_SERIAL_IMPL_HPP_

/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

#include "KokkosBatched_Iamax_Serial_Internal.hpp"

namespace KokkosBatched {

template <typename XViewType>
KOKKOS_INLINE_FUNCTION int SerialIamax::invoke(const XViewType &x) {
  if (x.extent(0) <= 0) return -1;
  if (x.extent(0) == 1) return 0;
  return KokkosBatched::Impl::SerialIamaxInternal::invoke(x.extent(0), x.data(), x.stride(0));
}

}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_IAMAX_SERIAL_IMPL_HPP_
