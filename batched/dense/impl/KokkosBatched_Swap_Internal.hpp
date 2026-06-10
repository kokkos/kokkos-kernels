// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBATCHED_SWAP_INTERNAL_HPP_
#define KOKKOSBATCHED_SWAP_INTERNAL_HPP_

#include <KokkosBatched_Util.hpp>

namespace KokkosBatched {
namespace Impl {

///
/// Serial Internal Impl
/// ====================
struct SerialSwapInternal {
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION static void invoke(const int n, ValueType *KOKKOS_RESTRICT x, const int xs0,
                                            ValueType *KOKKOS_RESTRICT y, const int ys0);
};

template <typename ValueType>
KOKKOS_INLINE_FUNCTION void SerialSwapInternal::invoke(const int n, ValueType *KOKKOS_RESTRICT x, const int xs0,
                                                       ValueType *KOKKOS_RESTRICT y, const int ys0) {
  for (int i = 0; i < n; ++i) {
    const ValueType temp = x[i * xs0];
    x[i * xs0]           = y[i * ys0];
    y[i * ys0]           = temp;
  }
}

///
/// Team Internal Impl
/// ==================
template <typename MemberType>
struct TeamSwapInternal {
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION static void invoke(const MemberType &member, const int n, ValueType *KOKKOS_RESTRICT x,
                                            const int xs0, ValueType *KOKKOS_RESTRICT y, const int ys0);
};

template <typename MemberType>
template <typename ValueType>
KOKKOS_INLINE_FUNCTION void TeamSwapInternal<MemberType>::invoke(const MemberType &member, const int n,
                                                                 ValueType *KOKKOS_RESTRICT x, const int xs0,
                                                                 ValueType *KOKKOS_RESTRICT y, const int ys0) {
  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n), [&](const int i) {
    const ValueType temp = x[i * xs0];
    x[i * xs0]           = y[i * ys0];
    y[i * ys0]           = temp;
  });
}

///
/// TeamVector Internal Impl
/// ========================
template <typename MemberType>
struct TeamVectorSwapInternal {
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION static void invoke(const MemberType &member, const int n, ValueType *KOKKOS_RESTRICT x,
                                            const int xs0, ValueType *KOKKOS_RESTRICT y, const int ys0);
};

template <typename MemberType>
template <typename ValueType>
KOKKOS_INLINE_FUNCTION void TeamVectorSwapInternal<MemberType>::invoke(const MemberType &member, const int n,
                                                                       ValueType *KOKKOS_RESTRICT x, const int xs0,
                                                                       ValueType *KOKKOS_RESTRICT y, const int ys0) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n), [&](const int i) {
    const ValueType temp = x[i * xs0];
    x[i * xs0]           = y[i * ys0];
    y[i * ys0]           = temp;
  });
}

}  // namespace Impl
}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_SWAP_INTERNAL_HPP_
