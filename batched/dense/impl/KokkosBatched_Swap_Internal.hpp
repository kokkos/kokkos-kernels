// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBATCHED_SWAP_INTERNAL_HPP_
#define KOKKOSBATCHED_SWAP_INTERNAL_HPP_

#include <concepts>
#include <KokkosBlas_util.hpp>
#include <KokkosBatched_Util.hpp>

namespace KokkosBatched {
namespace Impl {

// Concept to check if the value types of x and y are swappable
// (either the same type or both floating-point types)
template <typename T1, typename T2>
concept swappable_elements = std::same_as<T1, T2> || (std::floating_point<T1> && std::floating_point<T2>) ||
                             (KokkosKernels::ArithTraits<T1>::is_complex && KokkosKernels::ArithTraits<T2>::is_complex);

template <typename T1, typename T2>
  requires swappable_elements<T1, T2>
KOKKOS_INLINE_FUNCTION void swap_elements(T1 *KOKKOS_RESTRICT a, T2 *KOKKOS_RESTRICT b) {
  const T1 temp = static_cast<T1>(*b);
  *b            = static_cast<T2>(*a);
  *a            = temp;
}

///
/// Serial Internal Impl
/// ====================
struct SerialSwapInternal {
  template <typename XValueType, typename YValueType>
  KOKKOS_INLINE_FUNCTION static void invoke(const int n, XValueType *KOKKOS_RESTRICT x, const int xs0,
                                            YValueType *KOKKOS_RESTRICT y, const int ys0);
};

template <typename XValueType, typename YValueType>
KOKKOS_INLINE_FUNCTION void SerialSwapInternal::invoke(const int n, XValueType *KOKKOS_RESTRICT x, const int xs0,
                                                       YValueType *KOKKOS_RESTRICT y, const int ys0) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int i = 0; i < n; ++i) {
    swap_elements(&x[i * xs0], &y[i * ys0]);
  }
}

///
/// Team Internal Impl
/// ==================
template <typename MemberType>
struct TeamSwapInternal {
  template <typename XValueType, typename YValueType>
  KOKKOS_INLINE_FUNCTION static void invoke(const MemberType &member, const int n, XValueType *KOKKOS_RESTRICT x,
                                            const int xs0, YValueType *KOKKOS_RESTRICT y, const int ys0);
};

template <typename MemberType>
template <typename XValueType, typename YValueType>
KOKKOS_INLINE_FUNCTION void TeamSwapInternal<MemberType>::invoke(const MemberType &member, const int n,
                                                                 XValueType *KOKKOS_RESTRICT x, const int xs0,
                                                                 YValueType *KOKKOS_RESTRICT y, const int ys0) {
  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n),
                       [&](const int i) { swap_elements(&x[i * xs0], &y[i * ys0]); });
}

///
/// TeamVector Internal Impl
/// ========================
template <typename MemberType>
struct TeamVectorSwapInternal {
  template <typename XValueType, typename YValueType>
  KOKKOS_INLINE_FUNCTION static void invoke(const MemberType &member, const int n, XValueType *KOKKOS_RESTRICT x,
                                            const int xs0, YValueType *KOKKOS_RESTRICT y, const int ys0);
};

template <typename MemberType>
template <typename XValueType, typename YValueType>
KOKKOS_INLINE_FUNCTION void TeamVectorSwapInternal<MemberType>::invoke(const MemberType &member, const int n,
                                                                       XValueType *KOKKOS_RESTRICT x, const int xs0,
                                                                       YValueType *KOKKOS_RESTRICT y, const int ys0) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n),
                       [&](const int i) { swap_elements(&x[i * xs0], &y[i * ys0]); });
}

}  // namespace Impl
}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_SWAP_INTERNAL_HPP_
