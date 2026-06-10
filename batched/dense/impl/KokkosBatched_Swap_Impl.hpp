// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBATCHED_SWAP_IMPL_HPP_
#define KOKKOSBATCHED_SWAP_IMPL_HPP_

#include <concepts>
#include <Kokkos_Core.hpp>
#include <KokkosBlas_util.hpp>
#include <KokkosBatched_Util.hpp>
#include "KokkosBatched_Swap_Internal.hpp"

namespace KokkosBatched {
namespace Impl {

// Concept to check if the value types of x and y are swappable
// (either the same type or both floating-point types)
template <typename T1, typename T2>
concept swappable_elements = std::same_as<T1, T2> || (std::is_floating_point_v<T1> && std::is_floating_point_v<T2>) ||
                             (KokkosKernels::ArithTraits<T1>::is_complex && KokkosKernels::ArithTraits<T2>::is_complex);

template <typename XViewType, typename YViewType>
KOKKOS_INLINE_FUNCTION static int checkSwapInput([[maybe_unused]] const XViewType &x,
                                                 [[maybe_unused]] const YViewType &y) {
  static_assert(Kokkos::is_view_v<XViewType>, "KokkosBatched::swap: XViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view_v<YViewType>, "KokkosBatched::swap: YViewType is not a Kokkos::View.");
  static_assert(XViewType::rank() == 1, "KokkosBatched::swap: XViewType must have rank 1.");
  static_assert(YViewType::rank() == 1, "KokkosBatched::swap: YViewType must have rank 1.");
  static_assert(std::is_same_v<typename XViewType::value_type, typename XViewType::non_const_value_type>,
                "KokkosBatched::swap: XViewType must have non-const value type.");
  static_assert(std::is_same_v<typename YViewType::value_type, typename YViewType::non_const_value_type>,
                "KokkosBatched::swap: YViewType must have non-const value type.");
  static_assert(swappable_elements<typename XViewType::non_const_value_type, typename YViewType::non_const_value_type>,
                "KokkosBatched::swap: XViewType and YViewType must have swappable value types.");

#ifndef NDEBUG
  const int n = x.extent_int(0);
  if (n != y.extent_int(0)) {
    Kokkos::printf(
        "KokkosBatched::swap: x and y must have the same length: x length "
        "= "
        "%d, y length = %d\n",
        n, y.extent_int(0));
    return 1;  // Size mismatch
  }
#endif
  return 0;
}
}  // namespace Impl

///
/// Serial Impl
/// ===========
template <typename XViewType, typename YViewType>
KOKKOS_INLINE_FUNCTION int SerialSwap::invoke(const XViewType &x, const YViewType &y) {
  Impl::checkSwapInput(x, y);
  const int n = x.extent_int(0);
  if (n == 0) return 0;
  Impl::SerialSwapInternal::invoke(n, x.data(), x.stride(0), y.data(), y.stride(0));
  return 0;
}

///
/// Team Impl
/// =========

template <typename MemberType>
template <typename XViewType, typename YViewType>
KOKKOS_INLINE_FUNCTION int TeamSwap<MemberType>::invoke(const MemberType &member, const XViewType &x,
                                                        const YViewType &y) {
  Impl::checkSwapInput(x, y);
  const int n = x.extent_int(0);
  if (n == 0) return 0;
  Impl::TeamSwapInternal<MemberType>::invoke(member, n, x.data(), x.stride(0), y.data(), y.stride(0));
  return 0;
}

///
/// TeamVector Impl
/// ===============
template <typename MemberType>
template <typename XViewType, typename YViewType>
KOKKOS_INLINE_FUNCTION int TeamVectorSwap<MemberType>::invoke(const MemberType &member, const XViewType &x,
                                                              const YViewType &y) {
  Impl::checkSwapInput(x, y);
  const int n = x.extent_int(0);
  if (n == 0) return 0;
  Impl::TeamVectorSwapInternal<MemberType>::invoke(member, n, x.data(), x.stride(0), y.data(), y.stride(0));
  return 0;
}

}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_SWAP_IMPL_HPP_
