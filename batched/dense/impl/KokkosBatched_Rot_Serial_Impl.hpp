// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBATCHED_ROT_SERIAL_IMPL_HPP_
#define KOKKOSBATCHED_ROT_SERIAL_IMPL_HPP_

#include <KokkosBlas_util.hpp>
#include <KokkosBatched_Util.hpp>
#include "KokkosBatched_Rot_Serial_Internal.hpp"

namespace KokkosBatched {
namespace Impl {
template <typename ArgTrans, typename CType, typename SType, typename XViewType, typename YViewType>
KOKKOS_INLINE_FUNCTION static int checkRotInput([[maybe_unused]] const XViewType &x,
                                                [[maybe_unused]] const YViewType &y) {
  static_assert(Kokkos::is_view_v<XViewType>, "KokkosBatched::rot: XViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view_v<YViewType>, "KokkosBatched::rot: YViewType is not a Kokkos::View.");
  static_assert(XViewType::rank() == 1, "KokkosBatched::rot: XViewType must have rank 1.");
  static_assert(YViewType::rank() == 1, "KokkosBatched::rot: YViewType must have rank 1.");
  static_assert(std::is_same_v<typename XViewType::value_type, typename XViewType::non_const_value_type>,
                "KokkosBatched::rot: XViewType must have non-const value type.");
  static_assert(std::is_same_v<typename YViewType::value_type, typename YViewType::non_const_value_type>,
                "KokkosBatched::rot: YViewType must have non-const value type.");
  static_assert(!KokkosKernels::ArithTraits<CType>::is_complex, "KokkosBatched::rot: CType must be real.");
  using x_value_type = typename XViewType::non_const_value_type;
  using y_value_type = typename YViewType::non_const_value_type;
  static_assert(
      (KokkosKernels::ArithTraits<x_value_type>::is_complex && KokkosKernels::ArithTraits<y_value_type>::is_complex) ||
          (!KokkosKernels::ArithTraits<x_value_type>::is_complex &&
           !KokkosKernels::ArithTraits<y_value_type>::is_complex),
      "KokkosBatched::rot: XViewType and YViewType must be either both complex or both real.");
  if constexpr (std::is_same_v<ArgTrans, Trans::Transpose>) {
    // {s,d,cs,zd}rot, S must be real
    static_assert(!KokkosKernels::ArithTraits<SType>::is_complex,
                  "KokkosBatched::rot: SType must be real for Trans::Transpose.");
  } else {
    if constexpr (KokkosKernels::ArithTraits<x_value_type>::is_complex) {
      // {c,z}rot, S must be complex
      static_assert(KokkosKernels::ArithTraits<SType>::is_complex,
                    "KokkosBatched::rot: SType must be complex for complex input with Trans::ConjTranspose.");
    } else {
      // {s,d}rot, S must be real
      static_assert(!KokkosKernels::ArithTraits<SType>::is_complex,
                    "KokkosBatched::rot: SType must be real for real input with Trans::ConjTranspose.");
    }
  }

#ifndef NDEBUG
  const int n = x.extent_int(0);

  if (y.extent_int(0) != n) {
    Kokkos::printf(
        "KokkosBatched::rot: x and y must have the same length: x length "
        "= "
        "%d, y length = %d\n",
        n, y.extent_int(0));
    return 1;
  }
#endif
  return 0;
}
}  // namespace Impl

// {s,d,cs,zd}rot interface
// T
// x(i) := c*x(i) + s*y(i)
// y(i) := c*y(i) - s*x(i)
template <>
struct SerialRot<Trans::Transpose> {
  template <typename XViewType, typename YViewType, typename CType, typename SType>
  KOKKOS_INLINE_FUNCTION static int invoke(const XViewType &x, const YViewType &y, const CType c, const SType s) {
    // Quick return if possible
    const int n = x.extent_int(0);
    if (n == 0) return 0;

    auto info = Impl::checkRotInput<Trans::Transpose, CType, SType>(x, y);
    if (info) return info;

    return Impl::SerialRotInternal::invoke(KokkosBlas::Impl::OpID(), n, x.data(), x.stride(0), y.data(), y.stride(0), c,
                                           s);
  }
};

// {c,z}rot interface
// C
// x(i) := c*x(i) + s*y(i)
// y(i) := c*y(i) - conj(s)*x(i)
template <>
struct SerialRot<Trans::ConjTranspose> {
  template <typename XViewType, typename YViewType, typename CType, typename SType>
  KOKKOS_INLINE_FUNCTION static int invoke(const XViewType &x, const YViewType &y, const CType c, const SType s) {
    // Quick return if possible
    const int n = x.extent_int(0);
    if (n == 0) return 0;

    auto info = Impl::checkRotInput<Trans::ConjTranspose, CType, SType>(x, y);
    if (info) return info;

    return Impl::SerialRotInternal::invoke(KokkosBlas::Impl::OpConj(), n, x.data(), x.stride(0), y.data(), y.stride(0),
                                           c, s);
  }
};

}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_ROT_SERIAL_IMPL_HPP_
