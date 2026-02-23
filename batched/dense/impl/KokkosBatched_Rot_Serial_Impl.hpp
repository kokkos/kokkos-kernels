// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBATCHED_ROT_SERIAL_IMPL_HPP_
#define KOKKOSBATCHED_ROT_SERIAL_IMPL_HPP_

#include <KokkosBlas_util.hpp>
#include <KokkosBatched_Util.hpp>
#include "KokkosBatched_Rot_Serial_Internal.hpp"

namespace KokkosBatched {
namespace Impl {
template <typename XViewType, typename YViewType>
KOKKOS_INLINE_FUNCTION static int checkRotInput([[maybe_unused]] const XViewType &x,
                                                [[maybe_unused]] const YViewType &y) {
  static_assert(Kokkos::is_view_v<XViewType>, "KokkosBatched::rot: XViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view_v<YViewType>, "KokkosBatched::rot: YViewType is not a Kokkos::View.");
  static_assert(XViewType::rank == 1, "KokkosBatched::rot: XViewType must have rank 1.");
  static_assert(YViewType::rank == 1, "KokkosBatched::rot: YViewType must have rank 1.");
#ifndef NDEBUG
  const int n = x.extent_int(0);

  if (n < 0) {
    Kokkos::printf(
        "KokkosBatched::rot: input parameter n must not be less than 0: n "
        "= "
        "%d\n",
        n);
    return 1;
  }

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

// {s,d,c,z}rot / {c,z}drot interface
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

    auto info = Impl::checkRotInput(x, y);
    if (info) return info;

    return Impl::SerialRotInternal::invoke(KokkosBlas::Impl::OpID(), n, x.data(), x.stride(0),
                                           y.data(), y.stride(0), c, s);
  }
};

// zrot interface
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

    auto info = Impl::checkRotInput(x, y);
    if (info) return info;

    return Impl::SerialRotInternal::invoke(KokkosBlas::Impl::OpConj(), n, x.data(), x.stride(0),
                                           y.data(), y.stride(0), c, s);
  }
};

}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_ROT_SERIAL_IMPL_HPP_
