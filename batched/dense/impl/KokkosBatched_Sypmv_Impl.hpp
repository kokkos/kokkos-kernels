// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBATCHED_SYPMV_IMPL_HPP_
#define KOKKOSBATCHED_SYPMV_IMPL_HPP_

#include <concepts>
#include <KokkosBlas_util.hpp>
#include <KokkosBatched_Util.hpp>
#include "KokkosBatched_Sypmv_Internal.hpp"

namespace KokkosBatched {
namespace Impl {
template <typename APViewType, typename XViewType, typename YViewType>
KOKKOS_INLINE_FUNCTION static int checkSypmvInput([[maybe_unused]] const APViewType &ap,
                                                  [[maybe_unused]] const XViewType &x,
                                                  [[maybe_unused]] const YViewType &y) {
  static_assert(Kokkos::is_view_v<APViewType>, "KokkosBatched::sypmv: APViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view_v<XViewType>, "KokkosBatched::sypmv: XViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view_v<YViewType>, "KokkosBatched::sypmv: YViewType is not a Kokkos::View.");

  static_assert(APViewType::rank == 1, "KokkosBatched::sypmv: APViewType must have rank 1.");
  static_assert(XViewType::rank == 1, "KokkosBatched::sypmv: XViewType must have rank 1.");
  static_assert(YViewType::rank == 1, "KokkosBatched::sypmv: YViewType must have rank 1.");
  static_assert(!std::is_const_v<typename YViewType::value_type>,
                "KokkosBatched::sypmv: YViewType must have non-const value type.");
#ifndef NDEBUG
  const int x_len = x.extent_int(0), y_len = y.extent_int(0);
  if (x_len != y_len) {
    Kokkos::printf(
        "KokkosBatched::sypmv: x and y must have the same length: x_len = "
        "%d, y_len = %d\n",
        x_len, y_len);
    return 1;
  }

  if (ap.extent_int(0) < x_len * (x_len + 1) / 2) {
    Kokkos::printf(
        "KokkosBatched::sypmv: size of packed A must not be smaller than n*(n+1)/2: "
        "size = %d, n = %d\n",
        ap.extent_int(0), x_len);
    return 1;
  }
#endif
  return 0;
}
}  // namespace Impl

/// Serial Batched Sypmv:
template <typename ArgUplo, bool Conj>
template <typename ScalarType, typename APViewType, typename XViewType, typename YViewType>
KOKKOS_INLINE_FUNCTION int SerialSypmv<ArgUplo, Conj>::invoke(const ScalarType alpha, const APViewType &ap,
                                                              const XViewType &x, const ScalarType beta,
                                                              const YViewType &y) {
  // Quick return if possible
  const int n = x.extent_int(0);
  if (n == 0 || (alpha == ScalarType(0) && beta == ScalarType(1))) return 0;

  auto info = Impl::checkSypmvInput(ap, x, y);
  if (info) return info;

  using op     = std::conditional_t<Conj, KokkosBlas::Impl::OpConj, KokkosBlas::Impl::OpID>;
  using sym_op = std::conditional_t<Conj, KokkosBlas::Impl::OpReal, KokkosBlas::Impl::OpID>;

  return Impl::SerialSypmvInternal::invoke<ArgUplo>(op(), sym_op(), n, alpha, ap.data(), ap.stride(0), x.data(),
                                                    x.stride(0), beta, y.data(), y.stride(0));
}

/// Team Batched Sypmv:
template <typename MemberType, typename ArgUplo, bool Conj>
template <typename ScalarType, typename APViewType, typename XViewType, typename YViewType>
KOKKOS_INLINE_FUNCTION int TeamSypmv<MemberType, ArgUplo, Conj>::invoke(const MemberType &member,
                                                                        const ScalarType alpha, const APViewType &ap,
                                                                        const XViewType &x, const ScalarType beta,
                                                                        const YViewType &y) {
  // Quick return if possible
  const int n = x.extent_int(0);
  if (n == 0 || (alpha == ScalarType(0) && beta == ScalarType(1))) return 0;

  auto info = Impl::checkSypmvInput(ap, x, y);
  if (info) return info;

  using op     = std::conditional_t<Conj, KokkosBlas::Impl::OpConj, KokkosBlas::Impl::OpID>;
  using sym_op = std::conditional_t<Conj, KokkosBlas::Impl::OpReal, KokkosBlas::Impl::OpID>;

  return Impl::TeamSypmvInternal::invoke<ArgUplo>(member, op(), sym_op(), n, alpha, ap.data(), ap.stride(0), x.data(),
                                                  x.stride(0), beta, y.data(), y.stride(0));
}

/// TeamVector Batched Sypmv:
template <typename MemberType, typename ArgUplo, bool Conj>
template <typename ScalarType, typename APViewType, typename XViewType, typename YViewType>
KOKKOS_INLINE_FUNCTION int TeamVectorSypmv<MemberType, ArgUplo, Conj>::invoke(const MemberType &member,
                                                                              const ScalarType alpha,
                                                                              const APViewType &ap, const XViewType &x,
                                                                              const ScalarType beta,
                                                                              const YViewType &y) {
  // Quick return if possible
  const int n = x.extent_int(0);
  if (n == 0 || (alpha == ScalarType(0) && beta == ScalarType(1))) return 0;

  auto info = Impl::checkSypmvInput(ap, x, y);
  if (info) return info;

  using op     = std::conditional_t<Conj, KokkosBlas::Impl::OpConj, KokkosBlas::Impl::OpID>;
  using sym_op = std::conditional_t<Conj, KokkosBlas::Impl::OpReal, KokkosBlas::Impl::OpID>;

  return Impl::TeamVectorSypmvInternal::invoke<ArgUplo>(member, op(), sym_op(), n, alpha, ap.data(), ap.stride(0),
                                                        x.data(), x.stride(0), beta, y.data(), y.stride(0));
}

}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_SYPMV_IMPL_HPP_
