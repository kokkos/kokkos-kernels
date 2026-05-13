// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBATCHED_ROTM_IMPL_HPP_
#define KOKKOSBATCHED_ROTM_IMPL_HPP_

#include <KokkosBlas_util.hpp>
#include <KokkosBatched_Util.hpp>
#include "KokkosBatched_Rotm_Internal.hpp"

namespace KokkosBatched {
namespace Impl {
template <typename XViewType, typename YViewType, typename ParamViewType>
KOKKOS_INLINE_FUNCTION static int checkRotmInput([[maybe_unused]] const XViewType &x,
                                                 [[maybe_unused]] const YViewType &y,
                                                 [[maybe_unused]] const ParamViewType &param) {
  static_assert(Kokkos::is_view_v<XViewType>, "KokkosBatched::rot: XViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view_v<YViewType>, "KokkosBatched::rot: YViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view_v<ParamViewType>, "KokkosBatched::rot: ParamViewType is not a Kokkos::View.");
  static_assert(XViewType::rank() == 1, "KokkosBatched::rot: XViewType must have rank 1.");
  static_assert(YViewType::rank() == 1, "KokkosBatched::rot: YViewType must have rank 1.");
  static_assert(ParamViewType::rank() == 1, "KokkosBatched::rot: ParamViewType must have rank 1.");
  static_assert(std::is_same_v<typename XViewType::value_type, typename XViewType::non_const_value_type>,
                "KokkosBatched::rot: XViewType must have non-const value type.");
  static_assert(std::is_same_v<typename YViewType::value_type, typename YViewType::non_const_value_type>,
                "KokkosBatched::rot: YViewType must have non-const value type.");
  using x_value_type     = typename XViewType::non_const_value_type;
  using y_value_type     = typename YViewType::non_const_value_type;
  using param_value_type = typename ParamViewType::non_const_value_type;

  static_assert(!KokkosKernels::ArithTraits<x_value_type>::is_complex &&
                    !KokkosKernels::ArithTraits<y_value_type>::is_complex &&
                    !KokkosKernels::ArithTraits<param_value_type>::is_complex,
                "KokkosBatched::rotm: Complex types are not supported for Rotm.");

#ifndef NDEBUG
  const int n = x.extent_int(0);

  if (y.extent_int(0) != n) {
    Kokkos::printf(
        "KokkosBatched::rotm: x and y must have the same length: x length "
        "= "
        "%d, y length = %d\n",
        n, y.extent_int(0));
    return 1;
  }

  // We handle flag as a template parameter, so we only need to check that the length of param is 4, but not the value
  // of flag
  if (param.extent_int(0) != 4) {
    Kokkos::printf("KokkosBatched::rotm: param must have length 4: param length = %d\n", param.extent_int(0));
    return 1;
  }
#endif
  return 0;
}
}  // namespace Impl

///
/// Serial Impl
/// ===========

template <int Flag>
template <typename XViewType, typename YViewType, typename ParamViewType>
KOKKOS_INLINE_FUNCTION int SerialRotm<Flag>::invoke(const XViewType &x, const YViewType &y,
                                                    const ParamViewType &param) {
  // Quick return if possible
  const int n = x.extent_int(0);
  if (n == 0) return 0;

  auto info = Impl::checkRotmInput(x, y, param);
  if (info) return info;

  if constexpr (Flag != -2) {
    // flag == -2.0: identity, no need to do anything
    Impl::SerialRotmInternal<Flag>::invoke(n, x.data(), x.stride(0), y.data(), y.stride(0), param.data(),
                                           param.stride(0));
  }
  return 0;
}

///
/// Team Impl
/// ===========

template <typename MemberType, int Flag>
template <typename XViewType, typename YViewType, typename ParamViewType>
KOKKOS_INLINE_FUNCTION int TeamRotm<MemberType, Flag>::invoke(const MemberType &member, const XViewType &x,
                                                              const YViewType &y, const ParamViewType &param) {
  // Quick return if possible
  const int n = x.extent_int(0);
  if (n == 0) return 0;

  auto info = Impl::checkRotmInput(x, y, param);
  if (info) return info;

  if constexpr (Flag != -2) {
    // flag == -2.0: identity, no need to do anything
    Impl::TeamRotmInternal<Flag>::invoke(member, n, x.data(), x.stride(0), y.data(), y.stride(0), param.data(),
                                         param.stride(0));
  }
  return 0;
}

///
/// TeamVector Impl
/// ===============

template <typename MemberType, int Flag>
template <typename XViewType, typename YViewType, typename ParamViewType>
KOKKOS_INLINE_FUNCTION int TeamVectorRotm<MemberType, Flag>::invoke(const MemberType &member, const XViewType &x,
                                                                    const YViewType &y, const ParamViewType &param) {
  // Quick return if possible
  const int n = x.extent_int(0);
  if (n == 0) return 0;

  auto info = Impl::checkRotmInput(x, y, param);
  if (info) return info;

  if constexpr (Flag != -2) {
    // flag == -2.0: identity, no need to do anything
    Impl::TeamVectorRotmInternal<Flag>::invoke(member, n, x.data(), x.stride(0), y.data(), y.stride(0), param.data(),
                                               param.stride(0));
  }
  return 0;
}

}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_ROTM_IMPL_HPP_
