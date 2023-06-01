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
#ifndef KOKKOSBLAS1_SCAL_UNIFIED_SCALAR_VIEW_IMPL
#define KOKKOSBLAS1_SCAL_UNIFIED_SCALAR_VIEW_IMPL

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>

/*! \brief


Implements the following table:


Row | RMV    | AV               | XMV    | alpha_type
 1  | Rank-1 | S                | Rank-1 | S
 2  | Rank-2 | S                | Rank-2 | S
 3  | Rank-1 | View<S, host>    | Rank-1 | S
 4  | Rank-2 | View<S, host>    | Rank-2 | S
 5  | Rank-1 | View<S, dev>     | Rank-1 | View<S, dev>
 6  | Rank-2 | View<S, dev>     | Rank-2 | View<S, dev>
 7  | Rank-1 | View<S[1], host> | Rank-1 | S
 8  | Rank-2 | View<S[1], host> | Rank-2 | S
 9  | Rank-1 | View<S*, host>   | Rank-1 | S
10  | Rank-2 | View<S*, host>   | Rank-2 | View<S*, host>
11  | Rank-1 | View<S[1], dev>  | Rank-1 | View<S, dev>
12  | Rank-1 | View<S*, dev>    | Rank-1 | View<S, dev>
13  | Rank-2 | View<S[1], dev>  | Rank-2 | View<S, dev>
14  | Rank-2 | View<S*, dev>    | Rank-2 | View<S*, dev>

See comments on the implementation below for each rows
*/

namespace KokkosKernels::Impl {

template <typename T, typename ExecSpace, typename Enable = void>
struct is_host : std::false_type {};
template <typename T, typename ExecSpace>
struct is_host<
    T, ExecSpace,
    std::enable_if_t<Kokkos::is_view_v<T> &&
                     !Kokkos::SpaceAccessibility<
                         ExecSpace, typename T::memory_space>::accessible>>
    : std::true_type {};
template <typename T, typename ExecSpace>
constexpr inline bool is_host_v = is_host<T, ExecSpace>::value;

template <typename T, typename ExecSpace, typename Enable = void>
struct is_rank_0_host : std::false_type {};
template <typename T, typename ExecSpace>
struct is_rank_0_host<T, ExecSpace,
                      std::enable_if_t<is_host_v<T, ExecSpace> && T::rank == 0>>
    : std::true_type {};
template <typename T, typename ExecSpace>
constexpr inline bool is_rank_0_host_v = is_rank_0_host<T, ExecSpace>::value;

template <typename T, typename ExecSpace, typename Enable = void>
struct is_rank_1_host : std::false_type {};
template <typename T, typename ExecSpace>
struct is_rank_1_host<T, ExecSpace,
                      std::enable_if_t<is_host_v<T, ExecSpace> && T::rank == 1>>
    : std::true_type {};
template <typename T, typename ExecSpace>
constexpr inline bool is_rank_1_host_v = is_rank_1_host<T, ExecSpace>::value;

template <typename T, typename ExecSpace, typename Enable = void>
struct is_rank_1_host_static : std::false_type {};
template <typename T, typename ExecSpace>
struct is_rank_1_host_static<T, ExecSpace,
                             std::enable_if_t<is_rank_1_host_v<T, ExecSpace> &&
                                              T::static_extent(0) == 1>>
    : std::true_type {};
template <typename T, typename ExecSpace>
constexpr inline bool is_rank_1_host_static_v =
    is_rank_1_host_static<T, ExecSpace>::value;

template <typename T, typename ExecSpace, typename Enable = void>
struct is_dev : std::false_type {};
template <typename T, typename ExecSpace>
struct is_dev<
    T, ExecSpace,
    std::enable_if_t<Kokkos::is_view_v<T> &&
                     Kokkos::SpaceAccessibility<
                         ExecSpace, typename T::memory_space>::accessible>>
    : std::true_type {};
template <typename T, typename ExecSpace>
constexpr inline bool is_dev_v = is_dev<T, ExecSpace>::value;

template <typename T, typename ExecSpace, typename Enable = void>
struct is_rank_0_dev : std::false_type {};
template <typename T, typename ExecSpace>
struct is_rank_0_dev<T, ExecSpace,
                     std::enable_if_t<is_dev_v<T, ExecSpace> && T::rank == 0>>
    : std::true_type {};
template <typename T, typename ExecSpace>
constexpr inline bool is_rank_0_dev_v = is_rank_0_dev<T, ExecSpace>::value;

template <typename T, typename ExecSpace, typename Enable = void>
struct is_rank_1_dev : std::false_type {};
template <typename T, typename ExecSpace>
struct is_rank_1_dev<T, ExecSpace,
                     std::enable_if_t<is_dev_v<T, ExecSpace> && T::rank == 1>>
    : std::true_type {};
template <typename T, typename ExecSpace>
constexpr inline bool is_rank_1_dev_v = is_rank_1_dev<T, ExecSpace>::value;

template <typename T, typename ExecSpace, typename Enable = void>
struct is_rank_1_dev_static : std::false_type {};
template <typename T, typename ExecSpace>
struct is_rank_1_dev_static<
    T, ExecSpace,
    std::enable_if_t<is_rank_1_dev_v<T, ExecSpace> && T::static_extent(0) == 1>>
    : std::true_type {};
template <typename T, typename ExecSpace>
constexpr inline bool is_rank_1_dev_static_v =
    is_rank_1_dev_static<T, ExecSpace>::value;

template <typename RMV, typename AV, typename XMV, typename ExecSpace,
          typename Enable = void>
struct scal_unified_scalar_view;

// Rows 1,2: AV is a scalar
template <typename RMV, typename AV, typename XMV, typename ExecSpace>
struct scal_unified_scalar_view<RMV, AV, XMV, ExecSpace,
                                std::enable_if_t<!Kokkos::is_view_v<AV>>> {
  using alpha_type = AV;

  static alpha_type from(const AV &av) { return av; }
};

// Rows 3,4: AV is a rank 0 host view
template <typename RMV, typename AV, typename XMV, typename ExecSpace>
struct scal_unified_scalar_view<
    RMV, AV, XMV, ExecSpace,
    std::enable_if_t<is_rank_0_host_v<AV, ExecSpace>>> {
  using alpha_type = typename AV::data_type;

  static alpha_type from(const AV &av) { return av(); }
};

// Rows 5,6: AV is a rank 0 device view
template <typename RMV, typename AV, typename XMV, typename ExecSpace>
struct scal_unified_scalar_view<
    RMV, AV, XMV, ExecSpace, std::enable_if_t<is_rank_0_dev_v<AV, ExecSpace>>> {
  using alpha_type = Kokkos::View<const typename AV::data_type, typename AV::memory_space, Kokkos::MemoryUnmanaged>;

  static alpha_type from(const AV &av) { return av; }
};

// Rows 7,8: AV is a rank 1 host view with known extent
template <typename RMV, typename AV, typename XMV, typename ExecSpace>
struct scal_unified_scalar_view<
    RMV, AV, XMV, ExecSpace,
    std::enable_if_t<is_rank_1_host_static_v<AV, ExecSpace>>> {

  // FIXME: const?
  using alpha_type = typename AV::value_type;

  static alpha_type from(const AV &av) { return av(0); }
};

// Row 9: AV is a rank 1 host view of unknown size, but we assume it's
// a single scalar since XMV and YMV are rank 1
template <typename RMV, typename AV, typename XMV, typename ExecSpace>
struct scal_unified_scalar_view<
    RMV, AV, XMV, ExecSpace,
    std::enable_if_t<is_rank_1_host_v<AV, ExecSpace> && XMV::rank == 1 &&
                     RMV::rank == 1>> {

  // FIXME: const?
  using alpha_type = typename AV::value_type;

  static alpha_type from(const AV &av) { return av(0); }
};

// Row 10: AV is a rank 1 host view of unknown size, and we assume
// each element is to scale a vector in RMV and XMV
template <typename RMV, typename AV, typename XMV, typename ExecSpace>
struct scal_unified_scalar_view<
    RMV, AV, XMV, ExecSpace,
    std::enable_if_t<is_rank_1_host_v<AV, ExecSpace> && XMV::rank == 2 &&
                     RMV::rank == 2>> {

  // FIXME: const?
  using alpha_type = Kokkos::View<typename AV::data_type, typename AV::memory_space, Kokkos::MemoryUnmanaged>;

  static alpha_type from(const AV &av) { return av; }
};

// Row 11, 12: AV is a rank 1 dev view, but we assume its
// a single scalar since XMV and YMV are rank 1
template <typename RMV, typename AV, typename XMV, typename ExecSpace>
struct scal_unified_scalar_view<
    RMV, AV, XMV, ExecSpace,
    std::enable_if_t<is_rank_1_dev_v<AV, ExecSpace> && XMV::rank == 1 &&
                     RMV::rank == 1>> {
  
  using alpha_type =
      Kokkos::View<const typename AV::value_type, typename AV::memory_space,
                   Kokkos::MemoryUnmanaged>;

  static alpha_type from(const AV &av) { return Kokkos::subview(av, 0); }
};

// Row 13: AV is a rank 1 dev view of static size,
// so its a single scalar
template <typename RMV, typename AV, typename XMV, typename ExecSpace>
struct scal_unified_scalar_view<
    RMV, AV, XMV, ExecSpace,
    std::enable_if_t<is_rank_1_dev_static_v<AV, ExecSpace>>> {

  // FIXME: const?
  using alpha_type =
      Kokkos::View<const typename AV::value_type, typename AV::memory_space,
                   Kokkos::MemoryUnmanaged>;

  static alpha_type from(const AV &av) { return Kokkos::subview(av, 0); }
};

// Row 14: AV is a rank 1 dev view of unknown size,
// and XMV and YMV are rank 2, so assume each entry is
// used to scale each vector
template <typename RMV, typename AV, typename XMV, typename ExecSpace>
struct scal_unified_scalar_view<
    RMV, AV, XMV, ExecSpace,
    std::enable_if_t<is_rank_1_dev_v<AV, ExecSpace> && XMV::rank == 2 &&
                     RMV::rank == 2>> {
  // FIXME: const?
  using alpha_type = Kokkos::View<typename AV::data_type, typename AV::memory_space, Kokkos::MemoryUnmanaged>;

  static alpha_type from(const AV &av) { return av; }
};

}  // namespace KokkosKernels::Impl

#endif  // KOKKOSBLAS1_SCAL_UNIFIED_SCALAR_VIEW_IMPL