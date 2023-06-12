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

#include <KokkosKernels_ViewUtils.hpp>

/*! \brief

Canonicalizes a variety of different "scalar" values of AV to the most
restrictive version still consistent with the interface.

This may reduce the number of instantiations.

Transformations made:
* rank-1 AV, RMV, and XMV means AV is actually a scalar. On the host, we can go
further and convert to a true scalar since we can access the value. On device,
take a subview to convert to rank-0 scalar.
* S[1] -> S
* apply const to view data types
* TODO: simplify this by just applying these in turn when possible

Row | RMV / XMV | AV               | alpha_type
 1  | Rank-1    | S                | S
 2  | Rank-2    | S                | S
 3  | Rank-1    | View<S, host>    | S
 4  | Rank-2    | View<S, host>    | S
 5  | Rank-1    | View<S, dev>     | View<const S, dev, unmanaged>
 6  | Rank-2    | View<S, dev>     | View<const S, dev, unmanaged>
 7  | Rank-1    | View<S[1], host> | S
 8  | Rank-2    | View<S[1], host> | S
 9  | Rank-1    | View<S*, host>   | S
10  | Rank-2    | View<S*, host>   | View<const S*, host, unmanaged>
11  | Rank-1    | View<S[1], dev>  | View<const S, dev, unmanaged>
12  | Rank-1    | View<S*, dev>    | View<const S, dev, unmanaged>
13  | Rank-2    | View<S[1], dev>  | View<const S, dev, unmanaged>
14  | Rank-2    | View<S*, dev>    | View<const S*, dev, unmanaged>

See comments on the implementation below for each rows

This canonicalization strategy avoids:
* Calling Kokkos::deep_copy to convert S to View<S, host>
* Interacting with device scalars in the host code
*/

namespace KokkosBlas::Impl {

template <typename T, typename Enable = void>
struct is_host : std::false_type {};
template <typename T>
struct is_host<T, std::enable_if_t<Kokkos::is_view_v<T> &&
                                   Kokkos::Impl::MemorySpaceAccess<
                                       Kokkos::HostSpace,
                                       typename T::memory_space>::accessible>>
    : std::true_type {};
template <typename T>
constexpr inline bool is_host_v = is_host<T>::value;

template <typename T, typename Enable = void>
struct is_dev : std::false_type {};
template <typename T>
struct is_dev<T, std::enable_if_t<Kokkos::is_view_v<T> &&
                                  !Kokkos::Impl::MemorySpaceAccess<
                                       Kokkos::HostSpace,
                                       typename T::memory_space>::accessible>>
    : std::true_type {};
template <typename T>
constexpr inline bool is_dev_v = is_dev<T>::value;

template <typename T, typename Enable = void>
struct is_rank_0_host : std::false_type {};
template <typename T>
struct is_rank_0_host<T, std::enable_if_t<is_host_v<T> && T::rank == 0>>
    : std::true_type {};
template <typename T>
constexpr inline bool is_rank_0_host_v = is_rank_0_host<T>::value;

template <typename T, typename Enable = void>
struct is_rank_1_host : std::false_type {};
template <typename T>
struct is_rank_1_host<T, std::enable_if_t<is_host_v<T> && T::rank == 1>>
    : std::true_type {};
template <typename T>
constexpr inline bool is_rank_1_host_v = is_rank_1_host<T>::value;

template <typename T, typename Enable = void>
struct is_rank_1_host_static : std::false_type {};
template <typename T>
struct is_rank_1_host_static<
    T, std::enable_if_t<is_rank_1_host_v<T> && T::static_extent(0) == 1>>
    : std::true_type {};
template <typename T>
constexpr inline bool is_rank_1_host_static_v = is_rank_1_host_static<T>::value;

template <typename T, typename Enable = void>
struct is_rank_1_host_dynamic : std::false_type {};
template <typename T>
struct is_rank_1_host_dynamic<
    T, std::enable_if_t<is_rank_1_host_v<T> && T::rank_dynamic == 1>>
    : std::true_type {};
template <typename T>
constexpr inline bool is_rank_1_host_dynamic_v = is_rank_1_host_dynamic<T>::value;

template <typename T, typename Enable = void>
struct is_rank_0_dev : std::false_type {};
template <typename T>
struct is_rank_0_dev<T, std::enable_if_t<is_dev_v<T> && T::rank == 0>>
    : std::true_type {};
template <typename T>
constexpr inline bool is_rank_0_dev_v = is_rank_0_dev<T>::value;

template <typename T, typename Enable = void>
struct is_rank_1_dev : std::false_type {};
template <typename T>
struct is_rank_1_dev<T, std::enable_if_t<is_dev_v<T> && T::rank == 1>>
    : std::true_type {};
template <typename T>
constexpr inline bool is_rank_1_dev_v = is_rank_1_dev<T>::value;

template <typename T, typename Enable = void>
struct is_rank_1_dev_static : std::false_type {};
template <typename T>
struct is_rank_1_dev_static<
    T, std::enable_if_t<is_rank_1_dev_v<T> && T::static_extent(0) == 1>>
    : std::true_type {};
template <typename T>
constexpr inline bool is_rank_1_dev_static_v = is_rank_1_dev_static<T>::value;

template <typename T, typename Enable = void>
struct is_rank_1_dev_dynamic : std::false_type {};
template <typename T>
struct is_rank_1_dev_dynamic<
    T, std::enable_if_t<is_rank_1_dev_v<T> && T::rank_dynamic == 1>>
    : std::true_type {};
template <typename T>
constexpr inline bool is_rank_1_dev_dynamic_v = is_rank_1_dev_dynamic<T>::value;

template <typename RMV, typename AV, typename XMV, typename Enable = void>
struct scal_unified_scalar_view;

// Rows 1,2: S -> S
template <typename RMV, typename AV, typename XMV>
struct scal_unified_scalar_view<RMV, AV, XMV,
                                std::enable_if_t<!Kokkos::is_view_v<AV>>> {
  using alpha_type = AV;

  static alpha_type from(const AV &av) { return av; }
};

// Rows 3,4: AV is a rank 0 host view
template <typename RMV, typename AV, typename XMV>
struct scal_unified_scalar_view<RMV, AV, XMV,
                                std::enable_if_t<is_rank_0_host_v<AV>>> {
  using alpha_type = typename AV::non_const_data_type;

  static alpha_type from(const AV &av) { return av(); }
};

// Rows 5,6: AV is a rank 0 device view
template <typename RMV, typename AV, typename XMV>
struct scal_unified_scalar_view<RMV, AV, XMV,
                                std::enable_if_t<is_rank_0_dev_v<AV>>> {
  using alpha_type =
      Kokkos::View<const typename AV::data_type, typename AV::memory_space,
                   Kokkos::MemoryUnmanaged>;

  static alpha_type from(const AV &av) { return alpha_type(av); }
};

// Rows 7,8: AV is a rank 1 host view with known extent
template <typename RMV, typename AV, typename XMV>
struct scal_unified_scalar_view<RMV, AV, XMV,
                                std::enable_if_t<is_rank_1_host_static_v<AV>>> {
  using alpha_type = typename AV::non_const_value_type;

  static alpha_type from(const AV &av) { return av(0); }
};

// Row 9: AV is a rank 1 host view of unknown size, but we assume it's
// a single scalar since XMV and YMV are rank 1
template <typename RMV, typename AV, typename XMV>
struct scal_unified_scalar_view<
    RMV, AV, XMV,
    std::enable_if_t<is_rank_1_host_dynamic_v<AV> && RMV::rank == 1 && XMV::rank == 1>> {
  using alpha_type = typename AV::non_const_value_type;

  static alpha_type from(const AV &av) { return av(0); }
};


// Row 10: AV is a rank 1 host view of unknown size, and we assume
// each element is to scale a vector in RMV and XMV
template <typename RMV, typename AV, typename XMV>
struct scal_unified_scalar_view<
    RMV, AV, XMV,
    std::enable_if_t<is_rank_1_host_v<AV> && XMV::rank == 2 &&
                     RMV::rank == 2>> {
  using alpha_type =
      Kokkos::View<typename AV::data_type, typename AV::array_layout,
                   typename AV::memory_space, Kokkos::MemoryUnmanaged>;

  static alpha_type from(const AV &av) { return av; }
};

// Row 11, 12: AV is a rank 1 dev view, but we assume its
// a single scalar since XMV and YMV are rank 1
template <typename RMV, typename AV, typename XMV>
struct scal_unified_scalar_view<RMV, AV, XMV,
                                std::enable_if_t<is_rank_1_dev_dynamic_v<AV>>> {
  using alpha_type =
      Kokkos::View<const typename AV::value_type, typename AV::memory_space,
                   Kokkos::MemoryUnmanaged>;

  static alpha_type from(const AV &av) { return Kokkos::subview(av, 0); }
};

// Row 13: AV is a rank 1 dev view of static size,
// so its a single scalar
template <typename RMV, typename AV, typename XMV>
struct scal_unified_scalar_view<RMV, AV, XMV,
                                std::enable_if_t<is_rank_1_dev_static_v<AV>>> {
  using alpha_type =
      Kokkos::View<const typename AV::value_type, typename AV::memory_space,
                   Kokkos::MemoryUnmanaged>;

  static alpha_type from(const AV &av) { return Kokkos::subview(av, 0); }
};

// Row 14: AV is a rank 1 dev view of unknown size,
// and XMV and YMV are rank 2, so assume each entry is
// used to scale each vector
template <typename RMV, typename AV, typename XMV>
struct scal_unified_scalar_view<
    RMV, AV, XMV,
    std::enable_if_t<is_rank_1_dev_v<AV> && XMV::rank == 2 && RMV::rank == 2>> {
  using alpha_type =
      Kokkos::View<typename AV::data_type, typename AV::array_layout,
                   typename AV::memory_space, Kokkos::MemoryUnmanaged>;

  static alpha_type from(const AV &av) { return av; }
};


/*! \brief return av

Get a POD, Kokkos::complex, or 0D view as a scalar
*/
template <typename AV, std::enable_if_t<!Kokkos::is_view_v<AV>, bool> = true>
KOKKOS_INLINE_FUNCTION auto as_scalar(const AV &av) {
  return av;
}

/*! \brief return av()

Get a POD, Kokkos::complex, or 0D view as a scalar
*/
template <typename AV,
          std::enable_if_t<KokkosKernels::Impl::is_rank_0_v<AV>, bool> = true>
KOKKOS_INLINE_FUNCTION auto as_scalar(const AV &av) {
  return av();
}

/*! \brief return av
 */
template <typename AV, typename IndexType,
          std::enable_if_t<!Kokkos::is_view_v<AV>, bool> = true>
KOKKOS_INLINE_FUNCTION auto as_scalar(const AV &av, const IndexType & /*i*/) {
  return av;
}

/*! \brief return av()
 */
template <typename AV, typename IndexType,
          std::enable_if_t<KokkosKernels::Impl::is_rank_0_v<AV>, bool> = true>
KOKKOS_INLINE_FUNCTION auto as_scalar(const AV &av, const IndexType &i) {
  return av();
}

/*! \brief return av(i)
 */
template <typename AV, typename IndexType,
          std::enable_if_t<KokkosKernels::Impl::is_rank_1_v<AV>, bool> = true>
KOKKOS_INLINE_FUNCTION auto as_scalar(const AV &av, const IndexType &i) {
  return av(i);
}

}  // namespace KokkosBlas::Impl

#endif  // KOKKOSBLAS1_SCAL_UNIFIED_SCALAR_VIEW_IMPL