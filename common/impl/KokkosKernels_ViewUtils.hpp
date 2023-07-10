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
#include <type_traits>

#include "Kokkos_Core.hpp"

#ifndef KOKKOSKERNELS_VIEWUTILS_HPP
#define KOKKOSKERNELS_VIEWUTILS_HPP

namespace KokkosKernels::Impl {

template <typename T, typename Enable = void>
struct is_rank_0 : std::false_type {};
template <typename T>
struct is_rank_0<T,
                 std::enable_if_t<Kokkos::is_view_v<T> && T::rank == 0>>
    : std::true_type {};
template <typename T>
constexpr inline bool is_rank_0_v = is_rank_0<T>::value;

template <typename T, typename Enable = void>
struct is_rank_1 : std::false_type {};
template <typename T>
struct is_rank_1<T,
                 std::enable_if_t<Kokkos::is_view_v<T> && T::rank == 1>>
    : std::true_type {};
template <typename T>
constexpr inline bool is_rank_1_v = is_rank_1<T>::value;


} // namespace KokkosKernels::Impl

#endif // KOKKOSKERNELS_VIEWUTILS_HPP

