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

#ifndef KOKKOSKERNELS_UNIFIEDSCALARVIEW_HPP
#define KOKKOSKERNELS_UNIFIEDSCALARVIEW_HPP

#include <type_traits>

#include <Kokkos_Core.hpp>

#include <KokkosKernels_AlwaysFalse.hpp>
#include <KokkosKernels_IsKokkosComplex.hpp>


namespace KokkosKernels {
namespace Impl {


template <typename ScalarLike, typename = void>
struct is_scalar : std::false_type {};

// kokkos complex
template <typename T>
struct is_scalar<T, std::enable_if_t<is_kokkos_complex_v<T>>> : std::true_type {};

// other scalars
template <typename ScalarLike>
struct is_scalar<ScalarLike, std::enable_if_t<std::is_integral_v<ScalarLike> || std::is_floating_point_v<ScalarLike>>> : std::true_type {};

template <typename ScalarLike>
inline constexpr bool is_scalar_v = is_scalar<ScalarLike>::value;




template <typename ScalarLike, typename = void>
struct is_scalar_view : std::false_type {};

// rank 0
template <typename ScalarLike>
struct is_scalar_view<ScalarLike, std::enable_if_t<0 == ScalarLike::rank>> : std::true_type {};

// rank 1 and static extent is 1
template <typename ScalarLike>
struct is_scalar_view<ScalarLike, 
  std::enable_if_t<
    1 == ScalarLike::rank && 1 == ScalarLike::static_extent(0)
  >
> : std::true_type {};

template <typename ScalarLike>
inline constexpr bool is_scalar_view_v = is_scalar_view<ScalarLike>::value;

/*! \brief true iff ScalarLike is a scalar or a 0D or 1D view of a single thing
*/
template <typename ScalarLike>
inline constexpr bool is_scalar_or_scalar_view = is_scalar_v<ScalarLike> || is_scalar_view_v<ScalarLike>;





template <typename Value, typename = void>
struct unified_scalar;

template <typename Value>
struct unified_scalar<Value, std::enable_if_t<is_scalar_v<Value>>> {

    using type = Value;
    using non_const_type = std::remove_const_t<type>;
};

template <typename Value>
struct unified_scalar<Value, std::enable_if_t<is_scalar_view_v<Value>>> {

    using type = typename Value::value_type;
    using non_const_type = std::remove_const_t<type>;
};

template <typename Value>
using unified_scalar_t = typename unified_scalar<Value>::type;

template <typename Value>
using non_const_unified_scalar_t = typename unified_scalar<Value>::non_const_type;


template <typename Value>
constexpr unified_scalar_t<Value> get_scalar(const Value &v) {

    static_assert(is_scalar_or_scalar_view<Value>, "");

    unified_scalar_t<Value> ref;
    if constexpr (is_scalar_view_v<Value>) {
        if (0 == Value::rank) {
            ref = *v;
        } else if (1 == Value::rank) {
            ref = v[0];
        } else {
            static_assert(KokkosKernels::Impl::always_false_v<Value>, "");
        }
    } else {
        ref = v;
    }
    return ref;
} 

} // namespace Impl
} // namespace KokkosKernels

#endif // KOKKOSKERNELS_UNIFIEDSCALARVIEW_HPP