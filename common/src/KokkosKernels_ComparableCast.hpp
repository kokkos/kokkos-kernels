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

#ifndef KOKKOSKERNELS_COMPARABLE_CAST_HPP
#define KOKKOSKERNELS_COMPARABLE_CAST_HPP

namespace KokkosKernels {
namespace Impl {

/*! \brief cast `a` to a type that can be safely compared with an value of type
   T

    When comparing signed and unsigned types of the same size, the signed type
   is converted to unsigned which produces strange behavior like int32_t(-1) >
   uint32_t(1) This function casts its argument to a type that is safe to use to
   compare with T and U.

    Basically this boils down to:
    1. forbidding any comparisons between signed integers and uint64_t,
    since there's no reliable signed integer type larger than 64 bits.
    2. Using a type large enough to represent both sides of a comparison
   otherwise.

    If T and A are float, use the larger type
    Else If T or A are float, use the float type
    Else if T xor A are signed, we can have a problem. Choose a signed type at
   least: as large as the signed type large enough to represent the unsigned
   type Else, choose the larger type

    This function does not protect you from casting an int to a float where that
   value is not representable.
*/
template <typename T, typename A>
constexpr auto comparable_cast(const A &a) {
  // both floating point, use the larger type
  if constexpr (std::is_floating_point_v<T> && std::is_floating_point_v<A>) {
    if constexpr (sizeof(T) >= sizeof(A)) {
      return T(a);
    } else {
      return a;
    }
  }
  // one or the other floating point, use the floating point type
  else if constexpr (std::is_floating_point_v<T>) {
    return T(a);
  } else if constexpr (std::is_floating_point_v<A>) {
    return a;
  } else {
    // exactly one is signed integer, and both are the same size, choose a large
    // enough signed type
    if constexpr (std::is_signed_v<T> != std::is_signed_v<A>) {
      // how wide the signed type would need to be for T and U
      constexpr size_t t_width =
          std::is_signed_v<T> ? sizeof(T) : 2 * sizeof(T);
      constexpr size_t a_width =
          std::is_signed_v<A> ? sizeof(A) : 2 * sizeof(A);

      // how wide to compare T and U
      constexpr size_t width = std::max(t_width, a_width);
      if constexpr (width == 1) {
        return int8_t(a);
      } else if constexpr (width == 2) {
        return int16_t(a);
      } else if constexpr (width == 4) {
        return int32_t(a);
      } else if constexpr (width == 8) {
        return int64_t(a);
      } else {
        static_assert(std::is_same_v<T, A>, "no safe way to compare types");
      }
    }
    // both or neither are signedreturn the larger types
    else {
      if constexpr (sizeof(T) >= sizeof(A)) {
        return T(a);
      } else {
        return a;
      }
    }
  }
}

}  // namespace Impl
}  // namespace KokkosKernels

#endif  // KOKKOSKERNELS_COMPARABLE_CAST_HPP