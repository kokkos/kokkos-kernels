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

#ifndef KOKKOSKERNELS_ISKOKKOSCOMPLEX_HPP
#define KOKKOSKERNELS_ISKOKKOSCOMPLEX_HPP

#include <Kokkos_Core.hpp>

namespace KokkosKernels {
namespace Impl {

/// \class is_kokkos_complex
/// \brief is_kokkos_complex<T>::value is true if T is a Kokkos::complex<...>, false
/// otherwise
template <typename>
struct is_kokkos_complex : public std::false_type {};
template <typename... P>
struct is_kokkos_complex<Kokkos::complex<P...>> : public std::true_type {};
template <typename... P>
struct is_kokkos_complex<const Kokkos::complex<P...>> : public std::true_type {};

template <typename... P>
inline constexpr bool is_kokkos_complex_v = is_kokkos_complex<P...>::value;

}
}


#endif // KOKKOSKERNELS_ISKOKKOSCOMPLEX_HPP