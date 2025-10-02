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

#ifndef KOKKOSKERNELS_KOKKOS_ARITHTRAITS_HPP
#define KOKKOSKERNELS_KOKKOS_ARITHTRAITS_HPP

/// \file KokkosKernels_ArithTraits.hpp
/// \brief Declaration and definition of KokkosKernels::ArithTraits

#include <KokkosKernels_ArithTraits.hpp>

#ifndef KOKKOS_ENABLE_DEPRECATED_CODE_5
static_assert(false, "Header `Kokkos_ArithTraits.hpp` is deprecated, include `KokkosKernels_ArithTraits.hpp` instead");
#endif

namespace Kokkos {
template <typename T>
using ArithTraits [[deprecated("Use KokkosKernels::ArithTraits from KokkosKernels_ArithTraits.hpp header instead")]] =
    ::KokkosKernels::ArithTraits<T>;

namespace Details {
template <typename T>
using ArithTraits [[deprecated("Use KokkosKernels::ArithTraits from KokkosKernels_ArithTraits.hpp header instead")]] =
    ::KokkosKernels::ArithTraits<T>;

}  // namespace Details
}  // namespace Kokkos

#endif  // KOKKOSKERNELS_KOKKOS_ARITHTRAITS_HPP
