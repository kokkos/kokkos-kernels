/*
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
*/

#ifndef KOKKOSBLAS2_GER_SPEC_HPP_
#define KOKKOSBLAS2_GER_SPEC_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>

// Include the actual functors
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosBlas2_ger_impl.hpp>
#endif

namespace KokkosBlas {
namespace Impl {

  // EEP

#include <KokkosBlas2_ger_tpl_spec_decl.hpp>
#include <generated_specializations_hpp/KokkosBlas2_ger_eti_spec_decl.hpp>

#endif  // KOKKOSBLAS2_GER_SPEC_HPP_
