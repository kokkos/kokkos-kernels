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

#ifndef KK_EAGERINIT_TEST_HPP
#define KK_EAGERINIT_TEST_HPP

#include "KokkosKernels_EagerInitialize.hpp"

TEST_F(TestCategory, common_eager_initialize)
{
  KokkosKernels::eager_initialize();
  KokkosKernels::eager_initialize();
};

#endif
