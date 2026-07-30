// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef TEST_OPENMP_HPP
#define TEST_OPENMP_HPP

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosKernels_config.h>

#include "Test_common.hpp"

#if defined(KOKKOSKERNELS_TEST_ETI_ONLY) && !defined(KOKKOSKERNELS_ETI_ONLY)
#define KOKKOSKERNELS_ETI_ONLY
#endif

class openmp : public CommonParent {};

#define TestCategory openmp
#define TestDevice Kokkos::OpenMP

#endif  // TEST_OPENMP_HPP
