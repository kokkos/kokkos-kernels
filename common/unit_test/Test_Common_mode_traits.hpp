// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// \file Test_Common_IOUtils.hpp
/// \brief Tests for IO and print routines

#ifndef TEST_COMMON_MODE_TRAITS_HPP
#define TEST_COMMON_MODE_TRAITS_HPP

#include <Kokkos_Core.hpp>
#include <KokkosBlas_util.hpp>
#include <KokkosBatched_Util.hpp>

void test_mode_traits() {

  // Resource Mode
  EXPECT_TRUE(KokkosBlas::is_mode_v<KokkosBlas::Mode::Serial>);
  EXPECT_TRUE(KokkosBlas::is_mode_v<KokkosBlas::Mode::Team>);
  EXPECT_TRUE(KokkosBlas::is_mode_v<KokkosBlas::Mode::TeamVector>);

  EXPECT_FALSE(KokkosBlas::is_mode_v<KokkosBatched::Diag::Unit>);
  EXPECT_FALSE(KokkosBlas::is_mode_v<KokkosBatched::Uplo::Upper>);

  // Transpose Modes
  EXPECT_TRUE(KokkosBlas::is_trans_v<KokkosBlas::Trans::NoTranspose>);
  EXPECT_TRUE(KokkosBlas::is_trans_v<KokkosBlas::Trans::Transpose>);
  EXPECT_TRUE(KokkosBlas::is_trans_v<KokkosBlas::Trans::ConjNoTranspose>);
  EXPECT_TRUE(KokkosBlas::is_trans_v<KokkosBlas::Trans::ConjTranspose>);

  EXPECT_FALSE(KokkosBlas::is_trans_v<KokkosBlas::Mode::Team>);
  EXPECT_FALSE(KokkosBlas::is_trans_v<KokkosBatched::Side::Left>);

  // Side Modes
  EXPECT_TRUE(KokkosBatched::is_side_v<KokkosBatched::Side::Left>);
  EXPECT_TRUE(KokkosBatched::is_side_v<KokkosBatched::Side::Right>);

  EXPECT_FALSE(KokkosBatched::is_side_v<double>);
  EXPECT_FALSE(KokkosBatched::is_side_v<KokkosBatched::Uplo::Lower>);

  // Uplo Modes
  EXPECT_TRUE(KokkosBatched::is_uplo_v<KokkosBatched::Uplo::Upper>);
  EXPECT_TRUE(KokkosBatched::is_uplo_v<KokkosBatched::Uplo::Lower>);

  EXPECT_FALSE(KokkosBatched::is_uplo_v<KokkosBatched::Diag::NonUnit>);
  EXPECT_FALSE(KokkosBatched::is_uplo_v<KokkosBlas::Trans::Transpose>);

  // Diag Modes
  EXPECT_TRUE(KokkosBatched::is_diag_v<KokkosBatched::Diag::Unit>);
  EXPECT_TRUE(KokkosBatched::is_diag_v<KokkosBatched::Diag::NonUnit>);

  EXPECT_FALSE(KokkosBatched::is_diag_v<KokkosBlas::Trans::ConjTranspose>);
  EXPECT_FALSE(KokkosBatched::is_diag_v<KokkosBatched::Side::Right>);
}

TEST_F(TestCategory, common_mode_traits) { test_mode_traits(); }

#endif  // TEST_COMMON_MODE_TRAITS_HPP
