// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef TEST_COMMON_HPP
#define TEST_COMMON_HPP

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosKernels_config.h>

#if defined(KOKKOSKERNELS_TEST_ETI_ONLY) && !defined(KOKKOSKERNELS_ETI_ONLY)
#define KOKKOSKERNELS_ETI_ONLY
#endif

class CommonParent : public ::testing::Test {
 protected:
  static void SetUpTestCase() {}

  static void TearDownTestCase() {}

  static void SetUpTestSuite() {
    // Runs once before all tests in this fixture suite
  }

  void SetUp() override {
    const ::testing::TestInfo* const test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();

    std::string test_name = "UnknownTest";
    if (test_info != nullptr) {
      // Combines SuiteName.TestName (e.g., MyTrackingFixture.IntentionalFailureDemo)
      test_name = std::string(test_info->test_suite_name()) + "." + test_info->name();
    }

    // 1. Generate random integer dataset
    utils::initRandSeed();

    // 2. Format a string describing our dataset state
    std::string msg = test_name + " => rand seed: " + std::to_string(utils::getTestSeed());

    // 3. Keep trace alive for the whole test scope via modern pointer mechanics
    current_trace = std::make_unique<::testing::ScopedTrace>("", 0, msg);
  }

  void TearDown() override {
    current_trace.reset();
  }

  std::unique_ptr<::testing::ScopedTrace> current_trace;
};

#endif  // TEST_COMMON_HPP
