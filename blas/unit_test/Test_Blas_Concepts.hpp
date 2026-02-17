// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef TEST_BLAS_CONCEPTS_HPP
#define TEST_BLAS_CONCEPTS_HPP

#include <gtest/gtest.h>
#include "KokkosBlas_Concepts.hpp"
#include <KokkosKernels_TestUtils.hpp>

namespace Test {
void test_blas_concepts() {
  // Check that the concepts compile for valid types
  static_assert(KokkosBlas::TransposeOperation<KokkosBlas::Trans::Transpose>);
  static_assert(KokkosBlas::TransposeOperation<KokkosBlas::Trans::NoTranspose>);
  static_assert(KokkosBlas::TransposeOperation<KokkosBlas::Trans::ConjTranspose>);

  // Check for level 2 concepts
  static_assert(KokkosBlas::BlasLevel2<KokkosBlas::Algo::Level2::Unblocked>);
  static_assert(KokkosBlas::BlasLevel2<KokkosBlas::Algo::Level2::Blocked>);
  static_assert(KokkosBlas::BlasLevel2<KokkosBlas::Algo::Level2::MKL>);
  static_assert(KokkosBlas::BlasLevel2<KokkosBlas::Algo::Level2::CompactMKL>);

  static_assert(!KokkosBlas::BlasLevel2<KokkosBlas::Algo::Level3::Unblocked>);
  static_assert(!KokkosBlas::BlasLevel2<KokkosBlas::Algo::Level3::Blocked>);
  static_assert(!KokkosBlas::BlasLevel2<KokkosBlas::Algo::Level3::MKL>);
  static_assert(!KokkosBlas::BlasLevel2<KokkosBlas::Algo::Level3::CompactMKL>);

  // Check for level 3 concepts
  static_assert(KokkosBlas::BlasLevel3<KokkosBlas::Algo::Level3::Unblocked>);
  static_assert(KokkosBlas::BlasLevel3<KokkosBlas::Algo::Level3::Blocked>);
  static_assert(KokkosBlas::BlasLevel3<KokkosBlas::Algo::Level3::MKL>);
  static_assert(KokkosBlas::BlasLevel3<KokkosBlas::Algo::Level3::CompactMKL>);

  static_assert(!KokkosBlas::BlasLevel3<KokkosBlas::Algo::Level2::Unblocked>);
  static_assert(!KokkosBlas::BlasLevel3<KokkosBlas::Algo::Level2::Blocked>);
  static_assert(!KokkosBlas::BlasLevel3<KokkosBlas::Algo::Level2::MKL>);
  static_assert(!KokkosBlas::BlasLevel3<KokkosBlas::Algo::Level2::CompactMKL>);
}
}  // namespace Test

TEST_F(TestCategory, blas_concepts) { ::Test::test_blas_concepts(); }

#endif  // TEST_BLAS_CONCEPTS_HPP
