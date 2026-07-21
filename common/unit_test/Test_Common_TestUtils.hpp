// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// \file Test_Common_TestUtils.hpp
/// \brief Tests for KokkosKernels_TestUtils.hpp random-generation utilities.
///
/// Verifies that constructing RandCooMat and RandCsMatrix twice with the same
/// seed produces bit-for-bit identical data.  A FNV-1a hash of each view's raw
/// bytes is computed and compared.  If the hashes differ, the test fails and
/// the offending seed is reported so the failure can be reproduced with
/// --gtest_random_seed=<seed>.

#ifndef TEST_COMMON_TESTUTILS_HPP
#define TEST_COMMON_TESTUTILS_HPP

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace {

/// FNV-1a hash over the raw bytes of a contiguous host-accessible range.
template <class T>
uint64_t fnv1a(const T* data, size_t count) {
  constexpr uint64_t kBasis = 14695981039346656037ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t h                = kBasis;
  const uint8_t* bytes      = reinterpret_cast<const uint8_t*>(data);
  for (size_t i = 0; i < count * sizeof(T); ++i) {
    h ^= static_cast<uint64_t>(bytes[i]);
    h *= kPrime;
  }
  return h;
}

/// Copy view to host and hash its contiguous data.
template <class View>
uint64_t hashView(const View& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  Kokkos::fence();
  return fnv1a(h.data(), h.extent(0));
}

// ============================================================================
// RandCooMat determinism
// ============================================================================

template <class Device>
void test_rand_coo_mat_determinism() {
  using Scalar = double;
  using Layout = Kokkos::LayoutLeft;
  using RCM    = Test::RandCooMat<Scalar, Layout, Device>;

  // Two constructions each preceded by initRandSeed() simulate two separate
  // test runs with the same seed.  Both must produce identical output.
  Test::initRandSeed();
  RCM mat1(8, 8, 20, -1.0, 1.0);
  Test::initRandSeed();
  RCM mat2(8, 8, 20, -1.0, 1.0);

  uint64_t row_hash1 = hashView(mat1.get_row());
  uint64_t row_hash2 = hashView(mat2.get_row());
  EXPECT_EQ(row_hash1, row_hash2)
      << "RandCooMat rows differ between two constructions with the same seed.\n"
      << "Seed: " << Test::getTestSeed();

  uint64_t col_hash1 = hashView(mat1.get_col());
  uint64_t col_hash2 = hashView(mat2.get_col());
  EXPECT_EQ(col_hash1, col_hash2)
      << "RandCooMat cols differ between two constructions with the same seed.\n"
      << "Seed: " << Test::getTestSeed();

  uint64_t data_hash1 = hashView(mat1.get_data());
  uint64_t data_hash2 = hashView(mat2.get_data());
  EXPECT_EQ(data_hash1, data_hash2)
      << "RandCooMat data differ between two constructions with the same seed.\n"
      << "Seed: " << Test::getTestSeed();
}

// ============================================================================
// RandCsMatrix determinism
// ============================================================================

template <class Device>
void test_rand_cs_matrix_determinism() {
  using Scalar = double;
  using Layout = Kokkos::LayoutLeft;
  using RCS    = Test::RandCsMatrix<Scalar, Layout, Device>;

  // RandCsMatrix uses both std::rand (structure) and Kokkos::fill_random
  // (values).  Two runs each preceded by initRandSeed() must produce identical
  // output because both rand streams start from the same seed.
  Test::initRandSeed();
  RCS mat1(6, 6, -1.0, 1.0);
  Test::initRandSeed();
  RCS mat2(6, 6, -1.0, 1.0);

  uint64_t map_hash1 = hashView(mat1.get_map());
  uint64_t map_hash2 = hashView(mat2.get_map());
  EXPECT_EQ(map_hash1, map_hash2)
      << "RandCsMatrix map (structure) differs between two constructions with the same seed.\n"
      << "Seed: " << Test::getTestSeed();

  // The nnz may differ in principle only if std::rand is not reset, so check
  // it explicitly for a clearer message.
  EXPECT_EQ(mat1.get_nnz(), mat2.get_nnz())
      << "RandCsMatrix nnz differs between two constructions with the same seed.\n"
      << "Seed: " << Test::getTestSeed();

  if (mat1.get_nnz() == mat2.get_nnz()) {
    uint64_t ids_hash1 = hashView(mat1.get_ids());
    uint64_t ids_hash2 = hashView(mat2.get_ids());
    EXPECT_EQ(ids_hash1, ids_hash2)
        << "RandCsMatrix ids differ between two constructions with the same seed.\n"
        << "Seed: " << Test::getTestSeed();

    uint64_t vals_hash1 = hashView(mat1.get_vals());
    uint64_t vals_hash2 = hashView(mat2.get_vals());
    EXPECT_EQ(vals_hash1, vals_hash2)
        << "RandCsMatrix values differ between two constructions with the same seed.\n"
        << "Seed: " << Test::getTestSeed();
  }
}

// ============================================================================
// create_random_x_vector determinism
//
// create_random_x_vector uses std::rand() and requires the caller to have
// called Test::initRandSeed() first.  Two calls each preceded by initRandSeed()
// simulate two separate test runs and must produce identical output.
// ============================================================================

template <class Device>
void test_create_random_x_vector_determinism() {
  using Scalar  = double;
  using Layout  = Kokkos::LayoutLeft;
  using vec1d_t = Kokkos::View<Scalar*, Layout, Device>;
  using vec2d_t = Kokkos::View<Scalar**, Layout, Device>;

  // --- rank-1 ---
  vec1d_t x1("x1", 32), x2("x1_dup", 32);
  Test::initRandSeed();
  Test::create_random_x_vector(x1, 10.0);
  Test::initRandSeed();
  Test::create_random_x_vector(x2, 10.0);

  EXPECT_EQ(hashView(x1), hashView(x2))
      << "create_random_x_vector (rank-1) differs between two calls with the same seed.\n"
      << "Seed: " << Test::getTestSeed();

  // --- rank-2 ---
  vec2d_t X1("X1", 32, 4), X2("X1_dup", 32, 4);
  Test::initRandSeed();
  Test::create_random_x_vector(X1, 10.0);
  Test::initRandSeed();
  Test::create_random_x_vector(X2, 10.0);

  auto h1    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, X1);
  auto h2    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, X2);
  Kokkos::fence();
  EXPECT_EQ(fnv1a(h1.data(), h1.extent(0) * h1.extent(1)),
            fnv1a(h2.data(), h2.extent(0) * h2.extent(1)))
      << "create_random_x_vector (rank-2) differs between two calls with the same seed.\n"
      << "Seed: " << Test::getTestSeed();
}

}  // namespace

TEST_F(TestCategory, common_rand_coo_mat_determinism) {
  test_rand_coo_mat_determinism<TestDevice>();
}

TEST_F(TestCategory, common_rand_cs_matrix_determinism) {
  test_rand_cs_matrix_determinism<TestDevice>();
}

TEST_F(TestCategory, common_create_random_x_vector_determinism) {
  test_create_random_x_vector_determinism<TestDevice>();
}

#endif  // TEST_COMMON_TESTUTILS_HPP
