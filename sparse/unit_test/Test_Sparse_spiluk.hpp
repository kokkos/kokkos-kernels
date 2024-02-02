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

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <string>
#include <stdexcept>

#include "KokkosSparse_Utils.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include <KokkosKernels_IOUtils.hpp>
#include "KokkosBlas1_nrm2.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_spiluk.hpp"
#include "KokkosSparse_crs_to_bsr_impl.hpp"
#include "KokkosSparse_bsr_to_crs_impl.hpp"

#include "Test_vector_fixtures.hpp"

#include <tuple>
#include <random>

using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

using kokkos_complex_double = Kokkos::complex<double>;
using kokkos_complex_float  = Kokkos::complex<float>;

// Comment this out to do focussed debugging
// #define TEST_SPILUK_FULL_CHECKS

// Test verbosity level. 0 = none, 1 = print residuals, 2 = print L,U
#define TEST_SPILUK_VERBOSE_LEVEL 2

// #define TEST_SPILUK_TINY_TEST

namespace Test {

#ifdef TEST_SPILUK_TINY_TEST
template <typename scalar_t>
std::vector<std::vector<scalar_t>> get_fixture() {
  std::vector<std::vector<scalar_t>> A = {{10.00, 1.00, 0.00, 0.00},
                                          {0.00, 11.00, 0.00, 0.00},
                                          {0.00, 2.00, 12.00, 0.00},
                                          {5.00, 0.00, 3.00, 13.00}};
  return A;
}
#else
template <typename scalar_t>
std::vector<std::vector<scalar_t>> get_fixture() {
  std::vector<std::vector<scalar_t>> A = {
      {10.00, 0.00, 0.30, 0.00, 0.00, 0.60, 0.00, 0.00, 0.00},
      {0.00, 11.00, 0.00, 0.00, 0.00, 0.00, 0.70, 0.00, 0.00},
      {0.00, 0.00, 12.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
      {5.00, 0.00, 0.00, 13.00, 1.00, 0.00, 0.00, 0.00, 0.00},
      {4.00, 0.00, 0.00, 0.00, 14.00, 0.00, 0.00, 0.00, 0.00},
      {0.00, 3.00, 0.00, 0.00, 0.00, 15.00, 0.00, 0.00, 0.00},
      {0.00, 0.00, 7.00, 0.00, 0.00, 0.00, 16.00, 0.00, 0.00},
      {0.00, 0.00, 0.00, 6.00, 5.00, 0.00, 0.00, 17.00, 0.00},
      {0.00, 0.00, 0.00, 2.00, 2.50, 0.00, 0.00, 0.00, 18.00}};
  return A;
}
#endif

static constexpr double EPS = 1e-7;

template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
struct SpilukTest {
  using RowMapType  = Kokkos::View<size_type*, device>;
  using EntriesType = Kokkos::View<lno_t*, device>;
  using ValuesType  = Kokkos::View<scalar_t*, device>;
  using AT          = Kokkos::ArithTraits<scalar_t>;

  using RowMapType_hostmirror  = typename RowMapType::HostMirror;
  using EntriesType_hostmirror = typename EntriesType::HostMirror;
  using ValuesType_hostmirror  = typename ValuesType::HostMirror;
  using execution_space        = typename device::execution_space;
  using memory_space           = typename device::memory_space;
  using range_policy           = Kokkos::RangePolicy<execution_space>;

  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
      size_type, lno_t, scalar_t, typename device::execution_space,
      typename device::memory_space, typename device::memory_space>;

  using Crs = CrsMatrix<scalar_t, lno_t, device, void, size_type>;
  using Bsr = BsrMatrix<scalar_t, lno_t, device, void, size_type>;

  template <typename AType, typename LType, typename UType>
  static typename AT::mag_type check_result_impl(
      const AType& A, const LType& L, const UType& U, const size_type nrows,
      const size_type block_size = 1) {
    const scalar_t ZERO = scalar_t(0);
    const scalar_t ONE  = scalar_t(1);
    const scalar_t MONE = scalar_t(-1);

    // Create a reference view e set to all 1's
    ValuesType e_one("e_one", nrows * block_size);
    Kokkos::deep_copy(e_one, ONE);

    // Create two views for spmv results
    ValuesType bb("bb", nrows * block_size);
    ValuesType bb_tmp("bb_tmp", nrows * block_size);

    // Compute norm2(L*U*e_one - A*e_one)/norm2(A*e_one)
    KokkosSparse::spmv("N", ONE, A, e_one, ZERO, bb);

    typename AT::mag_type bb_nrm = KokkosBlas::nrm2(bb);

    KokkosSparse::spmv("N", ONE, U, e_one, ZERO, bb_tmp);
    KokkosSparse::spmv("N", ONE, L, bb_tmp, MONE, bb);

    typename AT::mag_type diff_nrm = KokkosBlas::nrm2(bb);

    return diff_nrm / bb_nrm;
  }

  static bool is_triangular(const RowMapType& drow_map, const EntriesType& dentries, const ValuesType& dvalues, bool check_lower, const size_type block_size = 1)
  {
    const auto nrows = drow_map.extent(0) - 1;
    const auto block_items = block_size * block_size;

    auto row_map = Kokkos::create_mirror_view(drow_map);
    auto entries = Kokkos::create_mirror_view(dentries);
    auto values  = Kokkos::create_mirror_view(dvalues);
    Kokkos::deep_copy(row_map, drow_map);
    Kokkos::deep_copy(entries, dentries);
    Kokkos::deep_copy(values, dvalues);

    for (size_type row = 0; row < nrows; ++row) {
      const auto row_nnz_begin = row_map(row);
      const auto row_nnz_end   = row_map(row+1);
      for (size_type nnz = row_nnz_begin; nnz < row_nnz_end; ++nnz) {
        const auto col = entries(nnz);
        if (col > row && check_lower) {
          return false;
        }
        else if (col < row && !check_lower) {
          return false;
        }
        else if (col == row && block_size > 1) {
          // Do the diagonal dense blocks also have to be upper/lower?
          // Check diagonal block
          // scalar_t* block = values.data() + nnz * block_items;
          // for (size_type i = 0; i < block_size; ++i) {
          //   for (size_type j = 0; j < block_size; ++j) {
          //     if ( (j > i && check_lower && block[i*block_size + j] != 0.0 ) ||
          //          (j < i && !check_lower && block[i*block_size + j] != 0.0) ) {
          //       std::cout << "Bad block entry is: " << block[i*block_size + j] << std::endl;
          //       return false;
          //     }
          //   }
          // }
        }
      }
    }
    return true;
  }

  static void populate_bsr(const ValuesType& dvalues, const size_type block_size)
  {
    using RPDF = std::uniform_real_distribution<scalar_t>;

    const size_type block_items = block_size * block_size;
    const size_type num_blocks  = dvalues.extent(0) / block_items;
    const scalar_t ZERO = scalar_t(0);
    std::mt19937_64 engine;

    auto values  = Kokkos::create_mirror_view(dvalues);
    Kokkos::deep_copy(values, dvalues);

    for (size_type block = 0; block < num_blocks; ++block) {
      scalar_t min = std::numeric_limits<scalar_t>::max();
      scalar_t max = std::numeric_limits<scalar_t>::min();
      // Get the range of values in this block
      for (size_type block_item = 0; block_item < block_items; ++block_item) {
        const scalar_t val = values(block * block_items + block_item);
        if (val < min) min = val;
        if (val > max) max = val;
      }
      // Set the zeros to a random value in this range
      RPDF val_gen(min, max);
      for (size_type block_item = 0; block_item < block_items; ++block_item) {
        scalar_t& val = values(block * block_items + block_item);
        if (val == ZERO) {
          val = val_gen(engine);
        }
      }
    }

    Kokkos::deep_copy(dvalues, values);
  }

  static void check_result(const RowMapType& row_map,
                           const EntriesType& entries, const ValuesType& values,
                           const RowMapType& L_row_map,
                           const EntriesType& L_entries,
                           const ValuesType& L_values,
                           const RowMapType& U_row_map,
                           const EntriesType& U_entries,
                           const ValuesType& U_values) {
    // Checking
    const auto nrows = row_map.extent(0) - 1;
    Crs A("A_Mtx", nrows, nrows, values.extent(0), values, row_map, entries);
    Crs L("L_Mtx", nrows, nrows, L_values.extent(0), L_values, L_row_map,
          L_entries);
    Crs U("U_Mtx", nrows, nrows, U_values.extent(0), U_values, U_row_map,
          U_entries);

    EXPECT_TRUE(is_triangular(L_row_map, L_entries, L_values, true));
    EXPECT_TRUE(is_triangular(U_row_map, U_entries, U_values, false));

    const auto result = check_result_impl(A, L, U, nrows);
    if (TEST_SPILUK_VERBOSE_LEVEL > 0) {
      std::cout << "For nrows=" << nrows << ", unblocked had residual: " << result << std::endl;
    }
    if (TEST_SPILUK_VERBOSE_LEVEL > 1) {
      std::cout << "L result" << std::endl;
      print_matrix(decompress_matrix(L_row_map, L_entries, L_values));
      std::cout << "U result" << std::endl;
      print_matrix(decompress_matrix(U_row_map, U_entries, U_values));
    }

    EXPECT_LT(result, 1e-4);
  }

  static void check_result_block(
      const RowMapType& row_map, const EntriesType& entries,
      const ValuesType& values, const RowMapType& L_row_map,
      const EntriesType& L_entries, const ValuesType& L_values,
      const RowMapType& U_row_map, const EntriesType& U_entries,
      const ValuesType& U_values, const size_type block_size) {
    // Checking
    const auto nrows = row_map.extent(0) - 1;
    Bsr A("A_Mtx", nrows, nrows, values.extent(0), values, row_map, entries,
          block_size);
    Bsr L("L_Mtx", nrows, nrows, L_values.extent(0), L_values, L_row_map,
          L_entries, block_size);
    Bsr U("U_Mtx", nrows, nrows, U_values.extent(0), U_values, U_row_map,
          U_entries, block_size);

    EXPECT_TRUE(is_triangular(L_row_map, L_entries, L_values, true, block_size));
    EXPECT_TRUE(is_triangular(U_row_map, U_entries, U_values, false, block_size));

    const auto result = check_result_impl(A, L, U, nrows, block_size);
    if (TEST_SPILUK_VERBOSE_LEVEL > 0) {
      std::cout << "For nrows=" << nrows << ", block_size=" << block_size << " had residual: " << result << std::endl;
    }
    if (TEST_SPILUK_VERBOSE_LEVEL > 1) {
      std::cout << "L result" << std::endl;
      print_matrix(decompress_matrix(L_row_map, L_entries, L_values, block_size));
      std::cout << "U result" << std::endl;
      print_matrix(decompress_matrix(U_row_map, U_entries, U_values, block_size));
    }

    EXPECT_LT(result, 1e-2);
  }

  static std::tuple<RowMapType, EntriesType, ValuesType, RowMapType,
                    EntriesType, ValuesType>
  run_and_check_spiluk(KernelHandle& kh, const RowMapType& row_map,
                       const EntriesType& entries, const ValuesType& values,
                       SPILUKAlgorithm alg, const lno_t fill_lev, const int team_size = -1) {
    const size_type nrows = row_map.extent(0) - 1;
    kh.create_spiluk_handle(alg, nrows, 40 * nrows, 40 * nrows);

    auto spiluk_handle = kh.get_spiluk_handle();
    if (team_size != -1) {
      spiluk_handle->set_team_size(team_size);
    }

    // Allocate L and U as outputs
    RowMapType L_row_map("L_row_map", nrows + 1);
    EntriesType L_entries("L_entries", spiluk_handle->get_nnzL());
    RowMapType U_row_map("U_row_map", nrows + 1);
    EntriesType U_entries("U_entries", spiluk_handle->get_nnzU());

    spiluk_symbolic(&kh, fill_lev, row_map, entries, L_row_map, L_entries,
                    U_row_map, U_entries);

    Kokkos::fence();

    Kokkos::resize(L_entries, spiluk_handle->get_nnzL());
    Kokkos::resize(U_entries, spiluk_handle->get_nnzU());
    ValuesType L_values("L_values", spiluk_handle->get_nnzL());
    ValuesType U_values("U_values", spiluk_handle->get_nnzU());

    spiluk_numeric(&kh, fill_lev, row_map, entries, values, L_row_map,
                   L_entries, L_values, U_row_map, U_entries, U_values);

    Kokkos::fence();

    if (TEST_SPILUK_VERBOSE_LEVEL > 0) {
      std::cout << "For fill_level=" << fill_lev << ", ";
    }
    check_result(row_map, entries, values, L_row_map, L_entries, L_values,
                 U_row_map, U_entries, U_values);

    kh.destroy_spiluk_handle();

#ifdef TEST_SPILUK_FULL_CHECKS
    // Check that team size = 1 produces same result
    if (team_size != 1) {
      const auto [L_row_map_ts1, L_entries_ts1, L_values_ts1, U_row_map_ts1,
                  U_entries_ts1, U_values_ts1] =
          run_and_check_spiluk(kh, row_map, entries, values, alg, fill_lev, 1);

      EXPECT_NEAR_KK_1DVIEW(L_row_map, L_row_map_ts1, EPS);
      EXPECT_NEAR_KK_1DVIEW(L_entries, L_entries_ts1, EPS);
      EXPECT_NEAR_KK_1DVIEW(L_values, L_values_ts1, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_row_map, U_row_map_ts1, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_entries, U_entries_ts1, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_values, U_values_ts1, EPS);
    }
#endif

    return std::make_tuple(L_row_map, L_entries, L_values, U_row_map, U_entries,
                           U_values);
  }

  static std::tuple<RowMapType, EntriesType, ValuesType, RowMapType,
                    EntriesType, ValuesType>
  run_and_check_spiluk_block(
      KernelHandle& kh, const RowMapType& row_map, const EntriesType& entries,
      const ValuesType& values, SPILUKAlgorithm alg, const lno_t fill_lev,
      const size_type block_size, const int team_size = -1) {
    const size_type block_items = block_size * block_size;
    const size_type nrows       = row_map.extent(0) - 1;
    kh.create_spiluk_handle(alg, nrows, 40 * nrows, 40 * nrows, block_size);

    auto spiluk_handle = kh.get_spiluk_handle();
    if (team_size != -1) {
      spiluk_handle->set_team_size(team_size);
    }

    // Allocate L and U as outputs
    RowMapType L_row_map("L_row_map", nrows + 1);
    EntriesType L_entries("L_entries", spiluk_handle->get_nnzL());
    RowMapType U_row_map("U_row_map", nrows + 1);
    EntriesType U_entries("U_entries", spiluk_handle->get_nnzU());

    spiluk_symbolic(&kh, fill_lev, row_map, entries, L_row_map, L_entries,
                    U_row_map, U_entries);

    Kokkos::fence();

    Kokkos::resize(L_entries, spiluk_handle->get_nnzL());
    Kokkos::resize(U_entries, spiluk_handle->get_nnzU());
    ValuesType L_values("L_values", spiluk_handle->get_nnzL() * block_items);
    ValuesType U_values("U_values", spiluk_handle->get_nnzU() * block_items);

    spiluk_numeric(&kh, fill_lev, row_map, entries, values, L_row_map,
                   L_entries, L_values, U_row_map, U_entries, U_values);

    Kokkos::fence();

    if (TEST_SPILUK_VERBOSE_LEVEL > 0) {
      std::cout << "For fill_level=" << fill_lev << ", ";
    }
    check_result_block(row_map, entries, values, L_row_map, L_entries, L_values,
                       U_row_map, U_entries, U_values, block_size);

    kh.destroy_spiluk_handle();

#ifdef TEST_SPILUK_FULL_CHECKS
    // If block_size is 1, results should exactly match unblocked results
    if (block_size == 1) {
      const auto [L_row_map_u, L_entries_u, L_values_u, U_row_map_u,
                  U_entries_u, U_values_u] =
          run_and_check_spiluk(kh, row_map, entries, values, alg, fill_lev, team_size);

      EXPECT_NEAR_KK_1DVIEW(L_row_map, L_row_map_u, EPS);
      EXPECT_NEAR_KK_1DVIEW(L_entries, L_entries_u, EPS);
      EXPECT_NEAR_KK_1DVIEW(L_values, L_values_u, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_row_map, U_row_map_u, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_entries, U_entries_u, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_values, U_values_u, EPS);
    }

    // Check that team size = 1 produces same result
    if (team_size != 1) {
      const auto [L_row_map_ts1, L_entries_ts1, L_values_ts1, U_row_map_ts1,
                  U_entries_ts1, U_values_ts1] =
          run_and_check_spiluk_block(kh, row_map, entries, values, alg, fill_lev, block_size, 1);

      EXPECT_NEAR_KK_1DVIEW(L_row_map, L_row_map_ts1, EPS);
      EXPECT_NEAR_KK_1DVIEW(L_entries, L_entries_ts1, EPS);
      EXPECT_NEAR_KK_1DVIEW(L_values, L_values_ts1, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_row_map, U_row_map_ts1, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_entries, U_entries_ts1, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_values, U_values_ts1, EPS);
    }
#endif

    return std::make_tuple(L_row_map, L_entries, L_values, U_row_map, U_entries,
                           U_values);
  }

  static void run_test_spiluk() {
    std::vector<std::vector<scalar_t>> A = get_fixture<scalar_t>();

    if (TEST_SPILUK_VERBOSE_LEVEL > 1) {
      std::cout << "A input" << std::endl;
      print_matrix(A);
    }

    RowMapType row_map;
    EntriesType entries;
    ValuesType values;

    compress_matrix(row_map, entries, values, A);

    const lno_t fill_lev = 2;

    KernelHandle kh;

    run_and_check_spiluk(kh, row_map, entries, values,
                         SPILUKAlgorithm::SEQLVLSCHD_TP1, fill_lev);
  }

  static void run_test_spiluk_blocks() {
    std::vector<std::vector<scalar_t>> A = get_fixture<scalar_t>();

    if (TEST_SPILUK_VERBOSE_LEVEL > 1) {
      std::cout << "A input" << std::endl;
      print_matrix(A);
    }

    RowMapType row_map, brow_map;
    EntriesType entries, bentries;
    ValuesType values, bvalues;

    compress_matrix(row_map, entries, values, A);

    const size_type nrows      = A.size();
    const size_type nnz        = values.extent(0);
    const lno_t fill_lev       = 2;
    const size_type block_size = nrows % 2 == 0 ? 2 : 3;
    ASSERT_EQ(nrows % block_size, 0);

    KernelHandle kh;

#ifdef TEST_SPILUK_FULL_CHECKS
    // Check block_size=1 produces identical result to unblocked
    run_and_check_spiluk_block(kh, row_map, entries, values,
                               SPILUKAlgorithm::SEQLVLSCHD_TP1, fill_lev, 1);
#endif

    // Convert to BSR
    Crs crs("crs for block spiluk test", nrows, nrows, nnz, values, row_map,
            entries);
    Bsr bsr(crs, block_size);

    // Pull out views from BSR
    Kokkos::resize(brow_map, bsr.graph.row_map.extent(0));
    Kokkos::resize(bentries, bsr.graph.entries.extent(0));
    Kokkos::resize(bvalues, bsr.values.extent(0));
    Kokkos::deep_copy(brow_map, bsr.graph.row_map);
    Kokkos::deep_copy(bentries, bsr.graph.entries);
    Kokkos::deep_copy(bvalues, bsr.values);

    run_and_check_spiluk_block(kh, brow_map, bentries, bvalues,
                               SPILUKAlgorithm::SEQLVLSCHD_TP1, fill_lev,
                               block_size);
  }

  static void run_test_spiluk_scale() {
    // Create a diagonally dominant sparse matrix to test:
    constexpr auto nrows         = 5000;
    constexpr auto diagDominance = 2;

    size_type nnz = 10 * nrows;
    auto A =
        KokkosSparse::Impl::kk_generate_diagonally_dominant_sparse_matrix<Crs>(
            nrows, nrows, nnz, 0, lno_t(0.01 * nrows), diagDominance);

    // Pull out views from CRS
    RowMapType row_map("row_map", A.graph.row_map.extent(0));
    EntriesType entries("entries", A.graph.entries.extent(0));
    ValuesType values("values", A.values.extent(0));
    Kokkos::deep_copy(row_map, A.graph.row_map);
    Kokkos::deep_copy(entries, A.graph.entries);
    Kokkos::deep_copy(values, A.values);

    for (lno_t fill_lev = 2; fill_lev < 3; ++fill_lev) {

      KernelHandle kh;

      run_and_check_spiluk(kh, row_map, entries, values,
                           SPILUKAlgorithm::SEQLVLSCHD_TP1, fill_lev);
    }
  }

  static void run_test_spiluk_scale_blocks() {
    // Create a diagonally dominant sparse matrix to test:
    constexpr auto nrows         = 5000;
    constexpr auto diagDominance = 2;

    RowMapType brow_map;
    EntriesType bentries;
    ValuesType bvalues;

    //const size_type block_size = 10;

    size_type nnz = 10 * nrows;
    auto A =
      KokkosSparse::Impl::kk_generate_diagonally_dominant_sparse_matrix<Crs>(
        nrows, nrows, nnz, 0, lno_t(0.01 * nrows), diagDominance);

    std::vector<size_type> block_sizes = {/*1, 2, 4,*/ 10};

    for (auto block_size : block_sizes) {

      // Pull out views from CRS
      Bsr bsr(A, block_size);

      // Pull out views from BSR
      Kokkos::resize(brow_map, bsr.graph.row_map.extent(0));
      Kokkos::resize(bentries, bsr.graph.entries.extent(0));
      Kokkos::resize(bvalues, bsr.values.extent(0));
      Kokkos::deep_copy(brow_map, bsr.graph.row_map);
      Kokkos::deep_copy(bentries, bsr.graph.entries);
      Kokkos::deep_copy(bvalues, bsr.values);

      // Fully fill / populate the dense blocks of the BSR?
      populate_bsr(bvalues, block_size);
      Kokkos::deep_copy(bsr.values, bvalues);

      for (lno_t fill_lev = 2; fill_lev < 3; ++fill_lev) {

        KernelHandle kh;

        // auto crs = KokkosSparse::Impl::bsr_to_crs<Crs>(bsr);

        // RowMapType row_map("row_map", crs.graph.row_map.extent(0));
        // EntriesType entries("entries", crs.graph.entries.extent(0));
        // ValuesType values("values", crs.values.extent(0));
        // Kokkos::deep_copy(row_map, crs.graph.row_map);
        // Kokkos::deep_copy(entries, crs.graph.entries);
        // Kokkos::deep_copy(values, crs.values);

        // run_and_check_spiluk(kh, row_map, entries, values,
        //                      SPILUKAlgorithm::SEQLVLSCHD_TP1, fill_lev);

        run_and_check_spiluk_block(kh, brow_map, bentries, bvalues,
                                   SPILUKAlgorithm::SEQLVLSCHD_TP1, fill_lev,
                                   block_size);
      }
    }
  }

  static void run_test_spiluk_streams(SPILUKAlgorithm test_algo, int nstreams) {
    // Workaround for OpenMP: skip tests if concurrency < nstreams because of
    // not enough resource to partition
    bool run_streams_test = true;
#ifdef KOKKOS_ENABLE_OPENMP
    if (std::is_same<typename device::execution_space, Kokkos::OpenMP>::value) {
      int exec_concurrency = execution_space().concurrency();
      if (exec_concurrency < nstreams) {
        run_streams_test = false;
        std::cout << "  Skip stream test: concurrency = " << exec_concurrency
                  << std::endl;
      }
    }
#endif
    if (!run_streams_test) return;

    std::vector<int> weights(nstreams, 1);
    std::vector<execution_space> instances =
        Kokkos::Experimental::partition_space(execution_space(), weights);

    std::vector<KernelHandle> kh_v(nstreams);
    std::vector<KernelHandle*> kh_ptr_v(nstreams);
    std::vector<RowMapType> A_row_map_v(nstreams);
    std::vector<EntriesType> A_entries_v(nstreams);
    std::vector<ValuesType> A_values_v(nstreams);
    std::vector<RowMapType> L_row_map_v(nstreams);
    std::vector<EntriesType> L_entries_v(nstreams);
    std::vector<ValuesType> L_values_v(nstreams);
    std::vector<RowMapType> U_row_map_v(nstreams);
    std::vector<EntriesType> U_entries_v(nstreams);
    std::vector<ValuesType> U_values_v(nstreams);

    std::vector<std::vector<scalar_t>> Afix = get_fixture<scalar_t>();

    RowMapType row_map;
    EntriesType entries;
    ValuesType values;

    compress_matrix(row_map, entries, values, Afix);

    const size_type nrows = Afix.size();
    const size_type nnz   = values.extent(0);

    RowMapType_hostmirror hrow_map("hrow_map", nrows + 1);
    EntriesType_hostmirror hentries("hentries", nnz);
    ValuesType_hostmirror hvalues("hvalues", nnz);

    Kokkos::deep_copy(hrow_map, row_map);
    Kokkos::deep_copy(hentries, entries);
    Kokkos::deep_copy(hvalues, values);

    typename KernelHandle::const_nnz_lno_t fill_lev = 2;

    for (int i = 0; i < nstreams; i++) {
      // Allocate A as input
      A_row_map_v[i] = RowMapType("A_row_map", nrows + 1);
      A_entries_v[i] = EntriesType("A_entries", nnz);
      A_values_v[i]  = ValuesType("A_values", nnz);

      // Copy from host to device
      Kokkos::deep_copy(A_row_map_v[i], hrow_map);
      Kokkos::deep_copy(A_entries_v[i], hentries);
      Kokkos::deep_copy(A_values_v[i], hvalues);

      // Create handle
      kh_v[i] = KernelHandle();
      kh_v[i].create_spiluk_handle(test_algo, nrows, 4 * nrows, 4 * nrows);
      kh_ptr_v[i] = &kh_v[i];

      auto spiluk_handle = kh_v[i].get_spiluk_handle();

      // Allocate L and U as outputs
      L_row_map_v[i] = RowMapType("L_row_map", nrows + 1);
      L_entries_v[i] = EntriesType("L_entries", spiluk_handle->get_nnzL());
      U_row_map_v[i] = RowMapType("U_row_map", nrows + 1);
      U_entries_v[i] = EntriesType("U_entries", spiluk_handle->get_nnzU());

      // Symbolic phase
      spiluk_symbolic(kh_ptr_v[i], fill_lev, A_row_map_v[i], A_entries_v[i],
                      L_row_map_v[i], L_entries_v[i], U_row_map_v[i],
                      U_entries_v[i], nstreams);

      Kokkos::fence();

      Kokkos::resize(L_entries_v[i], spiluk_handle->get_nnzL());
      Kokkos::resize(U_entries_v[i], spiluk_handle->get_nnzU());
      L_values_v[i] = ValuesType("L_values", spiluk_handle->get_nnzL());
      U_values_v[i] = ValuesType("U_values", spiluk_handle->get_nnzU());
    }  // Done handle creation and spiluk_symbolic on all streams

    // Numeric phase
    spiluk_numeric_streams(instances, kh_ptr_v, fill_lev, A_row_map_v,
                           A_entries_v, A_values_v, L_row_map_v, L_entries_v,
                           L_values_v, U_row_map_v, U_entries_v, U_values_v);

    for (int i = 0; i < nstreams; i++) instances[i].fence();

    // Checking
    for (int i = 0; i < nstreams; i++) {
      check_result(A_row_map_v[i], A_entries_v[i], A_values_v[i],
                   L_row_map_v[i], L_entries_v[i], L_values_v[i],
                   U_row_map_v[i], U_entries_v[i], U_values_v[i]);

      kh_v[i].destroy_spiluk_handle();
    }
  }

  static void run_test_spiluk_streams_blocks(SPILUKAlgorithm test_algo,
                                             int nstreams) {
    // Workaround for OpenMP: skip tests if concurrency < nstreams because of
    // not enough resource to partition
    bool run_streams_test = true;
#ifdef KOKKOS_ENABLE_OPENMP
    if (std::is_same<typename device::execution_space, Kokkos::OpenMP>::value) {
      int exec_concurrency = execution_space().concurrency();
      if (exec_concurrency < nstreams) {
        run_streams_test = false;
        std::cout << "  Skip stream test: concurrency = " << exec_concurrency
                  << std::endl;
      }
    }
#endif
    if (!run_streams_test) return;

    std::vector<int> weights(nstreams, 1);
    std::vector<execution_space> instances =
        Kokkos::Experimental::partition_space(execution_space(), weights);

    std::vector<KernelHandle> kh_v(nstreams);
    std::vector<KernelHandle*> kh_ptr_v(nstreams);
    std::vector<RowMapType> A_row_map_v(nstreams);
    std::vector<EntriesType> A_entries_v(nstreams);
    std::vector<ValuesType> A_values_v(nstreams);
    std::vector<RowMapType> L_row_map_v(nstreams);
    std::vector<EntriesType> L_entries_v(nstreams);
    std::vector<ValuesType> L_values_v(nstreams);
    std::vector<RowMapType> U_row_map_v(nstreams);
    std::vector<EntriesType> U_entries_v(nstreams);
    std::vector<ValuesType> U_values_v(nstreams);

    std::vector<std::vector<scalar_t>> Afix = get_fixture<scalar_t>();

    RowMapType row_map, brow_map;
    EntriesType entries, bentries;
    ValuesType values, bvalues;

    compress_matrix(row_map, entries, values, Afix);

    const size_type nrows       = Afix.size();
    const size_type block_size  = nrows % 2 == 0 ? 2 : 3;
    const size_type block_items = block_size * block_size;
    ASSERT_EQ(nrows % block_size, 0);

    // Convert to BSR
    Crs crs("crs for block spiluk test", nrows, nrows, values.extent(0), values,
            row_map, entries);
    Bsr bsr(crs, block_size);

    // Pull out views from BSR
    Kokkos::resize(brow_map, bsr.graph.row_map.extent(0));
    Kokkos::resize(bentries, bsr.graph.entries.extent(0));
    Kokkos::resize(bvalues, bsr.values.extent(0));
    Kokkos::deep_copy(brow_map, bsr.graph.row_map);
    Kokkos::deep_copy(bentries, bsr.graph.entries);
    Kokkos::deep_copy(bvalues, bsr.values);

    const size_type bnrows = brow_map.extent(0) - 1;
    const size_type bnnz   = bentries.extent(0);

    RowMapType_hostmirror hrow_map("hrow_map", bnrows + 1);
    EntriesType_hostmirror hentries("hentries", bnnz);
    ValuesType_hostmirror hvalues("hvalues", bnnz * block_items);

    Kokkos::deep_copy(hrow_map, brow_map);
    Kokkos::deep_copy(hentries, bentries);
    Kokkos::deep_copy(hvalues, bvalues);

    typename KernelHandle::const_nnz_lno_t fill_lev = 2;

    for (int i = 0; i < nstreams; i++) {
      // Allocate A as input
      A_row_map_v[i] = RowMapType("A_row_map", bnrows + 1);
      A_entries_v[i] = EntriesType("A_entries", bnnz);
      A_values_v[i]  = ValuesType("A_values", bnnz * block_items);

      // Copy from host to device
      Kokkos::deep_copy(A_row_map_v[i], hrow_map);
      Kokkos::deep_copy(A_entries_v[i], hentries);
      Kokkos::deep_copy(A_values_v[i], hvalues);

      // Create handle
      kh_v[i] = KernelHandle();
      kh_v[i].create_spiluk_handle(test_algo, bnrows, 4 * bnrows, 4 * bnrows,
                                   block_size);
      kh_ptr_v[i] = &kh_v[i];

      auto spiluk_handle = kh_v[i].get_spiluk_handle();

      // Allocate L and U as outputs
      L_row_map_v[i] = RowMapType("L_row_map", bnrows + 1);
      L_entries_v[i] = EntriesType("L_entries", spiluk_handle->get_nnzL());
      U_row_map_v[i] = RowMapType("U_row_map", bnrows + 1);
      U_entries_v[i] = EntriesType("U_entries", spiluk_handle->get_nnzU());

      // Symbolic phase
      spiluk_symbolic(kh_ptr_v[i], fill_lev, A_row_map_v[i], A_entries_v[i],
                      L_row_map_v[i], L_entries_v[i], U_row_map_v[i],
                      U_entries_v[i], nstreams);

      Kokkos::fence();

      Kokkos::resize(L_entries_v[i], spiluk_handle->get_nnzL());
      Kokkos::resize(U_entries_v[i], spiluk_handle->get_nnzU());
      L_values_v[i] =
          ValuesType("L_values", spiluk_handle->get_nnzL() * block_items);
      U_values_v[i] =
          ValuesType("U_values", spiluk_handle->get_nnzU() * block_items);
    }  // Done handle creation and spiluk_symbolic on all streams

    // Numeric phase
    spiluk_numeric_streams(instances, kh_ptr_v, fill_lev, A_row_map_v,
                           A_entries_v, A_values_v, L_row_map_v, L_entries_v,
                           L_values_v, U_row_map_v, U_entries_v, U_values_v);

    for (int i = 0; i < nstreams; i++) instances[i].fence();

    // Checking
    for (int i = 0; i < nstreams; i++) {
      check_result_block(A_row_map_v[i], A_entries_v[i], A_values_v[i],
                         L_row_map_v[i], L_entries_v[i], L_values_v[i],
                         U_row_map_v[i], U_entries_v[i], U_values_v[i],
                         block_size);

      kh_v[i].destroy_spiluk_handle();
    }
  }
};

}  // namespace Test

template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
void test_spiluk() {
  using TestStruct = Test::SpilukTest<scalar_t, lno_t, size_type, device>;
  // TestStruct::run_test_spiluk();
  TestStruct::run_test_spiluk_blocks();
  // TestStruct::run_test_spiluk_scale();
  // TestStruct::run_test_spiluk_scale_blocks();
}

template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
void test_spiluk_streams() {
  using TestStruct = Test::SpilukTest<scalar_t, lno_t, size_type, device>;

  // TestStruct::run_test_spiluk_streams(SPILUKAlgorithm::SEQLVLSCHD_TP1, 1);
  // TestStruct::run_test_spiluk_streams(SPILUKAlgorithm::SEQLVLSCHD_TP1, 2);
  // TestStruct::run_test_spiluk_streams(SPILUKAlgorithm::SEQLVLSCHD_TP1, 3);
  // TestStruct::run_test_spiluk_streams(SPILUKAlgorithm::SEQLVLSCHD_TP1, 4);

  // TestStruct::run_test_spiluk_streams_blocks(SPILUKAlgorithm::SEQLVLSCHD_TP1,
  //                                            1);
  // TestStruct::run_test_spiluk_streams_blocks(SPILUKAlgorithm::SEQLVLSCHD_TP1,
  //                                            2);
  // TestStruct::run_test_spiluk_streams_blocks(SPILUKAlgorithm::SEQLVLSCHD_TP1,
  //                                            3);
  // TestStruct::run_test_spiluk_streams_blocks(SPILUKAlgorithm::SEQLVLSCHD_TP1,
  //                                            4);
}

#define KOKKOSKERNELS_EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE)        \
  TEST_F(TestCategory,                                                     \
         sparse##_##spiluk##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) { \
    test_spiluk<SCALAR, ORDINAL, OFFSET, DEVICE>();                        \
    test_spiluk_streams<SCALAR, ORDINAL, OFFSET, DEVICE>();                \
  }

#define NO_TEST_COMPLEX

#include <Test_Common_Test_All_Type_Combos.hpp>

#undef KOKKOSKERNELS_EXECUTE_TEST
#undef NO_TEST_COMPLEX
