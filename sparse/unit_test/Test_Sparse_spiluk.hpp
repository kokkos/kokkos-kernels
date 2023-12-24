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

using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

using kokkos_complex_double = Kokkos::complex<double>;
using kokkos_complex_float  = Kokkos::complex<float>;

namespace Test {

template <typename scalar_t>
std::vector<std::vector<scalar_t>> get_9x9_fixture() {
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

template <typename scalar_t>
std::vector<std::vector<scalar_t>> get_4x4_fixture() {
  std::vector<std::vector<scalar_t>> A = {{10.00, 1.00, 0.00, 0.00},
                                          {0.00, 11.00, 0.00, 0.00},
                                          {0.00, 2.00, 12.00, 0.00},
                                          {5.00, 0.00, 0.00, 13.00}};
  return A;
}

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

    const auto result = check_result_impl(A, L, U, nrows);

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

    const auto result = check_result_impl(A, L, U, nrows, block_size);

    EXPECT_LT(result, 1e0);
  }

  static std::tuple<RowMapType, EntriesType, ValuesType, RowMapType,
                    EntriesType, ValuesType>
  run_and_check_spiluk(KernelHandle& kh, const RowMapType& row_map,
                       const EntriesType& entries, const ValuesType& values,
                       SPILUKAlgorithm alg, const size_type nrows,
                       const size_type nnz, const lno_t fill_lev) {
    kh.create_spiluk_handle(alg, nrows, 4 * nrows, 4 * nrows);

    auto spiluk_handle = kh.get_spiluk_handle();

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

    check_result(row_map, entries, values, L_row_map, L_entries, L_values,
                 U_row_map, U_entries, U_values);

    kh.destroy_spiluk_handle();

    // For team policy alg, check results against range policy
    if (alg == SPILUKAlgorithm::SEQLVLSCHD_TP1) {
      const auto [L_row_map_rp, L_entries_rp, L_values_rp, U_row_map_rp,
                  U_entries_rp, U_values_rp] =
          run_and_check_spiluk(kh, row_map, entries, values,
                               SPILUKAlgorithm::SEQLVLSCHD_RP, nrows, nnz,
                               fill_lev);

      EXPECT_NEAR_KK_1DVIEW(L_row_map, L_row_map_rp, EPS);
      EXPECT_NEAR_KK_1DVIEW(L_entries, L_entries_rp, EPS);
      EXPECT_NEAR_KK_1DVIEW(L_values, L_values_rp, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_row_map, U_row_map_rp, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_entries, U_entries_rp, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_values, U_values_rp, EPS);
    }

    return std::make_tuple(L_row_map, L_entries, L_values, U_row_map, U_entries,
                           U_values);
  }

  static void run_and_check_spiluk_block(
      KernelHandle& kh, const RowMapType& row_map, const EntriesType& entries,
      const ValuesType& values, SPILUKAlgorithm alg, const size_type nrows,
      const size_type nnz, const lno_t fill_lev, const size_type block_size) {
    const size_type block_items = block_size * block_size;
    kh.create_spiluk_handle(alg, nrows, 4 * nrows, 4 * nrows, block_size);

    auto spiluk_handle = kh.get_spiluk_handle();

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

    check_result_block(row_map, entries, values, L_row_map, L_entries, L_values,
                       U_row_map, U_entries, U_values, block_size);

    kh.destroy_spiluk_handle();

    // If block_size is 1, results should exactly match unblocked results
    if (block_size == 1) {
      const auto [L_row_map_u, L_entries_u, L_values_u, U_row_map_u,
                  U_entries_u, U_values_u] =
          run_and_check_spiluk(kh, row_map, entries, values, alg, nrows, nnz,
                               fill_lev);

      EXPECT_NEAR_KK_1DVIEW(L_row_map, L_row_map_u, EPS);
      EXPECT_NEAR_KK_1DVIEW(L_entries, L_entries_u, EPS);
      EXPECT_NEAR_KK_1DVIEW(L_values, L_values_u, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_row_map, U_row_map_u, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_entries, U_entries_u, EPS);
      EXPECT_NEAR_KK_1DVIEW(U_values, U_values_u, EPS);
    }
  }

  static void run_test_spiluk() {
    std::vector<std::vector<scalar_t>> A = get_9x9_fixture<scalar_t>();

    RowMapType row_map("row_map", 0);
    EntriesType entries("entries", 0);
    ValuesType values("values", 0);

    compress_matrix(row_map, entries, values, A);

    const size_type nrows = A.size();
    const size_type nnz   = values.extent(0);
    const lno_t fill_lev  = 2;

    KernelHandle kh;

    run_and_check_spiluk(kh, row_map, entries, values,
                         SPILUKAlgorithm::SEQLVLSCHD_RP, nrows, nnz, fill_lev);
    run_and_check_spiluk(kh, row_map, entries, values,
                         SPILUKAlgorithm::SEQLVLSCHD_TP1, nrows, nnz, fill_lev);
  }

  static void run_test_spiluk_streams(int test_algo, int nstreams) {
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

    std::vector<execution_space> instances;
    if (nstreams == 1)
      instances = Kokkos::Experimental::partition_space(execution_space(), 1);
    else if (nstreams == 2)
      instances =
          Kokkos::Experimental::partition_space(execution_space(), 1, 1);
    else if (nstreams == 3)
      instances =
          Kokkos::Experimental::partition_space(execution_space(), 1, 1, 1);
    else
      instances =
          Kokkos::Experimental::partition_space(execution_space(), 1, 1, 1, 1);

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

    std::vector<std::vector<scalar_t>> Afix = get_9x9_fixture<scalar_t>();

    RowMapType row_map("row_map", 0);
    EntriesType entries("entries", 0);
    ValuesType values("values", 0);

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
      if (test_algo == 0)
        kh_v[i].create_spiluk_handle(SPILUKAlgorithm::SEQLVLSCHD_RP, nrows,
                                     4 * nrows, 4 * nrows);
      else if (test_algo == 1)
        kh_v[i].create_spiluk_handle(SPILUKAlgorithm::SEQLVLSCHD_TP1, nrows,
                                     4 * nrows, 4 * nrows);
      kh_ptr_v[i] = &kh_v[i];

      auto spiluk_handle = kh_v[i].get_spiluk_handle();

      // Allocate L and U as outputs
      L_row_map_v[i] = RowMapType("L_row_map", nrows + 1);
      L_entries_v[i] = EntriesType("L_entries", spiluk_handle->get_nnzL());
      L_values_v[i]  = ValuesType("L_values", spiluk_handle->get_nnzL());
      U_row_map_v[i] = RowMapType("U_row_map", nrows + 1);
      U_entries_v[i] = EntriesType("U_entries", spiluk_handle->get_nnzU());
      U_values_v[i]  = ValuesType("U_values", spiluk_handle->get_nnzU());

      // Symbolic phase
      spiluk_symbolic(kh_ptr_v[i], fill_lev, A_row_map_v[i], A_entries_v[i],
                      L_row_map_v[i], L_entries_v[i], U_row_map_v[i],
                      U_entries_v[i], nstreams);

      Kokkos::fence();

      Kokkos::resize(L_entries_v[i], spiluk_handle->get_nnzL());
      Kokkos::resize(L_values_v[i], spiluk_handle->get_nnzL());
      Kokkos::resize(U_entries_v[i], spiluk_handle->get_nnzU());
      Kokkos::resize(U_values_v[i], spiluk_handle->get_nnzU());
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

  static void run_test_spiluk_blocks() {
    std::vector<std::vector<scalar_t>> A = get_9x9_fixture<scalar_t>();

    RowMapType row_map("row_map", 0), brow_map("brow_map", 0);
    EntriesType entries("entries", 0), bentries("bentries", 0);
    ValuesType values("values", 0), bvalues("bvalues", 0);

    compress_matrix(row_map, entries, values, A);

    const size_type nrows      = A.size();
    const size_type nnz        = values.extent(0);
    const lno_t fill_lev       = 2;
    const size_type block_size = nrows % 2 == 0 ? 2 : 3;
    ASSERT_EQ(nrows % block_size, 0);

    KernelHandle kh;

    // Check block_size=1 produces identical result to unblocked
    run_and_check_spiluk_block(kh, row_map, entries, values,
                               SPILUKAlgorithm::SEQLVLSCHD_RP, nrows, nnz,
                               fill_lev, 1);
    run_and_check_spiluk_block(kh, row_map, entries, values,
                               SPILUKAlgorithm::SEQLVLSCHD_TP1, nrows, nnz,
                               fill_lev, 1);

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
    const size_type bnnz   = values.extent(0);

    run_and_check_spiluk_block(kh, brow_map, bentries, bvalues,
                               SPILUKAlgorithm::SEQLVLSCHD_RP, bnrows, bnnz,
                               fill_lev, block_size);
    run_and_check_spiluk_block(kh, brow_map, bentries, bvalues,
                               SPILUKAlgorithm::SEQLVLSCHD_TP1, bnrows, bnnz,
                               fill_lev, block_size);
  }
};

}  // namespace Test

template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
void test_spiluk() {
  using TestStruct = Test::SpilukTest<scalar_t, lno_t, size_type, device>;
  TestStruct::run_test_spiluk();
  TestStruct::run_test_spiluk_blocks();
}

template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
void test_spiluk_streams() {
  using TestStruct = Test::SpilukTest<scalar_t, lno_t, size_type, device>;

  TestStruct::run_test_spiluk_streams(0, 1);
  TestStruct::run_test_spiluk_streams(0, 2);
  TestStruct::run_test_spiluk_streams(0, 3);
  TestStruct::run_test_spiluk_streams(0, 4);
  TestStruct::run_test_spiluk_streams(1, 1);
  TestStruct::run_test_spiluk_streams(1, 2);
  TestStruct::run_test_spiluk_streams(1, 3);
  TestStruct::run_test_spiluk_streams(1, 4);
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
