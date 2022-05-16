/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <Kokkos_Concepts.hpp>
#include <string>
#include <stdexcept>

#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include <KokkosKernels_IOUtils.hpp>
#include "KokkosBlas1_nrm2.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_par_ilut.hpp"

#include <gtest/gtest.h>

using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

// #ifndef kokkos_complex_double
// #define kokkos_complex_double Kokkos::complex<double>
// #define kokkos_complex_float Kokkos::complex<float>
// #endif

typedef Kokkos::complex<double> kokkos_complex_double;
typedef Kokkos::complex<float> kokkos_complex_float;

namespace Test {

template <typename scalar_t, typename lno_t, typename size_type, typename device>
std::vector<std::vector<scalar_t>> decompress_matrix(
  Kokkos::View<size_type*, device>& row_map,
  Kokkos::View<lno_t*, device>& entries,
  Kokkos::View<scalar_t*, device>& values
                                                       )
{
  const auto nrows = row_map.size() - 1;
  std::vector<std::vector<scalar_t> > result;
  result.resize(nrows);
  for (auto& row : result) {
    row.resize(nrows, 0.0);
  }

  auto hrow_map = Kokkos::create_mirror_view(row_map);
  auto hentries = Kokkos::create_mirror_view(entries);
  auto hvalues  = Kokkos::create_mirror_view(values);
  Kokkos::deep_copy(hrow_map, row_map);
  Kokkos::deep_copy(hentries, entries);
  Kokkos::deep_copy(hvalues, values);

  for (size_type row_idx = 0; row_idx < nrows; ++row_idx) {
    const size_type row_nnz_begin = row_map(row_idx);
    const size_type row_nnz_end   = row_map(row_idx+1);
    for (size_type row_nnz = row_nnz_begin; row_nnz < row_nnz_end; ++row_nnz) {
      const lno_t col_idx = entries(row_nnz);
      const scalar_t value = values(row_nnz);
      result[row_idx][col_idx] = value;
    }
  }

  return result;
}

template <typename scalar_t, typename lno_t, typename size_type, typename device>
void check_matrix(
  Kokkos::View<size_type*, device>& row_map,
  Kokkos::View<lno_t*, device>& entries,
  Kokkos::View<scalar_t*, device>& values,
  const std::vector<std::vector<scalar_t>>& expected)
{
  const auto decompressed_mtx = decompress_matrix(row_map, entries, values);

  const auto nrows = row_map.size() - 1;
  for (size_type row_idx = 0; row_idx < nrows; ++row_idx) {
    for (size_type col_idx = 0; col_idx < nrows; ++col_idx) {
      EXPECT_EQ(expected[row_idx][col_idx], decompressed_mtx[row_idx][col_idx]);
    }
  }
}


template <typename scalar_t>
void print_matrix(const std::vector<std::vector<scalar_t> >& matrix)
{
  for (const auto& row : matrix) {
    for (const auto& item : row) {
      std::printf("%.2f ", item);
    }
    std::cout << std::endl;
  }
}

template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
void run_test_par_ilut() {
  typedef Kokkos::View<size_type*, device> RowMapType;
  typedef Kokkos::View<lno_t*, device> EntriesType;
  typedef Kokkos::View<scalar_t*, device> ValuesType;
  typedef Kokkos::Details::ArithTraits<scalar_t> AT;

  // Simple test fixture A
  std::vector<std::vector<scalar_t> > A = {
    {1.,   6.,   4., 7.},
    {2.,  -5.,   0., 8.},
    {0.5, -3.,   6., 0.},
    {0.2, -0.5, -9., 0.}
  };

  const scalar_t ZERO = scalar_t(0);
  const scalar_t ONE  = scalar_t(1);
  const scalar_t MONE = scalar_t(-1);

  const size_type nrows = A.size();

  // Count A, L, U nnz's
  size_type nnz = 0, nnzL = 0, nnzU = 0;
  for (size_type row_idx = 0; row_idx < nrows; ++row_idx) {
    for (size_type col_idx = 0; col_idx < nrows; ++col_idx) {
      if (A[row_idx][col_idx] != ZERO) {
        ++nnz;
        nnzL += col_idx <= row_idx;
        nnzU += col_idx >= row_idx;
      }
    }
  }

  // Allocate device CRS views for A
  RowMapType row_map("row_map", nrows + 1);
  EntriesType entries("entries", nnz);
  ValuesType values("values", nnz);

  // Create host mirror views for CRS A
  auto hrow_map = Kokkos::create_mirror_view(row_map);
  auto hentries = Kokkos::create_mirror_view(entries);
  auto hvalues  = Kokkos::create_mirror_view(values);

  // Compress A into CRS (host views)
  size_type curr_nnz = 0;
  for (size_type row_idx = 0; row_idx < nrows; ++row_idx) {
    for (size_type col_idx = 0; col_idx < nrows; ++col_idx) {
      if (A[row_idx][col_idx] != ZERO) {
        hentries(curr_nnz) = col_idx;
        hvalues(curr_nnz) = A[row_idx][col_idx];
        ++curr_nnz;
      }
      hrow_map(row_idx+1) = curr_nnz;
    }
  }

  // Copy host A CRS views to device A CRS views
  Kokkos::deep_copy(row_map, hrow_map);
  Kokkos::deep_copy(entries, hentries);
  Kokkos::deep_copy(values, hvalues);

  // Make kernel handle
  typedef KokkosKernels::Experimental::KokkosKernelsHandle<
      size_type, lno_t, scalar_t, typename device::execution_space,
      typename device::memory_space, typename device::memory_space>
      KernelHandle;

  KernelHandle kh;

  kh.create_par_ilut_handle(nrows, nnzL, nnzU);

  auto par_ilut_handle = kh.get_par_ilut_handle();

  // Allocate L and U CRS views as outputs
  RowMapType L_row_map("L_row_map",  nrows + 1);
  EntriesType L_entries("L_entries", nnzL + nrows); // overallocate to be safe
  ValuesType L_values("L_values",    nnzL + nrows); // overallocate to be safe
  RowMapType U_row_map("U_row_map",  nrows + 1);
  EntriesType U_entries("U_entries", nnzU + nrows); // overallocate to be safe
  ValuesType U_values("U_values",    nnzU + nrows); // overallocate to be safe

  typename KernelHandle::const_nnz_lno_t fill_lev = 2; // Does par_ilut need this?

  // Initial L/U approximations for A
  par_ilut_symbolic(&kh, fill_lev,
                    row_map, entries, values,
                    L_row_map, L_entries, L_values,
                    U_row_map, U_entries, U_values);

  Kokkos::fence();

  EXPECT_EQ(par_ilut_handle->get_nnzL(), 10);
  EXPECT_EQ(par_ilut_handle->get_nnzU(), 8);

  Kokkos::resize(L_entries, par_ilut_handle->get_nnzL());
  Kokkos::resize(L_values, par_ilut_handle->get_nnzL());
  Kokkos::resize(U_entries, par_ilut_handle->get_nnzU());
  Kokkos::resize(U_values, par_ilut_handle->get_nnzU());

  std::vector<std::vector<scalar_t> > expected_L = {
    {1., 0., 0., 0.},
    {2., 1., 0., 0.},
    {0.50, -3., 1., 0.},
    {0.20, -0.50, -9., 1.}
  };
  check_matrix(L_row_map, L_entries, L_values, expected_L);

  std::vector<std::vector<scalar_t> > expected_U = {
    {1., 6., 4., 7.},
    {0., -5., 0., 8.},
    {0., 0., 6., 0.},
    {0., 0., 0., 1.}
  };
  check_matrix(U_row_map, U_entries, U_values, expected_U);

  par_ilut_numeric(&kh, row_map, entries, values,
                   L_row_map, L_entries, L_values,
                   U_row_map, U_entries, U_values);

  Kokkos::fence();

  std::vector<std::vector<scalar_t> > expected_L_candidates = {
    {1., 0., 0., 0.},
    {2., 1., 0., 0.},
    {0.50, -3., 1., 0.},
    {0.20, -0.50, -9., 1.}
  };
  std::cout << "L candidates: " << std::endl;
  print_matrix(decompress_matrix(L_row_map, L_entries, L_values));
  std::cout << "L_row_map: " << std::endl;
  for(size_type i = 0; i < L_row_map.extent(0); ++i) { std::cout << L_row_map(i) << " "; }
  std::cout << "\nL_entries: " << std::endl;
  for(size_type i = 0; i < L_entries.extent(0); ++i) { std::cout << L_entries(i) << " "; }
  std::cout << "\nL_values: " << std::endl;
  for(size_type i = 0; i < L_values.extent(0); ++i) { std::cout << L_values(i) << " "; }
  std::cout << "\nL candidates expected: " << std::endl;
  print_matrix(expected_L_candidates);

  check_matrix(L_row_map, L_entries, L_values, expected_L_candidates);

  std::vector<std::vector<scalar_t> > expected_U_candidates = {
    {1., 6., 4., 7.},
    {0., -5., -8., 8.},
    {0., 0., 6., 20.50},
    {0., 0., 0., 1.}
  };

  std::cout << "U candidates: " << std::endl;
  print_matrix(decompress_matrix(U_row_map, U_entries, U_values));
  std::cout << "U_row_map: " << std::endl;
  for(size_type i = 0; i < U_row_map.extent(0); ++i) { std::cout << U_row_map(i) << " "; }
  std::cout << "\nU_entries: " << U_entries.extent(0) << std::endl;
  for(size_type i = 0; i < U_entries.extent(0); ++i) { std::cout << U_entries(i) << " "; }
  std::cout << "\nU_values: " << std::endl;
  for(size_type i = 0; i < U_values.extent(0); ++i) { std::cout << U_values(i) << " "; }
  std::cout << "\nU candidates expected: " << std::endl;
  print_matrix(expected_U_candidates);

  check_matrix(U_row_map, U_entries, U_values, expected_U_candidates);

  // Checking

  kh.destroy_par_ilut_handle();
}

}  // namespace Test

template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
void test_par_ilut() {
  Test::run_test_par_ilut<scalar_t, lno_t, size_type, device>();
}

#define KOKKOSKERNELS_EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE)     \
  TEST_F(TestCategory,                                                  \
         sparse##_##par_ilut##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) { \
    test_par_ilut<SCALAR, ORDINAL, OFFSET, DEVICE>();                        \
  }

#define NO_TEST_COMPLEX

#include <Test_Common_Test_All_Type_Combos.hpp>

#undef KOKKOSKERNELS_EXECUTE_TEST
#define NO_TEST_COMPLEX
