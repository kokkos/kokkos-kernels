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

#include "KokkosSparse_coo2crs.hpp"
#include "KokkosKernels_TestUtils.hpp"

namespace Test {
template <class CrsType, class RowType, class ColType, class DataType>
void check_crs_matrix(CrsType crsMat, RowType row, ColType col, DataType data) {
  // Copy coo to host
  typename RowType::HostMirror row_h = Kokkos::create_mirror_view(row);
  Kokkos::deep_copy(row_h, row);
  typename ColType::HostMirror col_h = Kokkos::create_mirror_view(col);
  Kokkos::deep_copy(col_h, col);
  typename DataType::HostMirror data_h = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(data_h, data);

  auto crs_col_ids_d = crsMat.graph.entries;
  auto crs_row_map_d = crsMat.graph.row_map;
  auto crs_vals_d    = crsMat.values;

  using ViewTypeCrsColIds = decltype(crs_col_ids_d);
  using ViewTypeCrsRowMap = decltype(crs_row_map_d);
  using ViewTypeCrsVals   = decltype(crs_vals_d);

  // Copy crs to host
  typename ViewTypeCrsColIds::HostMirror crs_col_ids =
      Kokkos::create_mirror_view(crs_col_ids_d);
  Kokkos::deep_copy(crs_col_ids, crs_col_ids_d);
  typename ViewTypeCrsRowMap::HostMirror crs_row_map =
      Kokkos::create_mirror_view(crs_row_map_d);
  Kokkos::deep_copy(crs_row_map, crs_row_map_d);
  typename ViewTypeCrsVals::HostMirror crs_vals =
      Kokkos::create_mirror_view(crs_vals_d);
  Kokkos::deep_copy(crs_vals, crs_vals_d);

  Kokkos::fence();
}
template <class ScalarType, class LayoutType, class ExeSpaceType>
void doCoo2Crs(size_t m, size_t n, ScalarType min_val, ScalarType max_val) {
  RandCooMat<ScalarType, LayoutType, ExeSpaceType> cooMat(m, n, m * n, min_val,
                                                          max_val);
  auto randRow  = cooMat.get_row();
  auto randCol  = cooMat.get_col();
  auto randData = cooMat.get_data();

  // TODO: Uneven partition, Even partitions with multiple threads, Uneven
  // partitions with multiple threads
  auto crsMat = KokkosSparse::coo2crs(m, n, randRow, randCol, randData);

  /*
    auto csc_row_ids_d = cscMat.get_row_ids();
    auto csc_col_map_d = cscMat.get_col_map();
    auto csc_vals_d    = cscMat.get_vals();

    using ViewTypeRowIds = decltype(csc_row_ids_d);
    using ViewTypeColMap = decltype(csc_col_map_d);
    using ViewTypeVals   = decltype(csc_vals_d);

    // Copy to host
    typename ViewTypeRowIds::HostMirror csc_row_ids =
        Kokkos::create_mirror_view(csc_row_ids_d);
    Kokkos::deep_copy(csc_row_ids, csc_row_ids_d);
    typename ViewTypeColMap::HostMirror csc_col_map =
        Kokkos::create_mirror_view(csc_col_map_d);
    Kokkos::deep_copy(csc_col_map, csc_col_map_d);
    typename ViewTypeVals::HostMirror csc_vals =
        Kokkos::create_mirror_view(csc_vals_d);
    Kokkos::deep_copy(csc_vals, csc_vals_d);

    auto crs_col_ids_d = crsMat.graph.entries;
    auto crs_row_map_d = crsMat.graph.row_map;
    auto crs_vals_d    = crsMat.values;

    using ViewTypeCrsColIds = decltype(crs_col_ids_d);
    using ViewTypeCrsRowMap = decltype(crs_row_map_d);
    using ViewTypeCrsVals   = decltype(crs_vals_d);

    // Copy to host
    typename ViewTypeCrsColIds::HostMirror crs_col_ids =
        Kokkos::create_mirror_view(crs_col_ids_d);
    Kokkos::deep_copy(crs_col_ids, crs_col_ids_d);
    typename ViewTypeCrsRowMap::HostMirror crs_row_map =
        Kokkos::create_mirror_view(crs_row_map_d);
    Kokkos::deep_copy(crs_row_map, crs_row_map_d);
    typename ViewTypeCrsVals::HostMirror crs_vals =
        Kokkos::create_mirror_view(crs_vals_d);
    Kokkos::deep_copy(crs_vals, crs_vals_d);

    Kokkos::fence();

    for (int j = 0; j < cscMat.get_n(); ++j) {
      auto col_start = csc_col_map(j);
      auto col_len   = csc_col_map(j + 1) - col_start;

      for (int k = 0; k < col_len; ++k) {
        auto i = col_start + k;

        auto row_start = crs_row_map(csc_row_ids(i));
        auto row_len   = crs_row_map(csc_row_ids(i) + 1) - row_start;
        auto row_end   = row_start + row_len;

        if (row_len == 0) continue;

        // Linear search for corresponding element in crs matrix
        int l = row_start;
        while (l < row_end && crs_col_ids(l) != j) {
          ++l;
        }

        if (l == row_end)
          FAIL() << "crs element at (i: " << csc_row_ids(i) << ", j: " << j
                 << ") not found!" << std::endl;

        ASSERT_EQ(csc_vals(i), crs_vals(l))
            << "(i: " << csc_row_ids(i) << ", j: " << j << ")" << std::endl;
      }
    } */
}

template <class LayoutType, class ExeSpaceType>
void doAllScalarsCoo2Crs(size_t m, size_t n, int min, int max) {
  doCoo2Crs<float, LayoutType, ExeSpaceType>(m, n, min, max);
  /* doCoo2Crs<double, LayoutType, ExeSpaceType>(m, n, min, max); */
  /* doCoo2Crs<Kokkos::complex<float>, LayoutType, ExeSpaceType>(m, n, min,
  max);
  doCoo2Crs<Kokkos::complex<double>, LayoutType, ExeSpaceType>(m, n, min, max);
*/
}

template <class ExeSpaceType>
void doAllLayoutsCoo2Crs(size_t m, size_t n, int min, int max) {
  doAllScalarsCoo2Crs<Kokkos::LayoutLeft, ExeSpaceType>(m, n, min, max);
  doAllScalarsCoo2Crs<Kokkos::LayoutRight, ExeSpaceType>(m, n, min, max);
}

template <class ExeSpaceType>
void doAllCoo2crs(size_t m, size_t n) {
  int min = 1, max = 10;
  doAllLayoutsCoo2Crs<ExeSpaceType>(m, n, min, max);
}

TEST_F(TestCategory, sparse_coo2crs) {
  // Square cases
  for (size_t dim = 4; dim < 8 /* 1024 */; dim *= 4)
    doAllCoo2crs<TestExecSpace>(dim, dim);

  // Non-square cases
  /* for (size_t dim = 1; dim < 1024; dim *= 4) {
    doAllCoo2crs<TestExecSpace>(dim * 3, dim);
    doAllCoo2crs<TestExecSpace>(dim, dim * 3);
  } */

  // Fully sparse
  /* doCoo2Crs<float, Kokkos::LayoutLeft, TestExecSpace>(5, 5, 1, 10, true);
  doCoo2Crs<double, Kokkos::LayoutRight, TestExecSpace>(50, 10, 10, 100, true);
*/

  // Test edge case: len(coo) % team_size != 0
  RandCooMat<double, Kokkos::LayoutLeft, TestExecSpace> cooMat(4, 4, 4 * 4, 1,
                                                               10);
  auto row    = cooMat.get_row();
  auto col    = cooMat.get_col();
  auto data   = cooMat.get_data();
  auto crsMat = KokkosSparse::coo2crs(4, 4, row, col, data, 3);
}

TEST_F(TestCategory, sparse_coo2crs_staticMatrix_edgeCases) {
  int m = 4;
  int n = 4;
  long long staticRow[16]{0, 1, 3, 2, 3, 2, 2, 2, 0, 0, 0, 1, 2, 0, 3, 0};
  long long staticCol[16]{1, 1, 2, 3, 3, 2, 3, 2, 0, 0, 1, 3, 1, 2, 0, 0};
  float staticData[16]{7.28411, 8.17991, 8.84304, 5.01788, 9.85646, 5.79404,
                       8.42014, 1.90238, 8.24195, 4.39955, 3.2637,  5.4546,
                       6.51895, 8.09302, 9.36294, 3.44206};
  Kokkos::View<long long *, TestExecSpace> row("coo row", 16);
  Kokkos::View<long long *, TestExecSpace> col("coo col", 16);
  Kokkos::View<float *, TestExecSpace> data("coo data", 16);
  for (int i = 0; i < 16; i++) {
    row(i)  = staticRow[i];
    col(i)  = staticCol[i];
    data(i) = staticData[i];
  }
  // Even partitions with multiple threads
  auto crsMatTs4 = KokkosSparse::coo2crs(m, n, row, col, data, 4);
  printf("row_map: \n");
  for (long long i = 0; i < crsMatTs4.numRows(); i++)
    std::cout << crsMatTs4.graph.row_map(i) << " ";
  printf("\ncol_ids: \n");
  for (unsigned long i = 0; i < crsMatTs4.nnz(); i++)
    std::cout << crsMatTs4.graph.entries(i) << " ";
  printf("\nvals: \n");
  for (unsigned long i = 0; i < crsMatTs4.nnz(); i++)
    std::cout << crsMatTs4.values(i) << " ";
  std::cout << std::endl;

  // Uneven partitions with multiple threads
  auto crsMatTs3 = KokkosSparse::coo2crs(m, n, row, col, data, 3);
  printf("row_map: \n");
  for (long long i = 0; i < crsMatTs4.numRows(); i++)
    std::cout << crsMatTs3.graph.row_map(i) << " ";
  printf("\ncol_ids: \n");
  for (unsigned long i = 0; i < crsMatTs3.nnz(); i++)
    std::cout << crsMatTs3.graph.entries(i) << " ";
  printf("\nvals: \n");
  for (unsigned long i = 0; i < crsMatTs3.nnz(); i++)
    std::cout << crsMatTs3.values(i) << " ";
  std::cout << std::endl;
}

// TODO: Add reproducer for HashmapAccumulator vector atomic insert of same keys
// from multiple threads
}  // namespace Test