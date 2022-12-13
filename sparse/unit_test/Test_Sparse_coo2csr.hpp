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

#include "KokkosSparse_coo2csr.hpp"
#include "KokkosKernels_TestUtils.hpp"

namespace Test {
template <class ScalarType, class LayoutType, class ExeSpaceType>
void doCoo2Csr(size_t m, size_t n, ScalarType min_val, ScalarType max_val) {
  RandCooMat<ScalarType, LayoutType, ExeSpaceType> cooMat(m, n, m * n, min_val,
                                                          max_val);
  auto randRow  = cooMat.get_row();
  auto randCol  = cooMat.get_col();
  auto randData = cooMat.get_data();

#if 0
  long long staticRow[16] {0, 1, 3, 2, 3, 2, 2, 2, 0, 0, 0, 1, 2, 0, 3, 0};
  long long staticCol[16] {1, 1, 2, 3, 3, 2, 3, 2, 0, 0, 1, 3, 1, 2, 0, 0};
  float staticData[16] {7.28411, 8.17991, 8.84304, 5.01788, 9.85646, 
  5.79404, 8.42014, 1.90238, 8.24195, 4.39955, 3.2637, 5.4546, 
  6.51895, 8.09302, 9.36294, 3.44206};
  using rvt = decltype(randRow);
  using cvt = decltype(randCol);
  using dvt = decltype(randData);
  rvt row("coo row", 16);
  cvt col("coo col", 16);
  dvt data("coo data", 16);
  for (int i = 0; i < 16; i++) {
    row(i) = staticRow[i];
    col(i) = staticCol[i];
    data(i) = staticData[i];
  }
  // Even partitions with multiple threads
  auto csrMat = KokkosSparse::coo2csr(m, n, row, col, data, 4);
#else
  // TODO: Uneven partition, Even partitions with multiple threads, Uneven
  // partitions with multiple threads
  auto csrMat = KokkosSparse::coo2csr(m, n, randRow, randCol, randData);
#endif

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

    auto csr_col_ids_d = csrMat.graph.entries;
    auto csr_row_map_d = csrMat.graph.row_map;
    auto csr_vals_d    = csrMat.values;

    using ViewTypeCsrColIds = decltype(csr_col_ids_d);
    using ViewTypeCsrRowMap = decltype(csr_row_map_d);
    using ViewTypeCsrVals   = decltype(csr_vals_d);

    // Copy to host
    typename ViewTypeCsrColIds::HostMirror csr_col_ids =
        Kokkos::create_mirror_view(csr_col_ids_d);
    Kokkos::deep_copy(csr_col_ids, csr_col_ids_d);
    typename ViewTypeCsrRowMap::HostMirror csr_row_map =
        Kokkos::create_mirror_view(csr_row_map_d);
    Kokkos::deep_copy(csr_row_map, csr_row_map_d);
    typename ViewTypeCsrVals::HostMirror csr_vals =
        Kokkos::create_mirror_view(csr_vals_d);
    Kokkos::deep_copy(csr_vals, csr_vals_d);

    Kokkos::fence();

    for (int j = 0; j < cscMat.get_n(); ++j) {
      auto col_start = csc_col_map(j);
      auto col_len   = csc_col_map(j + 1) - col_start;

      for (int k = 0; k < col_len; ++k) {
        auto i = col_start + k;

        auto row_start = csr_row_map(csc_row_ids(i));
        auto row_len   = csr_row_map(csc_row_ids(i) + 1) - row_start;
        auto row_end   = row_start + row_len;

        if (row_len == 0) continue;

        // Linear search for corresponding element in csr matrix
        int l = row_start;
        while (l < row_end && csr_col_ids(l) != j) {
          ++l;
        }

        if (l == row_end)
          FAIL() << "csr element at (i: " << csc_row_ids(i) << ", j: " << j
                 << ") not found!" << std::endl;

        ASSERT_EQ(csc_vals(i), csr_vals(l))
            << "(i: " << csc_row_ids(i) << ", j: " << j << ")" << std::endl;
      }
    } */
}

template <class LayoutType, class ExeSpaceType>
void doAllScalarsCoo2Csr(size_t m, size_t n, int min, int max) {
  doCoo2Csr<float, LayoutType, ExeSpaceType>(m, n, min, max);
  doCoo2Csr<double, LayoutType, ExeSpaceType>(m, n, min, max);
  doCoo2Csr<Kokkos::complex<float>, LayoutType, ExeSpaceType>(m, n, min, max);
  doCoo2Csr<Kokkos::complex<double>, LayoutType, ExeSpaceType>(m, n, min, max);
}

template <class ExeSpaceType>
void doAllLayoutsCoo2Csr(size_t m, size_t n, int min, int max) {
  doAllScalarsCoo2Csr<Kokkos::LayoutLeft, ExeSpaceType>(m, n, min, max);
  doAllScalarsCoo2Csr<Kokkos::LayoutRight, ExeSpaceType>(m, n, min, max);
}

template <class ExeSpaceType>
void doAllCoo2csr(size_t m, size_t n) {
  int min = 1, max = 10;
  doAllLayoutsCoo2Csr<ExeSpaceType>(m, n, min, max);
}

TEST_F(TestCategory, sparse_coo2csr) {
  // Square cases
  for (size_t dim = 4; dim < 8 /* 1024 */; dim *= 4)
    doAllCoo2csr<TestExecSpace>(dim, dim);

  // Non-square cases
  /* for (size_t dim = 1; dim < 1024; dim *= 4) {
    doAllCoo2csr<TestExecSpace>(dim * 3, dim);
    doAllCoo2csr<TestExecSpace>(dim, dim * 3);
  } */

  // Fully sparse
  /* doCoo2Csr<float, Kokkos::LayoutLeft, TestExecSpace>(5, 5, 1, 10, true);
  doCoo2Csr<double, Kokkos::LayoutRight, TestExecSpace>(50, 10, 10, 100, true);
*/

  // Test edge case: len(coo) % team_size != 0
  RandCooMat<double, Kokkos::LayoutLeft, TestExecSpace> cooMat(4, 4, 4 * 4, 1,
                                                               10);
  auto row    = cooMat.get_row();
  auto col    = cooMat.get_col();
  auto data   = cooMat.get_data();
  auto csrMat = KokkosSparse::coo2csr(4, 4, row, col, data, 3);
}

#if 0
class CasBug {
  public:
  using tpType = Kokkos::TeamPolicy<Kokkos::OpenMP>;
  using tpMemberType = typename tpType::member_type;
  using svaType = Kokkos::View<int *, Kokkos::LayoutRight, typename Kokkos::OpenMP::scratch_memory_space>;
  Kokkos::View<int *, Kokkos::OpenMP> a;

  KOKKOS_INLINE_FUNCTION
  void operator()(const tpMemberType &member) const {
    int start_n = member.league_rank() * 4;
    svaType sva(member.team_scratch(0), 4);
    
    Kokkos::single(Kokkos::PerTeam(member), [&] {
        for (unsigned i = 0; i < 4; i++)
          sva(i) = -1;
    });
    member.team_barrier();

    Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, 0, 4),
          [&](const int &tid) {
            Kokkos::atomic_compare_exchange_strong(sva.data() + a(start_n + tid), int(-1), int(-2 - member.league_rank()));
          });
    member.team_barrier();

    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      int stop_n = start_n + 4;
      int j = 0;
      for (int i = start_n; i < stop_n; i++, j++) {
        a(i) = sva(j);
      }
    });
  }
};

TEST_F(TestCategory, kokkos_cas_bug) {
  CasBug casBug;
  casBug.a = Kokkos::View<int *, Kokkos::OpenMP>("a", 16);
  int staticCol[16] {1, 1, 2, 3, 3, 2, 3, 2, 0,  0, 1, 3, 1, 2, 0, 0};
  for (int i = 0; i < 16; i++)
    casBug.a(i) = staticCol[i];

  CasBug::tpType tp(4, 4);
  int shmem_size = CasBug::svaType::shmem_size(4);
  tp.set_scratch_size(0, Kokkos::PerTeam(shmem_size));
  Kokkos::parallel_for("casBug", tp, casBug);
  Kokkos::fence();
  for (int i = 0; i < 16; i++) {
    if (i % 4 == 0)
      printf("\nrank: %d:\n", i / 4);
    printf("%d\t", staticCol[i]);
  }
  for (int i = 0; i < 16; i++) {
    if (i % 4 == 0)
      printf("\nrank: %d:\n", i / 4);
    printf("%d\t", casBug.a(i));
  }
  printf("\n");
}
#endif
}  // namespace Test