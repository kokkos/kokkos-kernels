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
CrsType vanilla_coo2crs(size_t m, size_t n, RowType row, ColType col,
                        DataType data) {
  using RowIndexType = typename RowType::value_type;
  using ColIndexType = typename ColType::value_type;
  using ValueType    = typename DataType::value_type;
  std::unordered_map<RowIndexType,
                     std::unordered_map<ColIndexType, ValueType> *>
      umap;
  int nnz = 0;

  for (uint64_t i = 0; i < data.extent(0); i++) {
    auto r = row(i);
    auto c = col(i);
    auto d = data(i);

    if (r >= 0 && c >= 0) {
      if (umap.find(r) != umap.end()) {  // exists
        auto my_row = umap.at(r);
        if (my_row->find(c) != my_row->end())
          my_row->at(c) += d;
        else {
          my_row->insert(std::make_pair(c, d));
          nnz++;
        }
      } else {  // create a new row.
        auto new_row = new std::unordered_map<ColIndexType, ValueType>();
        umap.insert(std::make_pair(r, new_row));
        new_row->insert(std::make_pair(c, d));
        nnz++;
      }
    }
  }

  typename CrsType::row_map_type::non_const_type row_map("vanilla_row_map",
                                                         m + 1);
  typename CrsType::values_type values("vanilla_values", nnz);
  typename CrsType::staticcrsgraph_type::entries_type col_ids("vanilla_col_ids",
                                                              nnz);

  int row_len = 0;
  for (uint64_t i = 0; i < m; i++) {
    row_len += umap.at(i)->size();
    row_map(i + 1) = row_len;
  }

  for (uint64_t i = 0; i < m; i++) {
    auto row_start = row_map(i);
    auto row_end   = row_map(i + 1);
    auto my_row    = umap.at(i);
    auto iter      = my_row->begin();
    for (uint64_t j = row_start; j < row_end; j++, iter++) {
      col_ids(j) = iter->first;
      values(j)  = iter->second;
    }
    delete my_row;
  }

  printf("vanilla_row_map:\n");
  for (uint64_t i = 0; i < m; i++) printf("%lu ", row_map(i));
  printf("\n");
  printf("vanilla_col_ids:\n");
  for (int i = 0; i < nnz; i++) printf("%lld ", col_ids(i));
  printf("\n");
  for (int i = 0; i < nnz; i++) printf("%g ", values(i));
  printf("\n");

  return CrsType("vanilla_coo2csr", m, n, nnz, values, row_map, col_ids);
}

template <class CrsType, class RowType, class ColType, class DataType>
void check_crs_matrix(CrsType crsMat, RowType row, ColType col, DataType data) {
  // Copy coo to host
  typename RowType::HostMirror row_h = Kokkos::create_mirror_view(row);
  Kokkos::deep_copy(row_h, row);
  typename ColType::HostMirror col_h = Kokkos::create_mirror_view(col);
  Kokkos::deep_copy(col_h, col);
  typename DataType::HostMirror data_h = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(data_h, data);

  auto crsMatRef = vanilla_coo2crs<CrsType, RowType, ColType, DataType>(
      crsMat.numRows(), crsMat.numCols(), row_h, col_h, data_h);

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

  auto crsMat = KokkosSparse::coo2crs(m, n, randRow, randCol, randData);
  check_crs_matrix(crsMat, randRow, randCol, randData);

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
  check_crs_matrix(crsMatTs4, row, col, data);
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
  check_crs_matrix(crsMatTs3, row, col, data);
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

// Reading III.D of https://ieeexplore.ieee.org/abstract/document/7965111/
// indicates that the vector atomic insert routines were originally intended
// to be used by a set of vector lanes that performs insertions for a single row
// with a set of unique column ids. The part that worries me from a correctness
// perspective in the paper is:
// "If a key already exists in the hashmap, values are accumulated with “add”
// and Bitwiseor in numeric and symbolic phases, respectively.".
//
// I think a major challenge with the class is that
// it's not encapsulate and its behavior varies widely based on each insertion
// routine. This makes the class very difficult to understand and use unless you
// read through it in detail.
//
// The behavior this test demonstrates may not be a bug but an undocumented
// feature added for performance. Based on my experience trying to use this
// class when implementing coo2csr, my recommendation would be to rewrite the
// HashmapAccumulator. With the rewrite, any customization of the class for an
// algorithm would not be permitted within the class but instead would require a
// subclass such as: SpgemmHashmapAccumulator : public HashmapAccumulator. In
// this way, it's clear that anything using the child types is using a different
// behavior from the standard HashmapAccumulator insertion and lookup routines
// supported in the parent class.
//
// With that said, my assumption (after reading the insertion routine code) was
// that for any of the vector atomic insert routines a given key will be
// inserted exactly once and subsequent insertions would either SUM values or
// bitise-OR values. In this way, I thought of a new insertion as the creation
// of a bucket and the SUMing or bitwise-ORing of values as addition to that
// bucket.
//
// This test will fail if the same key appears twice in the hashmap.
namespace HashmapAccumulator_RaceToInsert {
struct myFunctor {
  using PolicyType = Kokkos::TeamPolicy<TestExecSpace>;
  using MemberType = typename PolicyType::member_type;

  using SizeType  = int32_t;
  using KeyType   = int32_t;
  using ValueType = int32_t;

  using ScratchSpaceType = typename TestExecSpace::scratch_memory_space;
  using KeyViewScratch =
      Kokkos::View<KeyType *, Kokkos::LayoutRight, ScratchSpaceType>;
  using SizeViewScratch =
      Kokkos::View<SizeType *, Kokkos::LayoutRight, ScratchSpaceType>;

  using UmapType = KokkosKernels::Experimental::HashmapAccumulator<
      SizeType, KeyType, ValueType,
      KokkosKernels::Experimental::HashOpType::bitwiseAnd>;

  using KeyViewType =
      Kokkos::View<KeyType *, Kokkos::LayoutRight, TestExecSpace>;
  using KeyViewType2 =
      Kokkos::View<KeyType **, Kokkos::LayoutRight, TestExecSpace>;

  KeyViewType keys_to_insert;
  KeyType key_len      = 4;
  KeyType pow2_key_len = 4;

  KeyViewType2 keys_inserted;
  KeyViewType key_count;

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType &member) const {
    KeyViewScratch keys(member.team_scratch(0), key_len);
    SizeViewScratch hash_ll(member.team_scratch(0), pow2_key_len + key_len + 1);
    volatile SizeType *used_size = hash_ll.data();
    auto *hash_begins            = hash_ll.data() + 1;
    auto *hash_nexts             = hash_begins + pow2_key_len;

    // Initialize hash_begins to -1
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, 0, pow2_key_len),
                         [&](const int &tid) { hash_begins[tid] = -1; });
    *used_size = 0;
    // Wait for each team's hashmap to be initialized.
    member.team_barrier();

    UmapType team_uset(key_len, pow2_key_len - 1, hash_begins, hash_nexts,
                       keys.data(), nullptr);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, 0, 4), [&](const int &i) {
          KeyType key = keys_to_insert(i);
          team_uset.vector_atomic_insert_into_hash(key, used_size);
        });

    member.team_barrier();
    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      int rank        = member.league_rank();
      key_count(rank) = *used_size;
      for (int i = 0; i < *used_size; i++) keys_inserted(rank, i) = keys(i);
    });
  }
};
}  // namespace HashmapAccumulator_RaceToInsert
TEST_F(TestCategory, HashmapAccumulator_RaceToInsertion) {
  using namespace Test::HashmapAccumulator_RaceToInsert;
  myFunctor functor;

  int team_size, n_teams;
  team_size = n_teams = 4;
  typename myFunctor::PolicyType policy(n_teams, team_size);

  functor.keys_inserted =
      myFunctor::KeyViewType2("keys_inserted", n_teams, functor.key_len);
  functor.key_count = myFunctor::KeyViewType("key_count", n_teams);
  functor.keys_to_insert =
      myFunctor::KeyViewType("keys_to_insert", functor.key_len);

  typename myFunctor::KeyViewType::HostMirror kti =
      Kokkos::create_mirror_view(functor.keys_to_insert);
  kti(0) = 0;
  kti(1) = 0;
  kti(2) = 2;
  kti(3) = 2;
  Kokkos::deep_copy(functor.keys_to_insert, kti);
  typename myFunctor::KeyViewType2::HostMirror ki =
      Kokkos::create_mirror_view(functor.keys_inserted);
  for (int i = 0; i < n_teams; i++) {
    for (int j = 0; j < functor.key_len; j++) ki(i, j) = -1;
  }
  Kokkos::deep_copy(functor.keys_inserted, ki);

  int scratch = myFunctor::KeyViewScratch::shmem_size(functor.key_len) +
                myFunctor::SizeViewScratch::shmem_size(n_teams) +
                myFunctor::SizeViewScratch::shmem_size(functor.pow2_key_len +
                                                       functor.key_len);
  policy.set_scratch_size(0, Kokkos::PerTeam(scratch));
  Kokkos::parallel_for("Test::HashmapAccumulator_RaceToInsert", policy,
                       functor);
  TestExecSpace().fence();
  Kokkos::deep_copy(ki, functor.keys_inserted);
  typename myFunctor::KeyViewType::HostMirror kc =
      Kokkos::create_mirror_view(functor.key_count);
  Kokkos::deep_copy(kc, functor.key_count);

  bool passed = true;

  for (int i = 0; i < n_teams; i++) {
    // We only have 2 unique keys (0 and 2) above
    if (kc(i) > 2) {
      printf("team %d has duplicate insertions. keys inserted:\n", i);
      for (int j = 0; j < functor.key_len; j++)
        printf("ki(%d, %d): %d\n", i, j, ki(i, j));
      passed = false;
    }
  }

  if (!passed) printf("%s:%d: FAILED.\n", __FILE__, __LINE__);
}
}  // namespace Test