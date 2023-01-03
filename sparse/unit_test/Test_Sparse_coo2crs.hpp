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
  // TODO: speed this up on device with a kokkos unordered_map
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

  typename CrsType::row_map_type::non_const_type::HostMirror row_map_h =
      Kokkos::create_mirror_view(row_map);
  typename CrsType::values_type::HostMirror values_h =
      Kokkos::create_mirror_view(values);
  typename CrsType::staticcrsgraph_type::entries_type::HostMirror col_ids_h =
      Kokkos::create_mirror_view(col_ids);

  int row_len = 0;
  for (uint64_t i = 0; i < m; i++) {
    if (umap.find(i) != umap.end()) row_len += umap.at(i)->size();
    row_map_h(i + 1) = row_len;
  }

  for (uint64_t i = 0; i < m; i++) {
    if (umap.find(i) == umap.end())  // Fully sparse row
      continue;

    auto row_start = row_map_h(i);
    auto row_end   = row_map_h(i + 1);
    auto my_row    = umap.at(i);
    auto iter      = my_row->begin();
    for (uint64_t j = row_start; j < row_end; j++, iter++) {
      col_ids_h(j) = iter->first;
      values_h(j)  = iter->second;
    }
    delete my_row;
  }

  Kokkos::deep_copy(row_map, row_map_h);
  Kokkos::deep_copy(col_ids, col_ids_h);
  Kokkos::deep_copy(values, values_h);

  return CrsType("vanilla_coo2csr", m, n, nnz, values, row_map, col_ids);
}

template <class CrsType, class RowType, class ColType, class DataType>
void check_crs_matrix(CrsType crsMat, RowType row, ColType col, DataType data) {
  using value_type = typename DataType::value_type;
  using ats        = Kokkos::Details::ArithTraits<value_type>;

  // Copy coo to host
  typename RowType::HostMirror row_h = Kokkos::create_mirror_view(row);
  Kokkos::deep_copy(row_h, row);
  typename ColType::HostMirror col_h = Kokkos::create_mirror_view(col);
  Kokkos::deep_copy(col_h, col);
  typename DataType::HostMirror data_h = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(data_h, data);

  auto crsMatRef = vanilla_coo2crs<CrsType, typename RowType::HostMirror,
                                   typename ColType::HostMirror,
                                   typename DataType::HostMirror>(
      crsMat.numRows(), crsMat.numCols(), row_h, col_h, data_h);

  auto crs_col_ids_ref_d = crsMatRef.graph.entries;
  auto crs_row_map_ref_d = crsMatRef.graph.row_map;
  auto crs_vals_ref_d    = crsMatRef.values;

  using ViewTypeCrsColIdsRef = decltype(crs_col_ids_ref_d);
  using ViewTypeCrsRowMapRef = decltype(crs_row_map_ref_d);
  using ViewTypeCrsValsRef   = decltype(crs_vals_ref_d);

  // Copy crs to host
  typename ViewTypeCrsColIdsRef::HostMirror crs_col_ids_ref =
      Kokkos::create_mirror_view(crs_col_ids_ref_d);
  Kokkos::deep_copy(crs_col_ids_ref, crs_col_ids_ref_d);
  typename ViewTypeCrsRowMapRef::HostMirror crs_row_map_ref =
      Kokkos::create_mirror_view(crs_row_map_ref_d);
  Kokkos::deep_copy(crs_row_map_ref, crs_row_map_ref_d);
  typename ViewTypeCrsValsRef::HostMirror crs_vals_ref =
      Kokkos::create_mirror_view(crs_vals_ref_d);
  Kokkos::deep_copy(crs_vals_ref, crs_vals_ref_d);

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

  ASSERT_EQ(crsMatRef.nnz(), crsMat.nnz());

  for (int i = 0; i < crsMatRef.numRows(); i++) {
    ASSERT_EQ(crs_row_map_ref(i), crs_row_map(i))
        << "crs_row_map_ref(" << i << " = " << crs_row_map_ref(i) << " != "
        << "crs_row_map(" << i << " = " << crs_row_map(i) << " -- ";
  }

  for (int i = 0; i < crsMatRef.numRows(); ++i) {
    auto row_start_ref = crs_row_map_ref(i);
    auto row_stop_ref  = crs_row_map_ref(i + 1);
    auto row_len_ref   = row_stop_ref - row_start_ref;

    auto row_start = crs_row_map(i);
    auto row_len   = crs_row_map(i + 1) - row_start;

    ASSERT_EQ(row_start_ref, row_start);
    ASSERT_EQ(row_len_ref, row_len);

    for (auto j = row_start_ref; j < row_stop_ref; ++j) {
      // Look for the corresponding col_id
      auto col_id_ref      = crs_col_ids_ref(j);
      std::string fail_msg = "row: " + std::to_string(i) +
                             ", crs_col_ids_ref(" + std::to_string(j) +
                             ") = " + std::to_string(col_id_ref);

      auto k = row_start_ref;
      for (; k < row_stop_ref; ++k)
        if (crs_col_ids(k) == col_id_ref) break;
      if (k == row_stop_ref) FAIL() << fail_msg << " not found in crs_col_ids!";

      // ASSERT_EQ doesn't work -- values may be summed in different orders
      // We sum at most m x n values.
      auto eps =
          crsMatRef.numCols() * crsMatRef.numRows() * 10e1 * ats::epsilon();
      EXPECT_NEAR_KK(crs_vals_ref(j), crs_vals(k), eps,
                     fail_msg + " mismatched values!");
    }
  }
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
}

template <class LayoutType, class ExeSpaceType>
void doAllScalarsCoo2Crs(size_t m, size_t n, int min, int max) {
  doCoo2Crs<float, LayoutType, ExeSpaceType>(m, n, min, max);
  doCoo2Crs<double, LayoutType, ExeSpaceType>(m, n, min, max);
  doCoo2Crs<Kokkos::complex<float>, LayoutType, ExeSpaceType>(m, n, min, max);
  doCoo2Crs<Kokkos::complex<double>, LayoutType, ExeSpaceType>(m, n, min, max);
}

template <class ExeSpaceType>
void doAllLayoutsCoo2Crs(size_t m, size_t n, int min, int max) {
  doAllScalarsCoo2Crs<Kokkos::LayoutLeft, ExeSpaceType>(m, n, min, max);
  doAllScalarsCoo2Crs<Kokkos::LayoutRight, ExeSpaceType>(m, n, min, max);
}

template <class ExeSpaceType>
void doAllCoo2Crs(size_t m, size_t n) {
  int min = 1, max = 10;
  doAllLayoutsCoo2Crs<ExeSpaceType>(m, n, min, max);
}

TEST_F(TestCategory, sparse_coo2crs) {
  uint64_t ticks =
      std::chrono::high_resolution_clock::now().time_since_epoch().count() %
      UINT32_MAX;
  std::srand(ticks);

  // Square cases
  for (size_t i = 1; i < 256; i *= 4) {
    size_t dim = (std::rand() % 511) + 1;
    doAllCoo2Crs<TestExecSpace>(dim, dim);
  }

  // Non-square cases
  for (size_t i = 1; i < 256; i *= 4) {
    size_t m = (std::rand() % 511) + 1;
    size_t n = (std::rand() % 511) + 1;
    while (n == m) n = (std::rand() % 511) + 1;
    doAllCoo2Crs<TestExecSpace>(m, n);
  }
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

  typename Kokkos::View<long long *, TestExecSpace>::HostMirror row_h =
      Kokkos::create_mirror_view(row);
  typename Kokkos::View<long long *, TestExecSpace>::HostMirror col_h =
      Kokkos::create_mirror_view(col);
  typename Kokkos::View<float *, TestExecSpace>::HostMirror data_h =
      Kokkos::create_mirror_view(data);
  for (int i = 0; i < 16; i++) {
    row_h(i)  = staticRow[i];
    col_h(i)  = staticCol[i];
    data_h(i) = staticData[i];
  }

  Kokkos::deep_copy(row, row_h);
  Kokkos::deep_copy(col, col_h);
  Kokkos::deep_copy(data, data_h);

  // Even partitions with multiple threads
  auto crsMatTs4 = KokkosSparse::coo2crs(m, n, row, col, data, 4);
  check_crs_matrix(crsMatTs4, row_h, col_h, data_h);

  // Uneven partitions with multiple threads
  auto crsMatTs3 = KokkosSparse::coo2crs(m, n, row, col, data, 3);
  check_crs_matrix(crsMatTs3, row_h, col_h, data_h);

  // Team size too large
  auto crsMatTsTooLarge =
      KokkosSparse::coo2crs(m, n, row, col, data, UINT32_MAX);
  check_crs_matrix(crsMatTsTooLarge, row_h, col_h, data_h);

  // Even partitions, single thread, fully sparse row
  long long staticRowTs1[16]{0, 3, 0, 2, 2, 3, 0, 3, 2, 0, 0, 0, 0, 3, 3, 0};
  long long staticColTs1[16]{3, 1, 3, 1, 2, 2, 1, 1, 2, 3, 3, 1, 1, 0, 0, 0};
  float staticDataTs1[16]{6.1355,  6.53989, 8.58559, 6.37476, 4.18964, 2.41146,
                          1.82177, 1.4249,  1.52659, 5.50521, 8.0484,  3.98874,
                          6.74709, 3.35072, 7.81944, 5.83494};
  for (int i = 0; i < 16; i++) {
    row_h(i)  = staticRowTs1[i];
    col_h(i)  = staticColTs1[i];
    data_h(i) = staticDataTs1[i];
  }
  Kokkos::deep_copy(row, row_h);
  Kokkos::deep_copy(col, col_h);
  Kokkos::deep_copy(data, data_h);

  auto crsMatTs1 = KokkosSparse::coo2crs(m, n, row, col, data, 1);
  check_crs_matrix(crsMatTs1, row_h, col_h, data_h);

  // Fully sparse
  for (int i = 0; i < 16; i++) {
    row_h(i) = -staticRowTs1[i];
    col_h(i) = -staticColTs1[i];
  }
  Kokkos::deep_copy(row, row_h);
  Kokkos::deep_copy(col, col_h);

  auto crsMatFsTs1 = KokkosSparse::coo2crs(m, n, row, col, data, 1);
  check_crs_matrix(crsMatFsTs1, row_h, col_h, data);
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
namespace HashmapAccumulatorRaceToInsert {
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
}  // namespace HashmapAccumulatorRaceToInsert
TEST_F(TestCategory, HashmapAccumulator_RaceToInsertion) {
  using functorType = typename HashmapAccumulatorRaceToInsert::myFunctor;
  functorType functor;

  int team_size, n_teams;
  team_size = n_teams = 4;
  functorType::PolicyType policy(1, 1);

  int team_size_max = policy.team_size_max(functor, Kokkos::ParallelForTag());

  if (team_size > team_size_max) {
    team_size = 1;
    n_teams   = 16;
  }

  policy = functorType::PolicyType(n_teams, team_size);

  functor.keys_inserted =
      functorType::KeyViewType2("keys_inserted", n_teams, functor.key_len);
  functor.key_count = functorType::KeyViewType("key_count", n_teams);
  functor.keys_to_insert =
      functorType::KeyViewType("keys_to_insert", functor.key_len);

  typename functorType::KeyViewType::HostMirror kti =
      Kokkos::create_mirror_view(functor.keys_to_insert);
  kti(0) = 0;
  kti(1) = 0;
  kti(2) = 2;
  kti(3) = 2;
  Kokkos::deep_copy(functor.keys_to_insert, kti);
  typename functorType::KeyViewType2::HostMirror ki =
      Kokkos::create_mirror_view(functor.keys_inserted);
  for (int i = 0; i < n_teams; i++) {
    for (int j = 0; j < functor.key_len; j++) ki(i, j) = -1;
  }
  Kokkos::deep_copy(functor.keys_inserted, ki);

  int scratch = functorType::KeyViewScratch::shmem_size(functor.key_len) +
                functorType::SizeViewScratch::shmem_size(n_teams) +
                functorType::SizeViewScratch::shmem_size(functor.pow2_key_len +
                                                         functor.key_len);
  policy.set_scratch_size(0, Kokkos::PerTeam(scratch));
  Kokkos::parallel_for("Test::HashmapAccumulator_RaceToInsert", policy,
                       functor);
  TestExecSpace().fence();
  Kokkos::deep_copy(ki, functor.keys_inserted);
  typename functorType::KeyViewType::HostMirror kc =
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