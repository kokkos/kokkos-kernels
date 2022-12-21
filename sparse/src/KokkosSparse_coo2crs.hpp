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

#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_Utils.hpp"
#include "KokkosKernels_HashmapAccumulator.hpp"
#include <Kokkos_StdAlgorithms.hpp>

#ifndef _KOKKOSSPARSE_COO2CRS_HPP
#define _KOKKOSSPARSE_COO2CRS_HPP
namespace KokkosSparse {
namespace Impl {
template <class DimType, class RowViewType, class ColViewType,
          class DataViewType>
class Coo2Crs {
 private:
  using CrsST             = typename DataViewType::value_type;
  using CrsOT             = int64_t;
  using CrsET             = typename DataViewType::execution_space;
  using CrsMT             = void;
  using CrsSzT            = size_t;
  using CrsType           = CrsMatrix<CrsST, CrsOT, CrsET, CrsMT, CrsSzT>;
  using CrsValsViewType   = typename CrsType::values_type;
  using CrsRowMapViewType = typename CrsType::row_map_type::non_const_type;
  using CrsColIdViewType  = typename CrsType::index_type;
  using RowIdViewType     = RowViewType;
  using AtomicRowIdViewType =
      Kokkos::View<typename RowIdViewType::value_type *,
                   typename RowIdViewType::array_layout, CrsET,
                   Kokkos::MemoryTraits<Kokkos::Atomic>>;
  using RowViewScalarType = typename AtomicRowIdViewType::value_type;
  using OrdinalType       = CrsOT;

  using SizeType         = int;  // Must be int for HashmapAccumulator...
  using KeyType          = typename CrsType::index_type::value_type;
  using ValueType        = typename DataViewType::value_type;
  using ScratchSpaceType = typename CrsET::scratch_memory_space;

  // Unordered set types. KeyViewScratch and SizeViewScratch types are used for
  // the hashmaps too.
  using L0HmapType = KokkosKernels::Experimental::HashmapAccumulator<
      SizeType, KeyType, ValueType,
      KokkosKernels::Experimental::HashOpType::bitwiseAnd>;
  using L0HmapIdxType = uint32_t;
  using L1HmapType    = KokkosKernels::Experimental::HashmapAccumulator<
      SizeType, KeyType, L0HmapIdxType,
      KokkosKernels::Experimental::HashOpType::bitwiseAnd>;
  using KeyViewScratch =
      Kokkos::View<KeyType *, Kokkos::LayoutRight, ScratchSpaceType>;
  using SizeViewScratch =
      Kokkos::View<SizeType *, Kokkos::LayoutRight, ScratchSpaceType>;
  using UsedSizePtrType = volatile SizeType *;
  using UsedSizePtrView =
      Kokkos::View<volatile UsedSizePtrType **, Kokkos::LayoutRight, CrsET>;
  using SizeTypeView = Kokkos::View<SizeType *, Kokkos::LayoutRight, CrsET>;

  using L0HmapView = Kokkos::View<L0HmapType **, Kokkos::LayoutRight, CrsET>;

  // Hashmap types.
  using L0HmapIdxViewScratch =
      Kokkos::View<L0HmapIdxType *, Kokkos::LayoutRight, ScratchSpaceType>;
  using ValViewScratch =
      Kokkos::View<ValueType *, Kokkos::LayoutRight, ScratchSpaceType>;

  using GlobalHmapType = KokkosKernels::Experimental::HashmapAccumulator<
      SizeType, KeyType, ValueType,
      KokkosKernels::Experimental::HashOpType::bitwiseAnd>;
  using GlobalHmapView =
      Kokkos::View<GlobalHmapType *, Kokkos::LayoutRight, CrsET>;
  using SizeView = Kokkos::View<SizeType *, Kokkos::LayoutRight, CrsET>;

  OrdinalType __nrows;
  OrdinalType __ncols;
  CrsSzT __nnz;

  AtomicRowIdViewType __crs_row_cnt;
  CrsValsViewType __crs_vals_tight, __crs_vals;
  CrsRowMapViewType __crs_row_map_tight, __crs_row_map;
  CrsColIdViewType __crs_col_ids_tight, __crs_col_ids;

  size_t __n_tuples;
  bool __insert_mode;
  unsigned int __team_size, __suggested_team_size, __n_teams;

 public:
  struct phase1Tags {
    struct s1RowCntDup {};
    struct s2MaxRowCnt {};
    struct s3UniqRows {};
    struct s4MaxUniqueRows {};
    struct s4RowCnt {};
  };

  using s3Policy = Kokkos::TeamPolicy<typename phase1Tags::s3UniqRows, CrsET>;
  using s4Policy = Kokkos::TeamPolicy<typename phase1Tags::s4RowCnt, CrsET>;

  /**
   * @brief A functor used for parsing the coo matrix and determining how to
   * allocate the crs matrix. The output of this class is: __crs_row_cnt,
   * max_row_cnt, pow2_max_row_cnt, max_n_unique_rows, and
   * pow2_max_n_unique_rows.
   */
  class __Phase1Functor {
   private:
    using s3MemberType = typename s3Policy::member_type;
    using s4MemberType = typename s4Policy::member_type;
    unsigned __n;
    AtomicRowIdViewType __crs_row_cnt;
    RowViewType __row;
    ColViewType __col;
    DataViewType __data;

   public:
    using RowViewScalarTypeView = Kokkos::View<RowViewScalarType *, CrsET>;
    unsigned teams_work, last_teams_work;
    RowViewScalarType max_row_cnt;
    RowViewScalarType pow2_max_row_cnt;
    RowViewScalarTypeView n_unique_rows_per_team;
    L0HmapView usets;
    UsedSizePtrView uset_used_sizes;
    RowViewScalarType max_n_unique_rows;
    RowViewScalarType pow2_max_n_unique_rows;

    __Phase1Functor(RowViewType row, ColViewType col, DataViewType data,
                    AtomicRowIdViewType crs_row_cnt)
        : __crs_row_cnt(crs_row_cnt), __row(row), __col(col), __data(data) {
      __n               = data.extent(0);
      max_row_cnt       = 0;
      max_n_unique_rows = 0;
    };

    KOKKOS_INLINE_FUNCTION
    void operator()(const typename phase1Tags::s1RowCntDup &,
                    const int &thread_id) const {
      auto i = __row(thread_id);
      auto j = __col(thread_id);

      if (i >= 0 && j >= 0) __crs_row_cnt(i)++;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const typename phase1Tags::s2MaxRowCnt,
                    const unsigned long &thread_id,
                    RowViewScalarType &value) const {
      if (__crs_row_cnt(thread_id) > value) value = __crs_row_cnt(thread_id);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const typename phase1Tags::s3UniqRows &,
                    const s3MemberType &member) const {
      unsigned n = member.league_rank() == member.league_size() - 1
                       ? last_teams_work
                       : teams_work;
      unsigned start_n = teams_work * member.league_rank();
      unsigned stop_n  = start_n + n;

      KeyViewScratch keys(member.team_scratch(0), max_row_cnt);
      SizeViewScratch hash_ll(member.team_scratch(0),
                              pow2_max_row_cnt + max_row_cnt + 1);
      volatile SizeType *used_size = hash_ll.data();
      auto *hash_begins            = hash_ll.data() + 1;
      auto *hash_nexts             = hash_begins + pow2_max_row_cnt;
      unsigned unique_row_cnt      = 0;  // thread-local row count
      n_unique_rows_per_team[member.league_rank()] = unique_row_cnt;

      // Initialize hash_begins to -1
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, 0, pow2_max_row_cnt),
                           [&](const int &tid) { hash_begins[tid] = -1; });
      *used_size = 0;
      // Wait for each team's hashmap to be initialized.
      member.team_barrier();

      L0HmapType team_uset(max_row_cnt, pow2_max_row_cnt - 1, hash_begins,
                           hash_nexts, keys.data(), nullptr);

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, start_n, stop_n),
          [&](const int &tid) {
            KeyType i = __row(tid);
            auto j    = __col(tid);

            if (i >= 0 && j >= 0)
              team_uset.vector_atomic_insert_into_hash_KeyCounter(i, used_size);
          });

      member.team_barrier();

      // All teams write their unique row counts to global memory
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        n_unique_rows_per_team[member.league_rank()] = *used_size;
      });
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const typename phase1Tags::s4MaxUniqueRows,
                    const unsigned long &thread_id,
                    RowViewScalarType &value) const {
      if (n_unique_rows_per_team(thread_id) > value)
        value = n_unique_rows_per_team(thread_id);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const typename phase1Tags::s4RowCnt &,
                    const s4MemberType &member) const {
      // Partitions into coo tuples
      unsigned n = member.league_rank() == member.league_size() - 1
                       ? last_teams_work
                       : teams_work;
      unsigned start_n = teams_work * member.league_rank();
      unsigned stop_n  = start_n + n;

      // Top-level hashmap
      KeyViewScratch hmap_keys(member.team_scratch(0), max_n_unique_rows);
      L0HmapIdxViewScratch hmap_values(member.team_scratch(0),
                                       max_n_unique_rows);
      SizeViewScratch hmap_ll(member.team_scratch(0),
                              pow2_max_n_unique_rows + max_n_unique_rows + 1);
      volatile SizeType *hmap_used_size = hmap_ll.data();
      auto *hmap_begins                 = hmap_ll.data() + 1;
      auto *hmap_nexts =
          hmap_begins +
          pow2_max_n_unique_rows;  // hash_nexts is max_n_unique_rows long

      // Initialize hash_begins to -1
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, 0, pow2_max_n_unique_rows),
          [&](const int &tid) { hmap_begins[tid] = -1; });
      *hmap_used_size = 0;

      // This is a hashmap key'd on row ids. Each value is a unordered set of
      // column ids.
      L1HmapType hmap(max_n_unique_rows, pow2_max_n_unique_rows - 1,
                      hmap_begins, hmap_nexts, hmap_keys.data(),
                      hmap_values.data());

      // Unordered sets which hmap points to
      KeyViewScratch uset_keys(member.team_scratch(0),
                               max_n_unique_rows * max_row_cnt);
      SizeViewScratch uset_ll(
          member.team_scratch(0),
          max_n_unique_rows * (pow2_max_row_cnt + max_row_cnt + 1));

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, 0, max_n_unique_rows),
          [&](const int &i) {
            size_t ll_stride = (pow2_max_row_cnt + max_row_cnt + 1) * i;
            volatile SizeType *used_size = uset_ll.data() + ll_stride;
            auto *uset_begins            = uset_ll.data() + ll_stride + 1;
            auto *uset_nexts =
                uset_begins +
                pow2_max_row_cnt;  // uset_nexts is max_row_cnt long
            auto *keys_ptr = uset_keys.data() + (max_row_cnt * i);

            // Initialize uset_begins to -1
            for (unsigned j = 0; j < pow2_max_row_cnt; j++) uset_begins[j] = -1;
            *used_size = 0;

            // This is an unordered set key'd on col ids. Each value is a
            // unordered set of column ids.
            L0HmapType uset(max_row_cnt, pow2_max_row_cnt - 1, uset_begins,
                            uset_nexts, keys_ptr, nullptr);

            usets(member.league_rank(), i)           = uset;
            uset_used_sizes(member.league_rank(), i) = used_size;
          });

      // Wait for the scratch memory initialization
      member.team_barrier();

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, start_n, stop_n),
          [&](const int &tid) {
            KeyType i = __row(tid);
            auto j    = __col(tid);

            if (i >= 0 && j >= 0) {
              // Possibly insert a new row id, i, if it hasn't already been
              // inserted.
              int uset_idx =
                  hmap.vector_atomic_insert_into_hash_once(i, hmap_used_size);

              if (uset_idx < 0) Kokkos::abort("uset_idx < 0");

              // Get the unordered set mapped to row i
              auto uset = usets(member.league_rank(), uset_idx);
              auto uset_used_size =
                  uset_used_sizes(member.league_rank(), uset_idx);

              // Insert the column id into this unordered set
              uset.vector_atomic_insert_into_hash_KeyCounter(j, uset_used_size);
            }
          });

      member.team_barrier();

      // Add up the "tight" row count
      // Can we use a thread's unique_col_count safely here? We have to
      // increment used_size anyways, so let's use that instead.
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, 0, *hmap_used_size),
                           [&](const int &i) {
                             auto uset_idx  = hmap_values(i);
                             auto col_count = uset_used_sizes(
                                 member.league_rank(), uset_idx);
                             auto row_id = hmap_keys(i);
                             __crs_row_cnt(row_id) += *col_count;
                           });
    }
  };

  struct phase2Tags {
    struct s6GlobalHmapSetup {};
    struct s7CopyCoo {};
  };

  using s7Policy = Kokkos::TeamPolicy<typename phase2Tags::s7CopyCoo, CrsET>;

  class __Phase2Functor {
   private:
    using s7MemberType = typename s7Policy::member_type;

    RowIdViewType __row;
    ColViewType __col;
    DataViewType __data;
    OrdinalType __nrows;
    OrdinalType __ncols;
    SizeType __nnz;

    CrsValsViewType __crs_vals;
    CrsRowMapViewType __crs_row_map;
    CrsColIdViewType __crs_col_ids;

    SizeTypeView __global_hmap_begins;
    SizeTypeView __global_hmap_nexts;
    GlobalHmapView __global_hmap;

   public:
    unsigned teams_work, last_teams_work;
    RowViewScalarType max_row_cnt;
    RowViewScalarType pow2_max_row_cnt;
    RowViewScalarType max_n_unique_rows;
    RowViewScalarType pow2_max_n_unique_rows;
    L0HmapView l0_hmaps;
    UsedSizePtrView l0_hmap_used_sizes;
    SizeTypeView global_hmap_used_sizes;

    __Phase2Functor(RowIdViewType row, ColViewType col, DataViewType data,
                    OrdinalType nrows, OrdinalType ncols, SizeType nnz,
                    CrsValsViewType crs_vals, CrsRowMapViewType crs_row_map,
                    CrsColIdViewType crs_col_ids,
                    RowViewScalarType max_row_cnt_in,
                    RowViewScalarType max_n_unique_rows_in,
                    RowViewScalarType pow2_max_n_unique_rows_in)
        : __row(row),
          __col(col),
          __data(data),
          __nrows(nrows),
          __ncols(ncols),
          __nnz(nnz),
          __crs_vals(crs_vals),
          __crs_row_map(crs_row_map),
          __crs_col_ids(crs_col_ids),
          max_row_cnt(max_row_cnt_in),
          max_n_unique_rows(max_n_unique_rows_in),
          pow2_max_n_unique_rows(pow2_max_n_unique_rows_in) {
      __global_hmap = GlobalHmapView(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "__global_hmap"),
          __nrows);
      global_hmap_used_sizes =
          SizeTypeView("global_hmap_used_sizes", __nrows + 1);
      pow2_max_row_cnt = 2;
      while (pow2_max_row_cnt < max_row_cnt) pow2_max_row_cnt *= 2;

      __global_hmap_begins =
          SizeView(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                      "__global_hmap_begins"),
                   pow2_max_row_cnt * __nrows);
      Kokkos::deep_copy(__global_hmap_begins, -1);
      __global_hmap_nexts =
          SizeView(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                      "__global_hmap_nexts"),
                   max_row_cnt * __nrows);
    };

    KOKKOS_INLINE_FUNCTION
    void operator()(const typename phase2Tags::s6GlobalHmapSetup &,
                    const int &row_idx) const {
      auto row_start = __crs_row_map(row_idx);
      auto row_len   = __crs_row_map(row_idx + 1) - row_start;
      // To use this for each hmap begins and nexts, you would
      // need to store the previous pow2_row_len for offsetting
      // below. That would require coordination across threads.
      // Since we already allocate pow2_max_row_cnt * nrows space
      // and max_row_cnt * nrows for hmap_begins and hmap_nexts,
      // let's just use it.
      // decltype(row_len) pow2_row_len = 2;
      // while (pow2_row_len < row_len) pow2_row_len *= 2;

      auto hmap_begins =
          __global_hmap_begins.data() + pow2_max_row_cnt * row_idx;
      auto hmap_nexts = __global_hmap_nexts.data() + max_row_cnt * row_idx;
      auto keys       = __crs_col_ids.data() + row_start;
      auto values     = __crs_vals.data() + row_start;
      GlobalHmapType hmap(row_len, pow2_max_row_cnt - 1, hmap_begins,
                          hmap_nexts, keys, values);
      __global_hmap(row_idx) = hmap;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const typename phase2Tags::s7CopyCoo &,
                    const s7MemberType &member) const {
      // Partitions into coo tuples
      unsigned n = member.league_rank() == member.league_size() - 1
                       ? last_teams_work
                       : teams_work;
      unsigned start_n = teams_work * member.league_rank();
      unsigned stop_n  = start_n + n;

#if 1
      printf("%d-%d-%d-%d-%d\n", member.league_rank(), member.league_size(), n,
             start_n, stop_n);
#endif

      // Top-level hashmap that point to l0 hashmaps
      KeyViewScratch l1_hmap_keys(member.team_scratch(0), max_n_unique_rows);
      L0HmapIdxViewScratch l1_hmap_values(member.team_scratch(0),
                                          max_n_unique_rows);
      SizeViewScratch l1_hmap_ll(
          member.team_scratch(0),
          pow2_max_n_unique_rows + max_n_unique_rows + 1);
      volatile SizeType *l1_hmap_used_size = l1_hmap_ll.data();
      auto *l1_hmap_begins                 = l1_hmap_ll.data() + 1;
      auto *l1_hmap_nexts =
          l1_hmap_begins +
          pow2_max_n_unique_rows;  // hash_nexts is max_n_unique_rows long

      // Initialize hash_begins to -1
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, 0, pow2_max_n_unique_rows),
          [&](const int &tid) { l1_hmap_begins[tid] = -1; });
      *l1_hmap_used_size = 0;

      // This is a hashmap key'd on row ids. Each value is a unordered set of
      // column ids.
      L1HmapType l1_hmap(max_n_unique_rows, pow2_max_n_unique_rows - 1,
                         l1_hmap_begins, l1_hmap_nexts, l1_hmap_keys.data(),
                         l1_hmap_values.data());

      // Level 0 hashmaps
      KeyViewScratch l0_hmap_keys(member.team_scratch(0),
                                  max_n_unique_rows * max_row_cnt);
      ValViewScratch l0_hmap_values(member.team_scratch(0),
                                    max_n_unique_rows * max_row_cnt);
      SizeViewScratch l0_hmap_ll(
          member.team_scratch(0),
          max_n_unique_rows * (pow2_max_row_cnt + max_row_cnt + 1));

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, 0, max_n_unique_rows),
          [&](const int &i) {
            size_t ll_stride = (pow2_max_row_cnt + max_row_cnt + 1) * i;
            volatile SizeType *used_size = l0_hmap_ll.data() + ll_stride;
            auto *l0_hmap_begins         = l0_hmap_ll.data() + ll_stride + 1;
            auto *l0_hmap_nexts =
                l0_hmap_begins +
                pow2_max_row_cnt;  // l0_hmap_nexts is max_row_cnt long
            auto *keys_ptr   = l0_hmap_keys.data() + (max_row_cnt * i);
            auto *values_ptr = l0_hmap_values.data() + (max_row_cnt * i);

            // Initialize l0_hmap_begins to -1
            for (unsigned j = 0; j < pow2_max_row_cnt; j++)
              l0_hmap_begins[j] = -1;
            *used_size = 0;

            // This is an unordered set key'd on col ids. Each value is a
            // unordered set of column ids.
            L0HmapType l0_hmap(max_row_cnt, pow2_max_row_cnt - 1,
                               l0_hmap_begins, l0_hmap_nexts, keys_ptr,
                               values_ptr);

            l0_hmaps(member.league_rank(), i)           = l0_hmap;
            l0_hmap_used_sizes(member.league_rank(), i) = used_size;
          });

      // Wait for the scratch memory initialization
      member.team_barrier();

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, start_n, stop_n),
          [&](const int &tid) {
            KeyType i = __row(tid);
            auto j    = __col(tid);
            auto v    = __data(tid);

            if (i >= 0 && j >= 0) {
              // Possibly insert a new row id, i, if it hasn't already been
              // inserted.
              int l0_hmap_idx = l1_hmap.vector_atomic_insert_into_hash_once(
                  i, l1_hmap_used_size);

              if (l0_hmap_idx < 0) Kokkos::abort("l0_hmap_idx < 0");

              // Get the unordered set mapped to row i
              auto l0_hmap = l0_hmaps(member.league_rank(), l0_hmap_idx);
              auto l0_hmap_used_size =
                  l0_hmap_used_sizes(member.league_rank(), l0_hmap_idx);

              l0_hmap.vector_atomic_insert_into_hash_once_mergeAtomicAdd(
                  j, v, l0_hmap_used_size);
            }
          });

      member.team_barrier();

      printf("%d - Accumulate into global hashmap...\n", member.league_rank());
      // Accumulate into crs matrix via __global_hmap
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, 0, *l1_hmap_used_size),
          [&](const int &i) {
            auto l0_hmap_idx = l1_hmap_values(i);
            auto col_count =
                *l0_hmap_used_sizes(member.league_rank(), l0_hmap_idx);
            auto uset   = l0_hmaps(member.league_rank(), l0_hmap_idx);
            auto row_id = l1_hmap_keys(i);
            volatile SizeType *used_size =
                &global_hmap_used_sizes.data()[row_id];
            for (int j = 0; j < col_count; j++) {
              auto col_id = uset.keys[j];
              auto val    = uset.values[j];
              __global_hmap(row_id)
                  .vector_atomic_insert_into_hash_once_mergeAtomicAdd(
                      col_id, val, used_size);
            }
          });
      printf("%d - done\n", member.league_rank());
    }
  };

  struct phase3Tags {
    struct s8CopyCrs {};
  };

  using s8Policy = Kokkos::TeamPolicy<typename phase3Tags::s8CopyCrs, CrsET>;

  class __Phase3Functor {
   private:
    using s8MemberType = typename s8Policy::member_type;

    OrdinalType __nrows;
    OrdinalType __ncols;

    CrsValsViewType __crs_vals_tight;
    CrsRowMapViewType __crs_row_map_tight;
    CrsColIdViewType __crs_col_ids_tight;
    SizeType __nnz;
    CrsValsViewType __crs_vals_out;
    CrsRowMapViewType __crs_row_map_out;
    CrsColIdViewType __crs_col_ids_out;

   public:
    __Phase3Functor(OrdinalType nrows, OrdinalType ncols,
                    CrsValsViewType crs_vals_tight,
                    CrsRowMapViewType crs_row_map_tight,
                    CrsColIdViewType crs_col_ids_tight, SizeType nnz,
                    CrsValsViewType crs_vals_out,
                    CrsRowMapViewType crs_row_map_out,
                    CrsColIdViewType crs_col_ids_out)
        : __nrows(nrows),
          __ncols(ncols),
          __crs_vals_tight(crs_vals_tight),
          __crs_row_map_tight(crs_row_map_tight),
          __crs_col_ids_tight(crs_col_ids_tight),
          __nnz(nnz),
          __crs_vals_out(crs_vals_out),
          __crs_row_map_out(crs_row_map_out),
          __crs_col_ids_out(crs_col_ids_out){};

    KOKKOS_INLINE_FUNCTION
    void operator()(const typename phase3Tags::s8CopyCrs &,
                    const s8MemberType &member) const {
      unsigned team_row_out_start = __crs_row_map_out(member.league_rank());
      unsigned team_row_out_stop  = __crs_row_map_out(member.league_rank() + 1);
      unsigned row_len            = team_row_out_stop - team_row_out_start;
      unsigned team_row_tight_start = __crs_row_map_tight(member.league_rank());

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, 0, row_len), [&](const int &tid) {
            __crs_col_ids_out(tid + team_row_out_start) =
                __crs_col_ids_tight(tid + team_row_tight_start);
            __crs_vals_out(tid + team_row_out_start) =
                __crs_vals_tight(tid + team_row_tight_start);
          });
    }
  };

  template <class FunctorType>
  void __runPhase1(FunctorType &functor) {
    {
      Kokkos::parallel_for(
          "Coo2Crs::phase1Tags::s1RowCntDup",
          Kokkos::RangePolicy<typename phase1Tags::s1RowCntDup, CrsET>(
              0, __n_tuples),
          functor);
      CrsET().fence();

      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<CrsET, typename phase1Tags::s2MaxRowCnt>(0,
                                                                       __nrows),
          functor, Kokkos::Max<RowViewScalarType>(functor.max_row_cnt));
      CrsET().fence();

      // Reset to 0 for s4RowCnt
      Kokkos::deep_copy(__crs_row_cnt, 0);

      functor.pow2_max_row_cnt = 2;
      while (functor.pow2_max_row_cnt < functor.max_row_cnt)
        functor.pow2_max_row_cnt *= 2;

      s3Policy s3p;
      __suggested_team_size =
          __team_size == 0
              ? s3p.team_size_recommended(functor, Kokkos::ParallelForTag())
              : __team_size;
      __n_teams               = __n_tuples / __suggested_team_size;
      functor.teams_work      = __n_tuples / __n_teams;
      functor.last_teams_work = __n_tuples - (functor.teams_work * __n_teams);
      functor.last_teams_work = functor.last_teams_work == 0
                                    ? functor.teams_work
                                    : functor.last_teams_work;
      __n_teams += !!(__n_tuples % __suggested_team_size);
      functor.n_unique_rows_per_team =
          typename __Phase1Functor::RowViewScalarTypeView(
              Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                 "n_unique_rows_per_team"),
              __n_teams);

      // clang-format: off
      int s3_shmem_size =
          KeyViewScratch::shmem_size(functor.max_row_cnt) +  // keys
          SizeViewScratch::shmem_size(__n_teams) +           // used_sizes
          SizeViewScratch::shmem_size(
              functor.pow2_max_row_cnt +
              functor.max_row_cnt);  // hash_begins and hash_nexts
                                     // clang-format: on

      s3p = s3Policy(__n_teams, __suggested_team_size);
      s3p.set_scratch_size(0, Kokkos::PerTeam(s3_shmem_size));
      Kokkos::parallel_for("Coo2Crs::phase1Tags::s3UniqRows", s3p, functor);
      CrsET().fence();

      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<CrsET, typename phase1Tags::s4MaxUniqueRows>(
              0, __n_teams),
          functor, Kokkos::Max<RowViewScalarType>(functor.max_n_unique_rows));
      CrsET().fence();

      // Between s3 and s4, the number of teams does not change
      // since the coo partitions are the same. We do need new scratch
      // memory allocations and suggested team sizes though.
      s4Policy s4p;
      __suggested_team_size =
          __team_size == 0
              ? s4p.team_size_recommended(functor, Kokkos::ParallelForTag())
              : __team_size;

      functor.pow2_max_n_unique_rows = 2;
      while (functor.pow2_max_n_unique_rows < functor.max_n_unique_rows)
        functor.pow2_max_n_unique_rows *= 2;

      // clang-format: off
      // Calculate size of each team's outer hashmap to row unordered sets
      int s4_shmem_size =
          KeyViewScratch::shmem_size(functor.max_n_unique_rows) +  // keys
          L0HmapIdxViewScratch::shmem_size(
              functor.max_n_unique_rows) +          // values
          SizeViewScratch::shmem_size(__n_teams) +  // used_sizes
          SizeViewScratch::shmem_size(
              functor.pow2_max_n_unique_rows +
              functor.max_n_unique_rows);  // hash_begins and hash_nexts
      // Calculate size of each team's unordered sets for counting unique
      // columns
      int s4_shmem_per_row =
          KeyViewScratch::shmem_size(functor.max_row_cnt) +  // keys
          SizeViewScratch::shmem_size(__n_teams) +           // used_sizes
          SizeViewScratch::shmem_size(
              functor.pow2_max_row_cnt +
              functor.max_row_cnt);  // hash_begins and hash_nexts
      // Each team has up to max_n_unique_rows with up to max_row_cnt columns
      s4_shmem_size += s4_shmem_per_row * functor.max_n_unique_rows;
      // clang-format: on

      functor.n_unique_rows_per_team =
          typename __Phase1Functor::RowViewScalarTypeView(
              Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                 "n_unique_rows_per_team"),
              __n_teams);

      functor.usets =
          L0HmapView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "usets"),
                     __n_teams, functor.max_n_unique_rows);
      functor.uset_used_sizes = UsedSizePtrView(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "uset_used_sizes"),
          __n_teams, functor.max_n_unique_rows);

      s4p = s4Policy(__n_teams, __suggested_team_size);
      s4p.set_scratch_size(0, Kokkos::PerTeam(s4_shmem_size));
      Kokkos::parallel_for("Coo2Crs::phase1Tags::s4RowCnt", s4p, functor);
      CrsET().fence();

      functor.max_row_cnt = 0;
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<CrsET, typename phase1Tags::s2MaxRowCnt>(0,
                                                                       __nrows),
          functor, Kokkos::Max<RowViewScalarType>(functor.max_row_cnt));
      CrsET().fence();
    }
    return;
  }

  template <class FunctorType>
  void __runPhase2(FunctorType &functor) {
    // Partition __crs_vals and __crs_col_ids into hashmaps inside
    // __global_hmap[__nrows]
    Kokkos::parallel_for(
        "Coo2Crs::phase2Tags::s6GlobalHmapSetup",
        Kokkos::RangePolicy<typename phase2Tags::s6GlobalHmapSetup, CrsET>(
            0, __nrows),
        functor);
    CrsET().fence();

    s7Policy s7p;
    __suggested_team_size =
        __team_size == 0
            ? s7p.team_size_recommended(functor, Kokkos::ParallelForTag())
            : __team_size;
    __n_teams               = __n_tuples / __suggested_team_size;
    functor.teams_work      = __n_tuples / __n_teams;
    functor.last_teams_work = __n_tuples - (functor.teams_work * __n_teams);
    functor.last_teams_work = functor.last_teams_work == 0
                                  ? functor.teams_work
                                  : functor.last_teams_work;
    __n_teams += !!(__n_tuples % __suggested_team_size);

    functor.l0_hmaps =
        L0HmapView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "usets"),
                   __n_teams, functor.max_n_unique_rows);
    functor.l0_hmap_used_sizes = UsedSizePtrView(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "uset_used_sizes"),
        __n_teams, functor.max_n_unique_rows);

    // clang-format: off
    // Calculate size of each team's outer hashmap to row unordered sets
    int s7_shmem_size =
        KeyViewScratch::shmem_size(functor.max_n_unique_rows) +        // keys
        L0HmapIdxViewScratch::shmem_size(functor.max_n_unique_rows) +  // values
        SizeViewScratch::shmem_size(__n_teams) +  // used_sizes
        SizeViewScratch::shmem_size(
            functor.pow2_max_n_unique_rows +
            functor.max_n_unique_rows);  // hash_begins and hash_nexts
    // Calculate size of each team's unordered sets
    int s7_shmem_per_row =
        KeyViewScratch::shmem_size(functor.max_row_cnt) +  // keys
        SizeViewScratch::shmem_size(__n_teams) +           // used_sizes
        ValViewScratch::shmem_size(functor.max_row_cnt) +  // values
        SizeViewScratch::shmem_size(
            functor.pow2_max_row_cnt +
            functor.max_row_cnt);  // hash_begins and hash_nexts
    // Each team has up to max_n_unique_rows with up to max_row_cnt columns
    s7_shmem_size += s7_shmem_per_row * functor.max_n_unique_rows;
    // clang-format: on

    printf("__n_teams: %d, __suggested_team_size: %d\n", __n_teams,
           __suggested_team_size);
    printf("functors.teams_work: %d, functor.last_teams_work: %d\n",
           functor.teams_work, functor.last_teams_work);
    printf(
        "functor.max_row_cnt: %lld, functor.max_n_unique_rows: %lld, "
        "functor.pow2_max_row_cnt: %lld, functor.pow2_max_n_unique_rows: "
        "%lld\n",
        functor.max_row_cnt, functor.max_n_unique_rows,
        functor.pow2_max_row_cnt, functor.pow2_max_n_unique_rows);
    s7p = s7Policy(__n_teams, __suggested_team_size);
    s7p.set_scratch_size(0, Kokkos::PerTeam(s7_shmem_size));
    Kokkos::parallel_for("Coo2Crs::Phase2Tags::s7CopyCoo", s7p, functor);
    CrsET().fence();
    return;
  }

  template <class FunctorType>
  void __runPhase3(FunctorType &functor) {
    s8Policy s8p;

    __suggested_team_size =
        __team_size == 0
            ? s8p.team_size_recommended(functor, Kokkos::ParallelForTag())
            : __team_size;

    s8p = s8Policy(__nrows, __suggested_team_size);
    Kokkos::parallel_for("Coo2Crs::Phase3Tags::s8CopyCrs", s8p, functor);
    CrsET().fence();
  }

 public:
  Coo2Crs(DimType m, DimType n, RowViewType row, ColViewType col,
          DataViewType data, unsigned team_size, bool insert_mode) {
    __insert_mode = insert_mode;
    __n_tuples    = data.extent(0);
    __nrows       = m;
    __ncols       = n;
    __team_size   = team_size;

    // Scalar results passed to phase2.
    RowViewScalarType phase1_max_row_cnt, phase1_max_n_unique_rows,
        phase1_pow2_max_n_unique_rows;

    // Get an estimate of the number of columns per row.
    {
      __crs_row_cnt = AtomicRowIdViewType("__crs_row_cnt", m + 1);
      __Phase1Functor phase1Functor(row, col, data, __crs_row_cnt);
      __runPhase1(phase1Functor);
      phase1_max_row_cnt            = phase1Functor.max_row_cnt;
      phase1_max_n_unique_rows      = phase1Functor.max_n_unique_rows;
      phase1_pow2_max_n_unique_rows = phase1Functor.pow2_max_n_unique_rows;
    }

    // Allocate and compute tight crs.
    {
      namespace KE = Kokkos::Experimental;
      CrsET crsET;

      __crs_row_map_tight =
          CrsRowMapViewType(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                               "__crs_row_map_tight"),
                            __nrows + 1);
      KE::exclusive_scan(crsET, KE::cbegin(__crs_row_cnt),
                         KE::cend(__crs_row_cnt),
                         KE::begin(__crs_row_map_tight), 0);
      CrsET().fence();

      __nnz = __crs_row_map_tight(__nrows);

      __crs_vals_tight = CrsValsViewType(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "__crs_vals_tight"),
          __nnz);
      __crs_col_ids_tight =
          CrsColIdViewType(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                              "__crs_col_ids_tight"),
                           __nnz);

      __Phase2Functor phase2Functor(
          row, col, data, __nrows, __ncols, __nnz, __crs_vals_tight,
          __crs_row_map_tight, __crs_col_ids_tight, phase1_max_row_cnt,
          phase1_max_n_unique_rows, phase1_pow2_max_n_unique_rows);

      __runPhase2(phase2Functor);

      // Setup exact row map
      __crs_row_map = CrsRowMapViewType(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "__crs_row_map"),
          __nrows + 1);

      for (int i = 0; i < __nrows + 1; i++)
        printf("global_hmap_used_size(%d) = %d\n", i,
               phase2Functor.global_hmap_used_sizes(i));

      // Find populate the exact row map and nnz
      KE::exclusive_scan(crsET,
                         KE::cbegin(phase2Functor.global_hmap_used_sizes),
                         KE::cend(phase2Functor.global_hmap_used_sizes),
                         KE::begin(__crs_row_map), 0);
      CrsET().fence();
    }

    // Allocate and populate exact crs
    {
      __nnz = __crs_row_map(__nrows);

      __crs_vals = CrsValsViewType(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "__crs_vals"), __nnz);
      __crs_col_ids = CrsColIdViewType(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "__crs_col_ids"),
          __nnz);

      __Phase3Functor phase3Functor(
          __nrows, __ncols, __crs_vals_tight, __crs_row_map_tight,
          __crs_col_ids_tight, __nnz, __crs_vals, __crs_row_map, __crs_col_ids);

      __runPhase3(phase3Functor);
    }
  }

  CrsType get_crsMat() {
    return CrsType("coo2crs", __nrows, __ncols, __nnz, __crs_vals,
                   __crs_row_map, __crs_col_ids);
  }
};
}  // namespace Impl
///
/// \brief Converts a coo matrix to a CrsMatrix.
/// \tparam DimType the dimension type
/// \tparam RowViewType The row array view type
/// \tparam ColViewType The column array view type
/// \tparam DataViewType The data array view type
/// \param m the number of rows
/// \param n the number of columns
/// \param row the array of row ids
/// \param col the array of col ids
/// \param data the array of data
/// \param team_size the requested team_size. By default, team_size = 0 uses the
/// recommended team size. \param insert_mode whether to insert values. By
/// default, values are added. \return A KokkosSparse::CrsMatrix.
template <class DimType, class RowViewType, class ColViewType,
          class DataViewType>
auto coo2crs(DimType m, DimType n, RowViewType row, ColViewType col,
             DataViewType data, unsigned team_size = 0,
             bool insert_mode = false) {
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  static_assert(Kokkos::is_view<RowViewType>::value,
                "RowViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<ColViewType>::value,
                "CalViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<DataViewType>::value,
                "DataViewType must be a Kokkos::View.");
  static_assert(static_cast<int>(RowViewType::rank) == 1,
                "RowViewType must have rank 1.");
  static_assert(static_cast<int>(ColViewType::rank) == 1,
                "ColViewType must have rank 1.");
  static_assert(static_cast<int>(DataViewType::rank) == 1,
                "DataViewType must have rank 1.");
#endif

  static_assert(std::is_integral<typename RowViewType::value_type>::value,
                "RowViewType::value_type must be an integral.");
  static_assert(std::is_integral<typename ColViewType::value_type>::value,
                "ColViewType::value_type must be an integral.");

  if (insert_mode) Kokkos::abort("insert_mode not supported yet.");

  if (row.extent(0) != col.extent(0) || row.extent(0) != data.extent(0))
    Kokkos::abort("row.extent(0) = col.extent(0) = data.extent(0) required.");

  using Coo2crsType =
      Impl::Coo2Crs<DimType, RowViewType, ColViewType, DataViewType>;
  Coo2crsType Coo2Crs(m, n, row, col, data, team_size, insert_mode);
  return Coo2Crs.get_crsMat();
}

#if 0
/// \brief Inserts new values into the given CrsMatrix.
/// \tparam DimType the dimension type
/// \tparam RowViewType The row array view type
/// \tparam ColViewType The column array view type
/// \tparam DataViewType The data array view type
/// \param m the number of rows
/// \param n the number of columns
/// \param row the array of row ids
/// \param col the array of col ids
/// \param data the array of data
/// \param insert_mode whether to insert values. By default, values are added.
/// \return A KokkosSparse::CrsMatrix.
template <class DimType, class RowViewType, class ColViewType, class DataViewType, class CrsMatrixType>
auto coo2crs(DimType m, DimType n, RowViewType row, ColViewType col, DataViewType data, CrsMatrixType crsMatrix) {
  // TODO: Run phase2 only.
}
#endif
}  // namespace KokkosSparse
#endif  //  _KOKKOSSPARSE_COO2CRS_HPP
