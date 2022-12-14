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
                   typename RowIdViewType::array_layout,
                   typename RowIdViewType::execution_space,
                   Kokkos::MemoryTraits<Kokkos::Atomic>>;
  using ColMapViewType = ColViewType;
  using ValViewType    = DataViewType;
  using OrdinalType    = CrsOT;
  using SizeType       = int;  // Must be int for HashmapAccumulator...

  using KeyType          = int;
  using ValueType        = typename DataViewType::value_type;
  using ScratchSpaceType = typename CrsET::scratch_memory_space;

  // Unordered set types. KeyViewScratch and SizeViewScratch types are used for
  // the hashmaps too.
  using UsetType = KokkosKernels::Experimental::HashmapAccumulator<
      SizeType, KeyType, ValueType,
      KokkosKernels::Experimental::HashOpType::bitwiseAnd>;
  using UsetIdxType = uint32_t;
  using HmapType    = KokkosKernels::Experimental::HashmapAccumulator<
      SizeType, KeyType, UsetIdxType,
      KokkosKernels::Experimental::HashOpType::bitwiseAnd>;
  using KeyViewScratch =
      Kokkos::View<KeyType *, Kokkos::LayoutRight, ScratchSpaceType>;
  using SizeViewScratch =
      Kokkos::View<SizeType *, Kokkos::LayoutRight, ScratchSpaceType>;
  using UsedSizePtrType = volatile SizeType *;
  using UsedSizePtrView =
      Kokkos::View<volatile UsedSizePtrType **, Kokkos::LayoutRight>;
  using UsetView = Kokkos::View<UsetType **, Kokkos::LayoutRight>;

  // Hashmap types.
  using UsetIdxViewScratch =
      Kokkos::View<UsetIdxType *, Kokkos::LayoutRight, ScratchSpaceType>;
  using ValViewScratch =
      Kokkos::View<ValueType *, Kokkos::LayoutRight, ScratchSpaceType>;

  OrdinalType __nrows;
  OrdinalType __ncols;
  SizeType __nnz;
  ValViewType __vals;
  RowIdViewType __row_ids;
  ColMapViewType __col_map;

  AtomicRowIdViewType __crs_row_cnt;
  using RowViewScalarType = typename AtomicRowIdViewType::value_type;

  CrsValsViewType __crs_vals;
  CrsRowMapViewType __crs_row_map;
  CrsRowMapViewType __crs_row_map_scratch;
  CrsColIdViewType __crs_col_ids;

  size_t __n_tuples;
  bool __insert_mode;
  unsigned __team_size;

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

  class __Phase1Functor {
   private:
    using s3MemberType = typename s3Policy::member_type;
    using s4MemberType = typename s4Policy::member_type;
    unsigned __n;

   public:
    AtomicRowIdViewType __crs_row_cnt;
    RowViewType __row;
    ColViewType __col;
    DataViewType __data;
    unsigned teams_work, last_teams_work;
    RowViewScalarType max_row_cnt;
    RowViewScalarType pow2_max_row_cnt;
    Kokkos::View<unsigned *, CrsET> n_unique_rows_per_team;
    UsetView usets;
    UsedSizePtrView uset_used_sizes;
    unsigned max_n_unique_rows;
    unsigned pow2_max_n_unique_rows;

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

      UsetType team_uset(max_row_cnt, pow2_max_row_cnt - 1, hash_begins,
                         hash_nexts, keys.data(), nullptr);

#if 0
      std::cout << "------------------" << std::endl;
      std::cout << "rank: " << member.league_rank() << std::endl;
      std::cout << "league_size: " << member.league_size() << std::endl;
      std::cout << "n: " << n << std::endl;
      std::cout << "start_n: " << start_n << std::endl;
      std::cout << "stop_n: " << stop_n << std::endl;
#endif

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
                    const unsigned long &thread_id, unsigned &value) const {
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
      UsetIdxViewScratch hmap_values(member.team_scratch(0), max_n_unique_rows);
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
      HmapType hmap(max_n_unique_rows, pow2_max_n_unique_rows - 1, hmap_begins,
                    hmap_nexts, hmap_keys.data(), hmap_values.data());

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
            UsetType uset(max_row_cnt, pow2_max_row_cnt - 1, uset_begins,
                          uset_nexts, keys_ptr, nullptr);

            usets(member.league_rank(), i)           = uset;
            uset_used_sizes(member.league_rank(), i) = used_size;
          });

      // Wait for the scratch memory initialization
      member.team_barrier();

#if 0
      std::cout << "------------------" << std::endl;
      std::cout << "rank: " << member.league_rank() << std::endl;
      std::cout << "league_size: " << member.league_size() << std::endl;
      std::cout << "n: " << n << std::endl;
      std::cout << "start_n: " << start_n << std::endl;
      std::cout << "stop_n: " << stop_n << std::endl;
#endif

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
#if 0
          std::cout << "rank: " << member.league_rank() << ", uset_idx: " << uset_idx << ", col_count: " << *col_count << ", row_id: " << row_id << std::endl;
#endif
                             __crs_row_cnt(row_id) += *col_count;
                           });
    }
  };

  struct phase2Tags {
    struct s6RowMap {};
    struct s7Copy {};
  };

  class __Phase2Functor {
   private:
    OrdinalType __nrows;
    OrdinalType __ncols;
    SizeType __nnz;
    ValViewType __vals;
    CrsValsViewType __crs_vals;
    RowIdViewType __row_ids;
    CrsRowMapViewType __crs_row_map;
    CrsRowMapViewType __crs_row_map_scratch;
    ColMapViewType __col_map;
    CrsColIdViewType __crs_col_ids;
    AtomicRowIdViewType __crs_row_cnt;

   public:
    __Phase2Functor(OrdinalType nrows, OrdinalType ncols, SizeType nnz,
                    ValViewType vals, CrsValsViewType crs_vals,
                    RowIdViewType row_ids, CrsRowMapViewType crs_row_map,
                    CrsRowMapViewType crs_row_map_scratch,
                    ColMapViewType col_map, CrsColIdViewType crs_col_ids,
                    AtomicRowIdViewType crs_row_cnt)
        : __nrows(nrows),
          __ncols(ncols),
          __nnz(nnz),
          __vals(vals),
          __crs_vals(crs_vals),
          __row_ids(row_ids),
          __crs_row_map(crs_row_map),
          __crs_row_map_scratch(crs_row_map_scratch),
          __col_map(col_map),
          __crs_col_ids(crs_col_ids),
          __crs_row_cnt(crs_row_cnt){};
  };

 private:
  unsigned int __suggested_team_size, __suggested_vec_size, __n_teams;

  template <class FunctorType>
  void __runPhase1(FunctorType &functor) {
    {
#if 0
      for (size_t i = 0; i < __n_tuples; i++) {
        std::cout << "(" << functor.__row(i) << ", " << functor.__col(i) << ", "
                  << functor.__data(i) << ")" << std::endl;
      }
#endif

      Kokkos::parallel_for(
          "Coo2Crs::phase1Tags::s1RowCntDup",
          Kokkos::RangePolicy<typename phase1Tags::s1RowCntDup, CrsET>(
              0, __n_tuples),
          functor);
      CrsET().fence();
#if 0
      std::cout << "phase1Functor.__crs_row_cnt: " << std::endl;
      for (unsigned i = 0; i < __nrows; i++) {
        std::cout << i << ":" << functor.__crs_row_cnt[i] << std::endl;
      }
#endif

      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<CrsET, typename phase1Tags::s2MaxRowCnt>(0,
                                                                       __nrows),
          functor, Kokkos::Max<RowViewScalarType>(functor.max_row_cnt));
      CrsET().fence();
#if 0
      std::cout << "phase1Functor.max_row_cnt: " << functor.max_row_cnt
                << std::endl;
#endif
      // Reset to 0 for s4RowCnt
      Kokkos::deep_copy(__crs_row_cnt, 0);

      functor.pow2_max_row_cnt = 1;
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
      functor.n_unique_rows_per_team = Kokkos::View<unsigned *, CrsET>(
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

#if 0
      std::cout << "__n_tuples: " << __n_tuples << std::endl;
      std::cout << "__n_teams: " << __n_teams << std::endl;
      std::cout << "__suggested_team_size: " << __suggested_team_size
                << std::endl;
      std::cout << "functor.teams_work: " << functor.teams_work << std::endl;
      std::cout << "functor.last_teams_work: " << functor.last_teams_work
                << std::endl;
      std::cout << "s3_shmem_size: " << s3_shmem_size << std::endl;
#endif

      s3p = s3Policy(__n_teams, __suggested_team_size);
      s3p.set_scratch_size(0, Kokkos::PerTeam(s3_shmem_size));
      Kokkos::parallel_for("Coo2Crs::phase1Tags::s3UniqRows", s3p, functor);
      CrsET().fence();
#if 0
      std::cout << "phase1Functor.n_unique_rows_per_team: " << std::endl;
      for (unsigned i = 0; i < __n_teams; i++) {
        std::cout << i << ":" << functor.n_unique_rows_per_team[i] << std::endl;
      }
#endif

      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<CrsET, typename phase1Tags::s4MaxUniqueRows>(
              0, __n_teams),
          functor, Kokkos::Max<unsigned>(functor.max_n_unique_rows));
      CrsET().fence();
#if 0
      std::cout << "phase1Functor.max_n_unique_rows: "
                << functor.max_n_unique_rows << std::endl;
#endif

      // Between s3 and s4, the number of teams does not change
      // since the coo partitions are the same. We do need new scratch
      // memory allocations and suggested team sizes though.
      s4Policy s4p;
      // TODO: put this in a function
      __suggested_team_size =
          __team_size == 0
              ? s4p.team_size_recommended(functor, Kokkos::ParallelForTag())
              : __team_size;

      functor.pow2_max_n_unique_rows = 1;
      while (functor.pow2_max_n_unique_rows < functor.max_n_unique_rows)
        functor.pow2_max_n_unique_rows *= 2;

      // clang-format: off
      // Calculate size of each team's outer hashmap to row unordered sets
      int s4_shmem_size =
          KeyViewScratch::shmem_size(functor.max_n_unique_rows) +      // keys
          UsetIdxViewScratch::shmem_size(functor.max_n_unique_rows) +  // values
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

      functor.n_unique_rows_per_team = Kokkos::View<unsigned *, CrsET>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "n_unique_rows_per_team"),
          __n_teams);

      functor.usets =
          UsetView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "usets"),
                   __n_teams, functor.max_n_unique_rows);
      functor.uset_used_sizes = UsedSizePtrView(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "uset_used_sizes"),
          __n_teams, functor.max_n_unique_rows);

#if 0
      std::cout << "__n_tuples: " << __n_tuples << std::endl;
      std::cout << "__n_teams: " << __n_teams << std::endl;
      std::cout << "__suggested_team_size: " << __suggested_team_size
                << std::endl;
      std::cout << "functor.teams_work: " << functor.teams_work << std::endl;
      std::cout << "functor.last_teams_work: " << functor.last_teams_work
                << std::endl;
      std::cout << "s4_shmem_size: " << s4_shmem_size << std::endl;
#endif

#if 0
      std::cout << "phase1Functor.__crs_row_cnt: " << std::endl;
      for (unsigned i = 0; i < __nrows; i++) {
        std::cout << i << ":" << functor.__crs_row_cnt[i] << std::endl;
      }
#endif

      s4p = s4Policy(__n_teams, __suggested_team_size);
      s4p.set_scratch_size(0, Kokkos::PerTeam(s4_shmem_size));
      Kokkos::parallel_for("Coo2Crs::phase1Tags::s4RowCnt", s4p, functor);
      CrsET().fence();
#if 0
      std::cout << "phase1Functor.__crs_row_cnt: " << std::endl;
      for (unsigned i = 0; i < __nrows; i++) {
        std::cout << i << ":" << functor.__crs_row_cnt[i] << std::endl;
      }
#endif

      functor.max_row_cnt = 0;
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<CrsET, typename phase1Tags::s2MaxRowCnt>(0,
                                                                       __nrows),
          functor, Kokkos::Max<RowViewScalarType>(functor.max_row_cnt));
      CrsET().fence();

#if 0
      std::cout << "phase1Functor.max_row_cnt: " << functor.max_row_cnt
                << std::endl;
#endif

#if 0
      for (size_t i = 0; i < __n_tuples; i++) {
        std::cout << "(" << functor.__row(i) << ", " << functor.__col(i) << ", "
                  << functor.__data(i) << ")" << std::endl;
      }
#endif
    }
    return;
  }

  template <class FunctorType>
  void __runPhase2(FunctorType &functor) {
    return;
  }

 public:
  Coo2Crs(DimType m, DimType n, RowViewType row, ColViewType col,
          DataViewType data, unsigned team_size, bool insert_mode) {
    __insert_mode = insert_mode;
    __n_tuples    = data.extent(0);
    __nrows       = m;
    __ncols       = n;
    __team_size   = team_size;

    // Get an estimate of the number of columns per row.
    __crs_row_cnt = AtomicRowIdViewType("__crs_row_cnt", m);
    {
      __Phase1Functor phase1Functor(row, col, data, __crs_row_cnt);
      __runPhase1(phase1Functor);
    }

    // Allocate and populate crs.

    /*    __crs_vals = CrsValsViewType(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "__crs_vals"), nnz);
    __crs_row_map = CrsRowMapViewType(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "__crs_row_map"),
        nrows + 1);
    __crs_row_map_scratch =
        CrsRowMapViewType(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             "__crs_row_map_scratch"),
                          nrows + 1);
    __crs_col_ids = CrsColIdViewType(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "__crs_col_ids"), nnz);

        KokkosKernels::Impl::get_suggested_vector_size<int64_t, CrsET>(
        __suggested_vec_size, __nrows, __nnz);
    __suggested_team_size =
        KokkosKernels::Impl::get_suggested_team_size<TeamPolicyType>(
            functor, __suggested_vec_size);

    __Phase2Functor<typename TeamPolicyType::member_type> phase2Functor(__nrows,
    __ncols, __nnz, __vals, __crs_vals, __row_ids, __crs_row_map,
        __crs_row_map_scratch, __col_map, __crs_col_ids, __crs_row_cnt);
    __runPhase2(phase2Functor);
    */
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
