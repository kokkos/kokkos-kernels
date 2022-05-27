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

#ifndef KOKKOSSPARSE_IMPL_SPILUK_NUMERIC_HPP_
#define KOKKOSSPARSE_IMPL_SPILUK_NUMERIC_HPP_

/// \file KokkosSparse_spiluk_numeric_impl.hpp
/// \brief Implementation(s) of the numeric phase of sparse ILU(k).

#include <KokkosKernels_config.h>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_spiluk_handle.hpp>

//#define NUMERIC_OUTPUT_INFO

namespace KokkosSparse {
namespace Impl {
namespace Experimental {

// struct UnsortedTag {};

template <class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType,
          class LevelViewType, class WorkViewType, class nnz_lno_t>
struct ILUKLvlSchedRPNumericFunctor {
  using lno_t    = typename AEntriesType::non_const_value_type;
  using scalar_t = typename AValuesType::non_const_value_type;
  ARowMapType A_row_map;
  AEntriesType A_entries;
  AValuesType A_values;
  LRowMapType L_row_map;
  LEntriesType L_entries;
  LValuesType L_values;
  URowMapType U_row_map;
  UEntriesType U_entries;
  UValuesType U_values;
  LevelViewType level_idx;
  WorkViewType iw;
  nnz_lno_t lev_start;

  ILUKLvlSchedRPNumericFunctor(
      const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
      const AValuesType &A_values_, const LRowMapType &L_row_map_,
      const LEntriesType &L_entries_, LValuesType &L_values_,
      const URowMapType &U_row_map_, const UEntriesType &U_entries_,
      UValuesType &U_values_, const LevelViewType &level_idx_,
      WorkViewType &iw_, const nnz_lno_t &lev_start_)
      : A_row_map(A_row_map_),
        A_entries(A_entries_),
        A_values(A_values_),
        L_row_map(L_row_map_),
        L_entries(L_entries_),
        L_values(L_values_),
        U_row_map(U_row_map_),
        U_entries(U_entries_),
        U_values(U_values_),
        level_idx(level_idx_),
        iw(iw_),
        lev_start(lev_start_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const lno_t i) const {
    auto rowid = level_idx(i);
    auto tid   = i - lev_start;
    auto k1    = L_row_map(rowid);
    auto k2    = L_row_map(rowid + 1);
#ifdef KEEP_DIAG
    for (auto k = k1; k < k2 - 1; ++k) {
#else
    for (auto k = k1; k < k2; ++k) {
#endif
      auto col     = L_entries(k);
      L_values(k)  = 0.0;
      iw(tid, col) = k;
    }
#ifdef KEEP_DIAG
    L_values(k2 - 1) = scalar_t(1.0);
#endif

    k1 = U_row_map(rowid);
    k2 = U_row_map(rowid + 1);
    for (auto k = k1; k < k2; ++k) {
      auto col     = U_entries(k);
      U_values(k)  = 0.0;
      iw(tid, col) = k;
    }

    // Unpack the ith row of A
    k1 = A_row_map(rowid);
    k2 = A_row_map(rowid + 1);
    for (auto k = k1; k < k2; ++k) {
      auto col  = A_entries(k);
      auto ipos = iw(tid, col);
      if (col < rowid)
        L_values(ipos) = A_values(k);
      else
        U_values(ipos) = A_values(k);
    }

    // Eliminate prev rows
    k1 = L_row_map(rowid);
    k2 = L_row_map(rowid + 1);
#ifdef KEEP_DIAG
    for (auto k = k1; k < k2 - 1; ++k) {
#else
    for (auto k = k1; k < k2; ++k) {
#endif
      auto prev_row = L_entries(k);
#ifdef KEEP_DIAG
      auto fact = L_values(k) / U_values(U_row_map(prev_row));
#else
      auto fact = L_values(k) * U_values(U_row_map(prev_row));
#endif
      L_values(k) = fact;
      for (auto kk = U_row_map(prev_row) + 1; kk < U_row_map(prev_row + 1);
           ++kk) {
        auto col  = U_entries(kk);
        auto ipos = iw(tid, col);
        if (ipos == -1) continue;
        auto lxu = -U_values(kk) * fact;
        if (col < rowid)
          L_values(ipos) += lxu;
        else
          U_values(ipos) += lxu;
      }  // end for kk
    }    // end for k

#ifdef KEEP_DIAG
    if (U_values(iw(tid, rowid)) == 0.0) {
      U_values(iw(tid, rowid)) = 1e6;
    }
#else
    if (U_values(iw(tid, rowid)) == 0.0) {
      U_values(iw(tid, rowid)) = 1e6;
    } else {
      U_values(iw(tid, rowid)) = 1.0 / U_values(iw(tid, rowid));
    }
#endif

    // Reset
    k1 = L_row_map(rowid);
    k2 = L_row_map(rowid + 1);
#ifdef KEEP_DIAG
    for (auto k = k1; k < k2 - 1; ++k)
#else
    for (auto k = k1; k < k2; ++k)
#endif
      iw(tid, L_entries(k)) = -1;

    k1 = U_row_map(rowid);
    k2 = U_row_map(rowid + 1);
    for (auto k = k1; k < k2; ++k) iw(tid, U_entries(k)) = -1;
  }
};

template <class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType,
          class LevelViewType, class WorkViewType, class nnz_lno_t>
struct ILUKLvlSchedTP1NumericFunctor {
  using execution_space = typename ARowMapType::execution_space;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;
  using member_type     = typename policy_type::member_type;
  using size_type       = typename ARowMapType::non_const_value_type;
  using lno_t           = typename AEntriesType::non_const_value_type;
  using scalar_t        = typename AValuesType::non_const_value_type;

  ARowMapType A_row_map;
  AEntriesType A_entries;
  AValuesType A_values;
  LRowMapType L_row_map;
  LEntriesType L_entries;
  LValuesType L_values;
  URowMapType U_row_map;
  UEntriesType U_entries;
  UValuesType U_values;
  LevelViewType level_idx;
  WorkViewType iw;
  nnz_lno_t lev_start;

  ILUKLvlSchedTP1NumericFunctor(
      const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
      const AValuesType &A_values_, const LRowMapType &L_row_map_,
      const LEntriesType &L_entries_, LValuesType &L_values_,
      const URowMapType &U_row_map_, const UEntriesType &U_entries_,
      UValuesType &U_values_, const LevelViewType &level_idx_,
      WorkViewType &iw_, const nnz_lno_t &lev_start_)
      : A_row_map(A_row_map_),
        A_entries(A_entries_),
        A_values(A_values_),
        L_row_map(L_row_map_),
        L_entries(L_entries_),
        L_values(L_values_),
        U_row_map(U_row_map_),
        U_entries(U_entries_),
        U_values(U_values_),
        level_idx(level_idx_),
        iw(iw_),
        lev_start(lev_start_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type &team) const {
    auto my_league = team.league_rank();  // map to rowid
    auto rowid     = level_idx(my_league + lev_start);
    //auto my_team   = team.team_rank();

    auto k1 = L_row_map(rowid);
    auto k2 = L_row_map(rowid + 1);
#ifdef KEEP_DIAG
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2 - 1),
                         [&](const nnz_lno_t k) {
                           nnz_lno_t col      = static_cast<nnz_lno_t>(L_entries(k));
                           L_values(k)        = 0.0;
                           //if (iw(my_league, col) != -1) printf("L initialize k %d, col %d\n", k, col);
                           iw(my_league, col) = k;
                         });
#else
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const nnz_lno_t k) {
                           nnz_lno_t col      = static_cast<nnz_lno_t>(L_entries(k));
                           L_values(k)        = 0.0;
                           iw(my_league, col) = k;
                         });
#endif

#ifdef KEEP_DIAG
    //if (my_team == 0) L_values(k2 - 1) = scalar_t(1.0);
    Kokkos::single(Kokkos::PerTeam(team),
                   [&]() { L_values(k2 - 1) = scalar_t(1.0); });
#endif

    team.team_barrier();

    k1 = U_row_map(rowid);
    k2 = U_row_map(rowid + 1);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const nnz_lno_t k) {
                           nnz_lno_t col      = static_cast<nnz_lno_t>(U_entries(k));
                           U_values(k)        = 0.0;
                           //if (iw(my_league, col) != -1) printf("U initialize k %d, col %d\n", k, col);
                           iw(my_league, col) = k;
                         });

    team.team_barrier();

    // Unpack the ith row of A
    k1 = A_row_map(rowid);
    k2 = A_row_map(rowid + 1);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const nnz_lno_t k) {
                           nnz_lno_t col  = static_cast<nnz_lno_t>(A_entries(k));
                           nnz_lno_t ipos = iw(my_league, col);
                           //if (ipos == -1) printf("A populate k %d, col %d\n", k, col);
                           if (col < rowid)
                             L_values(ipos) = A_values(k);
                           else
                             U_values(ipos) = A_values(k);
                         });

    team.team_barrier();

    // Eliminate prev rows
    k1 = L_row_map(rowid);
    k2 = L_row_map(rowid + 1);
#ifdef KEEP_DIAG
    for (auto k = k1; k < k2 - 1; ++k) {
#else
    for (auto k = k1; k < k2; ++k) {
#endif
      auto prev_row = L_entries(k);
#ifdef KEEP_DIAG
      auto fact = L_values(k) / U_values(U_row_map(prev_row));
#else
      auto fact = L_values(k) * U_values(U_row_map(prev_row));
#endif
      //if (my_team == 0) L_values(k) = fact;
      Kokkos::single(Kokkos::PerTeam(team), [&]() { L_values(k) = fact; });

      team.team_barrier();

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, U_row_map(prev_row) + 1,
                                  U_row_map(prev_row + 1)),
          [&](const size_type kk) {
            nnz_lno_t col  = static_cast<nnz_lno_t>(U_entries(kk));
            nnz_lno_t ipos = iw(my_league, col);
            if (ipos != -1) {
              auto lxu = -U_values(kk) * fact;
              if (col < rowid)
                Kokkos::atomic_add(&L_values(ipos), lxu);
              else
                Kokkos::atomic_add(&U_values(ipos), lxu);
            }
          });  // end for kk

      //Kokkos::single(Kokkos::PerTeam(team), [&]() { 
      //  for (size_type kk = U_row_map(prev_row) + 1; kk < U_row_map(prev_row + 1); kk++) {
      //      nnz_lno_t col  = static_cast<nnz_lno_t>(U_entries(kk));
      //      nnz_lno_t ipos = iw(my_league, col);
      //      if (ipos != -1) {
      //        auto lxu = -U_values(kk) * fact;
      //        if (col < rowid)
      //          Kokkos::atomic_add(&L_values(ipos), lxu);
      //        else
      //          Kokkos::atomic_add(&U_values(ipos), lxu);
      //      }
      //  }  // end for kk
      //});

      team.team_barrier();
    }  // end for k

    //if (my_team == 0) {
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      nnz_lno_t ipos = iw(my_league, rowid);
#ifdef KEEP_DIAG
      if (U_values(ipos) == 0.0) {
        U_values(ipos) = 1e6;
      }
#else
      if (U_values(ipos) == 0.0) {
        U_values(ipos) = 1e6;
      } else {
        U_values(ipos) = 1.0 / U_values(ipos);
      }
#endif
    });
    //}

    team.team_barrier();

    // Reset
    k1 = L_row_map(rowid);
    k2 = L_row_map(rowid + 1);
#ifdef KEEP_DIAG
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, k1, k2 - 1),
        [&](const nnz_lno_t k) {
        nnz_lno_t col  = static_cast<nnz_lno_t>(L_entries(k));
        iw(my_league, col) = -1;
    });
#else
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, k1, k2),
        [&](const nnz_lno_t k) {
        nnz_lno_t col  = static_cast<nnz_lno_t>(L_entries(k));
        iw(my_league, col) = -1;
    });
#endif

    k1 = U_row_map(rowid);
    k2 = U_row_map(rowid + 1);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, k1, k2),
        [&](const nnz_lno_t k) {
        nnz_lno_t col  = static_cast<nnz_lno_t>(U_entries(k));
        iw(my_league, col) = -1;
    });
  }
};

template <class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType,
          class LevelViewType, class nnz_lno_t>
struct ILUKLvlSchedTP1HashMapNumericFunctor {
  using execution_space = typename ARowMapType::execution_space;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;
  using member_type     = typename policy_type::member_type;
  using size_type       = typename ARowMapType::non_const_value_type;
  using scalar_t        = typename AValuesType::non_const_value_type;
  using hashmap_type    = KokkosKernels::Experimental::HashmapAccumulator<
      nnz_lno_t, nnz_lno_t, nnz_lno_t,
      KokkosKernels::Experimental::HashOpType::bitwiseAnd>;

  ARowMapType A_row_map;
  AEntriesType A_entries;
  AValuesType A_values;
  LRowMapType L_row_map;
  LEntriesType L_entries;
  LValuesType L_values;
  URowMapType U_row_map;
  UEntriesType U_entries;
  UValuesType U_values;
  LevelViewType level_idx;
  nnz_lno_t lev_start;
  nnz_lno_t shmem_hash_size;
  nnz_lno_t shmem_key_size;
  nnz_lno_t shared_memory_hash_func;
  nnz_lno_t shmem_size;

  ILUKLvlSchedTP1HashMapNumericFunctor(
      const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
      const AValuesType &A_values_, const LRowMapType &L_row_map_,
      const LEntriesType &L_entries_, LValuesType &L_values_,
      const URowMapType &U_row_map_, const UEntriesType &U_entries_,
      UValuesType &U_values_, const LevelViewType &level_idx_,
      const nnz_lno_t &lev_start_, const nnz_lno_t &shmem_hash_size_,
      const nnz_lno_t &shmem_key_size_,
      const nnz_lno_t &shared_memory_hash_func_, const nnz_lno_t &shmem_size_)
      : A_row_map(A_row_map_),
        A_entries(A_entries_),
        A_values(A_values_),
        L_row_map(L_row_map_),
        L_entries(L_entries_),
        L_values(L_values_),
        U_row_map(U_row_map_),
        U_entries(U_entries_),
        U_values(U_values_),
        level_idx(level_idx_),
        lev_start(lev_start_),
        shmem_hash_size(shmem_hash_size_),
        shmem_key_size(shmem_key_size_),
        shared_memory_hash_func(shared_memory_hash_func_),
        shmem_size(shmem_size_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type &team) const {
    auto my_league = team.league_rank();                // teamid
    auto rowid     = level_idx(my_league + lev_start);  // teamid-->rowid
    // auto my_team   = team.team_rank();

    // START shared hash map initialization
    char *all_shared_memory = (char *)(team.team_shmem().get_shmem(shmem_size));

    // Threads in a team share 4 arrays: begin, next, keys, values
    // used_hash_sizes hold the size of 1st and 2nd level hashes (not using 2nd
    // level hash right now)
    volatile nnz_lno_t *used_hash_sizes =
        (volatile nnz_lno_t *)(all_shared_memory);
    all_shared_memory += sizeof(nnz_lno_t) * 2;

    // points to begin array
    nnz_lno_t *begins = (nnz_lno_t *)(all_shared_memory);
    all_shared_memory += sizeof(nnz_lno_t) * shmem_hash_size;

    // points to the next elements
    nnz_lno_t *nexts = (nnz_lno_t *)(all_shared_memory);
    all_shared_memory += sizeof(nnz_lno_t) * shmem_key_size;

    // holds the keys and vals
    nnz_lno_t *keys = (nnz_lno_t *)(all_shared_memory);
    all_shared_memory += sizeof(nnz_lno_t) * shmem_key_size;
    nnz_lno_t *vals = (nnz_lno_t *)(all_shared_memory);

    hashmap_type hm(shmem_key_size, shared_memory_hash_func, begins, nexts,
                    keys, vals);

    // initialize begins
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, shmem_hash_size),
                         [&](int i) { begins[i] = -1; });

    // initialize hash usage sizes
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      used_hash_sizes[0] = 0;
      used_hash_sizes[1] = 0;
    });

    team.team_barrier();
    // Shared hash map initialization DONE

    auto k1 = L_row_map(rowid);
    auto k2 = L_row_map(rowid + 1);
#ifdef KEEP_DIAG
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, k1, k2 - 1), [&](const nnz_lno_t k) {
          nnz_lno_t col     = static_cast<nnz_lno_t>(L_entries(k));
          L_values(k)       = 0.0;
          int num_unsuccess = hm.vector_atomic_insert_into_hash_mergeOr(
              col, k, used_hash_sizes);
        });
#else
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, k1, k2), [&](const nnz_lno_t k) {
          nnz_lno_t col     = static_cast<nnz_lno_t>(L_entries(k));
          L_values(k)       = 0.0;
          int num_unsuccess = hm.vector_atomic_insert_into_hash_mergeOr(
              col, k, used_hash_sizes);
        });
#endif

#ifdef KEEP_DIAG
    // if ( my_team == 0 ) L_values(k2-1) = scalar_t(1.0);
    Kokkos::single(Kokkos::PerTeam(team),
                   [&]() { L_values(k2 - 1) = scalar_t(1.0); });
#endif

    team.team_barrier();

    k1 = U_row_map(rowid);
    k2 = U_row_map(rowid + 1);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, k1, k2), [&](const nnz_lno_t k) {
          nnz_lno_t col     = static_cast<nnz_lno_t>(U_entries(k));
          U_values(k)       = 0.0;
          int num_unsuccess = hm.vector_atomic_insert_into_hash_mergeOr(
              col, k, used_hash_sizes);
        });

    // Kokkos::single(Kokkos::PerTeam(team),[&] () {
    //  if (temp_nnz_cnt > shmem_key_size)
    //    printf("VINHVINH teamid %d, rowid %d (at level %d), temp_nnz_cnt %d,
    //    shmem_key_size %d\n", my_league, rowid, lvl+1, temp_nnz_cnt,
    //    shmem_key_size);
    //});

    team.team_barrier();

    // Unpack the ith row of A
    k1 = A_row_map(rowid);
    k2 = A_row_map(rowid + 1);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const nnz_lno_t k) {
                           nnz_lno_t col = static_cast<nnz_lno_t>(A_entries(k));
                           nnz_lno_t hashmap_idx = hm.find(col);
                           if (hashmap_idx != -1) {
                             nnz_lno_t ipos = hm.values[hashmap_idx];
                             if (col < rowid)
                               L_values(ipos) = A_values(k);
                             else
                               U_values(ipos) = A_values(k);
                           }
                         });

    team.team_barrier();

    // Eliminate prev rows
    k1 = L_row_map(rowid);
    k2 = L_row_map(rowid + 1);
#ifdef KEEP_DIAG
    for (auto k = k1; k < k2 - 1; ++k)
#else
    for (auto k = k1; k < k2; ++k)
#endif
    {
      auto prev_row = L_entries(k);
#ifdef KEEP_DIAG
      auto fact = L_values(k) / U_values(U_row_map(prev_row));
#else
      auto fact = L_values(k) * U_values(U_row_map(prev_row));
#endif
      // if ( my_team == 0 ) L_values(k) = fact;
      Kokkos::single(Kokkos::PerTeam(team), [&]() { L_values(k) = fact; });

      team.team_barrier();

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, U_row_map(prev_row) + 1,
                                  U_row_map(prev_row + 1)),
          [&](const size_type kk) {
            nnz_lno_t col         = static_cast<nnz_lno_t>(U_entries(kk));
            nnz_lno_t hashmap_idx = hm.find(col);
            if (hashmap_idx != -1) {
              nnz_lno_t ipos = hm.values[hashmap_idx];
              auto lxu       = -U_values(kk) * fact;
              if (col < rowid)
                // L_values(ipos) += lxu;
                Kokkos::atomic_add(&L_values(ipos), lxu);
              else
                // U_values(ipos) += lxu;
                Kokkos::atomic_add(&U_values(ipos), lxu);
            }
          });  // end for kk

      team.team_barrier();
    }  // end for k

    // if ( my_team == 0 ) {
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      nnz_lno_t hashmap_idx = hm.find(rowid);
      if (hashmap_idx != -1) {
        nnz_lno_t ipos = hm.values[hashmap_idx];
#ifdef KEEP_DIAG
        if (U_values(ipos) == 0.0) {
          U_values(ipos) = 1e6;
        }
#else
        if (U_values(ipos) == 0.0) {
          U_values(ipos) = 1e6;
        }
        else {
          U_values(ipos) = 1.0 / U_values(ipos);
        }
#endif
      }
    });
    //}
  }

  // nnz_lno_t team_shmem_size(int /* team_size */) const {
  //  return shmem_size;
  //}
};

template <class IlukHandle, class ARowMapType, class AEntriesType,
          class AValuesType, class LRowMapType, class LEntriesType,
          class LValuesType, class URowMapType, class UEntriesType,
          class UValuesType>
void iluk_numeric(IlukHandle &thandle, const ARowMapType &A_row_map,
                  const AEntriesType &A_entries, const AValuesType &A_values,
                  const LRowMapType &L_row_map, const LEntriesType &L_entries,
                  LValuesType &L_values, const URowMapType &U_row_map,
                  const UEntriesType &U_entries, UValuesType &U_values) {
  using execution_space         = typename IlukHandle::execution_space;
  using memory_space            = typename IlukHandle::memory_space;
  using size_type               = typename IlukHandle::size_type;
  using nnz_lno_t               = typename IlukHandle::nnz_lno_t;
  using HandleDeviceEntriesType = typename IlukHandle::nnz_lno_view_t;
  using WorkViewType =
      Kokkos::View<nnz_lno_t **, Kokkos::Device<execution_space, memory_space>>;
  using LevelHostViewType = Kokkos::View<nnz_lno_t *, Kokkos::HostSpace>;

  size_type nlevels = thandle.get_num_levels();
  size_type nrows   = thandle.get_nrows();

  // Keep these as host View, create device version and copy back to host
  HandleDeviceEntriesType level_ptr     = thandle.get_level_ptr();
  HandleDeviceEntriesType level_idx     = thandle.get_level_idx();
  HandleDeviceEntriesType level_nchunks = thandle.get_level_nchunks();
  HandleDeviceEntriesType level_nrowsperchunk =
      thandle.get_level_nrowsperchunk();

  // Make level_ptr_h a separate allocation, since it will be accessed on host
  // between kernel launches. If a mirror were used and level_ptr is in UVM
  // space, a fence would be required before each access since UVM views can
  // share pages.
  LevelHostViewType level_ptr_h, level_nchunks_h, level_nrowsperchunk_h;
  WorkViewType iw;

  level_ptr_h = LevelHostViewType(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Host level pointers"),
      level_ptr.extent(0));
  Kokkos::deep_copy(level_ptr_h, level_ptr);

  if (thandle.get_algorithm() ==
      KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1HASHMAP) {
    auto level_shmem_hash_size = thandle.get_level_shmem_hash_size();
    auto level_shmem_key_size  = thandle.get_level_shmem_key_size();

    for (size_type lvl = 0; lvl < nlevels; ++lvl) {
      nnz_lno_t lev_start = level_ptr_h(lvl);
      nnz_lno_t lev_end   = level_ptr_h(lvl + 1);

      if ((lev_end - lev_start) != 0) {
        using policy_type = Kokkos::TeamPolicy<execution_space>;

        nnz_lno_t shmem_hash_size =
            static_cast<nnz_lno_t>(level_shmem_hash_size(lvl));
        nnz_lno_t shmem_key_size =
            static_cast<nnz_lno_t>(level_shmem_key_size(lvl));

        nnz_lno_t shared_memory_hash_func =
            shmem_hash_size - 1;  // for AND operation we use -1

        // shmem needs the first 2 entries for sizes
        nnz_lno_t shmem_size =
            (2 + shmem_hash_size + shmem_key_size * 3) * sizeof(nnz_lno_t);

        int team_size = thandle.get_team_size();
        ILUKLvlSchedTP1HashMapNumericFunctor<
            ARowMapType, AEntriesType, AValuesType, LRowMapType, LEntriesType,
            LValuesType, URowMapType, UEntriesType, UValuesType,
            HandleDeviceEntriesType, nnz_lno_t>
            tstf(A_row_map, A_entries, A_values, L_row_map, L_entries, L_values,
                 U_row_map, U_entries, U_values, level_idx, lev_start,
                 shmem_hash_size, shmem_key_size, shared_memory_hash_func,
                 shmem_size);
        if (team_size == -1) {
          policy_type team_policy(lev_end - lev_start, Kokkos::AUTO);
          team_policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));
          Kokkos::parallel_for("parfor_l_team", team_policy, tstf);
        } else {
          policy_type team_policy(lev_end - lev_start, team_size);
          team_policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));
          Kokkos::parallel_for("parfor_l_team", team_policy, tstf);
        }
      }  // end if
    }    // end for lvl
  }      // End SEQLVLSCHD_TP1HASHMAP
  else {
    if (thandle.get_algorithm() ==
        KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1) {
      level_nchunks_h = LevelHostViewType(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "Host level nchunks"),
          level_nchunks.extent(0));
      level_nrowsperchunk_h =
          LevelHostViewType(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                               "Host level nrowsperchunk"),
                            level_nrowsperchunk.extent(0));
      Kokkos::deep_copy(level_nchunks_h, level_nchunks);
      Kokkos::deep_copy(level_nrowsperchunk_h, level_nrowsperchunk);
      iw = WorkViewType(Kokkos::view_alloc(Kokkos::WithoutInitializing, "iw"),
                        thandle.get_level_maxrowsperchunk(), nrows);
      Kokkos::deep_copy(iw, nnz_lno_t(-1));
    } else {
      iw = WorkViewType(Kokkos::view_alloc(Kokkos::WithoutInitializing, "iw"),
                        thandle.get_level_maxrows(), nrows);
      Kokkos::deep_copy(iw, nnz_lno_t(-1));
    }

    // Main loop must be performed sequential. Question: Try out Cuda's graph
    // stuff to reduce kernel launch overhead
    printf("work array iw %d x %d\n",iw.extent(0),iw.extent(1));
    int tmpcnt = 0;
    int tmpnrows = 0;
    for (size_type lvl = 0; lvl < nlevels; ++lvl) {
      nnz_lno_t lev_start = level_ptr_h(lvl);
      nnz_lno_t lev_end   = level_ptr_h(lvl + 1);

      if ((lev_end - lev_start) != 0) {
        if (thandle.get_algorithm() ==
            KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_RP) {
          Kokkos::parallel_for(
              "parfor_fixed_lvl",
              Kokkos::RangePolicy<execution_space>(lev_start, lev_end),
              ILUKLvlSchedRPNumericFunctor<
                  ARowMapType, AEntriesType, AValuesType, LRowMapType,
                  LEntriesType, LValuesType, URowMapType, UEntriesType,
                  UValuesType, HandleDeviceEntriesType, WorkViewType,
                  nnz_lno_t>(A_row_map, A_entries, A_values, L_row_map,
                             L_entries, L_values, U_row_map, U_entries,
                             U_values, level_idx, iw, lev_start));
        } else if (thandle.get_algorithm() ==
                   KokkosSparse::Experimental::SPILUKAlgorithm::
                       SEQLVLSCHD_TP1) {
          using policy_type = Kokkos::TeamPolicy<execution_space>;
          int team_size     = thandle.get_team_size();

          nnz_lno_t lvl_rowid_start = 0;
          nnz_lno_t lvl_nrows_chunk;
          for (int chunkid = 0; chunkid < level_nchunks_h(lvl); chunkid++) {
            if ((lvl_rowid_start + level_nrowsperchunk_h(lvl)) >
                (lev_end - lev_start))
              lvl_nrows_chunk = (lev_end - lev_start) - lvl_rowid_start;
            else
              lvl_nrows_chunk = level_nrowsperchunk_h(lvl);

            ILUKLvlSchedTP1NumericFunctor<
                ARowMapType, AEntriesType, AValuesType, LRowMapType,
                LEntriesType, LValuesType, URowMapType, UEntriesType,
                UValuesType, HandleDeviceEntriesType, WorkViewType, nnz_lno_t>
                tstf(A_row_map, A_entries, A_values, L_row_map, L_entries,
                     L_values, U_row_map, U_entries, U_values, level_idx, iw,
                     lev_start + lvl_rowid_start);

            if (team_size == -1)
              Kokkos::parallel_for("parfor_l_team",
                                   policy_type(lvl_nrows_chunk, Kokkos::AUTO),
                                   tstf);
            else
              Kokkos::parallel_for("parfor_l_team",
                                   policy_type(lvl_nrows_chunk, team_size),
                                   tstf);
            Kokkos::fence();
            lvl_rowid_start += lvl_nrows_chunk;
            tmpcnt++;
            tmpnrows += lvl_nrows_chunk;
          }
        }
      }  // end if
    }    // end for lvl
    printf("Total kernel calls %d, total nrows %d\n",tmpcnt, tmpnrows);
  }

// Output check
#ifdef NUMERIC_OUTPUT_INFO
  std::cout << "  iluk_numeric result: " << std::endl;

  std::cout << "  nnzL: " << thandle.get_nnzL() << std::endl;
  std::cout << "  L_row_map = ";
  for (size_type i = 0; i < nrows + 1; ++i) {
    std::cout << L_row_map(i) << " ";
  }
  std::cout << std::endl;

  std::cout << "  L_entries = ";
  for (size_type i = 0; i < thandle.get_nnzL(); ++i) {
    std::cout << L_entries(i) << " ";
  }
  std::cout << std::endl;

  std::cout << "  L_values = ";
  for (size_type i = 0; i < thandle.get_nnzL(); ++i) {
    std::cout << L_values(i) << " ";
  }
  std::cout << std::endl;

  std::cout << "  nnzU: " << thandle.get_nnzU() << std::endl;
  std::cout << "  U_row_map = ";
  for (size_type i = 0; i < nrows + 1; ++i) {
    std::cout << U_row_map(i) << " ";
  }
  std::cout << std::endl;

  std::cout << "  U_entries = ";
  for (size_type i = 0; i < thandle.get_nnzU(); ++i) {
    std::cout << U_entries(i) << " ";
  }
  std::cout << std::endl;

  std::cout << "  U_values = ";
  for (size_type i = 0; i < thandle.get_nnzU(); ++i) {
    std::cout << U_values(i) << " ";
  }
  std::cout << std::endl;
#endif

}  // end iluk_numeric

}  // namespace Experimental
}  // namespace Impl
}  // namespace KokkosSparse

#endif
