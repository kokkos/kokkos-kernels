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

#ifndef KOKKOSSPARSE_IMPL_SPTRSV_SOLVE_HPP_
#define KOKKOSSPARSE_IMPL_SPTRSV_SOLVE_HPP_

/// \file KokkosSparse_impl_sptrsv.hpp
/// \brief Implementation(s) of sparse triangular solve.

#include <KokkosKernels_config.h>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_sptrsv_handle.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_CrsMatrix.hpp>

#ifdef KOKKOSKERNELS_ENABLE_SUPERNODAL_SPTRSV
// Enable supernodal sptrsv
#include "KokkosBlas3_trsm.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosBatched_Util.hpp"
#include "KokkosBlas2_team_gemv_spec.hpp"
#include "KokkosBatched_Trsm_Team_Impl.hpp"
#endif

//#define SERIAL_FOR_LOOP

#define KOKKOSKERNELS_SPTRSV_TRILVLSCHED

//#define KOKKOSPSTRSV_SOLVE_IMPL_PROFILE 1
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
#include "cuda_profiler_api.h"
#endif

#if defined(KOKKOS_ENABLE_CUDA) && 10000 < CUDA_VERSION && \
    defined(KOKKOSKERNELS_ENABLE_EXP_CUDAGRAPH)
#define KOKKOSKERNELS_SPTRSV_CUDAGRAPHSUPPORT
#endif

namespace KokkosSparse {
namespace Impl {
namespace Experimental {

template <class TriSolveHandle>
struct SptrsvWrap {
  //
  // Useful types
  //
  using execution_space = typename TriSolveHandle::execution_space;
  using memory_space    = typename TriSolveHandle::memory_space;
  using temp_mem_space  = typename TriSolveHandle::HandleTempMemorySpace;
  using lno_t           = typename TriSolveHandle::nnz_lno_t;
  using size_type       = typename TriSolveHandle::size_type;
  using scalar_t        = typename TriSolveHandle::scalar_t;
  using row_map_t       = typename TriSolveHandle::nnz_row_view_t;
  using entries_t       = typename TriSolveHandle::nnz_lno_view_t;
  using values_t        = typename TriSolveHandle::nnz_scalar_view_t;
  using work_view_t =
      Kokkos::View<scalar_t *, Kokkos::Device<execution_space, temp_mem_space>>;
  using work_view_int_t =
      Kokkos::View<int *, Kokkos::Device<execution_space, temp_mem_space>>;
  using karith       = typename Kokkos::ArithTraits<scalar_t>;
  using team_policy  = typename TriSolveHandle::TeamPolicy;
  using member_type  = typename team_policy::member_type;
  using range_policy = typename TriSolveHandle::RangePolicy;
  using range_type   = Kokkos::pair<int, int>;

  // Tag structs
  struct UnsortedTag {};
  struct LargerCutoffTag {};
  struct UnsortedLargerCutoffTag {};

  template <class ViewType>
  static void print_view1d_solve(const ViewType dv, size_t range = 0) {
    auto v = Kokkos::create_mirror_view(dv);
    Kokkos::deep_copy(v, dv);
    std::cout << "Output for view " << v.label() << std::endl;
    range = range == 0 ? dv.extent(0) : range;
    for (size_t i = 0; i < range; ++i) {
      std::cout << "v(" << i << ") = " << v(i) << " , ";
    }
    std::cout << std::endl;
  }

  // Needed for cudagraphs
  struct EmptyFunctor {
    KOKKOS_INLINE_FUNCTION
    void operator()(const int) const {}
  };

  // This functor unifies the lower and upper implementations, the hope is the
  // "is_lowertri" check does not add noticable time on larger problems
  template <class RowMapType, class EntriesType, class ValuesType,
            class LHSType, class RHSType, bool IsLower>
  struct TriLvlSchedTP1SolverFunctor {
    RowMapType row_map;
    EntriesType entries;
    ValuesType values;
    LHSType lhs;
    RHSType rhs;
    entries_t nodes_grouped_by_level;

    long node_count;  // like "block" offset into ngbl, my_league is the "local"
                      // offset

    TriLvlSchedTP1SolverFunctor(const RowMapType &row_map_,
                                const EntriesType &entries_,
                                const ValuesType &values_, LHSType &lhs_,
                                const RHSType &rhs_,
                                const entries_t &nodes_grouped_by_level_,
                                const long &node_count_)
        : row_map(row_map_),
          entries(entries_),
          values(values_),
          lhs(lhs_),
          rhs(rhs_),
          nodes_grouped_by_level(nodes_grouped_by_level_),
          node_count(node_count_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type &team) const {
      auto my_league = team.league_rank();  // map to rowid
      auto rowid     = nodes_grouped_by_level(my_league + node_count);
      auto my_rank   = team.team_rank();

      auto soffset   = row_map(rowid);
      auto eoffset   = row_map(rowid + 1);
      auto rhs_rowid = rhs(rowid);
      scalar_t diff  = scalar_t(0.0);

      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, soffset, eoffset),
          [&](const long ptr, scalar_t &tdiff) {
            auto colid = entries(ptr);
            auto val   = values(ptr);
            if (colid != rowid) {
              tdiff = tdiff - val * lhs(colid);
            }
          },
          diff);

      team.team_barrier();

      // At end, finalize rowid == colid
      // only one thread should do this; can also use Kokkos::single
      if (my_rank == 0) {
        // ASSUMPTION: sorted diagonal value located at eoffset - 1
        lhs(rowid) = IsLower ? (rhs_rowid + diff) / values(eoffset - 1)
                             : (rhs_rowid + diff) / values(soffset);
      }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const UnsortedTag &, const member_type &team) const {
      auto my_league = team.league_rank();  // map to rowid
      auto rowid     = nodes_grouped_by_level(my_league + node_count);
      auto my_rank   = team.team_rank();

      auto soffset   = row_map(rowid);
      auto eoffset   = row_map(rowid + 1);
      auto rhs_rowid = rhs(rowid);
      scalar_t diff  = scalar_t(0.0);

      auto diag = -1;

      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, soffset, eoffset),
          [&](const long ptr, scalar_t &tdiff) {
            auto colid = entries(ptr);
            auto val   = values(ptr);
            if (colid != rowid) {
              tdiff = tdiff - val * lhs(colid);
            } else {
              diag = ptr;
            }
          },
          diff);
      team.team_barrier();

      // At end, finalize rowid == colid
      // only one thread should do this; can also use Kokkos::single
      if (my_rank == 0) {
        lhs(rowid) = (rhs_rowid + diff) / values(diag);
      }
    }
  };

  template <class RowMapType, class EntriesType, class ValuesType,
            class LHSType, class RHSType>
  struct TriLvlSchedTP1SolverFunctorDiagValues {
    RowMapType row_map;
    EntriesType entries;
    ValuesType values;
    LHSType lhs;
    RHSType rhs;
    entries_t nodes_grouped_by_level;
    ValuesType diagonal_values;  // inserted according to rowid

    const bool is_lowertri;

    long node_count;  // like "block" offset into ngbl, my_league is the "local"
                      // offset
    long dense_nrows;

    TriLvlSchedTP1SolverFunctorDiagValues(
        const RowMapType &row_map_, const EntriesType &entries_,
        const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_,
        const entries_t &nodes_grouped_by_level_,
        const ValuesType &diagonal_values_, const bool is_lowertri_,
        long node_count_, long dense_nrows_ = 0)
        : row_map(row_map_),
          entries(entries_),
          values(values_),
          lhs(lhs_),
          rhs(rhs_),
          nodes_grouped_by_level(nodes_grouped_by_level_),
          diagonal_values(diagonal_values_),
          is_lowertri(is_lowertri_),
          node_count(node_count_),
          dense_nrows(dense_nrows_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type &team) const {
      auto my_league = team.league_rank();  // map to rowid
      auto rowid     = nodes_grouped_by_level(my_league + node_count);
      auto my_rank   = team.team_rank();

      auto soffset   = row_map(rowid);
      auto eoffset   = row_map(rowid + 1);
      auto rhs_rowid = rhs(rowid);
      scalar_t diff  = scalar_t(0.0);

      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, soffset, eoffset),
          [&](const long ptr, scalar_t &tdiff) {
            auto colid = entries(ptr);
            auto val   = values(ptr);
            if (colid != rowid) {
              tdiff = tdiff - val * lhs(colid);
            }
          },
          diff);

      team.team_barrier();

      // At end, finalize rowid == colid
      // only one thread should do this; can also use Kokkos::single
      if (my_rank == 0) {
        // lhs(rowid) = is_lowertri ? (rhs_rowid+diff)/values(eoffset-1) :
        // (rhs_rowid+diff)/values(soffset);
        lhs(rowid) = (rhs_rowid + diff) / diagonal_values(rowid);
      }
    }
  };

  template <class RowMapType, class EntriesType, class ValuesType,
            class LHSType, class RHSType, bool IsLower>
  struct TriLvlSchedTP2SolverFunctor {
    RowMapType row_map;
    EntriesType entries;
    ValuesType values;
    LHSType lhs;
    RHSType rhs;
    entries_t nodes_grouped_by_level;

    long node_count;  // like "block" offset into ngbl, my_league is the "local"
                      // offset
    long node_groups;
    long dense_nrows;

    TriLvlSchedTP2SolverFunctor(const RowMapType &row_map_,
                                const EntriesType &entries_,
                                const ValuesType &values_, LHSType &lhs_,
                                const RHSType &rhs_,
                                const entries_t &nodes_grouped_by_level_,
                                long node_count_,
                                long node_groups_ = 0, long dense_nrows_ = 0)
        : row_map(row_map_),
          entries(entries_),
          values(values_),
          lhs(lhs_),
          rhs(rhs_),
          nodes_grouped_by_level(nodes_grouped_by_level_),
          node_count(node_count_),
          node_groups(node_groups_),
          dense_nrows(dense_nrows_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type &team) const {
      auto my_league = team.league_rank();  // map to rowid

      size_t nrows = row_map.extent(0) - 1;

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, 0, node_groups), [&](const long ng) {
            auto rowid = nodes_grouped_by_level(node_count +
                                                my_league * node_groups + ng);
            if (size_t(rowid) < nrows) {
              auto soffset   = row_map(rowid);
              auto eoffset   = row_map(rowid + 1);
              auto rhs_rowid = rhs(rowid);
              scalar_t diff  = scalar_t(0.0);

              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team, soffset, eoffset),
                  [&](const long ptr, scalar_t &tdiff) {
                    auto colid = entries(ptr);
                    auto val   = values(ptr);
                    if (colid != rowid) {
                      tdiff = tdiff - val * lhs(colid);
                    }
                  },
                  diff);

              // ASSUMPTION: sorted diagonal value located at eoffset - 1
              lhs(rowid) = IsLower
                               ? (rhs_rowid + diff) / values(eoffset - 1)
                               : (rhs_rowid + diff) / values(soffset);
            }  // end if
          });  // end TeamThreadRange

      team.team_barrier();
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const UnsortedTag &, const member_type &team) const {
      auto my_league = team.league_rank();  // map to rowid

      size_t nrows = row_map.extent(0) - 1;

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, 0, node_groups), [&](const long ng) {
            auto rowid = nodes_grouped_by_level(node_count +
                                                my_league * node_groups + ng);
            if (size_t(rowid) < nrows) {
              auto soffset   = row_map(rowid);
              auto eoffset   = row_map(rowid + 1);
              auto rhs_rowid = rhs(rowid);
              scalar_t diff  = scalar_t(0.0);

              auto diag = -1;
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team, soffset, eoffset),
                  [&](const long ptr, scalar_t &tdiff) {
                    auto colid = entries(ptr);
                    auto val   = values(ptr);
                    if (colid != rowid) {
                      tdiff = tdiff - val * lhs(colid);
                    } else {
                      diag = ptr;
                    }
                  },
                  diff);

              lhs(rowid) = (rhs_rowid + diff) / values(diag);
            }  // end if
          });  // end TeamThreadRange

      team.team_barrier();
    }
  };

  // Lower vs Upper Multi-block Functors

  template <class RowMapType, class EntriesType, class ValuesType,
            class LHSType, class RHSType, bool IsLower>
  struct TriLvlSchedRPSolverFunctor {
    RowMapType row_map;
    EntriesType entries;
    ValuesType values;
    LHSType lhs;
    RHSType rhs;
    entries_t nodes_grouped_by_level;

    TriLvlSchedRPSolverFunctor(const RowMapType &row_map_,
                               const EntriesType &entries_,
                               const ValuesType &values_, LHSType &lhs_,
                               const RHSType &rhs_,
                               const entries_t &nodes_grouped_by_level_)
        : row_map(row_map_),
          entries(entries_),
          values(values_),
          lhs(lhs_),
          rhs(rhs_),
          nodes_grouped_by_level(nodes_grouped_by_level_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const lno_t i) const {
      auto rowid = nodes_grouped_by_level(i);
      // Assuming indices are sorted per row, diag entry is final index in the
      // list
      long soffset   = row_map(rowid);
      long eoffset   = row_map(rowid + 1);
      auto rhs_rowid = rhs(rowid);
      for (long ptr = IsLower ? soffset : eoffset - 1;
           (IsLower && (ptr < eoffset)) || (!IsLower && (ptr >= soffset)); ptr+=(IsLower ? 1 : -1)) {
        auto colid = entries(ptr);
        auto val   = values(ptr);
        if (colid != rowid) {
          rhs_rowid = rhs_rowid - val * lhs(colid);
        } else {
          lhs(rowid) = rhs_rowid / val;
        }
      }  // end for ptr
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const UnsortedTag &, const lno_t i) const {
      auto rowid     = nodes_grouped_by_level(i);
      long soffset   = row_map(rowid);
      long eoffset   = row_map(rowid + 1);
      auto rhs_rowid = rhs(rowid);
      auto diag      = -1;
      for (long ptr = IsLower ? soffset : eoffset - 1;
           (IsLower && (ptr < eoffset)) || (!IsLower && (ptr >= soffset)); ptr+=(IsLower ? 1 : -1)) {
        auto colid = entries(ptr);
        auto val   = values(ptr);
        if (colid != rowid) {
          rhs_rowid = rhs_rowid - val * lhs(colid);
        } else {
          diag = ptr;
        }
      }  // end for ptr
      lhs(rowid) = rhs_rowid / values(diag);
    }
  };

#if defined(KOKKOSKERNELS_ENABLE_SUPERNODAL_SPTRSV)
  // -----------------------------------------------------------
  // Helper functors for Lower-triangular solve with SpMV
  template <class LHSType>
  struct SparseTriSupernodalSpMVFunctor {
    int flag;
    long node_count;
    entries_t nodes_grouped_by_level;

    const int *supercols;
    const int *workoffset;

    LHSType X;
    work_view_t work;

    // constructor
    SparseTriSupernodalSpMVFunctor(int flag_, long node_count_,
                                   const entries_t &nodes_grouped_by_level_,
                                   const int *supercols_,
                                   const int *workoffset_, LHSType &X_,
                                   work_view_t work_)
        : flag(flag_),
          node_count(node_count_),
          nodes_grouped_by_level(nodes_grouped_by_level_),
          supercols(supercols_),
          workoffset(workoffset_),
          X(X_),
          work(work_) {}

    // operator
    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type &team) const {
      const int league_rank = team.league_rank();  // batch id
      const int team_size   = team.team_size();
      const int team_rank   = team.team_rank();
      const scalar_t zero(0.0);

      auto s = nodes_grouped_by_level(node_count + league_rank);

      // copy vector elements for the diagonal to input vector (work)
      // and zero out the corresponding elements in output (X)
      int w1 = workoffset[s];
      int j1 = supercols[s];
      // number of columns in the s-th supernode column
      int nscol = supercols[s + 1] - j1;

      if (flag == -2) {
        // copy X to work
        for (int j = team_rank; j < nscol; j += team_size) {
          work(w1 + j) = X(j1 + j);
        }
      } else if (flag == -1) {
        // copy work to X
        for (int j = team_rank; j < nscol; j += team_size) {
          X(j1 + j) = work(w1 + j);
        }
      } else if (flag == 1) {
        for (int j = team_rank; j < nscol; j += team_size) {
          work(w1 + j) = X(j1 + j);
          X(j1 + j)    = zero;
        }
      } else {
        // reinitialize work to zero
        for (int j = team_rank; j < nscol; j += team_size) {
          work(w1 + j) = zero;
        }
      }
      team.team_barrier();
    }
  };

  // -----------------------------------------------------------
  // Functor for Lower-triangular solve
  template <class ColptrView, class RowindType, class ValuesType, class LHSType>
  struct LowerTriSupernodalFunctor {
    const bool unit_diagonal;
    const bool invert_diagonal;
    const bool invert_offdiagonal;
    const int *supercols;
    ColptrView colptr;
    RowindType rowind;
    ValuesType values;

    int level;
    work_view_int_t kernel_type;
    work_view_int_t diag_kernel_type;

    LHSType X;

    work_view_t work;  // needed with gemv for update&scatter
    work_view_int_t work_offset;

    entries_t nodes_grouped_by_level;

    long node_count;

    // constructor
    LowerTriSupernodalFunctor(  // supernode info
        const bool unit_diagonal_, const bool invert_diagonal_,
        const bool invert_offdiagonal_, const int *supercols_,
        // L in CSC
        const ColptrView &colptr_, const RowindType &rowind_,
        const ValuesType &values_,
        // options to pick kernel type
        int level_, work_view_int_t &kernel_type_,
        work_view_int_t &diag_kernel_type_,
        // right-hand-side (input), solution (output)
        LHSType &X_,
        // workspace
        work_view_t work_, work_view_int_t &work_offset_,
        //
        const entries_t &nodes_grouped_by_level_, long node_count_)
        : unit_diagonal(unit_diagonal_),
          invert_diagonal(invert_diagonal_),
          invert_offdiagonal(invert_offdiagonal_),
          supercols(supercols_),
          colptr(colptr_),
          rowind(rowind_),
          values(values_),
          level(level_),
          kernel_type(kernel_type_),
          diag_kernel_type(diag_kernel_type_),
          X(X_),
          work(work_),
          work_offset(work_offset_),
          nodes_grouped_by_level(nodes_grouped_by_level_),
          node_count(node_count_) {}

    // operator
    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type &team) const {
      /* ----------------------------------------------------------------------
       */
      /* get inputs */
      /* ----------------------------------------------------------------------
       */
      const int league_rank = team.league_rank();  // batch id
      const int team_size   = team.team_size();
      const int team_rank   = team.team_rank();
      const scalar_t zero(0.0);
      const scalar_t one(1.0);

      auto s = nodes_grouped_by_level(node_count + league_rank);

      // supernodal column size
      const int j1 = supercols[s];
      const int j2 = supercols[s + 1];
      // > number of columns in the s-th supernode column
      const int nscol = j2 - j1;
      // "total" number of rows in all the supernodes (diagonal+off-diagonal)
      const int i1    = colptr(j1);
      const int nsrow = colptr(j1 + 1) - i1;

      // create a view for the s-th supernocal column
      // NOTE: we currently supports only default_layout = LayoutLeft
      scalar_t *dataL = const_cast<scalar_t *>(values.data());
      Kokkos::View<scalar_t **, default_layout, temp_mem_space,
                   Kokkos::MemoryUnmanaged>
          viewL(&dataL[i1], nsrow, nscol);

      // extract part of the solution, corresponding to the diagonal block
      auto Xj = Kokkos::subview(X, range_type(j1, j2));

      // workspace
      const int workoffset = work_offset(s);
      auto Z               = Kokkos::subview(
          work, range_type(workoffset + nscol, workoffset + nsrow));

      if (diag_kernel_type(level) != 3) {  // not a device-level TRSM-solve
        if (invert_offdiagonal) {
          // combined TRSM solve with diagonal + GEMV update with off-diagonal
          auto Y = Kokkos::subview(
              work,
              range_type(
                  workoffset,
                  workoffset + nsrow));  // needed for gemv instead of trmv/trsv
          auto Ljj =
              Kokkos::subview(viewL, range_type(0, nsrow), Kokkos::ALL());
          KokkosBlas::TeamGemv<member_type, KokkosBlas::Trans::NoTranspose,
                               KokkosBlas::Algo::Gemv::Unblocked>::invoke(team,
                                                                          one,
                                                                          Ljj,
                                                                          Xj,
                                                                          zero,
                                                                          Y);
          team.team_barrier();
          for (int ii = team_rank; ii < nscol; ii += team_size) {
            Xj(ii) = Y(ii);
          }
          team.team_barrier();
        } else {
          /* TRSM with diagonal block */
          // extract diagonal and off-diagonal blocks of L
          auto Ljj =
              Kokkos::subview(viewL, range_type(0, nscol), Kokkos::ALL());
          if (invert_diagonal) {
            // workspace
            auto Y = Kokkos::subview(
                work,
                range_type(workoffset,
                           workoffset +
                               nscol));  // needed for gemv instead of trmv/trsv
            for (int ii = team_rank; ii < nscol; ii += team_size) {
              Y(ii) = Xj(ii);
            }
            team.team_barrier();
            // calling team-level "Unblocked" gemv on small-size diagonal in
            // KokkosBatched
            KokkosBlas::TeamGemv<
                member_type, KokkosBlas::Trans::NoTranspose,
                KokkosBlas::Algo::Gemv::Unblocked>::invoke(team, one, Ljj, Y,
                                                           zero, Xj);
          } else {
            // NOTE: we currently supports only default_layout = LayoutLeft
            Kokkos::View<scalar_t **, default_layout, temp_mem_space,
                         Kokkos::MemoryUnmanaged>
                Xjj(Xj.data(), nscol, 1);
            if (unit_diagonal) {
              KokkosBatched::TeamTrsm<
                  member_type, KokkosBatched::Side::Left,
                  KokkosBatched::Uplo::Lower, KokkosBatched::Trans::NoTranspose,
                  KokkosBatched::Diag::Unit,
                  KokkosBatched::Algo::Trsm::Unblocked>::invoke(team, one, Ljj,
                                                                Xjj);
            } else {
              KokkosBatched::TeamTrsm<
                  member_type, KokkosBatched::Side::Left,
                  KokkosBatched::Uplo::Lower, KokkosBatched::Trans::NoTranspose,
                  KokkosBatched::Diag::NonUnit,
                  KokkosBatched::Algo::Trsm::Unblocked>::invoke(team, one, Ljj,
                                                                Xjj);
            }
          }
          team.team_barrier();

          /* GEMM to update with off diagonal blocks */
          auto Lij =
              Kokkos::subview(viewL, range_type(nscol, nsrow), Kokkos::ALL());
          KokkosBlas::TeamGemv<member_type, KokkosBatched::Trans::NoTranspose,
                               KokkosBlas::Algo::Gemv::Unblocked>::invoke(team,
                                                                          one,
                                                                          Lij,
                                                                          Xj,
                                                                          zero,
                                                                          Z);
          team.team_barrier();
        }
      }

      /* scatter vectors back into X */
      int i2 = i1 + nscol;  // offset into rowind
      int nsrow2 =
          nsrow -
          nscol;  // "total" number of rows in all the off-diagonal supernodes
      Kokkos::View<scalar_t *, temp_mem_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic>>
          Xatomic(X.data(), X.extent(0));
      for (int ii = team_rank; ii < nsrow2; ii += team_size) {
        int i = rowind(i2 + ii);
        Xatomic(i) -= Z(ii);
      }
      team.team_barrier();
    }
  };

  // -----------------------------------------------------------
  // Functor for Upper-triangular solve in CSR
  template <class ColptrType, class RowindType, class ValuesType, class LHSType>
  struct UpperTriSupernodalFunctor {
    // NOTE: we currently supports only default_layout = LayoutLeft
    using SupernodeView =
        typename Kokkos::View<scalar_t **, default_layout, temp_mem_space,
                              Kokkos::MemoryUnmanaged>;

    bool invert_diagonal;
    const int *supercols;
    ColptrType colptr;
    RowindType rowind;
    ValuesType values;

    int level;
    work_view_int_t kernel_type;
    work_view_int_t diag_kernel_type;

    LHSType X;

    work_view_t work;  // needed with gemv for update&scatter
    work_view_int_t work_offset;

    entries_t nodes_grouped_by_level;

    long node_count;

    // constructor
    UpperTriSupernodalFunctor(  // supernode info
        bool invert_diagonal_, const int *supercols_,
        // U in CSR
        const ColptrType &colptr_, const RowindType &rowind_,
        const ValuesType &values_,
        // options to pick kernel type
        int level_, work_view_int_t &kernel_type_,
        work_view_int_t &diag_kernel_type_,
        // right-hand-side (input), solution (output)
        LHSType &X_,
        // workspace
        work_view_t &work_, work_view_int_t &work_offset_,
        //
        const entries_t &nodes_grouped_by_level_, long node_count_)
        : invert_diagonal(invert_diagonal_),
          supercols(supercols_),
          colptr(colptr_),
          rowind(rowind_),
          values(values_),
          level(level_),
          kernel_type(kernel_type_),
          diag_kernel_type(diag_kernel_type_),
          X(X_),
          work(work_),
          work_offset(work_offset_),
          nodes_grouped_by_level(nodes_grouped_by_level_),
          node_count(node_count_) {}

    // operator
    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type &team) const {
      /* ----------------------------------------------------------------------
       */
      /* get inputs */
      /* ----------------------------------------------------------------------
       */
      const int league_rank = team.league_rank();  // batch id
      const int team_size   = team.team_size();
      const int team_rank   = team.team_rank();
      const scalar_t zero(0.0);
      const scalar_t one(1.0);

      auto s = nodes_grouped_by_level(node_count + league_rank);

      // number of columns in the s-th supernode column
      int j1    = supercols[s];
      int j2    = supercols[s + 1];
      int nscol = j2 - j1;
      // "total" number of rows in all the supernodes (diagonal+off-diagonal)
      int i1    = colptr(j1);
      int nsrow = colptr(j1 + 1) - i1;

      // create a view of the s-th supernocal row of U
      scalar_t *dataU = const_cast<scalar_t *>(values.data());
      SupernodeView viewU(&dataU[i1], nsrow, nscol);

      // extract part of solution, corresponding to the diagonal block U(s, s)
      auto Xj       = Kokkos::subview(X, range_type(j1, j2));
      using Xj_type = decltype(Xj);

      // workspaces
      int workoffset = work_offset(s);

      // "total" number of rows in all the off-diagonal supernodes
      int nsrow2 = nsrow - nscol;
      /* gather vector into Z */
      int i2 = i1 + nscol;  // offset into rowind
      auto Z = Kokkos::subview(
          work, range_type(workoffset + nscol,
                           workoffset +
                               nsrow));  // needed with gemv for update&scatter
      using Z_type = decltype(Z);
      for (int ii = team_rank; ii < nsrow2; ii += team_size) {
        int i = rowind(i2 + ii);
        Z(ii) = X(i);
      }
      team.team_barrier();
      /* GEMM to update with off diagonal blocks, Xj = -Uij^T * Z */
      if (diag_kernel_type(level) != 3) {
        // not device-level GEMV-udpate
        auto Uij =
            Kokkos::subview(viewU, range_type(nscol, nsrow), Kokkos::ALL());
        using Uij_type = decltype(Uij);
        KokkosBlas::TeamGemv<member_type, KokkosBatched::Trans::Transpose,
                             KokkosBlas::Algo::Gemv::Unblocked>::
            template invoke<const scalar_t, Uij_type, Z_type, Xj_type>(
                team, -one, Uij, Z, one, Xj);
        team.team_barrier();

        /* TRSM with diagonal block */
        // extract diagonal and off-diagonal blocks of U
        auto Ujj = Kokkos::subview(viewU, range_type(0, nscol), Kokkos::ALL());
        using Ujj_type = decltype(Ujj);

        if (invert_diagonal) {
          // workspace
          auto Y = Kokkos::subview(
              work,
              range_type(
                  workoffset,
                  workoffset + nscol));  // needed for gemv instead of trmv/trsv
          using Y_type = decltype(Y);
          for (int ii = team_rank; ii < nscol; ii += team_size) {
            Y(ii) = Xj(ii);
          }
          team.team_barrier();

          // caling team-level kernel in KokkosBatched on a small-size diagonal
          KokkosBlas::TeamGemv<member_type, KokkosBatched::Trans::Transpose,
                               KokkosBlas::Algo::Gemv::Unblocked>::
              template invoke<const scalar_t, Ujj_type, Y_type, Xj_type>(
                  team, one, Ujj, Y, zero, Xj);
        } else {
          // NOTE: we currently supports only default_layout = LayoutLeft
          Kokkos::View<scalar_t **, default_layout, temp_mem_space,
                       Kokkos::MemoryUnmanaged>
              Xjj(Xj.data(), nscol, 1);
          KokkosBatched::TeamTrsm<
              member_type, KokkosBatched::Side::Left,
              KokkosBatched::Uplo::Lower, KokkosBatched::Trans::Transpose,
              KokkosBatched::Diag::NonUnit,
              KokkosBatched::Algo::Trsm::Unblocked>::invoke(team, one, Ujj,
                                                            Xjj);
        }
        team.team_barrier();
      }
    }
  };

  // -----------------------------------------------------------
  // Functor for Upper-triangular solve in CSC
  template <class ColptrType, class RowindType, class ValuesType, class LHSType>
  struct UpperTriTranSupernodalFunctor {
    const bool invert_diagonal;
    const bool invert_offdiagonal;
    const int *supercols;
    ColptrType colptr;
    RowindType rowind;
    ValuesType values;

    int level;
    work_view_int_t kernel_type;
    work_view_int_t diag_kernel_type;

    LHSType X;

    work_view_t work;  // needed with gemv for update&scatter
    work_view_int_t work_offset;

    entries_t nodes_grouped_by_level;

    long node_count;

    // constructor
    UpperTriTranSupernodalFunctor(  // supernode info
        const bool invert_diagonal_, const bool invert_offdiagonal_,
        const int *supercols_,

        // U in CSC
        const ColptrType &colptr_, const RowindType &rowind_,
        const ValuesType &values_,
        // options to pick kernel type
        const int level_, const work_view_int_t &kernel_type_,
        const work_view_int_t &diag_kernel_type_,
        // right-hand-side (input), solution (output)
        const LHSType &X_,
        // workspace
        const work_view_t &work_, const work_view_int_t &work_offset_,
        //
        const entries_t &nodes_grouped_by_level_, const long node_count_)
        : invert_diagonal(invert_diagonal_),
          invert_offdiagonal(invert_offdiagonal_),
          supercols(supercols_),
          colptr(colptr_),
          rowind(rowind_),
          values(values_),
          level(level_),
          kernel_type(kernel_type_),
          diag_kernel_type(diag_kernel_type_),
          X(X_),
          work(work_),
          work_offset(work_offset_),
          nodes_grouped_by_level(nodes_grouped_by_level_),
          node_count(node_count_) {}

    // operator
    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type &team) const {
      /* ----------------------------------------------------------------------
       */
      /* get inputs */
      /* ----------------------------------------------------------------------
       */
      const int league_rank = team.league_rank();  // batch id
      const int team_size   = team.team_size();
      const int team_rank   = team.team_rank();
      const scalar_t zero(0.0);
      const scalar_t one(1.0);

      auto s = nodes_grouped_by_level(node_count + league_rank);

      // number of columns in the s-th supernode column
      const int j1    = supercols[s];
      const int j2    = supercols[s + 1];
      const int nscol = j2 - j1;
      // "total" number of rows in all the supernodes (diagonal+off-diagonal)
      const int i1    = colptr(j1);
      const int nsrow = colptr(j1 + 1) - i1;
      // "total" number of rows in all the off-diagonal supernodes
      const int nsrow2 = nsrow - nscol;

      // create a view of the s-th supernocal column of U
      // NOTE: we currently supports only default_layout = LayoutLeft
      scalar_t *dataU = const_cast<scalar_t *>(values.data());
      Kokkos::View<scalar_t **, default_layout, temp_mem_space,
                   Kokkos::MemoryUnmanaged>
          viewU(&dataU[i1], nsrow, nscol);

      // extract part of solution, corresponding to the diagonal block U(s, s)
      auto Xj = Kokkos::subview(X, range_type(j1, j2));

      // workspaces
      int workoffset = work_offset(s);

      /* TRSM with diagonal block */
      if (diag_kernel_type(level) != 3) {
        // not device-level TRSM-solve
        if (invert_offdiagonal) {
          // extract diagonal + off-diagonal blocks of U
          auto Y = Kokkos::subview(
              work,
              range_type(
                  workoffset,
                  workoffset + nsrow));  // needed with gemv for update&scatter
          auto Uij =
              Kokkos::subview(viewU, range_type(0, nsrow), Kokkos::ALL());
          KokkosBlas::TeamGemv<member_type, KokkosBatched::Trans::NoTranspose,
                               KokkosBlas::Algo::Gemv::Unblocked>::invoke(team,
                                                                          one,
                                                                          Uij,
                                                                          Xj,
                                                                          zero,
                                                                          Y);
          team.team_barrier();
          // copy the diagonal back to output
          for (int ii = team_rank; ii < nscol; ii += team_size) {
            Xj(ii) = Y(ii);
          }
        } else {
          // extract diagonal block of U (stored on top)
          auto Ujj =
              Kokkos::subview(viewU, range_type(0, nscol), Kokkos::ALL());
          if (invert_diagonal) {
            auto Y = Kokkos::subview(
                work,
                range_type(workoffset,
                           workoffset +
                               nscol));  // needed for gemv instead of trmv/trsv
            for (int ii = team_rank; ii < nscol; ii += team_size) {
              Y(ii) = Xj(ii);
            }
            team.team_barrier();
            KokkosBlas::TeamGemv<
                member_type, KokkosBatched::Trans::NoTranspose,
                KokkosBlas::Algo::Gemv::Unblocked>::invoke(team, one, Ujj, Y,
                                                           zero, Xj);
          } else {
            // NOTE: we currently supports only default_layout = LayoutLeft
            Kokkos::View<scalar_t **, default_layout, temp_mem_space,
                         Kokkos::MemoryUnmanaged>
                Xjj(Xj.data(), nscol, 1);
            KokkosBatched::TeamTrsm<
                member_type, KokkosBatched::Side::Left,
                KokkosBatched::Uplo::Upper, KokkosBatched::Trans::NoTranspose,
                KokkosBatched::Diag::NonUnit,
                KokkosBatched::Algo::Trsm::Unblocked>::invoke(team, one, Ujj,
                                                              Xjj);
          }
        }
        team.team_barrier();
      }
      if (nsrow2 > 0) {
        /* GEMM to update off diagonal blocks, Z = Uij * Xj */
        auto Z = Kokkos::subview(
            work, range_type(workoffset + nscol, workoffset + nsrow));
        if (!invert_offdiagonal && diag_kernel_type(level) != 3) {
          // not device-level TRSM-solve
          auto Uij =
              Kokkos::subview(viewU, range_type(nscol, nsrow), Kokkos::ALL());
          KokkosBlas::TeamGemv<member_type, KokkosBatched::Trans::NoTranspose,
                               KokkosBlas::Algo::Gemv::Unblocked>::invoke(team,
                                                                          one,
                                                                          Uij,
                                                                          Xj,
                                                                          zero,
                                                                          Z);
          team.team_barrier();
        }

        /* scatter vector into Z */
        int i2 = i1 + nscol;  // offset into rowind
        Kokkos::View<scalar_t *, temp_mem_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic>>
            Xatomic(X.data(), X.extent(0));
        for (int ii = team_rank; ii < nsrow2; ii += team_size) {
          int i = rowind(i2 + ii);
          Xatomic(i) -= Z(ii);
        }
        team.team_barrier();
      }
    }
  };
#endif

  template <class RowMapType, class EntriesType, class ValuesType,
            class LHSType, class RHSType>
  struct UpperTriLvlSchedRPSolverFunctor {
    RowMapType row_map;
    EntriesType entries;
    ValuesType values;
    LHSType lhs;
    RHSType rhs;
    entries_t nodes_grouped_by_level;

    UpperTriLvlSchedRPSolverFunctor(const RowMapType &row_map_,
                                    const EntriesType &entries_,
                                    const ValuesType &values_, LHSType &lhs_,
                                    const RHSType &rhs_,
                                    const entries_t &nodes_grouped_by_level_)
        : row_map(row_map_),
          entries(entries_),
          values(values_),
          lhs(lhs_),
          rhs(rhs_),
          nodes_grouped_by_level(nodes_grouped_by_level_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const lno_t i) const {
      auto rowid = nodes_grouped_by_level(i);
      // Assuming indices are sorted per row, diag entry is final index in the
      // list
      long soffset   = row_map(rowid);
      long eoffset   = row_map(rowid + 1);
      auto rhs_rowid = rhs(rowid);
      for (long ptr = eoffset - 1; ptr >= soffset; --ptr) {
        auto colid = entries(ptr);
        auto val   = values(ptr);
        if (colid != rowid) {
          rhs_rowid = rhs_rowid - val * lhs(colid);
        } else {
          lhs(rowid) = rhs_rowid / val;
        }
      }  // end for ptr
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const UnsortedTag &, const lno_t i) const {
      auto rowid     = nodes_grouped_by_level(i);
      long soffset   = row_map(rowid);
      long eoffset   = row_map(rowid + 1);
      auto rhs_rowid = rhs(rowid);
      auto diag      = -1;
      for (long ptr = eoffset - 1; ptr >= soffset; --ptr) {
        auto colid = entries(ptr);
        auto val   = values(ptr);
        if (colid != rowid) {
          rhs_rowid = rhs_rowid - val * lhs(colid);
        } else {
          diag = ptr;
        }
      }  // end for ptr
      lhs(rowid) = rhs_rowid / values(diag);
    }
  };

  // --------------------------------
  // Single-block functors
  // --------------------------------

  template <class RowMapType, class EntriesType, class ValuesType,
            class LHSType, class RHSType>
  struct LowerTriLvlSchedTP1SingleBlockFunctor {
    RowMapType row_map;
    EntriesType entries;
    ValuesType values;
    LHSType lhs;
    RHSType rhs;
    entries_t nodes_grouped_by_level;
    entries_t nodes_per_level;

    long node_count;  // like "block" offset into ngbl, my_league is the "local"
                      // offset
    long lvl_start;
    long lvl_end;
    long cutoff;
    // team_size: each team can be assigned a row, if there are enough rows...

    LowerTriLvlSchedTP1SingleBlockFunctor(
        const RowMapType &row_map_, const EntriesType &entries_,
        const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_,
        const entries_t &nodes_grouped_by_level_, entries_t &nodes_per_level_,
        long node_count_, long lvl_start_, long lvl_end_, long cutoff_ = 0)
        : row_map(row_map_),
          entries(entries_),
          values(values_),
          lhs(lhs_),
          rhs(rhs_),
          nodes_grouped_by_level(nodes_grouped_by_level_),
          nodes_per_level(nodes_per_level_),
          node_count(node_count_),
          lvl_start(lvl_start_),
          lvl_end(lvl_end_),
          cutoff(cutoff_) {}

    // SingleBlock: Only one block (or league) executing; team_rank used to map
    // thread to row

    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);
      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_rank         = team.team_rank();
        diff                = scalar_t(0.0);

        if (my_rank < nodes_this_lvl) {
          // THIS is where the mapping of threadid to rowid happens
          rowid = nodes_grouped_by_level(my_rank + mut_node_count);

          soffset = row_map(rowid);
          eoffset = row_map(rowid + 1);
          rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
          for (auto ptr = soffset; ptr < eoffset; ++ptr) {
            auto colid = entries(ptr);
            auto val   = values(ptr);
            if (colid != rowid) {
              diff -= val * lhs(colid);
            }
          }
#else
          auto trange = eoffset - soffset;
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team, trange),
              [&](const int loffset, scalar_t &tdiff) {
                auto ptr   = soffset + loffset;
                auto colid = entries(ptr);
                auto val   = values(ptr);
                if (colid != rowid) {
                  tdiff -= val * lhs(colid);
                }
              },
              diff);
#endif
          // ASSUMPTION: sorted diagonal value located at eoffset - 1
          lhs(rowid) = (rhs_val + diff) / values(eoffset - 1);
        }  // end if team.team_rank() < nodes_this_lvl
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl per thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end operator

    KOKKOS_INLINE_FUNCTION
    void operator()(const UnsortedTag &, const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);
      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_rank         = team.team_rank();
        diff                = scalar_t(0.0);

        if (my_rank < nodes_this_lvl) {
          // THIS is where the mapping of threadid to rowid happens
          rowid   = nodes_grouped_by_level(my_rank + mut_node_count);
          soffset = row_map(rowid);
          eoffset = row_map(rowid + 1);
          rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
          for (auto ptr = soffset; ptr < eoffset; ++ptr) {
            auto colid = entries(ptr);
            auto val   = values(ptr);
            if (colid != rowid) {
              diff -= val * lhs(colid);
            }
          }
#else
          auto trange = eoffset - soffset;
          auto diag   = -1;

          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team, trange),
              [&](const int loffset, scalar_t &tdiff) {
                auto ptr = soffset + loffset;

                auto colid = entries(ptr);
                auto val   = values(ptr);
                if (colid != rowid) {
                  tdiff -= val * lhs(colid);
                } else {
                  diag = ptr;
                }
              },
              diff);
#endif
          lhs(rowid) = (rhs_val + diff) / values(diag);
        }  // end if team.team_rank() < nodes_this_lvl
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl per thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end operator

    KOKKOS_INLINE_FUNCTION
    void operator()(const LargerCutoffTag &, const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);
      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_team_rank    = team.team_rank();
        // If cutoff > team_size, then a thread will be responsible for multiple
        // rows - this may be a helpful scenario depending on occupancy etc.
        for (int my_rank = my_team_rank; my_rank < cutoff;
             my_rank += team.team_size()) {
          diff = scalar_t(0.0);
          if (my_rank < nodes_this_lvl) {
            // THIS is where the mapping of threadid to rowid happens
            rowid   = nodes_grouped_by_level(my_rank + mut_node_count);
            soffset = row_map(rowid);
            eoffset = row_map(rowid + 1);
            rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
            for (auto ptr = soffset; ptr < eoffset; ++ptr) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if (colid != rowid) {
                diff -= val * lhs(colid);
              }
            }
#else
            auto trange = eoffset - soffset;
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team, trange),
                [&](const int loffset, scalar_t &tdiff) {
                  auto ptr   = soffset + loffset;
                  auto colid = entries(ptr);
                  auto val   = values(ptr);
                  if (colid != rowid) {
                    tdiff -= val * lhs(colid);
                  }
                },
                diff);
#endif
            // ASSUMPTION: sorted diagonal value located at eoffset - 1 for
            // lower tri, soffset for upper tri
            lhs(rowid) = (rhs_val + diff) / values(eoffset - 1);
          }  // end if team.team_rank() < nodes_this_lvl
        }    // end for my_rank loop
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl per thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end tagged operator

    KOKKOS_INLINE_FUNCTION
    void operator()(const UnsortedLargerCutoffTag &,
                    const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);

      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_team_rank    = team.team_rank();
        // If cutoff > team_size, then a thread will be responsible for multiple
        // rows - this may be a helpful scenario depending on occupancy etc.
        for (int my_rank = my_team_rank; my_rank < cutoff;
             my_rank += team.team_size()) {
          diff = scalar_t(0.0);
          if (my_rank < nodes_this_lvl) {
            // THIS is where the mapping of threadid to rowid happens
            rowid   = nodes_grouped_by_level(my_rank + mut_node_count);
            soffset = row_map(rowid);
            eoffset = row_map(rowid + 1);
            rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
            for (auto ptr = soffset; ptr < eoffset; ++ptr) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if (colid != rowid) {
                diff -= val * lhs(colid);
              }
            }
#else
            auto trange = eoffset - soffset;
            auto diag   = -1;

            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team, trange),
                [&](const int loffset, scalar_t &tdiff) {
                  auto ptr   = soffset + loffset;
                  auto colid = entries(ptr);
                  auto val   = values(ptr);
                  if (colid != rowid) {
                    tdiff -= val * lhs(colid);
                  } else {
                    diag = ptr;
                  }
                },
                diff);
#endif
            lhs(rowid) = (rhs_val + diff) / values(diag);
          }  // end if team.team_rank() < nodes_this_lvl
        }    // end for my_rank loop
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl per thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end tagged operator
  };

  template <class RowMapType, class EntriesType, class ValuesType,
            class LHSType, class RHSType>
  struct UpperTriLvlSchedTP1SingleBlockFunctor {
    RowMapType row_map;
    EntriesType entries;
    ValuesType values;
    LHSType lhs;
    RHSType rhs;
    entries_t nodes_grouped_by_level;
    entries_t nodes_per_level;

    long node_count;  // like "block" offset into ngbl, my_league is the "local"
    // offset
    long lvl_start;
    long lvl_end;
    long cutoff;
    // team_size: each team can be assigned a row, if there are enough rows...

    UpperTriLvlSchedTP1SingleBlockFunctor(
        const RowMapType &row_map_, const EntriesType &entries_,
        const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_,
        const entries_t &nodes_grouped_by_level_, entries_t &nodes_per_level_,
        long node_count_, long lvl_start_, long lvl_end_, long cutoff_ = 0)
        : row_map(row_map_),
          entries(entries_),
          values(values_),
          lhs(lhs_),
          rhs(rhs_),
          nodes_grouped_by_level(nodes_grouped_by_level_),
          nodes_per_level(nodes_per_level_),
          node_count(node_count_),
          lvl_start(lvl_start_),
          lvl_end(lvl_end_),
          cutoff(cutoff_) {}

    // SingleBlock: Only one block (or league) executing; team_rank used to map
    // thread to row
    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);

      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_rank         = team.team_rank();
        diff                = scalar_t(0.0);

        if (my_rank < nodes_this_lvl) {
          // THIS is where the mapping of threadid to rowid happens
          rowid   = nodes_grouped_by_level(my_rank + mut_node_count);
          soffset = row_map(rowid);
          eoffset = row_map(rowid + 1);
          rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
          for (auto ptr = soffset; ptr < eoffset; ++ptr) {
            auto colid = entries(ptr);
            auto val   = values(ptr);
            if (colid != rowid) {
              diff -= val * lhs(colid);
            }
          }
#else
          auto trange = eoffset - soffset;
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team, trange),
              [&](const int loffset, scalar_t &tdiff) {
                auto ptr   = soffset + loffset;
                auto colid = entries(ptr);
                auto val   = values(ptr);
                if (colid != rowid) {
                  tdiff -= val * lhs(colid);
                }
              },
              diff);
#endif
          // ASSUMPTION: sorted diagonal value located at soffset
          lhs(rowid) = (rhs_val + diff) / values(soffset);
        }  // end if
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl each thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end operator

    KOKKOS_INLINE_FUNCTION
    void operator()(const UnsortedTag &, const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);

      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_rank         = team.team_rank();
        diff                = scalar_t(0.0);

        if (my_rank < nodes_this_lvl) {
          // THIS is where the mapping of threadid to rowid happens
          rowid   = nodes_grouped_by_level(my_rank + mut_node_count);
          soffset = row_map(rowid);
          eoffset = row_map(rowid + 1);
          rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
          auto diag = -1;
          for (auto ptr = soffset; ptr < eoffset; ++ptr) {
            auto colid = entries(ptr);
            auto val   = values(ptr);
            if (colid != rowid) {
              diff -= val * lhs(colid);
            } else {
              diag = ptr;
            }
          }
#else
          auto trange = eoffset - soffset;
          auto diag   = -1;

          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team, trange),
              [&](const int loffset, scalar_t &tdiff) {
                auto ptr   = soffset + loffset;
                auto colid = entries(ptr);
                auto val   = values(ptr);
                if (colid != rowid) {
                  tdiff -= val * lhs(colid);
                } else {
                  diag = ptr;
                }
              },
              diff);
#endif
          lhs(rowid) = (rhs_val + diff) / values(diag);
        }  // end if
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl each thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end operator

    KOKKOS_INLINE_FUNCTION
    void operator()(const LargerCutoffTag &, const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);

      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_team_rank    = team.team_rank();
        // If cutoff > team_size, then a thread will be responsible for multiple
        // rows - this may be a helpful scenario depending on occupancy etc.
        for (int my_rank = my_team_rank; my_rank < cutoff;
             my_rank += team.team_size()) {
          diff = scalar_t(0.0);
          if (my_rank < nodes_this_lvl) {
            // THIS is where the mapping of threadid to rowid happens
            rowid   = nodes_grouped_by_level(my_rank + mut_node_count);
            soffset = row_map(rowid);
            eoffset = row_map(rowid + 1);
            rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
            for (auto ptr = soffset; ptr < eoffset; ++ptr) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if (colid != rowid) {
                diff -= val * lhs(colid);
              }
            }
#else
            auto trange = eoffset - soffset;
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team, trange),
                [&](const int loffset, scalar_t &tdiff) {
                  auto ptr   = soffset + loffset;
                  auto colid = entries(ptr);
                  auto val   = values(ptr);
                  if (colid != rowid) {
                    tdiff -= val * lhs(colid);
                  }
                },
                diff);
#endif
            // ASSUMPTION: sorted diagonal value located at eoffset - 1 for
            // lower tri, soffset for upper tri
            lhs(rowid) = (rhs_val + diff) / values(soffset);
          }  // end if team.team_rank() < nodes_this_lvl
        }    // end for my_rank loop
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl per thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end tagged operator

    KOKKOS_INLINE_FUNCTION
    void operator()(const UnsortedLargerCutoffTag &,
                    const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);

      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_team_rank    = team.team_rank();
        // If cutoff > team_size, then a thread will be responsible for multiple
        // rows - this may be a helpful scenario depending on occupancy etc.
        for (int my_rank = my_team_rank; my_rank < cutoff;
             my_rank += team.team_size()) {
          diff = scalar_t(0.0);
          if (my_rank < nodes_this_lvl) {
            // THIS is where the mapping of threadid to rowid happens
            rowid   = nodes_grouped_by_level(my_rank + mut_node_count);
            soffset = row_map(rowid);
            eoffset = row_map(rowid + 1);
            rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
            auto diag = -1;
            for (auto ptr = soffset; ptr < eoffset; ++ptr) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if (colid != rowid) {
                diff -= val * lhs(colid);
              } else {
                diag = ptr;
              }
            }
#else
            auto trange = eoffset - soffset;
            auto diag   = -1;
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team, trange),
                [&](const int loffset, scalar_t &tdiff) {
                  auto ptr   = soffset + loffset;
                  auto colid = entries(ptr);
                  auto val   = values(ptr);
                  if (colid != rowid) {
                    tdiff -= val * lhs(colid);
                  } else {
                    diag = ptr;
                  }
                },
                diff);
#endif
            lhs(rowid) = (rhs_val + diff) / values(diag);
          }  // end if team.team_rank() < nodes_this_lvl
        }    // end for my_rank loop
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl per thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end tagged operator
  };

  template <class RowMapType, class EntriesType, class ValuesType,
            class LHSType, class RHSType>
  struct TriLvlSchedTP1SingleBlockFunctor {
    RowMapType row_map;
    EntriesType entries;
    ValuesType values;
    LHSType lhs;
    RHSType rhs;
    entries_t nodes_grouped_by_level;
    entries_t nodes_per_level;

    long node_count;  // like "block" offset into ngbl, my_league is the "local"
                      // offset
    long lvl_start;
    long lvl_end;
    const bool is_lowertri;
    const int dense_nrows;
    const int cutoff;
    // team_size: each team can be assigned a row, if there are enough rows...

    TriLvlSchedTP1SingleBlockFunctor(
        const RowMapType &row_map_, const EntriesType &entries_,
        const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_,
        const entries_t &nodes_grouped_by_level_, entries_t &nodes_per_level_,
        long node_count_, long lvl_start_, long lvl_end_, const bool is_lower_,
        const int dense_nrows_ = 0, const int cutoff_ = 0)
        : row_map(row_map_),
          entries(entries_),
          values(values_),
          lhs(lhs_),
          rhs(rhs_),
          nodes_grouped_by_level(nodes_grouped_by_level_),
          nodes_per_level(nodes_per_level_),
          node_count(node_count_),
          lvl_start(lvl_start_),
          lvl_end(lvl_end_),
          is_lowertri(is_lower_),
          dense_nrows(dense_nrows_),
          cutoff(cutoff_) {}

    // SingleBlock: Only one block (or league) executing; team_rank used to map
    // thread to row

    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);

      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_rank         = team.team_rank();
        diff                = scalar_t(0.0);

        if (my_rank < nodes_this_lvl) {
          // THIS is where the mapping of threadid to rowid happens
          rowid   = nodes_grouped_by_level(my_rank + mut_node_count);
          soffset = row_map(rowid);
          eoffset = row_map(rowid + 1);
          rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
          for (auto ptr = soffset; ptr < eoffset; ++ptr) {
            auto colid = entries(ptr);
            auto val   = values(ptr);
            if (colid != rowid) {
              diff -= val * lhs(colid);
            }
          }
#else
          auto trange = eoffset - soffset;
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team, trange),
              [&](const int loffset, scalar_t &tdiff) {
                auto ptr   = soffset + loffset;
                auto colid = entries(ptr);
                auto val   = values(ptr);
                if (colid != rowid) {
                  tdiff -= val * lhs(colid);
                }
              },
              diff);
#endif

          // ASSUMPTION: sorted diagonal value located at eoffset - 1 for lower
          // tri, soffset for upper tri
          if (is_lowertri)
            lhs(rowid) = (rhs_val + diff) / values(eoffset - 1);
          else
            lhs(rowid) = (rhs_val + diff) / values(soffset);
        }  // end if team.team_rank() < nodes_this_lvl
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl per thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end operator

    KOKKOS_INLINE_FUNCTION
    void operator()(const UnsortedTag &, const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);

      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_rank         = team.team_rank();
        diff                = scalar_t(0.0);

        if (my_rank < nodes_this_lvl) {
          // THIS is where the mapping of threadid to rowid happens
          rowid   = nodes_grouped_by_level(my_rank + mut_node_count);
          soffset = row_map(rowid);
          eoffset = row_map(rowid + 1);
          rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
          auto diag = -1;
          for (auto ptr = soffset; ptr < eoffset; ++ptr) {
            auto colid = entries(ptr);
            auto val   = values(ptr);
            if (colid != rowid) {
              diff -= val * lhs(colid);
            } else {
              diag = ptr;
            }
          }
#else
          auto trange = eoffset - soffset;
          auto diag   = -1;
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team, trange),
              [&](const int loffset, scalar_t &tdiff) {
                auto ptr   = soffset + loffset;
                auto colid = entries(ptr);
                auto val   = values(ptr);
                if (colid != rowid) {
                  tdiff -= val * lhs(colid);
                } else {
                  diag = ptr;
                }
              },
              diff);
#endif
          lhs(rowid) = (rhs_val + diff) / values(diag);
        }  // end if team.team_rank() < nodes_this_lvl
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl per thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end operator

    KOKKOS_INLINE_FUNCTION
    void operator()(const LargerCutoffTag &, const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);

      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_team_rank    = team.team_rank();
        // If cutoff > team_size, then a thread will be responsible for multiple
        // rows - this may be a helpful scenario depending on occupancy etc.
        for (int my_rank = my_team_rank; my_rank < cutoff;
             my_rank += team.team_size()) {
          diff = scalar_t(0.0);
          if (my_rank < nodes_this_lvl) {
            // THIS is where the mapping of threadid to rowid happens
            rowid   = nodes_grouped_by_level(my_rank + mut_node_count);
            soffset = row_map(rowid);
            eoffset = row_map(rowid + 1);
            rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
            for (auto ptr = soffset; ptr < eoffset; ++ptr) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if (colid != rowid) {
                diff -= val * lhs(colid);
              }
            }
#else
            auto trange = eoffset - soffset;
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team, trange),
                [&](const int loffset, scalar_t &tdiff) {
                  auto ptr   = soffset + loffset;
                  auto colid = entries(ptr);
                  auto val   = values(ptr);
                  if (colid != rowid) {
                    tdiff -= val * lhs(colid);
                  }
                },
                diff);
#endif

            // ASSUMPTION: sorted diagonal value located at eoffset - 1 for
            // lower tri, soffset for upper tri
            if (is_lowertri)
              lhs(rowid) = (rhs_val + diff) / values(eoffset - 1);
            else
              lhs(rowid) = (rhs_val + diff) / values(soffset);
          }  // end if team.team_rank() < nodes_this_lvl
        }    // end for my_rank loop
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl per thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end tagged operator

    KOKKOS_INLINE_FUNCTION
    void operator()(const UnsortedLargerCutoffTag &,
                    const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);

      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_team_rank    = team.team_rank();
        // If cutoff > team_size, then a thread will be responsible for multiple
        // rows - this may be a helpful scenario depending on occupancy etc.
        for (int my_rank = my_team_rank; my_rank < cutoff;
             my_rank += team.team_size()) {
          diff = scalar_t(0.0);
          if (my_rank < nodes_this_lvl) {
            // THIS is where the mapping of threadid to rowid happens
            rowid   = nodes_grouped_by_level(my_rank + mut_node_count);
            soffset = row_map(rowid);
            eoffset = row_map(rowid + 1);
            rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
            auto diag = -1;
            for (auto ptr = soffset; ptr < eoffset; ++ptr) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if (colid != rowid) {
                diff -= val * lhs(colid);
              } else {
                diag = ptr;
              }
            }
#else
            auto trange = eoffset - soffset;
            auto diag   = -1;
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team, trange),
                [&](const int loffset, scalar_t &tdiff) {
                  auto ptr   = soffset + loffset;
                  auto colid = entries(ptr);
                  auto val   = values(ptr);
                  if (colid != rowid) {
                    tdiff -= val * lhs(colid);
                  } else {
                    diag = ptr;
                  }
                },
                diff);
#endif
            lhs(rowid) = (rhs_val + diff) / values(diag);
          }  // end if team.team_rank() < nodes_this_lvl
        }    // end for my_rank loop
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl per thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end tagged operator
  };

  template <class RowMapType, class EntriesType, class ValuesType,
            class LHSType, class RHSType>
  struct TriLvlSchedTP1SingleBlockFunctorDiagValues {
    RowMapType row_map;
    EntriesType entries;
    ValuesType values;
    LHSType lhs;
    RHSType rhs;
    entries_t nodes_grouped_by_level;
    entries_t nodes_per_level;
    ValuesType diagonal_values;

    long node_count;  // like "block" offset into ngbl, my_league is the "local"
                      // offset
    long lvl_start;
    long lvl_end;
    const bool is_lowertri;
    const int dense_nrows;
    const int cutoff;
    // team_size: each team can be assigned a row, if there are enough rows...

    TriLvlSchedTP1SingleBlockFunctorDiagValues(
        const RowMapType &row_map_, const EntriesType &entries_,
        const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_,
        const entries_t &nodes_grouped_by_level_,
        const entries_t &nodes_per_level_, const ValuesType &diagonal_values_,
        long node_count_, const long lvl_start_, const long lvl_end_,
        const bool is_lower_, const int dense_nrows_ = 0, const int cutoff_ = 0)
        : row_map(row_map_),
          entries(entries_),
          values(values_),
          lhs(lhs_),
          rhs(rhs_),
          nodes_grouped_by_level(nodes_grouped_by_level_),
          nodes_per_level(nodes_per_level_),
          diagonal_values(diagonal_values_),
          node_count(node_count_),
          lvl_start(lvl_start_),
          lvl_end(lvl_end_),
          is_lowertri(is_lower_),
          dense_nrows(dense_nrows_),
          cutoff(cutoff_) {}

    // SingleBlock: Only one block (or league) executing; team_rank used to map
    // thread to row

    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);

      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_rank         = team.team_rank();
        diff                = scalar_t(0.0);

        if (my_rank < nodes_this_lvl) {
          // THIS is where the mapping of threadid to rowid happens
          rowid   = nodes_grouped_by_level(my_rank + mut_node_count);
          soffset = row_map(rowid);
          eoffset = row_map(rowid + 1);
          rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
          for (auto ptr = soffset; ptr < eoffset; ++ptr) {
            auto colid = entries(ptr);
            auto val   = values(ptr);
            if (colid != rowid) {
              diff -= val * lhs(colid);
            }
          }
#else
          auto trange = eoffset - soffset;
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team, trange),
              [&](const int loffset, scalar_t &tdiff) {
                auto ptr   = soffset + loffset;
                auto colid = entries(ptr);
                auto val   = values(ptr);

                if (colid != rowid) {
                  tdiff -= val * lhs(colid);
                }
              },
              diff);
#endif
          // ASSUMPTION: sorted diagonal value located at eoffset - 1 for lower
          // tri, soffset for upper tri
          lhs(rowid) = (rhs_val + diff) / diagonal_values(rowid);
        }  // end if team.team_rank() < nodes_this_lvl
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl per thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end operator

    KOKKOS_INLINE_FUNCTION
    void operator()(const LargerCutoffTag &, const member_type &team) const {
      long mut_node_count = node_count;
      typename entries_t::non_const_value_type rowid{0};
      typename RowMapType::non_const_value_type soffset{0};
      typename RowMapType::non_const_value_type eoffset{0};
      typename RHSType::non_const_value_type rhs_val{0};
      scalar_t diff = scalar_t(0.0);

      for (auto lvl = lvl_start; lvl < lvl_end; ++lvl) {
        auto nodes_this_lvl = nodes_per_level(lvl);
        int my_team_rank    = team.team_rank();
        // If cutoff > team_size, then a thread will be responsible for multiple
        // rows - this may be a helpful scenario depending on occupancy etc.
        for (int my_rank = my_team_rank; my_rank < cutoff;
             my_rank += team.team_size()) {
          diff = scalar_t(0.0);
          if (my_rank < nodes_this_lvl) {
            // THIS is where the mapping of threadid to rowid happens
            rowid   = nodes_grouped_by_level(my_rank + mut_node_count);
            soffset = row_map(rowid);
            eoffset = row_map(rowid + 1);
            rhs_val = rhs(rowid);

#ifdef SERIAL_FOR_LOOP
            for (auto ptr = soffset; ptr < eoffset; ++ptr) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if (colid != rowid) {
                diff -= val * lhs(colid);
              }
            }
#else
            auto trange = eoffset - soffset;
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team, trange),
                [&](const int loffset, scalar_t &tdiff) {
                  auto ptr   = soffset + loffset;
                  auto colid = entries(ptr);
                  auto val   = values(ptr);
                  if (colid != rowid) {
                    tdiff -= val * lhs(colid);
                  }
                },
                diff);
#endif
            lhs(rowid) = (rhs_val + diff) / diagonal_values(rowid);
          }  // end if team.team_rank() < nodes_this_lvl
        }    // end for my_rank loop
        {
          // Update mut_node_count from nodes_per_level(lvl) each iteration of
          // lvl per thread
          mut_node_count += nodes_this_lvl;
        }
        team.team_barrier();
      }  // end for lvl
    }    // end tagged operator
  };

  //
  // End of functors, begin external API
  //

#ifdef KOKKOSKERNELS_SPTRSV_CUDAGRAPHSUPPORT
  template <class RowMapType, class EntriesType, class ValuesType,
            class RHSType, class LHSType>
  static void lower_tri_solve_cg(TriSolveHandle &thandle,
                                 const RowMapType row_map,
                                 const EntriesType entries,
                                 const ValuesType values, const RHSType &rhs,
                                 LHSType &lhs) {
    typename TriSolveHandle::SPTRSVcudaGraphWrapperType *lcl_cudagraph =
        thandle.get_sptrsvCudaGraph();

    auto nlevels = thandle.get_num_levels();

    auto stream1 = lcl_cudagraph->stream;
    Kokkos::Cuda cuda1(stream1);
    auto graph = lcl_cudagraph->cudagraph;

    Kokkos::parallel_for("Init", Kokkos::RangePolicy<execution_space>(0, 1),
                         EmptyFunctor());
    Kokkos::Cuda().fence();
    cudaStreamSynchronize(stream1);
    // Kokkos::fence();

    auto hnodes_per_level       = thandle.get_host_nodes_per_level();
    auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

    size_type node_count = 0;

    int team_size = thandle.get_team_size();
    team_size     = team_size == -1 ? 64 : team_size;

    // Start capturing stream
    if (thandle.cudagraphCreated == false) {
      Kokkos::fence();
      cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
      {
        for (int iter = 0; iter < nlevels; ++iter) {
          size_type lvl_nodes = hnodes_per_level(iter);

          auto policy = std::is_same<execution_space, Kokkos::Cuda>::value
                            ? team_policy(lvl_nodes, team_size, cuda1)
                            : team_policy(lvl_nodes, team_size);

          Kokkos::parallel_for(
              "parfor_l_team_cudagraph",
              Kokkos::Experimental::require(
                  policy,
                  Kokkos::Experimental::WorkItemProperty::HintLightWeight),
              TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType,
                                          ValuesType, LHSType, RHSType, true>(
                  row_map, entries, values, lhs, rhs, nodes_grouped_by_level,
                  node_count));

          node_count += hnodes_per_level(iter);
        }
      }
      cudaStreamEndCapture(stream1, &graph);

      // Create graphExec
      cudaGraphInstantiate(&(lcl_cudagraph->cudagraphinstance), graph, NULL,
                           NULL, 0);
      thandle.cudagraphCreated = true;
    }
    // Run graph
    Kokkos::fence();
    cudaGraphLaunch(lcl_cudagraph->cudagraphinstance, stream1);

    cudaStreamSynchronize(stream1);
    Kokkos::fence();
  }  // end lower_tri_solve_cg

  template <class RowMapType, class EntriesType, class ValuesType,
            class RHSType, class LHSType>
  static void upper_tri_solve_cg(TriSolveHandle &thandle,
                                 const RowMapType row_map,
                                 const EntriesType entries,
                                 const ValuesType values, const RHSType &rhs,
                                 LHSType &lhs) {
    typename TriSolveHandle::SPTRSVcudaGraphWrapperType *lcl_cudagraph =
        thandle.get_sptrsvCudaGraph();

    auto nlevels = thandle.get_num_levels();

    auto stream1 = lcl_cudagraph->stream;
    Kokkos::Cuda cuda1(stream1);
    auto graph = lcl_cudagraph->cudagraph;

    Kokkos::parallel_for("Init", Kokkos::RangePolicy<execution_space>(0, 1),
                         EmptyFunctor());
    Kokkos::Cuda().fence();
    cudaStreamSynchronize(stream1);

    auto hnodes_per_level       = thandle.get_host_nodes_per_level();
    auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

    size_type node_count = 0;

    int team_size = thandle.get_team_size();
    team_size     = team_size == -1 ? 64 : team_size;

    // Start capturing stream
    if (thandle.cudagraphCreated == false) {
      Kokkos::fence();
      cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
      {
        for (int iter = 0; iter < nlevels; ++iter) {
          size_type lvl_nodes = hnodes_per_level(iter);

          auto policy = std::is_same<execution_space, Kokkos::Cuda>::value
                            ? team_policy(lvl_nodes, team_size, cuda1)
                            : team_policy(lvl_nodes, team_size);

          Kokkos::parallel_for(
              "parfor_u_team_cudagraph",
              Kokkos::Experimental::require(
                  policy,
                  Kokkos::Experimental::WorkItemProperty::HintLightWeight),
              TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType,
                                          ValuesType, LHSType, RHSType, false>(
                  row_map, entries, values, lhs, rhs, nodes_grouped_by_level,
                  node_count));

          node_count += hnodes_per_level(iter);
        }
      }
      cudaStreamEndCapture(stream1, &graph);

      // Create graphExec
      cudaGraphInstantiate(&(lcl_cudagraph->cudagraphinstance), graph, NULL,
                           NULL, 0);
      thandle.cudagraphCreated = true;
    }
    // Run graph
    Kokkos::fence();
    cudaGraphLaunch(lcl_cudagraph->cudagraphinstance, stream1);

    cudaStreamSynchronize(stream1);
    Kokkos::fence();
  }  // end upper_tri_solve_cg

#endif

#define FunctorTypeMacro(Functor, IsLower)                         \
  Functor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, IsLower>

  template <class RowMapType, class EntriesType, class ValuesType,
            class RHSType, class LHSType>
  static void lower_tri_solve(execution_space &space, TriSolveHandle &thandle,
                              const RowMapType row_map,
                              const EntriesType entries,
                              const ValuesType values, const RHSType &rhs,
                              LHSType &lhs) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
    cudaProfilerStop();
#endif
    const auto nlevels = thandle.get_num_levels();
    // Keep this a host View, create device version and copy to back to host
    // during scheduling This requires making sure the host view in the handle
    // is properly updated after the symbolic phase
    const auto nodes_per_level        = thandle.get_nodes_per_level();
    const auto hnodes_per_level       = thandle.get_host_nodes_per_level();
    const auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();
    //const auto block_size             = thandle.get_block_size();
    const auto block_enabled          = thandle.is_block_enabled();

    KK_REQUIRE_MSG(!block_enabled, "Block matrices not yet supported");

    // Set up functor types
    using LowerRPPoint = FunctorTypeMacro(TriLvlSchedRPSolverFunctor, true);
    //using LowerRPBlock = FunctorTypeMacro(TriLvlSchedRPSolverFunctor, true, true);
    using LowerTPPoint = FunctorTypeMacro(TriLvlSchedTP1SolverFunctor, true);
    //using LowerTPBlock = FunctorTypeMacro(TriLvlSchedTP1SolverFunctor, true, true);

#if defined(KOKKOSKERNELS_ENABLE_SUPERNODAL_SPTRSV)
    using namespace KokkosSparse::Experimental;
    using device_t            = Kokkos::Device<execution_space, temp_mem_space>;
    using integer_view_host_t = typename TriSolveHandle::integer_view_host_t;
    using row_map_host_view_t = Kokkos::View<size_type *, Kokkos::HostSpace>;

    row_map_host_view_t row_map_host;

    const scalar_t zero(0.0);
    const scalar_t one(1.0);

    auto nodes_grouped_by_level_host =
        thandle.get_host_nodes_grouped_by_level();

    if (thandle.get_algorithm() == SPTRSVAlgorithm::SUPERNODAL_NAIVE ||
        thandle.get_algorithm() == SPTRSVAlgorithm::SUPERNODAL_ETREE ||
        thandle.get_algorithm() == SPTRSVAlgorithm::SUPERNODAL_DAG) {
      Kokkos::deep_copy(nodes_grouped_by_level_host, nodes_grouped_by_level);

      row_map_host = row_map_host_view_t(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "host rowmap"),
          row_map.extent(0));
      Kokkos::deep_copy(row_map_host, row_map);
    }

    // inversion options
    const bool invert_diagonal    = thandle.get_invert_diagonal();
    const bool invert_offdiagonal = thandle.get_invert_offdiagonal();
    const bool unit_diagonal      = thandle.is_unit_diagonal();

    // supernode sizes
    const int *supercols      = thandle.get_supercols();
    const int *supercols_host = thandle.get_supercols_host();

    // kernel types
    work_view_int_t kernel_type      = thandle.get_kernel_type();
    work_view_int_t diag_kernel_type = thandle.get_diag_kernel_type();

    integer_view_host_t kernel_type_host = thandle.get_kernel_type_host();
    integer_view_host_t diag_kernel_type_host =
        thandle.get_diag_kernel_type_host();

    // workspaces
    work_view_int_t work_offset          = thandle.get_work_offset();
    integer_view_host_t work_offset_host = thandle.get_work_offset_host();
    auto work                            = thandle.get_workspace();
#endif

    size_type node_count = 0;

#ifdef profile_supernodal_etree
    Kokkos::Timer sptrsv_timer;
    sptrsv_timer.reset();
#endif

    for (size_type lvl = 0; lvl < nlevels; ++lvl) {
      size_type lvl_nodes = hnodes_per_level(lvl);

      if (lvl_nodes != 0) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
        cudaProfilerStart();
#endif
        if (thandle.get_algorithm() ==
            KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP) {
          LowerRPPoint lrpp(row_map, entries, values, lhs, rhs, nodes_grouped_by_level);

          Kokkos::parallel_for(
              "parfor_fixed_lvl",
              Kokkos::Experimental::require(
                range_policy(space, node_count, node_count + lvl_nodes),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight),
              lrpp);
        } else if (thandle.get_algorithm() ==
                   KokkosSparse::Experimental::SPTRSVAlgorithm::
                       SEQLVLSCHD_TP1) {
          LowerTPPoint ltpp(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
          int team_size = thandle.get_team_size();
          auto tp = team_size == -1 ? team_policy(space, lvl_nodes, Kokkos::AUTO) : team_policy(space, lvl_nodes, team_size);
          Kokkos::parallel_for(
            "parfor_l_team",
            Kokkos::Experimental::require(
              tp,
              Kokkos::Experimental::WorkItemProperty::HintLightWeight),
            ltpp);
        }
        // TP2 algorithm has issues with some offset-ordinal combo to be
        // addressed
        /*
        else if ( thandle.get_algorithm() ==
  KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHED_TP2 ) { typedef
  Kokkos::TeamPolicy<execution_space> tvt_policy_type;

          int team_size = thandle.get_team_size();
          if ( team_size == -1 ) {
            team_size = std::is_same< typename
  Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 :
  64;
          }
          int vector_size = thandle.get_team_size();
          if ( vector_size == -1 ) {
            vector_size = std::is_same< typename
  Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 :
  4;
          }

          // This impl: "chunk" lvl_nodes into node_groups; a league_rank is
  responsible for processing team_size # nodes
          //       TeamThreadRange over number nodes of node_groups
          //       To avoid masking threads, 1 thread (team) per node in
  node_group (thread has full ownership of a node)
          //       ThreadVectorRange responsible for the actual solve
  computation
          //const int node_groups = team_size;
          const int node_groups = vector_size;

  #ifdef KOKKOSKERNELS_SPTRSV_TRILVLSCHED
          TriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType,
  LHSType, RHSType> tstf(row_map, entries, values, lhs, rhs,
  nodes_grouped_by_level, true, node_count, vector_size, 0); #else
          LowerTriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType,
  LHSType, RHSType> tstf(row_map, entries, values, lhs, rhs,
  nodes_grouped_by_level, node_count, node_groups); #endif
          Kokkos::parallel_for("parfor_u_team_vector", tvt_policy_type(
  (int)std::ceil((float)lvl_nodes/(float)node_groups) , team_size, vector_size
  ), tstf); } // end elseif
        */
#if defined(KOKKOSKERNELS_ENABLE_SUPERNODAL_SPTRSV)
        else if (thandle.get_algorithm() == SPTRSVAlgorithm::SUPERNODAL_NAIVE ||
                 thandle.get_algorithm() == SPTRSVAlgorithm::SUPERNODAL_ETREE ||
                 thandle.get_algorithm() == SPTRSVAlgorithm::SUPERNODAL_DAG) {

#ifdef profile_supernodal_etree
          size_t flops = 0;
          Kokkos::Timer timer;
          timer.reset();
#endif

          // NOTE: we currently supports only default_layout = LayoutLeft
          using supernode_view_type =
              Kokkos::View<scalar_t **, default_layout, device_t,
                           Kokkos::MemoryUnmanaged>;
          if (diag_kernel_type_host(lvl) == 3) {
            // using device-level kernels (functor is called to scatter the
            // results)
            scalar_t *dataL = const_cast<scalar_t *>(values.data());

            if (invert_diagonal && !invert_offdiagonal) {
              // copy diagonals to workspaces
              const int *work_offset_data = work_offset.data();
              SparseTriSupernodalSpMVFunctor<LHSType> sptrsv_init_functor(
                  -2, node_count, nodes_grouped_by_level, supercols,
                  work_offset_data, lhs, work);
              Kokkos::parallel_for(
                  "parfor_tri_supernode_spmv",
                  Kokkos::Experimental::require(
                      team_policy(space, lvl_nodes, Kokkos::AUTO),
                      Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                  sptrsv_init_functor);
            }

            for (size_type league_rank = 0; league_rank < lvl_nodes;
                 league_rank++) {
              auto s = nodes_grouped_by_level_host(node_count + league_rank);

              // supernodal column size
              int j1 = supercols_host[s];
              int j2 = supercols_host[s + 1];
              // number of columns in the s-th supernode column
              int nscol = j2 - j1;
              // "total" number of rows in all the supernodes
              // (diagonal+off-diagonal)
              int i1    = row_map_host(j1);
              int nsrow = row_map_host(j1 + 1) - i1;
#ifdef profile_supernodal_etree
              flops += 2 * (nscol * nsrow);
#endif

              // workspace  (needed for gemv instead of trmv/trsv)
              int workoffset = work_offset_host(s);

              // create a view for the s-th supernocal block column
              supernode_view_type viewL(&dataL[i1], nsrow, nscol);

              // "triangular-solve" to compute Xj
              if (invert_offdiagonal) {
                auto Y = Kokkos::subview(
                    work, range_type(workoffset, workoffset + nsrow));
                auto Xj = Kokkos::subview(
                    lhs,
                    range_type(j1, j2));  // part of the solution, corresponding
                                          // to the diagonal block
                auto Ljj = Kokkos::subview(
                    viewL, range_type(0, nsrow),
                    Kokkos::ALL());  // s-th supernocal column of L
                KokkosBlas::gemv(space, "N", one, Ljj, Xj, zero, Y);
              } else {
                auto Xj = Kokkos::subview(
                    lhs,
                    range_type(j1, j2));  // part of the solution, corresponding
                                          // to the diagonal block
                auto Ljj =
                    Kokkos::subview(viewL, range_type(0, nscol),
                                    Kokkos::ALL());  // diagonal block of s-th
                                                     // supernocal column of L
                if (invert_diagonal) {
                  auto Y = Kokkos::subview(
                      work, range_type(workoffset, workoffset + nscol));
                  KokkosBlas::gemv(space, "N", one, Ljj, Y, zero, Xj);
                } else {
                  char unit_diag = (unit_diagonal ? 'U' : 'N');
                  // NOTE: we currently supports only default_layout =
                  // LayoutLeft
                  Kokkos::View<scalar_t **, default_layout, device_t,
                               Kokkos::MemoryUnmanaged>
                      Xjj(Xj.data(), nscol, 1);
                  KokkosBlas::trsm(space, "L", "L", "N", &unit_diag, one, Ljj,
                                   Xjj);
                  // TODO: space.fence();
                  Kokkos::fence();
                }
                // update off-diagonal blocks
                int nsrow2 = nsrow - nscol;  // "total" number of rows in all
                                             // the off-diagonal supernodes
                if (nsrow2 > 0) {
                  auto Z = Kokkos::subview(
                      work, range_type(workoffset + nscol,
                                       workoffset +
                                           nsrow));  // workspace, needed with
                                                     // gemv for update&scatter
                  auto Lij = Kokkos::subview(
                      viewL, range_type(nscol, nsrow),
                      Kokkos::ALL());  // off-diagonal blocks of s-th supernodal
                                       // column of L
                  KokkosBlas::gemv(space, "N", one, Lij, Xj, zero, Z);
                }
              }
            }
            if (invert_offdiagonal) {
              // copy diagonals from workspaces
              const int *work_offset_data = work_offset.data();
              SparseTriSupernodalSpMVFunctor<LHSType> sptrsv_init_functor(
                  -1, node_count, nodes_grouped_by_level, supercols,
                  work_offset_data, lhs, work);
              Kokkos::parallel_for(
                  "parfor_tri_supernode_spmv",
                  Kokkos::Experimental::require(
                      team_policy(space, lvl_nodes, Kokkos::AUTO),
                      Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                  sptrsv_init_functor);
            }
          }

          // launching sparse-triangular solve functor
          LowerTriSupernodalFunctor<RowMapType, EntriesType, ValuesType,
                                    LHSType>
              sptrsv_functor(unit_diagonal, invert_diagonal, invert_offdiagonal,
                             supercols, row_map, entries, values, lvl,
                             kernel_type, diag_kernel_type, lhs, work,
                             work_offset, nodes_grouped_by_level, node_count);
          Kokkos::parallel_for(
              "parfor_lsolve_supernode",
              Kokkos::Experimental::require(
                  team_policy(space, lvl_nodes, Kokkos::AUTO),
                  Kokkos::Experimental::WorkItemProperty::HintLightWeight),
              sptrsv_functor);

#ifdef profile_supernodal_etree
          Kokkos::fence();
          double time_seconds = timer.seconds();
          std::cout << " > SUPERNODAL LowerTri: " << lvl << " " << time_seconds
                    << " flop count: " << flops
                    << " kernel-type: " << kernel_type_host(lvl)
                    << " # of supernodes: " << lvl_nodes << std::endl;
#endif
        } else if (thandle.get_algorithm() ==
                       SPTRSVAlgorithm::SUPERNODAL_SPMV ||
                   thandle.get_algorithm() ==
                       SPTRSVAlgorithm::SUPERNODAL_SPMV_DAG) {
#ifdef profile_supernodal_etree
          Kokkos::Timer timer;
          timer.reset();
#endif

          // initialize input & output vectors

          // update with spmv (one or two SpMV)
          bool transpose_spmv =
              ((!thandle.transpose_spmv() && thandle.is_column_major()) ||
               (thandle.transpose_spmv() && !thandle.is_column_major()));
          const char *tran = (transpose_spmv ? "T" : "N");
          if (!invert_offdiagonal) {
            // solve with diagonals
            auto digmat = thandle.get_diagblock(lvl);
            KokkosSparse::spmv(space, tran, one, digmat, lhs, one, work);
            // copy from work to lhs corresponding to diagonal blocks
            SparseTriSupernodalSpMVFunctor<LHSType> sptrsv_init_functor(
                -1, node_count, nodes_grouped_by_level, supercols, supercols,
                lhs, work);
            Kokkos::parallel_for(
                "parfor_lsolve_supernode",
                Kokkos::Experimental::require(
                    team_policy(space, lvl_nodes, Kokkos::AUTO),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                sptrsv_init_functor);
          } else {
            // copy lhs corresponding to diagonal blocks to work and zero out in
            // lhs
            SparseTriSupernodalSpMVFunctor<LHSType> sptrsv_init_functor(
                1, node_count, nodes_grouped_by_level, supercols, supercols,
                lhs, work);
            Kokkos::parallel_for(
                "parfor_lsolve_supernode",
                Kokkos::Experimental::require(
                    team_policy(space, lvl_nodes, Kokkos::AUTO),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                sptrsv_init_functor);
          }
          // update off-diagonals (potentiall combined with solve with
          // diagonals)
          auto submat = thandle.get_submatrix(lvl);
          KokkosSparse::spmv(space, tran, one, submat, work, one, lhs);

          // reinitialize workspace
          SparseTriSupernodalSpMVFunctor<LHSType> sptrsv_finalize_functor(
              0, node_count, nodes_grouped_by_level, supercols, supercols, lhs,
              work);
          Kokkos::parallel_for(
              "parfor_lsolve_supernode",
              Kokkos::Experimental::require(
                  team_policy(space, lvl_nodes, Kokkos::AUTO),
                  Kokkos::Experimental::WorkItemProperty::HintLightWeight),
              sptrsv_finalize_functor);

#ifdef profile_supernodal_etree
          Kokkos::fence();
          double time_seconds = timer.seconds();
          std::cout << " > SUPERNODAL LowerTri: " << lvl << " " << time_seconds
                    << " kernel-type: " << kernel_type_host(lvl)
                    << " # of supernodes: " << lvl_nodes << std::endl;
#endif
        }
#endif
        node_count += lvl_nodes;

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
        cudaProfilerStop();
#endif
      }  // end if
    }    // end for lvl

#ifdef profile_supernodal_etree
    Kokkos::fence();
    double sptrsv_time_seconds = sptrsv_timer.seconds();
    std::cout << " + Execution space   : " << execution_space::name()
              << std::endl;
    std::cout << " + Memory space      : " << temp_mem_space::name()
              << std::endl;
    std::cout << " + SpTrsv(lower) time: " << sptrsv_time_seconds << std::endl
              << std::endl;
#endif

  }  // end lower_tri_solve

  template <class RowMapType, class EntriesType, class ValuesType,
            class RHSType, class LHSType>
  static void upper_tri_solve(execution_space &space, TriSolveHandle &thandle,
                              const RowMapType row_map,
                              const EntriesType entries,
                              const ValuesType values, const RHSType &rhs,
                              LHSType &lhs) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
    cudaProfilerStop();
#endif
    using device_t = Kokkos::Device<execution_space, temp_mem_space>;

    auto nlevels = thandle.get_num_levels();
    // Keep this a host View, create device version and copy to back to host
    // during scheduling This requires making sure the host view in the handle
    // is properly updated after the symbolic phase
    auto nodes_per_level  = thandle.get_nodes_per_level();
    auto hnodes_per_level = thandle.get_host_nodes_per_level();
    auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();
    //const auto block_size             = thandle.get_block_size();
    const auto block_enabled          = thandle.is_block_enabled();

    KK_REQUIRE_MSG(!block_enabled, "Block matrices not yet supported");

    // Set up functor types
    using UpperRPPoint = FunctorTypeMacro(TriLvlSchedRPSolverFunctor, false);
    //using LowerRPBlock = FunctorTypeMacro(TriLvlSchedRPSolverFunctor, false, true);
    using UpperTPPoint = FunctorTypeMacro(TriLvlSchedTP1SolverFunctor, false);
    //using LowerTPBlock = FunctorTypeMacro(TriLvlSchedTP1SolverFunctor, false, true);

#if defined(KOKKOSKERNELS_ENABLE_SUPERNODAL_SPTRSV)
    using namespace KokkosSparse::Experimental;
    using integer_view_host_t = typename TriSolveHandle::integer_view_host_t;
    using row_map_host_view_t = Kokkos::View<size_type *, Kokkos::HostSpace>;

    row_map_host_view_t row_map_host;

    const scalar_t zero(0.0);
    const scalar_t one(1.0);

    auto nodes_grouped_by_level_host =
        thandle.get_host_nodes_grouped_by_level();

    if (thandle.get_algorithm() == SPTRSVAlgorithm::SUPERNODAL_NAIVE ||
        thandle.get_algorithm() == SPTRSVAlgorithm::SUPERNODAL_ETREE ||
        thandle.get_algorithm() == SPTRSVAlgorithm::SUPERNODAL_DAG) {
      Kokkos::deep_copy(nodes_grouped_by_level_host, nodes_grouped_by_level);

      row_map_host = row_map_host_view_t(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "host rowmap"),
          row_map.extent(0));
      Kokkos::deep_copy(row_map_host, row_map);
    }

    // supernode sizes
    const int *supercols      = thandle.get_supercols();
    const int *supercols_host = thandle.get_supercols_host();

    // inversion option
    const bool invert_diagonal    = thandle.get_invert_diagonal();
    const bool invert_offdiagonal = thandle.get_invert_offdiagonal();

    // kernel types
    work_view_int_t kernel_type      = thandle.get_kernel_type();
    work_view_int_t diag_kernel_type = thandle.get_diag_kernel_type();

    integer_view_host_t kernel_type_host = thandle.get_kernel_type_host();
    integer_view_host_t diag_kernel_type_host =
        thandle.get_diag_kernel_type_host();

    // workspace
    work_view_int_t work_offset          = thandle.get_work_offset();
    integer_view_host_t work_offset_host = thandle.get_work_offset_host();
    auto work                            = thandle.get_workspace();
#endif

    size_type node_count = 0;

    // This must stay serial; would be nice to try out Cuda's graph stuff to
    // reduce kernel launch overhead
#ifdef profile_supernodal_etree
    Kokkos::Timer sptrsv_timer;
    sptrsv_timer.reset();
#endif
    for (size_type lvl = 0; lvl < nlevels; ++lvl) {
      size_type lvl_nodes = hnodes_per_level(lvl);

      if (lvl_nodes != 0) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
        cudaProfilerStart();
#endif

        if (thandle.get_algorithm() ==
            KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP) {
          UpperRPPoint urpp(row_map, entries, values, lhs, rhs, nodes_grouped_by_level);
          Kokkos::parallel_for(
              "parfor_fixed_lvl",
              Kokkos::Experimental::require(
                  range_policy(space, node_count, node_count + lvl_nodes),
                  Kokkos::Experimental::WorkItemProperty::HintLightWeight),
              urpp);
        } else if (thandle.get_algorithm() ==
                   KokkosSparse::Experimental::SPTRSVAlgorithm::
                       SEQLVLSCHD_TP1) {
          UpperTPPoint utpp(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
          int team_size = thandle.get_team_size();
          auto tp = team_size == -1 ? team_policy(space, lvl_nodes, Kokkos::AUTO) : team_policy(space, lvl_nodes, team_size);
          Kokkos::parallel_for(
            "parfor_u_team",
            Kokkos::Experimental::require(
              tp,
              Kokkos::Experimental::WorkItemProperty::HintLightWeight),
            utpp);
        }
        // TP2 algorithm has issues with some offset-ordinal combo to be
        // addressed
        /*
          else if ( thandle.get_algorithm() ==
          KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHED_TP2 ) {
typedef Kokkos::TeamPolicy<execution_space> tvt_policy_type;

          int team_size = thandle.get_team_size();
          if ( team_size == -1 ) {
          team_size = std::is_same< typename
          Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace
>::value ? 1 : 64;
        }
        int vector_size = thandle.get_team_size();
        if ( vector_size == -1 ) {
          vector_size = std::is_same< typename
Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 4;
        }

        // This impl: "chunk" lvl_nodes into node_groups; a league_rank is
responsible for processing that many nodes
        //       TeamThreadRange over number nodes of node_groups
        //       To avoid masking threads, 1 thread (team) per node in
node_group (thread has full ownership of a node)
        //       ThreadVectorRange responsible for the actual solve computation
        //const int node_groups = team_size;
        const int node_groups = vector_size;

#ifdef KOKKOSKERNELS_SPTRSV_TRILVLSCHED
        TriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType,
LHSType, RHSType> tstf(row_map, entries, values, lhs, rhs,
nodes_grouped_by_level, false, node_count, vector_size, 0); #else
        UpperTriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType,
LHSType, RHSType> tstf(row_map, entries, values, lhs, rhs,
nodes_grouped_by_level, node_count, node_groups); #endif

        Kokkos::parallel_for("parfor_u_team_vector", tvt_policy_type(
(int)std::ceil((float)lvl_nodes/(float)node_groups) , team_size, vector_size ),
tstf); } // end elseif
      */
#if defined(KOKKOSKERNELS_ENABLE_SUPERNODAL_SPTRSV)
        else if (thandle.get_algorithm() == SPTRSVAlgorithm::SUPERNODAL_NAIVE ||
                 thandle.get_algorithm() == SPTRSVAlgorithm::SUPERNODAL_ETREE ||
                 thandle.get_algorithm() == SPTRSVAlgorithm::SUPERNODAL_DAG) {

#ifdef profile_supernodal_etree
          size_t flops = 0;
          Kokkos::Timer timer;
          timer.reset();
#endif

          if (thandle.is_column_major()) {  // U stored in CSC
            if (diag_kernel_type_host(lvl) == 3) {
              // using device-level kernels (functor is called to gather the
              // input into workspace)
              scalar_t *dataU = const_cast<scalar_t *>(values.data());

              if (invert_diagonal && !invert_offdiagonal) {
                // copy diagonals to workspaces
                const int *work_offset_data = work_offset.data();
                SparseTriSupernodalSpMVFunctor<LHSType> sptrsv_init_functor(
                    -2, node_count, nodes_grouped_by_level, supercols,
                    work_offset_data, lhs, work);
                Kokkos::parallel_for(
                    "parfor_tri_supernode_spmv",
                    Kokkos::Experimental::require(
                        team_policy(space, lvl_nodes, Kokkos::AUTO),
                        Kokkos::Experimental::WorkItemProperty::
                            HintLightWeight),
                    sptrsv_init_functor);
              }
              for (size_type league_rank = 0; league_rank < lvl_nodes;
                   league_rank++) {
                auto s = nodes_grouped_by_level_host(node_count + league_rank);

                // supernodal column size
                int j1 = supercols_host[s];
                int j2 = supercols_host[s + 1];
                int nscol =
                    j2 - j1;  // number of columns in the s-th supernode column

                int i1    = row_map_host(j1);
                int i2    = row_map_host(j1 + 1);
                int nsrow = i2 - i1;  // "total" number of rows in all the
                                      // supernodes (diagonal+off-diagonal)
                int nsrow2 = nsrow - nscol;  // "total" number of rows in all
                                             // the off-diagonal supernodes
#ifdef profile_supernodal_etree
                flops += 2 * (nscol * nsrow);
#endif

                // workspace
                int workoffset = work_offset_host(s);

                // create a view for the s-th supernocal block column
                // NOTE: we currently supports only default_layout = LayoutLeft
                Kokkos::View<scalar_t **, default_layout, device_t,
                             Kokkos::MemoryUnmanaged>
                    viewU(&dataU[i1], nsrow, nscol);

                if (invert_offdiagonal) {
                  auto Uij = Kokkos::subview(viewU, range_type(0, nsrow),
                                             Kokkos::ALL());
                  auto Xj  = Kokkos::subview(lhs, range_type(j1, j2));
                  auto Z   = Kokkos::subview(
                      work,
                      range_type(
                          workoffset,
                          workoffset +
                              nsrow));  // needed with gemv for update&scatter
                  KokkosBlas::gemv(space, "N", one, Uij, Xj, zero, Z);
                } else {
                  // extract part of the solution, corresponding to the diagonal
                  // block
                  auto Xj = Kokkos::subview(lhs, range_type(j1, j2));

                  // "triangular-solve" to compute Xj
                  // extract the diagonal block of s-th supernocal column of U
                  auto Ujj = Kokkos::subview(viewU, range_type(0, nscol),
                                             Kokkos::ALL());
                  if (invert_diagonal) {
                    auto Y = Kokkos::subview(
                        work, range_type(
                                  workoffset,
                                  workoffset + nscol));  // needed for gemv
                                                         // instead of trmv/trsv
                    KokkosBlas::gemv(space, "N", one, Ujj, Y, zero, Xj);
                  } else {
                    // NOTE: we currently supports only default_layout =
                    // LayoutLeft
                    Kokkos::View<scalar_t **, default_layout, device_t,
                                 Kokkos::MemoryUnmanaged>
                        Xjj(Xj.data(), nscol, 1);
                    KokkosBlas::trsm(space, "L", "U", "N", "N", one, Ujj, Xjj);
                  }
                  // update off-diagonal blocks
                  if (nsrow2 > 0) {
                    // extract the off-diagonal blocks of s-th supernodal column
                    // of U
                    auto Uij = Kokkos::subview(viewU, range_type(nscol, nsrow),
                                               Kokkos::ALL());
                    auto Z   = Kokkos::subview(
                        work, range_type(workoffset + nscol,
                                         workoffset + nscol +
                                             nsrow2));  // needed with gemv for
                                                          // update&scatter
                    KokkosBlas::gemv(space, "N", one, Uij, Xj, zero, Z);
                  }
                }
              }
              if (invert_offdiagonal) {
                // copy diagonals from workspaces
                const int *work_offset_data = work_offset.data();
                SparseTriSupernodalSpMVFunctor<LHSType> sptrsv_init_functor(
                    -1, node_count, nodes_grouped_by_level, supercols,
                    work_offset_data, lhs, work);
                Kokkos::parallel_for(
                    "parfor_tri_supernode_spmv",
                    Kokkos::Experimental::require(
                        team_policy(space, lvl_nodes, Kokkos::AUTO),
                        Kokkos::Experimental::WorkItemProperty::
                            HintLightWeight),
                    sptrsv_init_functor);
              }
            }

            // launching sparse-triangular solve functor
            UpperTriTranSupernodalFunctor<RowMapType, EntriesType, ValuesType,
                                          LHSType>
                sptrsv_functor(invert_diagonal, invert_offdiagonal, supercols,
                               row_map, entries, values, lvl, kernel_type,
                               diag_kernel_type, lhs, work, work_offset,
                               nodes_grouped_by_level, node_count);

            Kokkos::parallel_for(
                "parfor_usolve_tran_supernode",
                Kokkos::Experimental::require(
                    team_policy(space, lvl_nodes, Kokkos::AUTO),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                sptrsv_functor);
          } else {  // U stored in CSR
            // launching sparse-triangular solve functor
            UpperTriSupernodalFunctor<RowMapType, EntriesType, ValuesType,
                                      LHSType>
                sptrsv_functor(invert_diagonal, supercols, row_map, entries,
                               values, lvl, kernel_type, diag_kernel_type, lhs,
                               work, work_offset, nodes_grouped_by_level,
                               node_count);

            Kokkos::parallel_for(
                "parfor_usolve_supernode",
                Kokkos::Experimental::require(
                    team_policy(space, lvl_nodes, Kokkos::AUTO),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                sptrsv_functor);

            if (diag_kernel_type_host(lvl) == 3) {
              // using device-level kernels (functor is called to gather the
              // input into workspace)
              scalar_t *dataU = const_cast<scalar_t *>(values.data());

              for (size_type league_rank = 0; league_rank < lvl_nodes;
                   league_rank++) {
                auto s = nodes_grouped_by_level_host(node_count + league_rank);

                // supernodal column size
                int j1 = supercols_host[s];
                int j2 = supercols_host[s + 1];
                int nscol =
                    j2 - j1;  // number of columns in the s-th supernode column

                // "total" number of rows in all the supernodes
                // (diagonal+off-diagonal)
                int i1    = row_map_host(j1);
                int i2    = row_map_host(j1 + 1);
                int nsrow = i2 - i1;
                // "total" number of rows in all the off-diagonal supernodes
                int nsrow2 = nsrow - nscol;

                // workspace
                int workoffset = work_offset_host(s);

                // create a view for the s-th supernocal block column
                // NOTE: we currently supports only default_layout = LayoutLeft
                Kokkos::View<scalar_t **, default_layout, device_t,
                             Kokkos::MemoryUnmanaged>
                    viewU(&dataU[i1], nsrow, nscol);

                // extract part of the solution, corresponding to the diagonal
                // block
                auto Xj = Kokkos::subview(lhs, range_type(j1, j2));
                auto Y  = Kokkos::subview(
                    work,
                    range_type(
                        workoffset,
                        workoffset +
                            nscol));  // needed for gemv instead of trmv/trsv

                // update with off-diagonal blocks
                if (nsrow2 > 0) {
                  // extract the off-diagonal blocks of s-th supernodal column
                  // of
                  // U
                  auto Uij = Kokkos::subview(viewU, range_type(nscol, nsrow),
                                             Kokkos::ALL());
                  auto Z   = Kokkos::subview(
                      work,
                      range_type(
                          workoffset + nscol,
                          workoffset + nscol +
                              nsrow2));  // needed with gemv for update&scatter
                  KokkosBlas::gemv(space, "T", -one, Uij, Z, one, Xj);
                }

                // "triangular-solve" to compute Xj
                // extract the diagonal block of s-th supernocal column of U
                auto Ujj =
                    Kokkos::subview(viewU, range_type(0, nscol), Kokkos::ALL());
                if (invert_diagonal) {
                  KokkosBlas::gemv(space, "T", one, Ujj, Xj, zero, Y);
                } else {
                  // NOTE: we currently supports only default_layout =
                  // LayoutLeft
                  Kokkos::View<scalar_t **, default_layout, device_t,
                               Kokkos::MemoryUnmanaged>
                      Xjj(Xj.data(), nscol, 1);
                  KokkosBlas::trsm(space, "L", "L", "T", "N", one, Ujj, Xjj);
                }
              }
              if (invert_diagonal) {
                // copy diagonals from workspaces
                const int *work_offset_data = work_offset.data();
                SparseTriSupernodalSpMVFunctor<LHSType> sptrsv_init_functor(
                    -1, node_count, nodes_grouped_by_level, supercols,
                    work_offset_data, lhs, work);
                Kokkos::parallel_for(
                    "parfor_tri_supernode_spmv",
                    Kokkos::Experimental::require(
                        team_policy(space, lvl_nodes, Kokkos::AUTO),
                        Kokkos::Experimental::WorkItemProperty::
                            HintLightWeight),
                    sptrsv_init_functor);
              }
            }
          }
#ifdef profile_supernodal_etree
          Kokkos::fence();
          double time_seconds = timer.seconds();
          std::cout << " > SUPERNODAL UpperTri: " << lvl << " " << time_seconds
                    << " flop count: " << flops
                    << " kernel-type: " << kernel_type_host(lvl)
                    << " # of supernodes: " << lvl_nodes << std::endl;
#endif
        } else if (thandle.get_algorithm() ==
                       SPTRSVAlgorithm::SUPERNODAL_SPMV ||
                   thandle.get_algorithm() ==
                       SPTRSVAlgorithm::SUPERNODAL_SPMV_DAG) {
#ifdef profile_supernodal_etree
          Kokkos::Timer timer;
          timer.reset();
#endif

          // initialize input & output vectors

          // update with one, or two, spmv
          bool transpose_spmv =
              ((!thandle.transpose_spmv() && thandle.is_column_major()) ||
               (thandle.transpose_spmv() && !thandle.is_column_major()));
          const char *tran = (transpose_spmv ? "T" : "N");
          if (!transpose_spmv) {  // U stored in CSR
            if (!invert_offdiagonal) {
              // solve with diagonals
              auto digmat = thandle.get_diagblock(lvl);
              KokkosSparse::spmv(space, tran, one, digmat, lhs, one, work);
              // copy from work to lhs corresponding to diagonal blocks
              SparseTriSupernodalSpMVFunctor<LHSType> sptrsv_init_functor(
                  -1, node_count, nodes_grouped_by_level, supercols, supercols,
                  lhs, work);
              Kokkos::parallel_for(
                  "parfor_lsolve_supernode",
                  Kokkos::Experimental::require(
                      team_policy(space, lvl_nodes, Kokkos::AUTO),
                      Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                  sptrsv_init_functor);
            } else {
              // zero out lhs corresponding to diagonal blocks in lhs, and copy
              // to work
              SparseTriSupernodalSpMVFunctor<LHSType> sptrsv_init_functor(
                  1, node_count, nodes_grouped_by_level, supercols, supercols,
                  lhs, work);
              Kokkos::parallel_for(
                  "parfor_lsolve_supernode",
                  Kokkos::Experimental::require(
                      team_policy(space, lvl_nodes, Kokkos::AUTO),
                      Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                  sptrsv_init_functor);
            }
            // update with off-diagonals (potentiall combined with diagonal
            // solves)
            auto submat = thandle.get_submatrix(lvl);
            KokkosSparse::spmv(space, tran, one, submat, work, one, lhs);
          } else {
            if (!invert_offdiagonal) {
              // zero out lhs corresponding to diagonal blocks in lhs, and copy
              // to work
              SparseTriSupernodalSpMVFunctor<LHSType> sptrsv_init_functor(
                  1, node_count, nodes_grouped_by_level, supercols, supercols,
                  lhs, work);
              Kokkos::parallel_for(
                  "parfor_lsolve_supernode",
                  Kokkos::Experimental::require(
                      team_policy(space, lvl_nodes, Kokkos::AUTO),
                      Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                  sptrsv_init_functor);

              // update with off-diagonals
              auto submat = thandle.get_submatrix(lvl);
              KokkosSparse::spmv(space, tran, one, submat, lhs, one, work);

              // solve with diagonals
              auto digmat = thandle.get_diagblock(lvl);
              KokkosSparse::spmv(space, tran, one, digmat, work, one, lhs);
            } else {
              std::cout << " ** invert_offdiag with U in CSR not supported **"
                        << std::endl;
            }
          }
          // reinitialize workspace
          SparseTriSupernodalSpMVFunctor<LHSType> sptrsv_finalize_functor(
              0, node_count, nodes_grouped_by_level, supercols, supercols, lhs,
              work);
          Kokkos::parallel_for(
              "parfor_lsolve_supernode",
              Kokkos::Experimental::require(
                  team_policy(space, lvl_nodes, Kokkos::AUTO),
                  Kokkos::Experimental::WorkItemProperty::HintLightWeight),
              sptrsv_finalize_functor);

#ifdef profile_supernodal_etree
          Kokkos::fence();
          double time_seconds = timer.seconds();
          std::cout << " > SUPERNODAL UpperTri: " << lvl << " " << time_seconds
                    << " kernel-type: " << kernel_type_host(lvl)
                    << " # of supernodes: " << lvl_nodes << std::endl;
#endif
        }
#endif
        node_count += lvl_nodes;

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
        cudaProfilerStop();
#endif
      }  // end if
    }    // end for lvl
#ifdef profile_supernodal_etree
    Kokkos::fence();
    double sptrsv_time_seconds = sptrsv_timer.seconds();
    std::cout << " + SpTrsv(uppper) time: " << sptrsv_time_seconds << std::endl
              << std::endl;
    std::cout << "  + Execution space    : " << execution_space::name()
              << std::endl;
    std::cout << " + Memory space       : " << temp_mem_space::name()
              << std::endl;
#endif

  }  // end upper_tri_solve

  template <class RowMapType, class EntriesType, class ValuesType,
            class RHSType, class LHSType>
  static void tri_solve_chain(execution_space &space, TriSolveHandle &thandle,
                              const RowMapType row_map,
                              const EntriesType entries,
                              const ValuesType values, const RHSType &rhs,
                              LHSType &lhs, const bool /*is_lowertri_*/) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
    cudaProfilerStop();
#endif
    // Algorithm is checked before this function is called
    auto h_chain_ptr            = thandle.get_host_chain_ptr();
    size_type num_chain_entries = thandle.get_num_chain_entries();

    // Keep this a host View, create device version and copy to back to host
    // during scheduling This requires making sure the host view in the handle
    // is properly updated after the symbolic phase
    auto nodes_per_level  = thandle.get_nodes_per_level();
    auto hnodes_per_level = thandle.get_host_nodes_per_level();

    auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

    const bool is_lowertri = thandle.is_lower_tri();

    size_type node_count = 0;

    // REFACTORED to cleanup; next, need debug and timer routines
    using large_cutoff_policy_type =
        Kokkos::TeamPolicy<LargerCutoffTag, execution_space>;
    using SingleBlockFunctor =
        TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType,
                                         LHSType, RHSType>;

    int team_size = thandle.get_team_size();
    int vector_size =
        thandle.get_vector_size() > 0 ? thandle.get_vector_size() : 1;

    auto cutoff               = thandle.get_chain_threshold();
    int team_size_singleblock = team_size;

    // Enumerate options
    // ts -1,0 | cu 0 - select default ts == 1
    // ts -1,0 | cu > 0 - select default ts; restriction: ts <= tsmax (auto)
    // ts > 0 | cu 0 - set
    // ts > 0 | cu > 0 - set
    // Controls ts,cu > 0
    // co > ts  - not all rows can be mapped to a thread - must call
    // largercutoff impl co <= ts - okay, kernel must be careful not to access
    // out-of-bounds; some threads idol
    if (team_size_singleblock <= 0 && cutoff == 0) {
      team_size_singleblock = 1;
      // If cutoff == 0, no single-block calls will be made,
      // team_size_singleblock is unimportant
    }

    // This is only necessary for Lower,UpperTri functor versions; else,
    // is_lowertri can be passed as arg to the generic Tri functor...
    if (is_lowertri) {
      for (size_type chainlink = 0; chainlink < num_chain_entries;
           ++chainlink) {
        size_type schain = h_chain_ptr(chainlink);
        size_type echain = h_chain_ptr(chainlink + 1);

        if (echain - schain == 1) {
          // if team_size is -1 (unset), get recommended size from Kokkos
          TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType,
                                      LHSType, RHSType, true>
              tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level,
                   node_count);
          if (team_size == -1) {
            team_size =
                team_policy(space, 1, 1, vector_size)
                    .team_size_recommended(tstf, Kokkos::ParallelForTag());
          }

          size_type lvl_nodes = hnodes_per_level(schain);  // lvl == echain????
          Kokkos::parallel_for(
              "parfor_l_team_chain1",
              Kokkos::Experimental::require(
                  team_policy(space, lvl_nodes, team_size, vector_size),
                  Kokkos::Experimental::WorkItemProperty::HintLightWeight),
              tstf);
          node_count += lvl_nodes;

        } else {
          size_type lvl_nodes = 0;

          for (size_type i = schain; i < echain; ++i) {
            lvl_nodes += hnodes_per_level(i);
          }

          if (team_size_singleblock <= 0) {
            team_size_singleblock =
                team_policy(space, 1, 1, vector_size)
                    .team_size_recommended(
                        SingleBlockFunctor(row_map, entries, values, lhs, rhs,
                                           nodes_grouped_by_level,
                                           nodes_per_level, node_count, schain,
                                           echain, is_lowertri),
                        Kokkos::ParallelForTag());
          }

          if (cutoff <= team_size_singleblock) {
#ifdef KOKKOSKERNELS_SPTRSV_TRILVLSCHED
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType,
                                             ValuesType, LHSType, RHSType>
                tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level,
                     nodes_per_level, node_count, schain, echain, true);
#else
            LowerTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType,
                                                  ValuesType, LHSType, RHSType>
                tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level,
                     nodes_per_level, node_count, schain, echain);
#endif
            Kokkos::parallel_for(
                "parfor_l_team_chainmulti",
                Kokkos::Experimental::require(
                    team_policy(space, 1, team_size_singleblock, vector_size),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                tstf);
          } else {
            // team_size_singleblock < cutoff => kernel must allow for a
            // block-stride internally
#ifdef KOKKOSKERNELS_SPTRSV_TRILVLSCHED
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType,
                                             ValuesType, LHSType, RHSType>
                tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level,
                     nodes_per_level, node_count, schain, echain, true, 0,
                     cutoff);
#else
            LowerTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType,
                                                  ValuesType, LHSType, RHSType>
                tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level,
                     nodes_per_level, node_count, schain, echain, cutoff);
#endif
            Kokkos::parallel_for(
                "parfor_l_team_chainmulti_cutoff",
                Kokkos::Experimental::require(
                    large_cutoff_policy_type(1, team_size_singleblock,
                                             vector_size),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                tstf);
          }
          node_count += lvl_nodes;
        }
        // TODO: space.fence()
        Kokkos::fence();  // TODO - is this necessary? that is, can the
                          // parallel_for launch before the s/echain values have
                          // been updated?
      }

    } else {
      for (size_type chainlink = 0; chainlink < num_chain_entries;
           ++chainlink) {
        size_type schain = h_chain_ptr(chainlink);
        size_type echain = h_chain_ptr(chainlink + 1);

        if (echain - schain == 1) {
          // if team_size is -1 (unset), get recommended size from Kokkos
          TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType,
                                      LHSType, RHSType, false>
              tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level,
                   node_count);
          if (team_size == -1) {
            team_size =
                team_policy(space, 1, 1, vector_size)
                    .team_size_recommended(tstf, Kokkos::ParallelForTag());
          }

          // TODO To use cudagraph here, need to know how many non-unit chains
          // there are, create a graph for each and launch accordingly
          size_type lvl_nodes = hnodes_per_level(schain);  // lvl == echain????
          Kokkos::parallel_for(
              "parfor_u_team_chain1",
              Kokkos::Experimental::require(
                  team_policy(space, lvl_nodes, team_size, vector_size),
                  Kokkos::Experimental::WorkItemProperty::HintLightWeight),
              tstf);
          node_count += lvl_nodes;

        } else {
          size_type lvl_nodes = 0;

          for (size_type i = schain; i < echain; ++i) {
            lvl_nodes += hnodes_per_level(i);
          }

          if (team_size_singleblock <= 0) {
            // team_size_singleblock = team_policy(1, 1,
            // 1).team_size_recommended(SingleBlockFunctor(row_map, entries,
            // values, lhs, rhs, nodes_grouped_by_level, is_lowertri,
            // node_count), Kokkos::ParallelForTag());
            team_size_singleblock =
                team_policy(space, 1, 1, vector_size)
                    .team_size_recommended(
                        SingleBlockFunctor(row_map, entries, values, lhs, rhs,
                                           nodes_grouped_by_level,
                                           nodes_per_level, node_count, schain,
                                           echain, is_lowertri),
                        Kokkos::ParallelForTag());
          }

          if (cutoff <= team_size_singleblock) {
#ifdef KOKKOSKERNELS_SPTRSV_TRILVLSCHED
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType,
                                             ValuesType, LHSType, RHSType>
                tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level,
                     nodes_per_level, node_count, schain, echain, is_lowertri);
#else
            UpperTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType,
                                                  ValuesType, LHSType, RHSType>
                tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level,
                     nodes_per_level, node_count, schain, echain);
#endif
            Kokkos::parallel_for(
                "parfor_u_team_chainmulti",
                Kokkos::Experimental::require(
                    team_policy(space, 1, team_size_singleblock, vector_size),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                tstf);
          } else {
            // team_size_singleblock < cutoff => kernel must allow for a
            // block-stride internally
#ifdef KOKKOSKERNELS_SPTRSV_TRILVLSCHED
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType,
                                             ValuesType, LHSType, RHSType>
                tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level,
                     nodes_per_level, node_count, schain, echain, is_lowertri,
                     0, cutoff);
#else
            UpperTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType,
                                                  ValuesType, LHSType, RHSType>
                tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level,
                     nodes_per_level, node_count, schain, echain, cutoff);
#endif
            Kokkos::parallel_for(
                "parfor_u_team_chainmulti_cutoff",
                Kokkos::Experimental::require(
                    large_cutoff_policy_type(1, team_size_singleblock,
                                             vector_size),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                tstf);
          }
          node_count += lvl_nodes;
        }
        // TODO: space.fence()
        Kokkos::fence();  // TODO - is this necessary? that is, can the
                          // parallel_for launch before the s/echain values have
                          // been updated?
      }
    }
  }  // end tri_solve_chain

  // --------------------------------
  // Stream interfaces
  // --------------------------------
  template <class RowMapType, class EntriesType, class ValuesType,
            class RHSType, class LHSType>
  static void lower_tri_solve_streams(
      const std::vector<execution_space> &execspace_v,
      const std::vector<TriSolveHandle *> &thandle_v,
      const std::vector<RowMapType> &row_map_v,
      const std::vector<EntriesType> &entries_v,
      const std::vector<ValuesType> &values_v,
      const std::vector<RHSType> &rhs_v, std::vector<LHSType> &lhs_v) {
    // NOTE: Only support SEQLVLSCHD_RP and SEQLVLSCHD_TP1 at this moment
    using nodes_per_level_type =
        typename TriSolveHandle::hostspace_nnz_lno_view_t;
    using nodes_grouped_by_level_type = typename TriSolveHandle::nnz_lno_view_t;
    using LowerRPPoint = FunctorTypeMacro(TriLvlSchedRPSolverFunctor, true);
    using LowerTPPoint = FunctorTypeMacro(TriLvlSchedTP1SolverFunctor, true);

    // Create vectors for handles' data in streams
    int nstreams = execspace_v.size();
    std::vector<size_type> nlevels_v(nstreams);
    std::vector<nodes_per_level_type> hnodes_per_level_v(nstreams);
    std::vector<nodes_grouped_by_level_type> nodes_grouped_by_level_v(nstreams);
    std::vector<size_type> node_count_v(nstreams);

    // Retrieve data from handles and find max. number of levels among streams
    size_type nlevels_max = 0;
    for (int i = 0; i < nstreams; i++) {
      nlevels_v[i]                = thandle_v[i]->get_num_levels();
      hnodes_per_level_v[i]       = thandle_v[i]->get_host_nodes_per_level();
      nodes_grouped_by_level_v[i] = thandle_v[i]->get_nodes_grouped_by_level();
      node_count_v[i]             = 0;
      if (nlevels_max < nlevels_v[i]) nlevels_max = nlevels_v[i];
    }

    // Main loop must be performed sequential
    for (size_type lvl = 0; lvl < nlevels_max; lvl++) {
      // 1. Launch work on all streams
      for (int i = 0; i < nstreams; i++) {
        // Only if stream i-th still has this level
        if (lvl < nlevels_v[i]) {
          size_type lvl_nodes = hnodes_per_level_v[i](lvl);
          if (lvl_nodes != 0) {
            if (thandle_v[i]->get_algorithm() ==
                KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP) {
              Kokkos::parallel_for(
                  "parfor_fixed_lvl",
                  range_policy(execspace_v[i], node_count_v[i],
                               node_count_v[i] + lvl_nodes),
                  LowerRPPoint(
                      row_map_v[i], entries_v[i], values_v[i], lhs_v[i],
                      rhs_v[i], nodes_grouped_by_level_v[i]));
            } else if (thandle_v[i]->get_algorithm() ==
                       KokkosSparse::Experimental::SPTRSVAlgorithm::
                           SEQLVLSCHD_TP1) {
              int team_size = thandle_v[i]->get_team_size();
              auto tp = team_size == -1 ? team_policy(execspace_v[i], lvl_nodes, Kokkos::AUTO) : team_policy(execspace_v[i], lvl_nodes, team_size);
              LowerTPPoint
                  tstf(row_map_v[i], entries_v[i], values_v[i], lhs_v[i],
                       rhs_v[i], nodes_grouped_by_level_v[i], node_count_v[i]);
              Kokkos::parallel_for("parfor_l_team", tp, tstf);
            }
            node_count_v[i] += lvl_nodes;
          }  // end if (lvl_nodes != 0)
        }    // end if (lvl < nlevels_v[i])
      }      // end for streams
    }        // end for lvl
  }          // end lower_tri_solve_streams

  template <class RowMapType, class EntriesType, class ValuesType,
            class RHSType, class LHSType>
  static void upper_tri_solve_streams(
      const std::vector<execution_space> &execspace_v,
      const std::vector<TriSolveHandle *> &thandle_v,
      const std::vector<RowMapType> &row_map_v,
      const std::vector<EntriesType> &entries_v,
      const std::vector<ValuesType> &values_v,
      const std::vector<RHSType> &rhs_v, std::vector<LHSType> &lhs_v) {
    // NOTE: Only support SEQLVLSCHD_RP and SEQLVLSCHD_TP1 at this moment
    using nodes_per_level_type =
        typename TriSolveHandle::hostspace_nnz_lno_view_t;
    using nodes_grouped_by_level_type = typename TriSolveHandle::nnz_lno_view_t;
    using UpperRPPoint = FunctorTypeMacro(TriLvlSchedRPSolverFunctor, false);
    using UpperTPPoint = FunctorTypeMacro(TriLvlSchedTP1SolverFunctor, false);

    // Create vectors for handles' data in streams
    int nstreams = execspace_v.size();
    std::vector<size_type> nlevels_v(nstreams);
    std::vector<nodes_per_level_type> hnodes_per_level_v(nstreams);
    std::vector<nodes_grouped_by_level_type> nodes_grouped_by_level_v(nstreams);
    std::vector<size_type> node_count_v(nstreams);

    // Retrieve data from handles and find max. number of levels among streams
    size_type nlevels_max = 0;
    for (int i = 0; i < nstreams; i++) {
      nlevels_v[i]                = thandle_v[i]->get_num_levels();
      hnodes_per_level_v[i]       = thandle_v[i]->get_host_nodes_per_level();
      nodes_grouped_by_level_v[i] = thandle_v[i]->get_nodes_grouped_by_level();
      node_count_v[i]             = 0;
      if (nlevels_max < nlevels_v[i]) nlevels_max = nlevels_v[i];
    }

    // Main loop must be performed sequential
    for (size_type lvl = 0; lvl < nlevels_max; lvl++) {
      // 1. Launch work on all streams
      for (int i = 0; i < nstreams; i++) {
        // Only if stream i-th still has this level
        if (lvl < nlevels_v[i]) {
          size_type lvl_nodes = hnodes_per_level_v[i](lvl);
          if (lvl_nodes != 0) {
            if (thandle_v[i]->get_algorithm() ==
                KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP) {
              Kokkos::parallel_for(
                  "parfor_fixed_lvl",
                  range_policy(execspace_v[i], node_count_v[i],
                               node_count_v[i] + lvl_nodes),
                  UpperRPPoint(
                      row_map_v[i], entries_v[i], values_v[i], lhs_v[i],
                      rhs_v[i], nodes_grouped_by_level_v[i]));
            } else if (thandle_v[i]->get_algorithm() ==
                       KokkosSparse::Experimental::SPTRSVAlgorithm::
                           SEQLVLSCHD_TP1) {
              int team_size = thandle_v[i]->get_team_size();
              auto tp = team_size == -1 ? team_policy(execspace_v[i], lvl_nodes, Kokkos::AUTO) : team_policy(execspace_v[i], lvl_nodes, team_size);
              UpperTPPoint
                  tstf(row_map_v[i], entries_v[i], values_v[i], lhs_v[i],
                       rhs_v[i], nodes_grouped_by_level_v[i], node_count_v[i]);
              Kokkos::parallel_for("parfor_l_team", tp, tstf);
            }
            node_count_v[i] += lvl_nodes;
          }  // end if (lvl_nodes != 0)
        }    // end if (lvl < nlevels_v[i])
      }      // end for streams
    }        // end for lvl
  }          // end upper_tri_solve_streams

};  // struct SptrsvWrap

}  // namespace Experimental
}  // namespace Impl
}  // namespace KokkosSparse

#endif
