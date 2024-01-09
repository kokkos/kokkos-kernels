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

#ifndef KOKKOSSPARSE_IMPL_SPILUK_NUMERIC_HPP_
#define KOKKOSSPARSE_IMPL_SPILUK_NUMERIC_HPP_

/// \file KokkosSparse_spiluk_numeric_impl.hpp
/// \brief Implementation(s) of the numeric phase of sparse ILU(k).

#include <KokkosKernels_config.h>
#include <KokkosKernels_Error.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_spiluk_handle.hpp>
#include "KokkosBatched_SetIdentity_Decl.hpp"
#include "KokkosBatched_SetIdentity_Impl.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Serial_Impl.hpp"
#include "KokkosBatched_Axpy.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBatched_Gemm_Serial_Impl.hpp"
#include "KokkosBlas1_set.hpp"

//#define NUMERIC_OUTPUT_INFO

namespace KokkosSparse {
namespace Impl {
namespace Experimental {

template <class IlukHandle>
struct IlukWrap {
  //
  // Useful types
  //
  using execution_space        = typename IlukHandle::execution_space;
  using memory_space           = typename IlukHandle::memory_space;
  using lno_t                  = typename IlukHandle::nnz_lno_t;
  using size_type              = typename IlukHandle::size_type;
  using scalar_t               = typename IlukHandle::nnz_scalar_t;
  using HandleDeviceRowMapType = typename IlukHandle::nnz_row_view_t;
  using HandleDeviceValueType  = typename IlukHandle::nnz_value_view_t;
  using WorkViewType           = typename IlukHandle::work_view_t;
  using LevelHostViewType      = typename IlukHandle::nnz_lno_view_host_t;
  using LevelViewType          = typename IlukHandle::nnz_lno_view_t;
  using karith                 = typename Kokkos::ArithTraits<scalar_t>;
  using team_policy            = typename IlukHandle::TeamPolicy;
  using member_type            = typename team_policy::member_type;
  using range_policy           = typename IlukHandle::RangePolicy;
  using sview_1d = typename Kokkos::View<scalar_t *, memory_space>;

  static team_policy get_team_policy(const size_type nrows,
                                     const int team_size) {
    team_policy rv;
    if (team_size == -1) {
      rv = team_policy(nrows, Kokkos::AUTO);
    } else {
      rv = team_policy(nrows, team_size);
    }
    return rv;
  }

  static team_policy get_team_policy(execution_space exe_space,
                                     const size_type nrows,
                                     const int team_size) {
    team_policy rv;
    if (team_size == -1) {
      rv = team_policy(exe_space, nrows, Kokkos::AUTO);
    } else {
      rv = team_policy(exe_space, nrows, team_size);
    }
    return rv;
  }

  static range_policy get_range_policy(const lno_t start, const lno_t end) {
    range_policy rv(start, end);
    return rv;
  }

  static range_policy get_range_policy(execution_space exe_space,
                                       const lno_t start, const lno_t end) {
    range_policy rv(exe_space, start, end);
    return rv;
  }

  /**
   * Common base class for SPILUK functors. Default version does not support
   * blocks
   */
  template <class ARowMapType, class AEntriesType, class AValuesType,
            class LRowMapType, class LEntriesType, class LValuesType,
            class URowMapType, class UEntriesType, class UValuesType,
            bool BlockEnabled>
  struct Common {
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
    lno_t lev_start;

    // unblocked does not require any buffer
    static constexpr size_type BUFF_SIZE = 1;

    Common(const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
           const AValuesType &A_values_, const LRowMapType &L_row_map_,
           const LEntriesType &L_entries_, LValuesType &L_values_,
           const URowMapType &U_row_map_, const UEntriesType &U_entries_,
           UValuesType &U_values_, const LevelViewType &level_idx_,
           WorkViewType &iw_, const lno_t &lev_start_,
           const size_type &block_size_)
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
          lev_start(lev_start_) {
      KK_REQUIRE_MSG(block_size_ == 0,
                     "Tried to use blocks with the unblocked Common?");
    }

    // lset
    KOKKOS_INLINE_FUNCTION
    void lset(const size_type nnz, const scalar_t &value) const {
      L_values(nnz) = value;
    }

    // uset
    KOKKOS_INLINE_FUNCTION
    void uset(const size_type nnz, const scalar_t &value) const {
      U_values(nnz) = value;
    }

    // lset_id
    KOKKOS_INLINE_FUNCTION
    void lset_id(const size_type nnz) const { L_values(nnz) = scalar_t(1.0); }

    KOKKOS_INLINE_FUNCTION
    void lset_id(const member_type &team, const size_type nnz) const {
      // Not sure a Kokkos::single is really needed here since the
      // race is harmless
      Kokkos::single(Kokkos::PerTeam(team),
                     [&]() { L_values(nnz) = scalar_t(1.0); });
    }

    // divide. lhs /= rhs
    KOKKOS_INLINE_FUNCTION
    void divide(scalar_t &lhs, const scalar_t &rhs) const { lhs /= rhs; }

    KOKKOS_INLINE_FUNCTION
    void divide(const member_type &team, scalar_t &lhs,
                const scalar_t &rhs) const {
      Kokkos::single(Kokkos::PerTeam(team), [&]() { lhs /= rhs; });
      team.team_barrier();
    }

    // add. lhs += rhs
    KOKKOS_INLINE_FUNCTION
    void add(scalar_t &lhs, const scalar_t &rhs) const { lhs += rhs; }

    // multiply: return (alpha * lhs) * rhs
    KOKKOS_INLINE_FUNCTION
    scalar_t multiply(const scalar_t &alpha, const scalar_t &lhs,
                      const scalar_t &rhs, scalar_t *) const {
      return alpha * lhs * rhs;
    }

    // lget
    KOKKOS_INLINE_FUNCTION
    scalar_t &lget(const size_type nnz) const { return L_values(nnz); }

    // uget
    KOKKOS_INLINE_FUNCTION
    scalar_t &uget(const size_type nnz) const { return U_values(nnz); }

    // aget
    KOKKOS_INLINE_FUNCTION
    scalar_t aget(const size_type nnz) const { return A_values(nnz); }

    // uequal
    KOKKOS_INLINE_FUNCTION
    bool uequal(const size_type nnz, const scalar_t &value) const {
      return U_values(nnz) == value;
    }
  };

  // Partial specialization for block support
  template <class ARowMapType, class AEntriesType, class AValuesType,
            class LRowMapType, class LEntriesType, class LValuesType,
            class URowMapType, class UEntriesType, class UValuesType>
  struct Common<ARowMapType, AEntriesType, AValuesType, LRowMapType,
                LEntriesType, LValuesType, URowMapType, UEntriesType,
                UValuesType, true> {
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
    lno_t lev_start;
    size_type block_size;
    size_type block_items;
    sview_1d ones;

    // blocked requires a buffer to store gemm output
    static constexpr size_type BUFF_SIZE = 128;

    using LValuesUnmanaged2DBlockType = Kokkos::View<
        typename LValuesType::value_type **,
        typename KokkosKernels::Impl::GetUnifiedLayout<
            LValuesType>::array_layout,
        typename LValuesType::device_type,
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >;

    using UValuesUnmanaged2DBlockType = Kokkos::View<
        typename UValuesType::value_type **,
        typename KokkosKernels::Impl::GetUnifiedLayout<
            UValuesType>::array_layout,
        typename UValuesType::device_type,
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >;

    using AValuesUnmanaged2DBlockType = Kokkos::View<
        typename AValuesType::value_type **,
        typename KokkosKernels::Impl::GetUnifiedLayout<
            AValuesType>::array_layout,
        typename AValuesType::device_type,
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >;

    Common(const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
           const AValuesType &A_values_, const LRowMapType &L_row_map_,
           const LEntriesType &L_entries_, LValuesType &L_values_,
           const URowMapType &U_row_map_, const UEntriesType &U_entries_,
           UValuesType &U_values_, const LevelViewType &level_idx_,
           WorkViewType &iw_, const lno_t &lev_start_,
           const size_type &block_size_)
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
          lev_start(lev_start_),
          block_size(block_size_),
          block_items(block_size * block_size),
          ones("ones", block_size) {
      Kokkos::deep_copy(ones, 1.0);
      KK_REQUIRE_MSG(block_size > 0,
                     "Tried to use block_size=0 with the blocked Common?");
      KK_REQUIRE_MSG(block_size <= 11, "Max supported block size is 11");
    }

    // lset
    KOKKOS_INLINE_FUNCTION
    void lset(const size_type block, const scalar_t &value) const {
      KokkosBlas::SerialSet::invoke(value, lget(block));
    }

    KOKKOS_INLINE_FUNCTION
    void lset(const size_type block,
              const AValuesUnmanaged2DBlockType &rhs) const {
      auto lblock = lget(block);
      for (size_type i = 0; i < block_size; ++i) {
        for (size_type j = 0; j < block_size; ++j) {
          lblock(i, j) = rhs(i, j);
        }
      }
    }

    // uset
    KOKKOS_INLINE_FUNCTION
    void uset(const size_type block, const scalar_t &value) const {
      KokkosBlas::SerialSet::invoke(value, uget(block));
    }

    KOKKOS_INLINE_FUNCTION
    void uset(const size_type block,
              const AValuesUnmanaged2DBlockType &rhs) const {
      auto ublock = uget(block);
      for (size_type i = 0; i < block_size; ++i) {
        for (size_type j = 0; j < block_size; ++j) {
          ublock(i, j) = rhs(i, j);
        }
      }
    }

    // lset_id
    KOKKOS_INLINE_FUNCTION
    void lset_id(const size_type block) const {
      KokkosBatched::SerialSetIdentity::invoke(lget(block));
    }

    KOKKOS_INLINE_FUNCTION
    void lset_id(const member_type &team, const size_type block) const {
      KokkosBatched::TeamSetIdentity<member_type>::invoke(team, lget(block));
    }

    // divide. lhs /= rhs
    KOKKOS_INLINE_FUNCTION
    void divide(const LValuesUnmanaged2DBlockType &lhs,
                const UValuesUnmanaged2DBlockType &rhs) const {
      KokkosBatched::SerialTrsm<
          KokkosBatched::Side::Right, KokkosBatched::Uplo::Upper,
          KokkosBatched::Trans::NoTranspose,  // not 100% on this
          KokkosBatched::Diag::NonUnit,
          KokkosBatched::Algo::Trsm::Unblocked>::  // not 100% on this
          invoke<scalar_t>(1.0, rhs, lhs);
    }

    KOKKOS_INLINE_FUNCTION
    void divide(const member_type &team, const LValuesUnmanaged2DBlockType &lhs,
                const UValuesUnmanaged2DBlockType &rhs) const {
      KokkosBatched::TeamTrsm<
          member_type, KokkosBatched::Side::Right, KokkosBatched::Uplo::Upper,
          KokkosBatched::Trans::NoTranspose,  // not 100% on this
          KokkosBatched::Diag::NonUnit,
          KokkosBatched::Algo::Trsm::Unblocked>::  // not 100% on this
          invoke(team, 1.0, rhs, lhs);
    }

    // add. lhs += rhs
    template <typename Lview, typename Rview>
    KOKKOS_INLINE_FUNCTION void add(Lview lhs, const Rview &rhs) const {
      KokkosBatched::SerialAxpy::invoke(ones, rhs, lhs);
    }

    // multiply: return (alpha * lhs) * rhs
    KOKKOS_INLINE_FUNCTION
    LValuesUnmanaged2DBlockType multiply(const scalar_t &alpha,
                                         const UValuesUnmanaged2DBlockType &lhs,
                                         const LValuesUnmanaged2DBlockType &rhs,
                                         scalar_t *buff) const {
      LValuesUnmanaged2DBlockType result(&buff[0], block_size, block_size);
      KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Gemm::Unblocked>::
          invoke<scalar_t, LValuesUnmanaged2DBlockType,
                 UValuesUnmanaged2DBlockType, LValuesUnmanaged2DBlockType>(
              alpha, lhs, rhs, 0.0, result);
      return result;
    }

    // lget
    KOKKOS_INLINE_FUNCTION
    LValuesUnmanaged2DBlockType lget(const size_type block) const {
      return LValuesUnmanaged2DBlockType(
          L_values.data() + (block * block_items), block_size, block_size);
    }

    // uget
    KOKKOS_INLINE_FUNCTION
    UValuesUnmanaged2DBlockType uget(const size_type block) const {
      return UValuesUnmanaged2DBlockType(
          U_values.data() + (block * block_items), block_size, block_size);
    }

    // aget
    KOKKOS_INLINE_FUNCTION
    AValuesUnmanaged2DBlockType aget(const size_type block) const {
      return AValuesUnmanaged2DBlockType(
          A_values.data() + (block * block_items), block_size, block_size);
    }

    // uequal
    KOKKOS_INLINE_FUNCTION
    bool uequal(const size_type block, const scalar_t &value) const {
      auto u_block = uget(block);
      for (size_type i = 0; i < block_size; ++i) {
        for (size_type j = 0; j < block_size; ++j) {
          if (u_block(i, j) != value) {
            return false;
          }
        }
      }
      return true;
    }
  };

  template <class ARowMapType, class AEntriesType, class AValuesType,
            class LRowMapType, class LEntriesType, class LValuesType,
            class URowMapType, class UEntriesType, class UValuesType,
            bool BlockEnabled>
  struct ILUKLvlSchedRPNumericFunctor
      : public Common<ARowMapType, AEntriesType, AValuesType, LRowMapType,
                      LEntriesType, LValuesType, URowMapType, UEntriesType,
                      UValuesType, BlockEnabled> {
    using Base = Common<ARowMapType, AEntriesType, AValuesType, LRowMapType,
                        LEntriesType, LValuesType, URowMapType, UEntriesType,
                        UValuesType, BlockEnabled>;

    ILUKLvlSchedRPNumericFunctor(
        const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
        const AValuesType &A_values_, const LRowMapType &L_row_map_,
        const LEntriesType &L_entries_, LValuesType &L_values_,
        const URowMapType &U_row_map_, const UEntriesType &U_entries_,
        UValuesType &U_values_, const LevelViewType &level_idx_,
        WorkViewType &iw_, const lno_t &lev_start_,
        const size_type &block_size_ = 0)
        : Base(A_row_map_, A_entries_, A_values_, L_row_map_, L_entries_,
               L_values_, U_row_map_, U_entries_, U_values_, level_idx_, iw_,
               lev_start_, block_size_) {}

    KOKKOS_FUNCTION
    void operator()(const lno_t i) const {
      scalar_t buff[Base::BUFF_SIZE];

      const auto rowid = Base::level_idx(i);
      const auto tid   = i - Base::lev_start;
      auto k1          = Base::L_row_map(rowid);
      auto k2          = Base::L_row_map(rowid + 1) - 1;
      Base::lset_id(k2);
      for (auto k = k1; k < k2; ++k) {
        const auto col = Base::L_entries(k);
        Base::lset(k, 0.0);
        Base::iw(tid, col) = k;
      }

      k1 = Base::U_row_map(rowid);
      k2 = Base::U_row_map(rowid + 1);
      for (auto k = k1; k < k2; ++k) {
        const auto col = Base::U_entries(k);
        Base::uset(k, 0.0);
        Base::iw(tid, col) = k;
      }

      k1 = Base::A_row_map(rowid);
      k2 = Base::A_row_map(rowid + 1);
      for (auto k = k1; k < k2; ++k) {
        const auto col  = Base::A_entries(k);
        const auto ipos = Base::iw(tid, col);
        if (col < rowid) {
          Base::lset(ipos, Base::aget(k));
        } else {
          Base::uset(ipos, Base::aget(k));
        }
      }

      // Eliminate prev rows
      k1 = Base::L_row_map(rowid);
      k2 = Base::L_row_map(rowid + 1) - 1;
      for (auto k = k1; k < k2; ++k) {
        const auto prev_row = Base::L_entries(k);
        const auto u_diag   = Base::uget(Base::U_row_map(prev_row));
        Base::divide(Base::lget(k), u_diag);
        auto fact = Base::lget(k);
        for (auto kk = Base::U_row_map(prev_row) + 1;
             kk < Base::U_row_map(prev_row + 1); ++kk) {
          const auto col  = Base::U_entries(kk);
          const auto ipos = Base::iw(tid, col);
          if (ipos == -1) continue;
          const auto lxu = Base::multiply(-1.0, Base::uget(kk), fact, &buff[0]);
          if (col < rowid) {
            Base::add(Base::lget(ipos), lxu);
          } else {
            Base::add(Base::uget(ipos), lxu);
          }
        }  // end for kk
      }    // end for k

      const auto ipos = Base::iw(tid, rowid);
      if (Base::uequal(ipos, 0.0)) {
        Base::uset(ipos, 1e6);
      }

      // Reset
      k1 = Base::L_row_map(rowid);
      k2 = Base::L_row_map(rowid + 1) - 1;
      for (auto k = k1; k < k2; ++k) Base::iw(tid, Base::L_entries(k)) = -1;

      k1 = Base::U_row_map(rowid);
      k2 = Base::U_row_map(rowid + 1);
      for (auto k = k1; k < k2; ++k) Base::iw(tid, Base::U_entries(k)) = -1;
    }
  };

  template <class ARowMapType, class AEntriesType, class AValuesType,
            class LRowMapType, class LEntriesType, class LValuesType,
            class URowMapType, class UEntriesType, class UValuesType,
            bool BlockEnabled>
  struct ILUKLvlSchedTP1NumericFunctor
      : public Common<ARowMapType, AEntriesType, AValuesType, LRowMapType,
                      LEntriesType, LValuesType, URowMapType, UEntriesType,
                      UValuesType, BlockEnabled> {
    using Base = Common<ARowMapType, AEntriesType, AValuesType, LRowMapType,
                        LEntriesType, LValuesType, URowMapType, UEntriesType,
                        UValuesType, BlockEnabled>;

    ILUKLvlSchedTP1NumericFunctor(
        const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
        const AValuesType &A_values_, const LRowMapType &L_row_map_,
        const LEntriesType &L_entries_, LValuesType &L_values_,
        const URowMapType &U_row_map_, const UEntriesType &U_entries_,
        UValuesType &U_values_, const LevelViewType &level_idx_,
        WorkViewType &iw_, const lno_t &lev_start_,
        const size_type &block_size_ = 0)
        : Base(A_row_map_, A_entries_, A_values_, L_row_map_, L_entries_,
               L_values_, U_row_map_, U_entries_, U_values_, level_idx_, iw_,
               lev_start_, block_size_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type &team) const {
      const auto my_team = team.league_rank();
      const auto rowid =
          Base::level_idx(my_team + Base::lev_start);  // map to rowid
      size_type k1 = Base::L_row_map(rowid);
      size_type k2 = Base::L_row_map(rowid + 1) - 1;
      Base::lset_id(team, k2);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                           [&](const size_type k) {
                             const auto col = Base::L_entries(k);
                             Base::lset(k, 0.0);
                             Base::iw(my_team, col) = k;
                           });

      team.team_barrier();

      k1 = Base::U_row_map(rowid);
      k2 = Base::U_row_map(rowid + 1);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                           [&](const size_type k) {
                             const auto col = Base::U_entries(k);
                             Base::uset(k, 0.0);
                             Base::iw(my_team, col) = k;
                           });

      team.team_barrier();

      // Unpack the ith row of A
      k1 = Base::A_row_map(rowid);
      k2 = Base::A_row_map(rowid + 1);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                           [&](const size_type k) {
                             const auto col  = Base::A_entries(k);
                             const auto ipos = Base::iw(my_team, col);
                             if (col < rowid) {
                               Base::lset(ipos, Base::aget(k));
                             } else {
                               Base::uset(ipos, Base::aget(k));
                             }
                           });

      team.team_barrier();

      // Eliminate prev rows
      k1 = Base::L_row_map(rowid);
      k2 = Base::L_row_map(rowid + 1) - 1;
      for (auto k = k1; k < k2; k++) {
        const auto prev_row = Base::L_entries(k);
        const auto udiag    = Base::uget(Base::U_row_map(prev_row));
        Base::divide(team, Base::lget(k), udiag);
        auto fact = Base::lget(k);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, Base::U_row_map(prev_row) + 1,
                                    Base::U_row_map(prev_row + 1)),
            [&](const size_type kk) {
              const auto col  = Base::U_entries(kk);
              const auto ipos = Base::iw(my_team, col);
              if (ipos != -1) {
                scalar_t buff[Base::BUFF_SIZE];
                auto lxu = Base::multiply(-1.0, Base::uget(kk), fact, &buff[0]);
                if (col < rowid) {
                  Base::add(Base::lget(ipos), lxu);
                } else {
                  Base::add(Base::uget(ipos), lxu);
                }
              }
            });  // end for kk

        team.team_barrier();
      }  // end for k

      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        const auto ipos = Base::iw(my_team, rowid);
        if (Base::uequal(ipos, 0.0)) {
          Base::uset(ipos, 1e6);
        }
      });

      team.team_barrier();

      // Reset
      k1 = Base::L_row_map(rowid);
      k2 = Base::L_row_map(rowid + 1) - 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                           [&](const size_type k) {
                             const auto col         = Base::L_entries(k);
                             Base::iw(my_team, col) = -1;
                           });

      k1 = Base::U_row_map(rowid);
      k2 = Base::U_row_map(rowid + 1);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                           [&](const size_type k) {
                             const auto col         = Base::U_entries(k);
                             Base::iw(my_team, col) = -1;
                           });
    }
  };

#define FunctorTypeMacro(Functor, BlockEnabled)                              \
  Functor<ARowMapType, AEntriesType, AValuesType, LRowMapType, LEntriesType, \
          LValuesType, URowMapType, UEntriesType, UValuesType, BlockEnabled>

#define KernelLaunchMacro(arow, aent, aval, lrow, lent, lval, urow, uent,   \
                          uval, polc, name, lidx, iwv, lstrt, ftf, ftb, be, \
                          bs)                                               \
  if (be) {                                                                 \
    ftb functor(arow, aent, aval, lrow, lent, lval, urow, uent, uval, lidx, \
                iwv, lstrt, bs);                                            \
    Kokkos::parallel_for(name, polc, functor);                              \
  } else {                                                                  \
    ftf functor(arow, aent, aval, lrow, lent, lval, urow, uent, uval, lidx, \
                iwv, lstrt);                                                \
    Kokkos::parallel_for(name, polc, functor);                              \
  }

  template <class ARowMapType, class AEntriesType, class AValuesType,
            class LRowMapType, class LEntriesType, class LValuesType,
            class URowMapType, class UEntriesType, class UValuesType>
  static void iluk_numeric(IlukHandle &thandle, const ARowMapType &A_row_map,
                           const AEntriesType &A_entries,
                           const AValuesType &A_values,
                           const LRowMapType &L_row_map,
                           const LEntriesType &L_entries, LValuesType &L_values,
                           const URowMapType &U_row_map,
                           const UEntriesType &U_entries,
                           UValuesType &U_values) {
    using RPF = FunctorTypeMacro(ILUKLvlSchedRPNumericFunctor, false);
    using RPB = FunctorTypeMacro(ILUKLvlSchedRPNumericFunctor, true);
    using TPF = FunctorTypeMacro(ILUKLvlSchedTP1NumericFunctor, false);
    using TPB = FunctorTypeMacro(ILUKLvlSchedTP1NumericFunctor, true);

    size_type nlevels     = thandle.get_num_levels();
    int team_size         = thandle.get_team_size();
    const auto block_size = thandle.get_block_size();

    LevelHostViewType level_ptr_h = thandle.get_host_level_ptr();
    LevelViewType level_idx       = thandle.get_level_idx();

    LevelHostViewType level_nchunks_h, level_nrowsperchunk_h;
    WorkViewType iw;

    if (thandle.get_algorithm() ==
        KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1) {
      level_nchunks_h       = thandle.get_level_nchunks();
      level_nrowsperchunk_h = thandle.get_level_nrowsperchunk();
    }
    iw = thandle.get_iw();

    // Main loop must be performed sequential. Question: Try out Cuda's graph
    // stuff to reduce kernel launch overhead
    for (size_type lvl = 0; lvl < nlevels; ++lvl) {
      lno_t lev_start = level_ptr_h(lvl);
      lno_t lev_end   = level_ptr_h(lvl + 1);

      if ((lev_end - lev_start) != 0) {
        if (thandle.get_algorithm() ==
            KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_RP) {
          range_policy rpolicy = get_range_policy(lev_start, lev_end);
          KernelLaunchMacro(A_row_map, A_entries, A_values, L_row_map,
                            L_entries, L_values, U_row_map, U_entries, U_values,
                            rpolicy, "parfor_fixed_lvl", level_idx, iw,
                            lev_start, RPF, RPB, thandle.is_block_enabled(),
                            block_size);
        } else if (thandle.get_algorithm() ==
                   KokkosSparse::Experimental::SPILUKAlgorithm::
                       SEQLVLSCHD_TP1) {
          lno_t lvl_rowid_start = 0;
          lno_t lvl_nrows_chunk;
          for (int chunkid = 0; chunkid < level_nchunks_h(lvl); chunkid++) {
            if ((lvl_rowid_start + level_nrowsperchunk_h(lvl)) >
                (lev_end - lev_start))
              lvl_nrows_chunk = (lev_end - lev_start) - lvl_rowid_start;
            else
              lvl_nrows_chunk = level_nrowsperchunk_h(lvl);

            team_policy tpolicy = get_team_policy(lvl_nrows_chunk, team_size);
            KernelLaunchMacro(A_row_map, A_entries, A_values, L_row_map,
                              L_entries, L_values, U_row_map, U_entries,
                              U_values, tpolicy, "parfor_tp1", level_idx, iw,
                              lev_start + lvl_rowid_start, TPF, TPB,
                              thandle.is_block_enabled(), block_size);
            Kokkos::fence();
            lvl_rowid_start += lvl_nrows_chunk;
          }
        }
      }  // end if
    }    // end for lvl
         //}

// Output check
#ifdef NUMERIC_OUTPUT_INFO
    std::cout << "  iluk_numeric result: " << std::endl;

    std::cout << "  nnzL: " << thandle.get_nnzL() << std::endl;
    std::cout << "  L_row_map = ";
    for (size_type i = 0; i < thandle.get_nrows() + 1; ++i) {
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
    for (size_type i = 0; i < thandle.get_nrows() + 1; ++i) {
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

  template <class ExecutionSpace, class ARowMapType, class AEntriesType,
            class AValuesType, class LRowMapType, class LEntriesType,
            class LValuesType, class URowMapType, class UEntriesType,
            class UValuesType>
  static void iluk_numeric_streams(
      const std::vector<ExecutionSpace> &execspace_v,
      const std::vector<IlukHandle *> &thandle_v,
      const std::vector<ARowMapType> &A_row_map_v,
      const std::vector<AEntriesType> &A_entries_v,
      const std::vector<AValuesType> &A_values_v,
      const std::vector<LRowMapType> &L_row_map_v,
      const std::vector<LEntriesType> &L_entries_v,
      std::vector<LValuesType> &L_values_v,
      const std::vector<URowMapType> &U_row_map_v,
      const std::vector<UEntriesType> &U_entries_v,
      std::vector<UValuesType> &U_values_v) {
    using RPF = FunctorTypeMacro(ILUKLvlSchedRPNumericFunctor, false);
    using RPB = FunctorTypeMacro(ILUKLvlSchedRPNumericFunctor, true);
    using TPF = FunctorTypeMacro(ILUKLvlSchedTP1NumericFunctor, false);
    using TPB = FunctorTypeMacro(ILUKLvlSchedTP1NumericFunctor, true);

    // Create vectors for handles' data in streams
    int nstreams = execspace_v.size();
    std::vector<size_type> nlevels_v(nstreams);
    std::vector<LevelHostViewType> lvl_ptr_h_v(nstreams);
    std::vector<LevelViewType> lvl_idx_v(nstreams);  // device views
    std::vector<lno_t> lvl_start_v(nstreams);
    std::vector<lno_t> lvl_end_v(nstreams);
    std::vector<WorkViewType> iw_v(nstreams);  // device views
    std::vector<bool> stream_have_level_v(nstreams);
    std::vector<bool> is_block_enabled_v(nstreams);
    std::vector<size_type> block_size_v(nstreams);

    // Retrieve data from handles and find max. number of levels among streams
    size_type nlevels_max = 0;
    for (int i = 0; i < nstreams; i++) {
      nlevels_v[i]           = thandle_v[i]->get_num_levels();
      lvl_ptr_h_v[i]         = thandle_v[i]->get_host_level_ptr();
      lvl_idx_v[i]           = thandle_v[i]->get_level_idx();
      iw_v[i]                = thandle_v[i]->get_iw();
      is_block_enabled_v[i]  = thandle_v[i]->is_block_enabled();
      block_size_v[i]        = thandle_v[i]->get_block_size();
      stream_have_level_v[i] = true;
      if (nlevels_max < nlevels_v[i]) nlevels_max = nlevels_v[i];
    }

    // Assume all streams use the same algorithm
    if (thandle_v[0]->get_algorithm() ==
        KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_RP) {
      // Main loop must be performed sequential
      for (size_type lvl = 0; lvl < nlevels_max; lvl++) {
        // Initial work across streams at each level
        for (int i = 0; i < nstreams; i++) {
          // Only do this if this stream has this level
          if (lvl < nlevels_v[i]) {
            lvl_start_v[i] = lvl_ptr_h_v[i](lvl);
            lvl_end_v[i]   = lvl_ptr_h_v[i](lvl + 1);
            if ((lvl_end_v[i] - lvl_start_v[i]) != 0)
              stream_have_level_v[i] = true;
            else
              stream_have_level_v[i] = false;
          } else
            stream_have_level_v[i] = false;
        }

        // Main work of the level across streams
        // 1. Launch work on all streams
        for (int i = 0; i < nstreams; i++) {
          // Launch only if stream i-th has this level
          if (stream_have_level_v[i]) {
            range_policy rpolicy =
                get_range_policy(execspace_v[i], lvl_start_v[i], lvl_end_v[i]);
            KernelLaunchMacro(A_row_map_v[i], A_entries_v[i], A_values_v[i],
                              L_row_map_v[i], L_entries_v[i], L_values_v[i],
                              U_row_map_v[i], U_entries_v[i], U_values_v[i],
                              rpolicy, "parfor_rp", lvl_idx_v[i], iw_v[i],
                              lvl_start_v[i], RPF, RPB, is_block_enabled_v[i],
                              block_size_v[i]);
          }  // end if (stream_have_level_v[i])
        }    // end for streams
      }      // end for lvl
    }        // end SEQLVLSCHD_RP
    else if (thandle_v[0]->get_algorithm() ==
             KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1) {
      std::vector<LevelHostViewType> lvl_nchunks_h_v(nstreams);
      std::vector<LevelHostViewType> lvl_nrowsperchunk_h_v(nstreams);
      std::vector<lno_t> lvl_rowid_start_v(nstreams);
      std::vector<int> team_size_v(nstreams);

      for (int i = 0; i < nstreams; i++) {
        lvl_nchunks_h_v[i]       = thandle_v[i]->get_level_nchunks();
        lvl_nrowsperchunk_h_v[i] = thandle_v[i]->get_level_nrowsperchunk();
        team_size_v[i]           = thandle_v[i]->get_team_size();
      }

      // Main loop must be performed sequential
      for (size_type lvl = 0; lvl < nlevels_max; lvl++) {
        // Initial work across streams at each level
        lno_t lvl_nchunks_max = 0;
        for (int i = 0; i < nstreams; i++) {
          // Only do this if this stream has this level
          if (lvl < nlevels_v[i]) {
            lvl_start_v[i] = lvl_ptr_h_v[i](lvl);
            lvl_end_v[i]   = lvl_ptr_h_v[i](lvl + 1);
            if ((lvl_end_v[i] - lvl_start_v[i]) != 0) {
              stream_have_level_v[i] = true;
              lvl_rowid_start_v[i]   = 0;
              if (lvl_nchunks_max < lvl_nchunks_h_v[i](lvl))
                lvl_nchunks_max = lvl_nchunks_h_v[i](lvl);
            } else
              stream_have_level_v[i] = false;
          } else
            stream_have_level_v[i] = false;
        }

        // Main work of the level across streams -- looping through chunnks
        for (int chunkid = 0; chunkid < lvl_nchunks_max; chunkid++) {
          // 1. Launch work on all streams (for each chunk)
          for (int i = 0; i < nstreams; i++) {
            // Launch only if stream i-th has this level
            if (stream_have_level_v[i]) {
              // Launch only if stream i-th has this chunk
              if (chunkid < lvl_nchunks_h_v[i](lvl)) {
                // 1.a. Specify number of rows (i.e. number of teams) to launch
                lno_t lvl_nrows_chunk = 0;
                if ((lvl_rowid_start_v[i] + lvl_nrowsperchunk_h_v[i](lvl)) >
                    (lvl_end_v[i] - lvl_start_v[i]))
                  lvl_nrows_chunk =
                      (lvl_end_v[i] - lvl_start_v[i]) - lvl_rowid_start_v[i];
                else
                  lvl_nrows_chunk = lvl_nrowsperchunk_h_v[i](lvl);

                // 1.b. Create functor for stream i-th and launch
                team_policy tpolicy = get_team_policy(
                    execspace_v[i], lvl_nrows_chunk, team_size_v[i]);
                KernelLaunchMacro(A_row_map_v[i], A_entries_v[i], A_values_v[i],
                                  L_row_map_v[i], L_entries_v[i], L_values_v[i],
                                  U_row_map_v[i], U_entries_v[i], U_values_v[i],
                                  tpolicy, "parfor_tp1", lvl_idx_v[i], iw_v[i],
                                  lvl_start_v[i] + lvl_rowid_start_v[i], TPF,
                                  TPB, is_block_enabled_v[i], block_size_v[i]);
                // 1.c. Ready to move to next chunk
                lvl_rowid_start_v[i] += lvl_nrows_chunk;
              }  // end if (chunkid < lvl_nchunks_h_v[i](lvl))
            }    // end if (stream_have_level_v[i])
          }      // end for streams
        }        // end for chunkid
      }          // end for lvl
    }            // end SEQLVLSCHD_TP1

  }  // end iluk_numeric_streams

};  // IlukWrap

}  // namespace Experimental
}  // namespace Impl
}  // namespace KokkosSparse

#undef FunctorTypeMacro
#undef KernelLaunchMacro

#endif
