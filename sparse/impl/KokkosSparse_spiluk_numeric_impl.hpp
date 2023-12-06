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
using execution_space         = typename IlukHandle::execution_space;
using memory_space            = typename IlukHandle::memory_space;
using lno_t                   = typename IlukHandle::nnz_lno_t;
using size_type               = typename IlukHandle::size_type;
using scalar_t                = typename IlukHandle::nnz_scalar_t;
using HandleDeviceRowMapType  = typename IlukHandle::nnz_row_view_t;
using HandleDeviceValueType   = typename IlukHandle::nnz_value_view_t;
using WorkViewType            = typename IlukHandle::work_view_t;
using LevelHostViewType       = typename IlukHandle::nnz_lno_view_host_t;
using LevelViewType           = typename IlukHandle::nnz_lno_view_t;
using karith                  = typename Kokkos::ArithTraits<scalar_t>;
using policy_type             = typename IlukHandle::TeamPolicy;
using member_type             = typename policy_type::member_type;
using range_policy            = typename IlukHandle::RangePolicy;
using sview_2d                = typename Kokkos::View<scalar_t**, memory_space>;
using sview_1d                = typename Kokkos::View<scalar_t*, memory_space>;

/**
 * Common base class for SPILUK functors. Default version does not support blocks
 */
template <class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType,
          bool BlockEnabled>
struct Common
{
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

  Common(
      const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
      const AValuesType &A_values_, const LRowMapType &L_row_map_,
      const LEntriesType &L_entries_, LValuesType &L_values_,
      const URowMapType &U_row_map_, const UEntriesType &U_entries_,
      UValuesType &U_values_, const LevelViewType &level_idx_,
      WorkViewType &iw_, const lno_t &lev_start_, const size_type& block_size_) :
    A_row_map(A_row_map_),
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
    lev_start(lev_start_)
  {
    KK_REQUIRE_MSG(block_size_ == 0, "Tried to use blocks with the unblocked Common?");
  }

  // lset
  KOKKOS_INLINE_FUNCTION
  void lset(const size_type nnz, const scalar_t& value) const
  { L_values(nnz) = value; }

  // uset
  KOKKOS_INLINE_FUNCTION
  void uset(const size_type nnz, const scalar_t& value) const
  { U_values(nnz) = value; }

  // lset_id
  KOKKOS_INLINE_FUNCTION
  void lset_id(const size_type nnz) const
  { L_values(nnz) = scalar_t(1.0); }

  KOKKOS_INLINE_FUNCTION
  void lset_id(const member_type& team, const size_type nnz) const
  {
    // Not sure a Kokkos::single is really needed here since the
    // race is harmless
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      L_values(nnz) = scalar_t(1.0);
    });
  }

  // divide. lhs /= rhs
  KOKKOS_INLINE_FUNCTION
  void divide(scalar_t& lhs, const scalar_t& rhs) const
  { lhs /= rhs; }

  KOKKOS_INLINE_FUNCTION
  void divide(const member_type& team, scalar_t& lhs, const scalar_t& rhs) const
  {
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
       lhs /= rhs;
    });
  }

  // add. lhs += rhs
  KOKKOS_INLINE_FUNCTION
  void add(scalar_t& lhs, const scalar_t& rhs) const
  { lhs += rhs; }

  // multiply: return (alpha * lhs) * rhs
  KOKKOS_INLINE_FUNCTION
  scalar_t multiply(const scalar_t& alpha, const scalar_t& lhs, const scalar_t& rhs) const
  { return alpha * lhs * rhs; }

  // lget
  KOKKOS_INLINE_FUNCTION
  scalar_t& lget(const size_type nnz) const
  { return L_values(nnz); }

  // uget
  KOKKOS_INLINE_FUNCTION
  scalar_t& uget(const size_type nnz) const
  { return U_values(nnz); }

  // aget
  KOKKOS_INLINE_FUNCTION
  scalar_t aget(const size_type nnz) const
  { return A_values(nnz); }

  // uequal
  KOKKOS_INLINE_FUNCTION
  bool uequal(const size_type nnz, const scalar_t& value) const
  { return U_values(nnz) == value; }
};

// Partial specialization for block support
template <class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType>
struct Common<ARowMapType, AEntriesType, AValuesType,
              LRowMapType, LEntriesType, LValuesType,
              URowMapType, UEntriesType, UValuesType, true>
{
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
  sview_2d temp_dense_block;
  sview_1d ones;

  using LValuesUnmanaged2DBlockType = Kokkos::View<
    typename LValuesType::value_type**,
    typename KokkosKernels::Impl::GetUnifiedLayout<LValuesType>::array_layout,
    typename LValuesType::device_type,
    Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >;

  using UValuesUnmanaged2DBlockType = Kokkos::View<
    typename UValuesType::value_type**,
    typename KokkosKernels::Impl::GetUnifiedLayout<UValuesType>::array_layout,
    typename UValuesType::device_type,
    Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >;

  using AValuesUnmanaged2DBlockType = Kokkos::View<
    typename AValuesType::value_type**,
    typename KokkosKernels::Impl::GetUnifiedLayout<AValuesType>::array_layout,
    typename AValuesType::device_type,
    Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >;

  Common(
      const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
      const AValuesType &A_values_, const LRowMapType &L_row_map_,
      const LEntriesType &L_entries_, LValuesType &L_values_,
      const URowMapType &U_row_map_, const UEntriesType &U_entries_,
      UValuesType &U_values_, const LevelViewType &level_idx_,
      WorkViewType &iw_, const lno_t &lev_start_, const size_type& block_size_) :
    A_row_map(A_row_map_),
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
    temp_dense_block("temp_dense_block", block_size, block_size), // this will have races unless Serial
    ones("ones", block_size)
  {
    Kokkos::deep_copy(ones, 1.0);
  }

  // lset
  KOKKOS_INLINE_FUNCTION
  void lset(const size_type block, const scalar_t& value) const
  { KokkosBlas::SerialSet::invoke(value, lget(block)); }

  KOKKOS_INLINE_FUNCTION
  void lset(const size_type block, const AValuesUnmanaged2DBlockType& rhs) const
  { Kokkos::deep_copy(lget(block), rhs); }

  // uset
  KOKKOS_INLINE_FUNCTION
  void uset(const size_type block, const scalar_t& value) const
  { KokkosBlas::SerialSet::invoke(value, uget(block)); }

  KOKKOS_INLINE_FUNCTION
  void uset(const size_type block, const AValuesUnmanaged2DBlockType& rhs) const
  { Kokkos::deep_copy(uget(block), rhs); }

  // lset_id
  KOKKOS_INLINE_FUNCTION
  void lset_id(const size_type block) const
  { KokkosBatched::SerialSetIdentity::invoke(lget(block)); }

  KOKKOS_INLINE_FUNCTION
  void lset_id(const member_type& team, const size_type block) const
  { KokkosBatched::TeamSetIdentity<member_type>::invoke(team, lget(block)); }

  // divide. lhs /= rhs
  KOKKOS_INLINE_FUNCTION
  void divide(LValuesUnmanaged2DBlockType lhs, const UValuesUnmanaged2DBlockType& rhs) const
  {
    KokkosBatched::SerialTrsm<KokkosBatched::Side::Right,
                              KokkosBatched::Uplo::Upper,
                              KokkosBatched::Trans::Transpose, // not 100% on this
                              KokkosBatched::Diag::NonUnit,
                              KokkosBatched::Algo::Trsm::Unblocked>:: // not 100% on this
      invoke<scalar_t>(1.0, rhs, lhs);
  }

  KOKKOS_INLINE_FUNCTION
  void divide(const member_type& team, LValuesUnmanaged2DBlockType lhs, const UValuesUnmanaged2DBlockType& rhs) const
  {
    KokkosBatched::TeamTrsm<member_type,
                            KokkosBatched::Side::Right,
                            KokkosBatched::Uplo::Upper,
                            KokkosBatched::Trans::Transpose, // not 100% on this
                            KokkosBatched::Diag::NonUnit,
                            KokkosBatched::Algo::Trsm::Unblocked>:: // not 100% on this
      invoke(team, 1.0, rhs, lhs);
  }


  // add. lhs += rhs
  template <typename Lview, typename Rview>
  KOKKOS_INLINE_FUNCTION
  void add(Lview lhs, const Rview& rhs) const
  {
    KokkosBatched::SerialAxpy::invoke(ones, rhs, lhs);
  }

  // multiply: return (alpha * lhs) * rhs
  KOKKOS_INLINE_FUNCTION
  sview_2d multiply(const scalar_t& alpha, const UValuesUnmanaged2DBlockType& lhs, const LValuesUnmanaged2DBlockType& rhs) const
  {
    KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose,
                              KokkosBatched::Trans::NoTranspose,
                              KokkosBatched::Algo::Gemm::Unblocked>::
      invoke(alpha, lhs, rhs, 0.0, temp_dense_block);
    return temp_dense_block;
  }

  // lget
  KOKKOS_INLINE_FUNCTION
  LValuesUnmanaged2DBlockType lget(const size_type block) const
  {
    return LValuesUnmanaged2DBlockType(L_values.data() + (block * block_items), block_size, block_size);
  }

  // uget
  KOKKOS_INLINE_FUNCTION
  UValuesUnmanaged2DBlockType uget(const size_type block) const
  {
    return UValuesUnmanaged2DBlockType(U_values.data() + (block * block_items), block_size, block_size);
  }

  // aget
  KOKKOS_INLINE_FUNCTION
  AValuesUnmanaged2DBlockType aget(const size_type block) const
  {
    return AValuesUnmanaged2DBlockType(A_values.data() + (block * block_items), block_size, block_size);
  }

  // uequal
  KOKKOS_INLINE_FUNCTION
  bool uequal(const size_type block, const scalar_t& value) const
  {
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
struct ILUKLvlSchedRPNumericFunctor :
    public Common<ARowMapType, AEntriesType, AValuesType, LRowMapType, LEntriesType, LValuesType,
                  URowMapType, UEntriesType, UValuesType, BlockEnabled>
{
  using Base = Common<ARowMapType, AEntriesType, AValuesType, LRowMapType, LEntriesType, LValuesType,
                      URowMapType, UEntriesType, UValuesType, BlockEnabled>;

  ILUKLvlSchedRPNumericFunctor(
      const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
      const AValuesType &A_values_, const LRowMapType &L_row_map_,
      const LEntriesType &L_entries_, LValuesType &L_values_,
      const URowMapType &U_row_map_, const UEntriesType &U_entries_,
      UValuesType &U_values_, const LevelViewType &level_idx_,
      WorkViewType &iw_, const lno_t &lev_start_, const size_type& block_size_ = 0) :
    Base(A_row_map_, A_entries_, A_values_, L_row_map_, L_entries_, L_values_,
         U_row_map_, U_entries_, U_values_, level_idx_, iw_, lev_start_, block_size_)
  {}

  KOKKOS_FUNCTION
  void operator()(const lno_t i) const {
    // Grab items from parent to make code more readable
    auto A_row_map = Base::A_row_map;
    auto A_entries = Base::A_row_map;
    auto L_row_map = Base::L_row_map;
    auto L_entries = Base::L_entries;
    auto U_row_map = Base::U_row_map;
    auto U_entries = Base::U_entries;
    auto level_idx = Base::level_idx;
    auto lev_start = Base::lev_start;
    auto iw        = Base::iw;

    const auto rowid = level_idx(i);
    const auto tid   = i - lev_start;
    auto k1    = L_row_map(rowid);
#ifdef KEEP_DIAG
    auto k2    = L_row_map(rowid + 1) - 1;
    Base::lset_id(k2);
#else
    auto k2    = L_row_map(rowid + 1);
#endif
    for (auto k = k1; k < k2; ++k) {
      const auto col = L_entries(k);
      Base::lset(k, 0.0);
      iw(tid, col) = k;
    }

    k1 = U_row_map(rowid);
    k2 = U_row_map(rowid + 1);
    for (auto k = k1; k < k2; ++k) {
      const auto col = U_entries(k);
      Base::uset(k, 0.0);
      iw(tid, col) = k;
    }

    k1 = A_row_map(rowid);
    k2 = A_row_map(rowid + 1);
    for (auto k = k1; k < k2; ++k) {
      const auto col  = A_entries(k);
      const auto ipos = iw(tid, col);
      if (col < rowid) {
        Base::lset(ipos, Base::aget(k));
      }
      else {
        Base::uset(ipos, Base::aget(k));
      }
    }

    // Eliminate prev rows
    k1 = L_row_map(rowid);
#ifdef KEEP_DIAG
    k2 = L_row_map(rowid + 1) - 1;
#else
    k2 = L_row_map(rowid + 1);
#endif
    for (auto k = k1; k < k2; ++k) {
      const auto prev_row = L_entries(k);
      const auto u_diag   = Base::uget(U_row_map(prev_row));
#ifdef KEEP_DIAG
      Base::divide(Base::lget(k), u_diag);
#else
      fact = Base::multiply(1.0, fact, u_diag);
#endif
      auto fact = Base::lget(k);
      for (auto kk = U_row_map(prev_row) + 1; kk < U_row_map(prev_row + 1);
           ++kk) {
        const auto col  = U_entries(kk);
        const auto ipos = iw(tid, col);
        if (ipos == -1) continue;
        const auto lxu = Base::multiply(-1.0, Base::uget(kk), fact);
        if (col < rowid) {
          Base::add(Base::lget(ipos), lxu);
        }
        else {
          Base::add(Base::uget(ipos), lxu);
        }
      }  // end for kk
    }    // end for k

    const auto ipos = iw(tid, rowid);
    if (Base::uequal(ipos, 0.0)) {
      Base::uset(ipos, 1e6);
    }
#ifndef KEEP_DIAG
    else {
      // U_values(ipos) = 1.0 / U_values(ipos);
      KK_KERNEL_REQUIRE(false);
    }
#endif

    // Reset
    k1 = L_row_map(rowid);
#ifdef KEEP_DIAG
    k2 = L_row_map(rowid + 1) - 1;
#else
    k2 = L_row_map(rowid + 1);
#endif
    for (auto k = k1; k < k2; ++k)
      iw(tid, L_entries(k)) = -1;

    k1 = U_row_map(rowid);
    k2 = U_row_map(rowid + 1);
    for (auto k = k1; k < k2; ++k) iw(tid, U_entries(k)) = -1;
  }
};

template <class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType,
          bool BlockEnabled>
struct ILUKLvlSchedTP1NumericFunctor :
    public Common<ARowMapType, AEntriesType, AValuesType, LRowMapType, LEntriesType, LValuesType,
                  URowMapType, UEntriesType, UValuesType, BlockEnabled>
{
  using Base = Common<ARowMapType, AEntriesType, AValuesType, LRowMapType, LEntriesType, LValuesType,
                      URowMapType, UEntriesType, UValuesType, BlockEnabled>;

  ILUKLvlSchedTP1NumericFunctor(
      const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
      const AValuesType &A_values_, const LRowMapType &L_row_map_,
      const LEntriesType &L_entries_, LValuesType &L_values_,
      const URowMapType &U_row_map_, const UEntriesType &U_entries_,
      UValuesType &U_values_, const LevelViewType &level_idx_,
        WorkViewType &iw_, const lno_t &lev_start_, const size_type& block_size_ = 0) :
    Base(A_row_map_, A_entries_, A_values_, L_row_map_, L_entries_, L_values_,
      U_row_map_, U_entries_, U_values_, level_idx_, iw_, lev_start_, block_size_)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type &team) const {
    // Grab items from parent to make code more readable
    auto A_row_map = Base::A_row_map;
    auto A_entries = Base::A_row_map;
    auto L_row_map = Base::L_row_map;
    auto L_entries = Base::L_entries;
    auto U_row_map = Base::U_row_map;
    auto U_entries = Base::U_entries;
    auto level_idx = Base::level_idx;
    auto lev_start = Base::lev_start;
    auto iw        = Base::iw;

    const auto my_team = team.league_rank();
    const auto rowid   = level_idx(my_team + lev_start);  // map to rowid
    size_type k1 = L_row_map(rowid);
#ifdef KEEP_DIAG
    size_type k2 = L_row_map(rowid + 1) - 1;
    Base::lset_id(team, k2);
#else
    size_type k2 = L_row_map(rowid + 1);
#endif
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
      const auto col = L_entries(k);
      Base::lset(k, 0.0);
      iw(my_team, col) = k;
    });

    team.team_barrier();

    k1 = U_row_map(rowid);
    k2 = U_row_map(rowid + 1);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
      const auto col = U_entries(k);
      Base::uset(k, 0.0);
      iw(my_team, col) = k;
    });

    team.team_barrier();

    // Unpack the ith row of A
    k1 = A_row_map(rowid);
    k2 = A_row_map(rowid + 1);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
      const auto col = A_entries(k);
      const auto ipos = iw(my_team, col);
      if (col < rowid) {
        Base::lset(ipos, Base::aget(k));
      }
      else {
        Base::uset(ipos, Base::aget(k));
      }
    });

    team.team_barrier();

    // Eliminate prev rows
    k1 = L_row_map(rowid);
#ifdef KEEP_DIAG
    k2 = L_row_map(rowid + 1) - 1;
#else
    k2 = L_row_map(rowid + 1);
#endif
    for (auto k = k1; k < k2; k++) {
      const auto prev_row = L_entries(k);
      const auto udiag   = Base::uget(U_row_map(prev_row));
#ifdef KEEP_DIAG
      Base::divide(team, Base::lget(k), udiag);
#else
      fact = Base::multiply(team, 1.0, fact, udiag);
#endif
      auto fact = Base::lget(k);
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, U_row_map(prev_row) + 1, U_row_map(prev_row + 1)),
                                [&](const size_type kk) {
        const auto col  = U_entries(kk);
        const auto ipos = iw(my_team, col);
        if (ipos != -1) {
          auto lxu = Base::multiply(-1.0, Base::uget(kk), fact);
          if (col < rowid) {
            Base::add(Base::lget(ipos), lxu);
          }
          else {
            Base::add(Base::uget(ipos), lxu);
          }
        }
      });  // end for kk

      team.team_barrier();
    }  // end for k

    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      const auto ipos = iw(my_team, rowid);
      if (Base::uequal(ipos, 0.0)) {
        Base::uset(ipos, 1e6);
      }
#ifndef KEEP_DIAG
      else {
        // U_values(ipos) = 1.0 / U_values(ipos);
        KK_KERNEL_REQUIRE(false);
      }
#endif
    });

    team.team_barrier();

    // Reset
    k1 = L_row_map(rowid);
#ifdef KEEP_DIAG
    k2 = L_row_map(rowid + 1) - 1;
#else
    k2 = L_row_map(rowid + 1);
#endif
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
      const auto col = L_entries(k);
      iw(my_team, col) = -1;
    });

    k1 = U_row_map(rowid);
    k2 = U_row_map(rowid + 1);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
      const auto col = U_entries(k);
      iw(my_team, col) = -1;
    });
  }
};

template <class ARowMapType, class AEntriesType,
          class AValuesType, class LRowMapType, class LEntriesType,
          class LValuesType, class URowMapType, class UEntriesType,
          class UValuesType>
static void iluk_numeric(IlukHandle &thandle, const ARowMapType &A_row_map,
                  const AEntriesType &A_entries, const AValuesType &A_values,
                  const LRowMapType &L_row_map, const LEntriesType &L_entries,
                  LValuesType &L_values, const URowMapType &U_row_map,
                  const UEntriesType &U_entries, UValuesType &U_values) {

  bool verbose = false;

  size_type nlevels = thandle.get_num_levels();
  if (verbose)
    std::cout << "JGF iluk_numeric with nlevels: " << nlevels << std::endl;
  int team_size     = thandle.get_team_size();
  const auto block_size = thandle.get_block_size();

  LevelHostViewType level_ptr_h = thandle.get_host_level_ptr();
  LevelViewType     level_idx   = thandle.get_level_idx();

  LevelHostViewType level_nchunks_h, level_nrowsperchunk_h;
  WorkViewType iw;

  //{
  if (thandle.get_algorithm() ==
      KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1) {
    level_nchunks_h       = thandle.get_level_nchunks();
    level_nrowsperchunk_h = thandle.get_level_nrowsperchunk();
  }
  iw = thandle.get_iw();

  // Main loop must be performed sequential. Question: Try out Cuda's graph
  // stuff to reduce kernel launch overhead
  for (size_type lvl = 0; lvl < nlevels; ++lvl) {
    if (verbose)
      std::cout << "  JGF starting level: " << lvl << std::endl;
    lno_t lev_start = level_ptr_h(lvl);
    lno_t lev_end   = level_ptr_h(lvl + 1);

    if ((lev_end - lev_start) != 0) {
      if (thandle.get_algorithm() ==
          KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_RP) {

        if (thandle.block_enabled()) {
          Kokkos::parallel_for(
            "parfor_fixed_lvl",
            Kokkos::RangePolicy<execution_space>(lev_start, lev_end),
            ILUKLvlSchedRPNumericFunctor<
                ARowMapType, AEntriesType, AValuesType, LRowMapType,
                LEntriesType, LValuesType, URowMapType, UEntriesType,
                UValuesType, true>(
                A_row_map, A_entries, A_values, L_row_map, L_entries, L_values,
                U_row_map, U_entries, U_values, level_idx, iw, lev_start, block_size));
        }
        else {
          Kokkos::parallel_for(
            "parfor_fixed_lvl",
            Kokkos::RangePolicy<execution_space>(lev_start, lev_end),
            ILUKLvlSchedRPNumericFunctor<
                ARowMapType, AEntriesType, AValuesType, LRowMapType,
                LEntriesType, LValuesType, URowMapType, UEntriesType,
                UValuesType, false>(
                A_row_map, A_entries, A_values, L_row_map, L_entries, L_values,
                U_row_map, U_entries, U_values, level_idx, iw, lev_start));
        }
      } else if (thandle.get_algorithm() ==
                 KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1) {
        using policy_type = Kokkos::TeamPolicy<execution_space>;

        lno_t lvl_rowid_start = 0;
        lno_t lvl_nrows_chunk;
        for (int chunkid = 0; chunkid < level_nchunks_h(lvl); chunkid++) {
          if ((lvl_rowid_start + level_nrowsperchunk_h(lvl)) >
              (lev_end - lev_start))
            lvl_nrows_chunk = (lev_end - lev_start) - lvl_rowid_start;
          else
            lvl_nrows_chunk = level_nrowsperchunk_h(lvl);

          if (thandle.block_enabled()) {
            ILUKLvlSchedTP1NumericFunctor<
              ARowMapType, AEntriesType, AValuesType, LRowMapType, LEntriesType,
              LValuesType, URowMapType, UEntriesType, UValuesType, true>
              tstf(A_row_map, A_entries, A_values, L_row_map, L_entries,
                   L_values, U_row_map, U_entries, U_values, level_idx, iw,
                   lev_start + lvl_rowid_start, block_size);

            if (team_size == -1)
              Kokkos::parallel_for(
                "parfor_tp1", policy_type(lvl_nrows_chunk, Kokkos::AUTO), tstf);
            else
              Kokkos::parallel_for("parfor_tp1",
                                   policy_type(lvl_nrows_chunk, team_size), tstf);
          }
          else {
            ILUKLvlSchedTP1NumericFunctor<
              ARowMapType, AEntriesType, AValuesType, LRowMapType, LEntriesType,
              LValuesType, URowMapType, UEntriesType, UValuesType, false>
              tstf(A_row_map, A_entries, A_values, L_row_map, L_entries,
                   L_values, U_row_map, U_entries, U_values, level_idx, iw,
                   lev_start + lvl_rowid_start);

            if (team_size == -1)
              Kokkos::parallel_for(
                "parfor_tp1", policy_type(lvl_nrows_chunk, Kokkos::AUTO), tstf);
            else
              Kokkos::parallel_for("parfor_tp1",
                                   policy_type(lvl_nrows_chunk, team_size), tstf);
          }
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

template <class ExecutionSpace, class ARowMapType,
          class AEntriesType, class AValuesType, class LRowMapType,
          class LEntriesType, class LValuesType, class URowMapType,
          class UEntriesType, class UValuesType>
static void iluk_numeric_streams(const std::vector<ExecutionSpace> &execspace_v,
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
  // Create vectors for handles' data in streams
  int nstreams = execspace_v.size();
  std::vector<size_type> nlevels_v(nstreams);
  std::vector<LevelHostViewType> lvl_ptr_h_v(nstreams);
  std::vector<LevelViewType> lvl_idx_v(nstreams);  // device views
  std::vector<lno_t> lvl_start_v(nstreams);
  std::vector<lno_t> lvl_end_v(nstreams);
  std::vector<WorkViewType> iw_v(nstreams);  // device views
  std::vector<bool> stream_have_level_v(nstreams);

  // Retrieve data from handles and find max. number of levels among streams
  size_type nlevels_max = 0;
  for (int i = 0; i < nstreams; i++) {
    nlevels_v[i]           = thandle_v[i]->get_num_levels();
    lvl_ptr_h_v[i]         = thandle_v[i]->get_host_level_ptr();
    lvl_idx_v[i]           = thandle_v[i]->get_level_idx();
    iw_v[i]                = thandle_v[i]->get_iw();
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
          ILUKLvlSchedRPNumericFunctor<
              ARowMapType, AEntriesType, AValuesType, LRowMapType, LEntriesType,
              LValuesType, URowMapType, UEntriesType, UValuesType, false>
              tstf(A_row_map_v[i], A_entries_v[i], A_values_v[i],
                   L_row_map_v[i], L_entries_v[i], L_values_v[i],
                   U_row_map_v[i], U_entries_v[i], U_values_v[i], lvl_idx_v[i],
                   iw_v[i], lvl_start_v[i]);
          Kokkos::parallel_for(
              "parfor_rp",
              Kokkos::RangePolicy<ExecutionSpace>(execspace_v[i],
                                                  lvl_start_v[i], lvl_end_v[i]),
              tstf);
        }  // end if (stream_have_level_v[i])
      }    // end for streams
    }      // end for lvl
  }        // end SEQLVLSCHD_RP
  else if (thandle_v[0]->get_algorithm() ==
           KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1) {
    using policy_type = Kokkos::TeamPolicy<ExecutionSpace>;

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
              ILUKLvlSchedTP1NumericFunctor<
                  ARowMapType, AEntriesType, AValuesType, LRowMapType,
                  LEntriesType, LValuesType, URowMapType, UEntriesType,
                  UValuesType, false>
                  tstf(A_row_map_v[i], A_entries_v[i], A_values_v[i],
                       L_row_map_v[i], L_entries_v[i], L_values_v[i],
                       U_row_map_v[i], U_entries_v[i], U_values_v[i],
                       lvl_idx_v[i], iw_v[i],
                       lvl_start_v[i] + lvl_rowid_start_v[i]);
              if (team_size_v[i] == -1)
                Kokkos::parallel_for(
                    "parfor_tp1",
                    policy_type(execspace_v[i], lvl_nrows_chunk, Kokkos::AUTO),
                    tstf);
              else
                Kokkos::parallel_for(
                    "parfor_tp1",
                    policy_type(execspace_v[i], lvl_nrows_chunk,
                                team_size_v[i]),
                    tstf);

              // 1.c. Ready to move to next chunk
              lvl_rowid_start_v[i] += lvl_nrows_chunk;
            }  // end if (chunkid < lvl_nchunks_h_v[i](lvl))
          }    // end if (stream_have_level_v[i])
        }      // end for streams
      }        // end for chunkid
    }          // end for lvl
  }            // end SEQLVLSCHD_TP1

}  // end iluk_numeric_streams

}; // IlukWrap

}  // namespace Experimental
}  // namespace Impl
}  // namespace KokkosSparse

#endif
