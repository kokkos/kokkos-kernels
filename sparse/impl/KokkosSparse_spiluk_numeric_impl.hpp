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

// struct UnsortedTag {};

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

static void print_matrix(const std::vector<std::vector<scalar_t>>& matrix) {
  for (const auto& row : matrix) {
    for (const auto& item : row) {
      std::printf("%.2f ", item);
    }
    std::cout << std::endl;
  }
}

static void print_iw(const WorkViewType& iw, const size_type block_size=1) {
  std::cout << "      IW:" << std::endl;
  for (auto i = 0; i < iw.extent(0); ++i) {
    if (block_size == 1) {
      for (auto j = 0; j < iw.extent(1); ++j) {
        std::cout << iw(i, j) << " ";
      }
      std::cout << std::endl;
    }
    else {
      const auto block_items = block_size * block_size;
      const auto num_blocks = iw.extent(1) / block_items;
      for (size_type block_row = 0; block_row < block_size; ++block_row) {
        for (size_type b = 0; b < num_blocks; ++b) {
          for (size_type block_col = 0; block_col < block_size; ++block_col) {
            std::cout << iw(i, b * block_items + block_row * block_size + block_col) << " ";
          }
        }
        std::cout << std::endl;
      }
    }
  }
}

template <class RowMapType, class EntriesType, class ValuesType>
static std::vector<std::vector<scalar_t>> decompress_matrix(
  const RowMapType& row_map,
  const EntriesType& entries,
  const ValuesType& values)
{
  const scalar_t ZERO = scalar_t(0);

  const size_type nrows = row_map.size() - 1;
  std::vector<std::vector<scalar_t>> result;
  result.resize(nrows);
  for (auto& row : result) {
    row.resize(nrows, ZERO);
  }

  auto hrow_map = Kokkos::create_mirror_view(row_map);
  auto hentries = Kokkos::create_mirror_view(entries);
  auto hvalues  = Kokkos::create_mirror_view(values);
  Kokkos::deep_copy(hrow_map, row_map);
  Kokkos::deep_copy(hentries, entries);
  Kokkos::deep_copy(hvalues, values);

  for (size_type row_idx = 0; row_idx < nrows; ++row_idx) {
    const size_type row_nnz_begin = hrow_map(row_idx);
    const size_type row_nnz_end   = hrow_map(row_idx + 1);
    for (size_type row_nnz = row_nnz_begin; row_nnz < row_nnz_end; ++row_nnz) {
      const lno_t col_idx      = hentries(row_nnz);
      const scalar_t value     = hvalues(row_nnz);
      result[row_idx][col_idx] = value;
    }
  }

  return result;
}

template <class RowMapType, class EntriesType, class ValuesType>
static std::vector<std::vector<scalar_t>> decompress_matrix(
  const RowMapType& row_map,
  const EntriesType& entries,
  const ValuesType& values,
  const int block_size)
{
  const scalar_t ZERO = scalar_t(0);

  const size_type nbrows   = row_map.extent(0) - 1;
  const size_type nrows    = nbrows * block_size;
  const size_type block_items = block_size * block_size;
  std::vector<std::vector<scalar_t>> result;
  result.resize(nrows);
  for (auto& row : result) {
    row.resize(nrows, ZERO);
  }

  auto hrow_map = Kokkos::create_mirror_view(row_map);
  auto hentries = Kokkos::create_mirror_view(entries);
  auto hvalues  = Kokkos::create_mirror_view(values);
  Kokkos::deep_copy(hrow_map, row_map);
  Kokkos::deep_copy(hentries, entries);
  Kokkos::deep_copy(hvalues, values);

  for (size_type row_idx = 0; row_idx < nbrows; ++row_idx) {
    const size_type row_nnz_begin = hrow_map(row_idx);
    const size_type row_nnz_end   = hrow_map(row_idx + 1);
    for (size_type row_nnz = row_nnz_begin; row_nnz < row_nnz_end; ++row_nnz) {
      const lno_t col_idx = hentries(row_nnz);
      for (size_type i = 0; i < block_size; ++i) {
        const size_type unc_row_idx = row_idx*block_size + i;
        for (size_type j = 0; j < block_size; ++j) {
          const size_type unc_col_idx = col_idx*block_size + j;
          result[unc_row_idx][unc_col_idx] = hvalues(row_nnz*block_items + i*block_size + j);
        }
      }
    }
  }

  return result;
}

template <class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType>
struct ILUKLvlSchedRPNumericFunctor {
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
  bool verbose;

  ILUKLvlSchedRPNumericFunctor(
      const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
      const AValuesType &A_values_, const LRowMapType &L_row_map_,
      const LEntriesType &L_entries_, LValuesType &L_values_,
      const URowMapType &U_row_map_, const UEntriesType &U_entries_,
      UValuesType &U_values_, const LevelViewType &level_idx_,
      WorkViewType &iw_, const lno_t &lev_start_)
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
        verbose(false) {}

  KOKKOS_INLINE_FUNCTION
  void verbose_lset(const size_type nnz, const scalar_t& value) const
  {
    const size_type nrows = L_row_map.extent(0) - 1;
    for (size_type row = 0; row < nrows; ++row) {
      const auto row_begin = L_row_map(row);
      const auto row_end   = L_row_map(row+1);
      if (nnz >= row_begin && nnz < row_end) {
        const auto col = L_entries(nnz);
        if (L_values(nnz) != value) {
          if (verbose)
            std::cout << "        JGF Setting L_values[" << row << "][" << col << "] = " << value << std::endl;
          L_values(nnz) = value;
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void verbose_uset(const size_type nnz, const scalar_t& value) const
  {
    const size_type nrows = U_row_map.extent(0) - 1;
    for (size_type row = 0; row < nrows; ++row) {
      const auto row_begin = U_row_map(row);
      const auto row_end   = U_row_map(row+1);
      if (nnz >= row_begin && nnz < row_end) {
        const auto col = U_entries(nnz);
        if (U_values(nnz) != value) {
          if (verbose)
            std::cout << "        JGF Setting U_values[" << row << "][" << col << "] = " << value << std::endl;
          U_values(nnz) = value;
        }
      }
    }
  }

  void verbose_iwset(const size_type tid, const size_type col, const lno_t value) const
  {
    if (verbose)
      std::cout << "        JGF Setting iw[" << tid << "][" << col << "] = " << value << std::endl;
    iw(tid, col) = value;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const lno_t i) const {
    auto rowid = level_idx(i);
    auto tid   = i - lev_start;
    auto k1    = L_row_map(rowid);
    auto k2    = L_row_map(rowid + 1);

    if (verbose)
      std::cout << "    JGF level ptr: " << i << ", tid: " << tid << ", rowid: " << rowid << std::endl;

    if (verbose)
      std::cout << "      JGF Block 1" << std::endl;
#ifdef KEEP_DIAG
    for (auto k = k1; k < k2 - 1; ++k) {
#else
    for (auto k = k1; k < k2; ++k) {
#endif
      auto col     = L_entries(k);
      verbose_lset(k, 0.0);
      verbose_iwset(tid, col, k);
    }
#ifdef KEEP_DIAG
    verbose_lset(k2 - 1, scalar_t(1.0));
#endif

    if (verbose) {
      print_iw(iw);

      std::cout << "      JGF Block 2" << std::endl;
    }
    k1 = U_row_map(rowid);
    k2 = U_row_map(rowid + 1);
    for (auto k = k1; k < k2; ++k) {
      auto col     = U_entries(k);
      verbose_uset(k, 0.0);
      verbose_iwset(tid, col, k);
    }

    // Unpack the ith row of A
    if (verbose) {
      print_iw(iw);

      std::cout << "      JGF Block 3" << std::endl;
    }
    k1 = A_row_map(rowid);
    k2 = A_row_map(rowid + 1);
    for (auto k = k1; k < k2; ++k) {
      auto col  = A_entries(k);
      auto ipos = iw(tid, col);
      if (col < rowid) {
        verbose_lset(ipos, A_values(k));
      }
      else {
        verbose_uset(ipos, A_values(k));
      }
    }

    // Eliminate prev rows
    if (verbose) {
      std::cout << "      JGF Block 4" << std::endl;
      std::cout << "      L:" << std::endl;
      print_matrix(decompress_matrix(L_row_map, L_entries, L_values));
      std::cout << "      U:" << std::endl;
      print_matrix(decompress_matrix(U_row_map, U_entries, U_values));
      print_iw(iw);
    }
    k1 = L_row_map(rowid);
    k2 = L_row_map(rowid + 1);
#ifdef KEEP_DIAG
    for (auto k = k1; k < k2 - 1; ++k) {
#else
    for (auto k = k1; k < k2; ++k) {
#endif
      auto prev_row = L_entries(k);
      if (verbose)
        std::cout << "        JGF Processing L[" << rowid << "][" << prev_row << "]" << std::endl;
#ifdef KEEP_DIAG
      auto fact = L_values(k) / U_values(U_row_map(prev_row));
#else
      auto fact = L_values(k) * U_values(U_row_map(prev_row));
#endif
      verbose_lset(k, fact);
      if (verbose)
        std::cout << "        JGF Block 4 trouble spot" << std::endl;
      for (auto kk = U_row_map(prev_row) + 1; kk < U_row_map(prev_row + 1);
           ++kk) {
        auto col  = U_entries(kk);
        auto ipos = iw(tid, col);
        if (verbose) {
          std::cout << "          JGF Processing U[" << prev_row << "][" << col << "]" << std::endl;
          std::cout << "          JGF rowid=" << rowid <<", prev_row=" << prev_row << ", kk=" << kk << ", col=" << col << ", ipos=" << ipos << std::endl;
        }
        if (ipos == -1) continue;
        auto lxu = -U_values(kk) * fact;
        if (col < rowid) {
          verbose_lset(ipos, L_values(ipos) + lxu);
        }
        else {
          verbose_uset(ipos, U_values(ipos) + lxu);
        }
      }  // end for kk
      if (verbose)
        std::cout << "        JGF Block 4 trouble end" << std::endl;
    }    // end for k

    if (verbose) {
      std::cout << "      L:" << std::endl;
      print_matrix(decompress_matrix(L_row_map, L_entries, L_values));
      std::cout << "      U:" << std::endl;
      print_matrix(decompress_matrix(U_row_map, U_entries, U_values));

      std::cout << "      JGF Block 5" << std::endl;
    }
#ifdef KEEP_DIAG
    if (U_values(iw(tid, rowid)) == 0.0) {
      verbose_uset(iw(tid, rowid), 1e6);
    }
#else
    if (U_values(iw(tid, rowid)) == 0.0) {
      verbose_uset(iw(tid, rowid), 1e6);
    } else {
      verbose_uset(iw(tid, rowid), 1.0 / U_values(iw(tid, rowid)));
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
          class URowMapType, class UEntriesType, class UValuesType>
struct ILUKLvlSchedRPNumericFunctorBlock {
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
  bool verbose;

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

  ILUKLvlSchedRPNumericFunctorBlock(
      const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
      const AValuesType &A_values_, const LRowMapType &L_row_map_,
      const LEntriesType &L_entries_, LValuesType &L_values_,
      const URowMapType &U_row_map_, const UEntriesType &U_entries_,
      UValuesType &U_values_, const LevelViewType &level_idx_,
      WorkViewType &iw_, const lno_t &lev_start_, const size_type& block_size_)
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
        temp_dense_block("temp_dense_block", block_size, block_size), // this will have races unless Serial
        ones("ones", block_size),
        verbose(false)
  {
    Kokkos::deep_copy(ones, 1.0);
  }

  KOKKOS_INLINE_FUNCTION
  void verbose_lset(const size_type block, const scalar_t& value) const
  {
    const size_type nrows = L_row_map.extent(0) - 1;
    for (size_type row = 0; row < nrows; ++row) {
      const auto row_begin = L_row_map(row);
      const auto row_end   = L_row_map(row+1);
      if (block >= row_begin && block < row_end) {
        const auto col = L_entries(block);
        if (verbose)
          std::cout << "        JGF Setting L_values[" << row << "][" << col << "] = " << value << std::endl;
        KokkosBlas::SerialSet::invoke(value, get_l_block(block));
      }
    }
  }

  template <typename BlockType>
  KOKKOS_INLINE_FUNCTION
  void verbose_lset_block(const size_type block, const BlockType& rhs) const
  {
    const size_type nrows = L_row_map.extent(0) - 1;
    for (size_type row = 0; row < nrows; ++row) {
      const auto row_begin = L_row_map(row);
      const auto row_end   = L_row_map(row+1);
      if (block >= row_begin && block < row_end) {
        const auto col = L_entries(block);
        if (verbose)
          std::cout << "        JGF Setting block L_values[" << row << "][" << col << "]" << std::endl;
        Kokkos::deep_copy(get_l_block(block), rhs);
      }
    }
  }


  KOKKOS_INLINE_FUNCTION
  void verbose_uset(const size_type block, const scalar_t& value) const
  {
    const size_type nrows = U_row_map.extent(0) - 1;
    for (size_type row = 0; row < nrows; ++row) {
      const auto row_begin = U_row_map(row);
      const auto row_end   = U_row_map(row+1);
      if (block >= row_begin && block < row_end) {
        const auto col = U_entries(block);
        if (verbose)
          std::cout << "        JGF Setting U_values[" << row << "][" << col << "] = " << value << std::endl;
        KokkosBlas::SerialSet::invoke(value, get_u_block(block));
      }
    }
  }

  template <typename BlockType>
  KOKKOS_INLINE_FUNCTION
  void verbose_uset_block(const size_type block, const BlockType& rhs) const
  {
    const size_type nrows = U_row_map.extent(0) - 1;
    for (size_type row = 0; row < nrows; ++row) {
      const auto row_begin = U_row_map(row);
      const auto row_end   = U_row_map(row+1);
      if (block >= row_begin && block < row_end) {
        const auto col = U_entries(block);
        std::cout << "        JGF Setting block U_values[" << row << "][" << col << "]" << std::endl;
        Kokkos::deep_copy(get_u_block(block), rhs);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  bool ublock_all_eq(const size_type block, const scalar_t& value) const
  {
    auto u_block = get_u_block(block);
    for (size_type i = 0; i < block_size; ++i) {
      for (size_type j = 0; j < block_size; ++j) {
        if (u_block(i, j) != value) {
          return false;
        }
      }
    }
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  void verbose_iwset(const size_type tid, const size_type offset, const lno_t value) const
  {
    if (verbose)
      std::cout << "        JGF Setting iw[" << tid << "][" << offset << "] = " << value << std::endl;
    iw(tid, offset) = value;
  }

  KOKKOS_INLINE_FUNCTION
  LValuesUnmanaged2DBlockType get_l_block(const size_type block) const
  {
    return LValuesUnmanaged2DBlockType(L_values.data() + (block * block_items), block_size, block_size);
  }

  KOKKOS_INLINE_FUNCTION
  UValuesUnmanaged2DBlockType get_u_block(const size_type block) const
  {
    return UValuesUnmanaged2DBlockType(U_values.data() + (block * block_items), block_size, block_size);
  }

  KOKKOS_INLINE_FUNCTION
  AValuesUnmanaged2DBlockType get_a_block(const size_type block) const
  {
    return AValuesUnmanaged2DBlockType(A_values.data() + (block * block_items), block_size, block_size);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const lno_t lvl) const {
    auto rowid = level_idx(lvl);
    auto tid   = lvl - lev_start;
    auto k1    = L_row_map(rowid);
    auto k2    = L_row_map(rowid + 1);

    if (verbose) {
      std::cout << "    JGF level ptr: " << lvl << ", tid: " << tid << ", rowid: " << rowid << std::endl;

      std::cout << "      JGF Block 1" << std::endl;
      std::cout << "      L:" << std::endl;
      print_matrix(decompress_matrix(L_row_map, L_entries, L_values, block_size));
      std::cout << "      U:" << std::endl;
      print_matrix(decompress_matrix(U_row_map, U_entries, U_values, block_size));
    }
#ifdef KEEP_DIAG
    for (auto k = k1; k < k2 - 1; ++k) {
#else
    for (auto k = k1; k < k2; ++k) {
#endif
      auto col     = L_entries(k);
      verbose_lset(k, 0.0);
      verbose_iwset(tid, col, k);
    }
#ifdef KEEP_DIAG
    KokkosBatched::SerialSetIdentity::invoke(get_l_block(k2 -1));
#endif

    if (verbose) {
      print_iw(iw);

      std::cout << "      JGF Block 2" << std::endl;
      std::cout << "      L:" << std::endl;
      print_matrix(decompress_matrix(L_row_map, L_entries, L_values, block_size));
      std::cout << "      U:" << std::endl;
      print_matrix(decompress_matrix(U_row_map, U_entries, U_values, block_size));
    }
    k1 = U_row_map(rowid);
    k2 = U_row_map(rowid + 1);
    for (auto k = k1; k < k2; ++k) {
      auto col     = U_entries(k);
      verbose_uset(k, 0.0);
      verbose_iwset(tid, col, k);
    }

    // Unpack the ith row of A
    if (verbose) {
      print_iw(iw);

      std::cout << "      JGF Block 3" << std::endl;
      std::cout << "      L:" << std::endl;
      print_matrix(decompress_matrix(L_row_map, L_entries, L_values, block_size));
      std::cout << "      U:" << std::endl;
      print_matrix(decompress_matrix(U_row_map, U_entries, U_values, block_size));
    }
    k1 = A_row_map(rowid);
    k2 = A_row_map(rowid + 1);
    for (auto k = k1; k < k2; ++k) {
      auto col  = A_entries(k);
      auto ipos = iw(tid, col);
      if (col < rowid) {
        verbose_lset_block(ipos, get_a_block(k));
      }
      else {
        verbose_uset_block(ipos, get_a_block(k));
      }
    }

    // Eliminate prev rows
    if (verbose) {
      std::cout << "      JGF Block 4" << std::endl;
      std::cout << "      L:" << std::endl;
      print_matrix(decompress_matrix(L_row_map, L_entries, L_values, block_size));
      std::cout << "      U:" << std::endl;
      print_matrix(decompress_matrix(U_row_map, U_entries, U_values, block_size));
      print_iw(iw);
    }
    k1 = L_row_map(rowid);
    k2 = L_row_map(rowid + 1);
#ifdef KEEP_DIAG
    for (auto k = k1; k < k2 - 1; ++k) {
#else
    for (auto k = k1; k < k2; ++k) {
#endif
      auto prev_row = L_entries(k);
      if (verbose)
        std::cout << "        JGF Processing L[" << rowid << "][" << prev_row << "]" << std::endl;
      auto fact = get_l_block(k);
      auto u_diag = get_u_block(U_row_map(prev_row));
#ifdef KEEP_DIAG
      KokkosBatched::SerialTrsm<KokkosBatched::Side::Right,
                                KokkosBatched::Uplo::Upper,
                                KokkosBatched::Trans::Transpose, // not 100% on this
                                KokkosBatched::Diag::NonUnit,
                                KokkosBatched::Algo::Trsm::Unblocked>:: // not 100% on this
        invoke<scalar_t>(1.0, u_diag, fact);
#else
      // This should be a gemm
      auto fact = L_values(k) * U_values(U_row_map(prev_row));
#endif
      if (verbose)
        std::cout << "        JGF Block 4 trouble spot" << std::endl;
      for (auto kk = U_row_map(prev_row) + 1; kk < U_row_map(prev_row + 1);
           ++kk) {
        auto col  = U_entries(kk);
        auto ipos = iw(tid, col);
        if (verbose) {
          std::cout << "          JGF Processing U[" << prev_row << "][" << col << "]" << std::endl;
          std::cout << "          JGF rowid=" << rowid <<", prev_row=" << prev_row << ", kk=" << kk << ", col=" << col << ", ipos=" << ipos << std::endl;
        }
        if (ipos == -1) continue;

        KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose,
                                  KokkosBatched::Trans::NoTranspose,
                                  KokkosBatched::Algo::Gemm::Unblocked>::
          invoke(-1.0, get_u_block(kk), fact, 0.0, temp_dense_block);
        if (col < rowid) {
          KokkosBatched::SerialAxpy::invoke(ones, temp_dense_block, get_l_block(ipos));
        }
        else {
          KokkosBatched::SerialAxpy::invoke(ones, temp_dense_block, get_u_block(ipos));
        }
      }  // end for kk
      if (verbose)
        std::cout << "        JGF Block 4 trouble end" << std::endl;
    }    // end for k

    if (verbose) {
      std::cout << "      L:" << std::endl;
      print_matrix(decompress_matrix(L_row_map, L_entries, L_values, block_size));
      std::cout << "      U:" << std::endl;
      print_matrix(decompress_matrix(U_row_map, U_entries, U_values, block_size));

      std::cout << "      JGF Block 5" << std::endl;
    }
    const auto diag_ipos = iw(tid, rowid);
    if (ublock_all_eq(diag_ipos, 0.0)) {
      verbose_uset(diag_ipos, 1e6);
    }
#ifndef KEEP_DIAG
    else {
      assert(false);
      //verbose_uset(diag_ipos, 1.0 / U_values(diag_ipos));
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
          class URowMapType, class UEntriesType, class UValuesType>
struct ILUKLvlSchedTP1NumericFunctor {
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

  ILUKLvlSchedTP1NumericFunctor(
      const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
      const AValuesType &A_values_, const LRowMapType &L_row_map_,
      const LEntriesType &L_entries_, LValuesType &L_values_,
      const URowMapType &U_row_map_, const UEntriesType &U_entries_,
      UValuesType &U_values_, const LevelViewType &level_idx_,
      WorkViewType &iw_, const lno_t &lev_start_)
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
    lno_t my_team = static_cast<lno_t>(team.league_rank());
    lno_t rowid =
        static_cast<lno_t>(level_idx(my_team + lev_start));  // map to rowid

    size_type k1 = static_cast<size_type>(L_row_map(rowid));
    size_type k2 = static_cast<size_type>(L_row_map(rowid + 1));
#ifdef KEEP_DIAG
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2 - 1),
                         [&](const size_type k) {
                           lno_t col = static_cast<lno_t>(L_entries(k));
                           L_values(k)   = 0.0;
                           iw(my_team, col) = static_cast<lno_t>(k);
                         });
#else
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
                           lno_t col = static_cast<lno_t>(L_entries(k));
                           L_values(k)   = 0.0;
                           iw(my_team, col) = static_cast<lno_t>(k);
                         });
#endif

#ifdef KEEP_DIAG
    // if (my_thread == 0) L_values(k2 - 1) = scalar_t(1.0);
    Kokkos::single(Kokkos::PerTeam(team),
                   [&]() { L_values(k2 - 1) = scalar_t(1.0); });
#endif

    team.team_barrier();

    k1 = static_cast<size_type>(U_row_map(rowid));
    k2 = static_cast<size_type>(U_row_map(rowid + 1));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
                           lno_t col = static_cast<lno_t>(U_entries(k));
                           U_values(k)   = 0.0;
                           iw(my_team, col) = static_cast<lno_t>(k);
                         });

    team.team_barrier();

    // Unpack the ith row of A
    k1 = static_cast<size_type>(A_row_map(rowid));
    k2 = static_cast<size_type>(A_row_map(rowid + 1));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
                           lno_t col = static_cast<lno_t>(A_entries(k));
                           lno_t ipos = iw(my_team, col);
                           if (col < rowid)
                             L_values(ipos) = A_values(k);
                           else
                             U_values(ipos) = A_values(k);
                         });

    team.team_barrier();

    // Eliminate prev rows
    k1 = static_cast<size_type>(L_row_map(rowid));
    k2 = static_cast<size_type>(L_row_map(rowid + 1));
#ifdef KEEP_DIAG
    for (size_type k = k1; k < k2 - 1; k++)
#else
    for (size_type k = k1; k < k2; k++)
#endif
    {
      lno_t prev_row = L_entries(k);

      scalar_t fact = scalar_t(0.0);
      Kokkos::single(
          Kokkos::PerTeam(team),
          [&](scalar_t &tmp_fact) {
#ifdef KEEP_DIAG
            tmp_fact = L_values(k) / U_values(U_row_map(prev_row));
#else
            tmp_fact = L_values(k) * U_values(U_row_map(prev_row));
#endif
            L_values(k) = tmp_fact;
          },
          fact);

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, U_row_map(prev_row) + 1,
                                  U_row_map(prev_row + 1)),
          [&](const size_type kk) {
            lno_t col  = static_cast<lno_t>(U_entries(kk));
            lno_t ipos = iw(my_team, col);
            auto lxu       = -U_values(kk) * fact;
            if (ipos != -1) {
              if (col < rowid)
                L_values(ipos) += lxu;
              else
                U_values(ipos) += lxu;
            }
          });  // end for kk

      team.team_barrier();
    }  // end for k

    // if (my_thread == 0) {
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      lno_t ipos = iw(my_team, rowid);
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
    k1 = static_cast<size_type>(L_row_map(rowid));
    k2 = static_cast<size_type>(L_row_map(rowid + 1));
#ifdef KEEP_DIAG
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2 - 1),
                         [&](const size_type k) {
                           lno_t col = static_cast<lno_t>(L_entries(k));
                           iw(my_team, col) = -1;
                         });
#else
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
                           lno_t col = static_cast<lno_t>(L_entries(k));
                           iw(my_team, col) = -1;
                         });
#endif

    k1 = static_cast<size_type>(U_row_map(rowid));
    k2 = static_cast<size_type>(U_row_map(rowid + 1));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
                           lno_t col = static_cast<lno_t>(U_entries(k));
                           iw(my_team, col) = -1;
                         });
  }
};

template <class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType>
struct ILUKLvlSchedTP1NumericFunctorBlock {
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
  bool verbose;

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

  ILUKLvlSchedTP1NumericFunctorBlock(
      const ARowMapType &A_row_map_, const AEntriesType &A_entries_,
      const AValuesType &A_values_, const LRowMapType &L_row_map_,
      const LEntriesType &L_entries_, LValuesType &L_values_,
      const URowMapType &U_row_map_, const UEntriesType &U_entries_,
      UValuesType &U_values_, const LevelViewType &level_idx_,
      WorkViewType &iw_, const lno_t &lev_start_, const size_type& block_size_)
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
        temp_dense_block("temp_dense_block", block_size, block_size), // this will have races unless Serial
        ones("ones", block_size),
        verbose(false)
  {
    Kokkos::deep_copy(ones, 1.0);
  }

    KOKKOS_INLINE_FUNCTION
  void verbose_lset(const size_type block, const scalar_t& value) const
  {
    const size_type nrows = L_row_map.extent(0) - 1;
    for (size_type row = 0; row < nrows; ++row) {
      const auto row_begin = L_row_map(row);
      const auto row_end   = L_row_map(row+1);
      if (block >= row_begin && block < row_end) {
        const auto col = L_entries(block);
        if (verbose)
          std::cout << "        JGF Setting L_values[" << row << "][" << col << "] = " << value << std::endl;
        KokkosBlas::SerialSet::invoke(value, get_l_block(block));
      }
    }
  }

  template <typename BlockType>
  KOKKOS_INLINE_FUNCTION
  void verbose_lset_block(const size_type block, const BlockType& rhs) const
  {
    const size_type nrows = L_row_map.extent(0) - 1;
    for (size_type row = 0; row < nrows; ++row) {
      const auto row_begin = L_row_map(row);
      const auto row_end   = L_row_map(row+1);
      if (block >= row_begin && block < row_end) {
        const auto col = L_entries(block);
        if (verbose)
          std::cout << "        JGF Setting block L_values[" << row << "][" << col << "]" << std::endl;
        Kokkos::deep_copy(get_l_block(block), rhs);
      }
    }
  }


  KOKKOS_INLINE_FUNCTION
  void verbose_uset(const size_type block, const scalar_t& value) const
  {
    const size_type nrows = U_row_map.extent(0) - 1;
    for (size_type row = 0; row < nrows; ++row) {
      const auto row_begin = U_row_map(row);
      const auto row_end   = U_row_map(row+1);
      if (block >= row_begin && block < row_end) {
        const auto col = U_entries(block);
        if (verbose)
          std::cout << "        JGF Setting U_values[" << row << "][" << col << "] = " << value << std::endl;
        KokkosBlas::SerialSet::invoke(value, get_u_block(block));
      }
    }
  }

  template <typename BlockType>
  KOKKOS_INLINE_FUNCTION
  void verbose_uset_block(const size_type block, const BlockType& rhs) const
  {
    const size_type nrows = U_row_map.extent(0) - 1;
    for (size_type row = 0; row < nrows; ++row) {
      const auto row_begin = U_row_map(row);
      const auto row_end   = U_row_map(row+1);
      if (block >= row_begin && block < row_end) {
        const auto col = U_entries(block);
        std::cout << "        JGF Setting block U_values[" << row << "][" << col << "]" << std::endl;
        Kokkos::deep_copy(get_u_block(block), rhs);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  bool ublock_all_eq(const size_type block, const scalar_t& value) const
  {
    auto u_block = get_u_block(block);
    for (size_type i = 0; i < block_size; ++i) {
      for (size_type j = 0; j < block_size; ++j) {
        if (u_block(i, j) != value) {
          return false;
        }
      }
    }
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  void verbose_iwset(const size_type tid, const size_type offset, const lno_t value) const
  {
    if (verbose)
      std::cout << "        JGF Setting iw[" << tid << "][" << offset << "] = " << value << std::endl;
    iw(tid, offset) = value;
  }

  KOKKOS_INLINE_FUNCTION
  LValuesUnmanaged2DBlockType get_l_block(const size_type block) const
  {
    return LValuesUnmanaged2DBlockType(L_values.data() + (block * block_items), block_size, block_size);
  }

  KOKKOS_INLINE_FUNCTION
  UValuesUnmanaged2DBlockType get_u_block(const size_type block) const
  {
    return UValuesUnmanaged2DBlockType(U_values.data() + (block * block_items), block_size, block_size);
  }

  KOKKOS_INLINE_FUNCTION
  AValuesUnmanaged2DBlockType get_a_block(const size_type block) const
  {
    return AValuesUnmanaged2DBlockType(A_values.data() + (block * block_items), block_size, block_size);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type &team) const {
    lno_t my_team = static_cast<lno_t>(team.league_rank());
    lno_t rowid =
        static_cast<lno_t>(level_idx(my_team + lev_start));  // map to rowid

    size_type k1 = static_cast<size_type>(L_row_map(rowid));
    size_type k2 = static_cast<size_type>(L_row_map(rowid + 1));
#ifdef KEEP_DIAG
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2 - 1),
#else
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
#endif
                         [&](const size_type k) {
                           lno_t col = static_cast<lno_t>(L_entries(k));
                           verbose_lset(k, 0.0);
                           verbose_iwset(my_team, col, k);
                         });

#ifdef KEEP_DIAG
    KokkosBatched::TeamSetIdentity<member_type>::invoke(team, get_l_block(k2 -1));
#endif

    team.team_barrier();

    k1 = static_cast<size_type>(U_row_map(rowid));
    k2 = static_cast<size_type>(U_row_map(rowid + 1));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
                           lno_t col = static_cast<lno_t>(U_entries(k));
                           verbose_uset(k, 0.0);
                           verbose_iwset(my_team, col, k);
                         });

    team.team_barrier();

    // Unpack the ith row of A
    k1 = static_cast<size_type>(A_row_map(rowid));
    k2 = static_cast<size_type>(A_row_map(rowid + 1));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
                           lno_t col = static_cast<lno_t>(A_entries(k));
                           lno_t ipos = iw(my_team, col);
                           if (col < rowid)
                             verbose_lset_block(ipos, get_a_block(k));
                           else
                             verbose_uset_block(ipos, get_a_block(k));
                         });

    team.team_barrier();

    // Eliminate prev rows
    k1 = static_cast<size_type>(L_row_map(rowid));
    k2 = static_cast<size_type>(L_row_map(rowid + 1));
#ifdef KEEP_DIAG
    for (size_type k = k1; k < k2 - 1; k++)
#else
    for (size_type k = k1; k < k2; k++)
#endif
    {
      lno_t prev_row = L_entries(k);

      auto fact = get_l_block(k);
      auto u_diag = get_u_block(U_row_map(prev_row));
#ifdef KEEP_DIAG
      KokkosBatched::TeamTrsm<member_type,
                              KokkosBatched::Side::Right,
                              KokkosBatched::Uplo::Upper,
                              KokkosBatched::Trans::Transpose, // not 100% on this
                              KokkosBatched::Diag::NonUnit,
                              KokkosBatched::Algo::Trsm::Unblocked>:: // not 100% on this
        invoke(team, 1.0, u_diag, fact);
#else
      // TeamGemm
#endif

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, U_row_map(prev_row) + 1,
                                  U_row_map(prev_row + 1)),
          [&](const size_type kk) {
            lno_t col  = static_cast<lno_t>(U_entries(kk));
            lno_t ipos = iw(my_team, col);
            if (ipos != -1) {
              KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose,
                                        KokkosBatched::Trans::NoTranspose,
                                        KokkosBatched::Algo::Gemm::Unblocked>::
                invoke(-1.0, get_u_block(kk), fact, 0.0, temp_dense_block);
              if (col < rowid) {
                KokkosBatched::SerialAxpy::invoke(ones, temp_dense_block, get_l_block(ipos));
              }
              else {
                KokkosBatched::SerialAxpy::invoke(ones, temp_dense_block, get_u_block(ipos));
              }
            }
          });  // end for kk

      team.team_barrier();
    }  // end for k

    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      lno_t ipos = iw(my_team, rowid);
      if (ublock_all_eq(ipos, 0.0)) {
        verbose_uset(ipos, 1e6);
      }
#ifndef KEEP_DIAG
      else {
        assert(false);
      }
#endif
    });

    team.team_barrier();

    // Reset
    k1 = static_cast<size_type>(L_row_map(rowid));
    k2 = static_cast<size_type>(L_row_map(rowid + 1));
#ifdef KEEP_DIAG
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2 - 1),
                         [&](const size_type k) {
                           lno_t col = static_cast<lno_t>(L_entries(k));
                           iw(my_team, col) = -1;
                         });
#else
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
                           lno_t col = static_cast<lno_t>(L_entries(k));
                           iw(my_team, col) = -1;
                         });
#endif

    k1 = static_cast<size_type>(U_row_map(rowid));
    k2 = static_cast<size_type>(U_row_map(rowid + 1));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k1, k2),
                         [&](const size_type k) {
                           lno_t col = static_cast<lno_t>(U_entries(k));
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
        Kokkos::parallel_for(
            "parfor_fixed_lvl",
            Kokkos::RangePolicy<execution_space>(lev_start, lev_end),
            ILUKLvlSchedRPNumericFunctor<
                ARowMapType, AEntriesType, AValuesType, LRowMapType,
                LEntriesType, LValuesType, URowMapType, UEntriesType,
                UValuesType>(
                A_row_map, A_entries, A_values, L_row_map, L_entries, L_values,
                U_row_map, U_entries, U_values, level_idx, iw, lev_start));
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

          ILUKLvlSchedTP1NumericFunctor<
              ARowMapType, AEntriesType, AValuesType, LRowMapType, LEntriesType,
              LValuesType, URowMapType, UEntriesType, UValuesType>
              tstf(A_row_map, A_entries, A_values, L_row_map, L_entries,
                   L_values, U_row_map, U_entries, U_values, level_idx, iw,
                   lev_start + lvl_rowid_start);

          if (team_size == -1)
            Kokkos::parallel_for(
                "parfor_tp1", policy_type(lvl_nrows_chunk, Kokkos::AUTO), tstf);
          else
            Kokkos::parallel_for("parfor_tp1",
                                 policy_type(lvl_nrows_chunk, team_size), tstf);
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

template <class ARowMapType, class AEntriesType,
          class AValuesType, class LRowMapType, class LEntriesType,
          class LValuesType, class URowMapType, class UEntriesType,
          class UValuesType>
static void iluk_numeric_block(IlukHandle &thandle, const ARowMapType &A_row_map,
                  const AEntriesType &A_entries, const AValuesType &A_values,
                  const LRowMapType &L_row_map, const LEntriesType &L_entries,
                  LValuesType &L_values, const URowMapType &U_row_map,
                  const UEntriesType &U_entries, UValuesType &U_values)
{
  bool verbose = false;
  const size_type nlevels    = thandle.get_num_levels();
  if (verbose)
    std::cout << "JGF iluk_numeric_block with nlevels: " << nlevels << std::endl;
  const int team_size        = thandle.get_team_size();
  const size_type block_size = thandle.get_block_size();

  LevelHostViewType level_ptr_h = thandle.get_host_level_ptr();
  LevelViewType     level_idx   = thandle.get_level_idx();

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
    if (verbose)
      std::cout << "  JGF starting level: " << lvl << std::endl;
    lno_t lev_start = level_ptr_h(lvl);
    lno_t lev_end   = level_ptr_h(lvl + 1);

    if ((lev_end - lev_start) != 0) {
      if (thandle.get_algorithm() ==
          KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_RP) {
        Kokkos::parallel_for(
            "parfor_fixed_lvl",
            Kokkos::RangePolicy<execution_space>(lev_start, lev_end),
            ILUKLvlSchedRPNumericFunctorBlock<
                ARowMapType, AEntriesType, AValuesType, LRowMapType,
                LEntriesType, LValuesType, URowMapType, UEntriesType,
                UValuesType>(
                A_row_map, A_entries, A_values, L_row_map, L_entries, L_values,
                U_row_map, U_entries, U_values, level_idx, iw, lev_start, block_size));
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

          ILUKLvlSchedTP1NumericFunctorBlock<
              ARowMapType, AEntriesType, AValuesType, LRowMapType, LEntriesType,
              LValuesType, URowMapType, UEntriesType, UValuesType>
              tstf(A_row_map, A_entries, A_values, L_row_map, L_entries,
                   L_values, U_row_map, U_entries, U_values, level_idx, iw,
                   lev_start + lvl_rowid_start, block_size);

          if (team_size == -1)
            Kokkos::parallel_for(
                "parfor_tp1", policy_type(lvl_nrows_chunk, Kokkos::AUTO), tstf);
          else
            Kokkos::parallel_for("parfor_tp1",
                                 policy_type(lvl_nrows_chunk, team_size), tstf);
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

}  // end iluk_numeric_block

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
              LValuesType, URowMapType, UEntriesType, UValuesType>
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
                  UValuesType>
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
