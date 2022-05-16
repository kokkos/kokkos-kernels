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

#ifndef KOKKOSSPARSE_IMPL_PAR_ILUT_NUMERIC_HPP_
#define KOKKOSSPARSE_IMPL_PAR_ILUT_NUMERIC_HPP_

/// \file KokkosSparse_par_ilut_numeric_impl.hpp
/// \brief Implementation(s) of the numeric phase of sparse parallel ILUT.

#include <KokkosKernels_config.h>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_par_ilut_handle.hpp>
#include "KokkosSparse_spgemm.hpp"

#include <limits>

//#define NUMERIC_OUTPUT_INFO

namespace KokkosSparse {
namespace Impl {
namespace Experimental {

template <typename scalar_t, typename RowMapType, typename EntriesType, typename ValuesType>
std::vector<std::vector<scalar_t>> decompress_matrix(
  const RowMapType& row_map,
  const EntriesType& entries,
  const ValuesType& values
                                                     )
{
  using size_type = typename RowMapType::non_const_value_type;
  using lno_t     = typename EntriesType::non_const_value_type;

  const auto nrows = row_map.size() - 1;
  std::vector<std::vector<scalar_t> > result;
  result.resize(nrows);
  for (auto& row : result) {
    row.resize(nrows, 0.0);
  }

  auto hrow_map = Kokkos::create_mirror_view(row_map);
  auto hentries = Kokkos::create_mirror_view(entries);
  auto hvalues  = Kokkos::create_mirror_view(values);
  Kokkos::deep_copy(hrow_map, row_map);
  Kokkos::deep_copy(hentries, entries);
  Kokkos::deep_copy(hvalues, values);

  for (size_type row_idx = 0; row_idx < nrows; ++row_idx) {
    const size_type row_nnz_begin = row_map(row_idx);
    const size_type row_nnz_end   = row_map(row_idx+1);
    for (size_type row_nnz = row_nnz_begin; row_nnz < row_nnz_end; ++row_nnz) {
      const lno_t col_idx = entries(row_nnz);
      const scalar_t value = values(row_nnz);
      result[row_idx][col_idx] = value;
    }
  }

  return result;
}

template <typename scalar_t>
void print_matrix(const std::vector<std::vector<scalar_t> >& matrix)
{
  for (const auto& row : matrix) {
    for (const auto& item : row) {
      std::printf("%.2f ", item);
    }
    std::cout << std::endl;
  }
}

template <class IlutHandle, class RowMapType, class PrefixSumView>
typename IlutHandle::size_type prefix_sum(IlutHandle& ih, RowMapType& row_map, PrefixSumView& prefix_sum_view)
{
  using size_type   = typename IlutHandle::size_type;
  using policy_type = typename IlutHandle::TeamPolicy;
  using member_type = typename policy_type::member_type;
  using RangePolicy = typename IlutHandle::RangePolicy;

  const auto policy = ih.get_default_team_policy();
  const size_type nteams = policy.league_size();
  const size_type nrows  = ih.get_nrows() + 1;
  const size_type rows_per_team = (nrows - 1) / nteams + 1;

  size_type total_sum = 0;
  Kokkos::parallel_for(
    "prefix_sum per-team sums",
    policy,
    KOKKOS_LAMBDA(const member_type& team) {
      const auto team_rank = team.league_rank();

      const size_type starti = team_rank * rows_per_team;
      const size_type endi   = std::min(nrows, (team_rank + 1) * rows_per_team);
      size_type sum = 0;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, starti, endi),
        [&](const size_type row, size_type& sum_inner) {
          sum_inner += row_map(row);
      }, sum);

      team.team_barrier();

      Kokkos::single(
        Kokkos::PerTeam(team), [&] () {
          prefix_sum_view(team_rank) = sum;
          size_type curr_sum = 0;
          for (size_type row = starti; row < endi; ++row) {
            const size_type curr_val = row_map(row);
            row_map(row) = curr_sum;
            curr_sum += curr_val;
          }
      });
  });

  Kokkos::fence();

  Kokkos::parallel_for(
    "prefix_sum per-team prefix",
    RangePolicy(0, 1), // No parallelism in this alg
    KOKKOS_LAMBDA(const size_type) {
      size_type team_sum = 0;
      for (size_type t = 0; t < nteams; ++t) {
        const size_type team_result = prefix_sum_view(t);
        prefix_sum_view(t) = team_sum;
        team_sum += team_result;
      }
  });

  Kokkos::fence();

  Kokkos::parallel_reduce(
    "prefix_sum finish",
    policy,
    KOKKOS_LAMBDA(const member_type& team, size_type& total_sum_outer) {
      const auto team_rank = team.league_rank();
      const size_type starti = team_rank * rows_per_team;
      const size_type endi   = std::min(nrows, (team_rank + 1) * rows_per_team);
      if (team_rank != 0) {
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, starti, endi),
          [&](const size_type row) {
            row_map(row) += prefix_sum_view(team_rank);
        });
      }

      total_sum_outer = row_map(endi - 1);

  }, Kokkos::Max<size_type>(total_sum));

  Kokkos::fence();

  return total_sum;
}

template <class IlutHandle,
          class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType,
          class LURowMapType, class LUEntriesType, class LUValuesType,
          class LNewRowMapType, class LNewEntriesType, class LNewValuesType,
          class UNewRowMapType, class UNewEntriesType, class UNewValuesType,
          class PrefixSumType>
void add_candidates(
  IlutHandle& ih,
  const ARowMapType& A_row_map, const AEntriesType& A_entries, const AValuesType& A_values,
  const LRowMapType& L_row_map, const LEntriesType& L_entries, const LValuesType& L_values,
  const URowMapType& U_row_map, const UEntriesType& U_entries, const UValuesType& U_values,
  const LURowMapType& LU_row_map, const LUEntriesType& LU_entries, const LUValuesType& LU_values,
  LNewRowMapType& L_new_row_map, LNewEntriesType& L_new_entries, LNewValuesType& L_new_values,
  UNewRowMapType& U_new_row_map, UNewEntriesType& U_new_entries, UNewValuesType& U_new_values,
  PrefixSumType& prefix_sum_view)
{
  using size_type       = typename IlutHandle::size_type;
  using policy_type     = typename IlutHandle::TeamPolicy;
  using member_type     = typename policy_type::member_type;
  using range_policy    = typename IlutHandle::RangePolicy;

  const auto policy = ih.get_default_team_policy();

  const size_type nrows = ih.get_nrows();

  Kokkos::parallel_for(
    "add_candidates sizing",
    policy,
    KOKKOS_LAMBDA(const member_type& team) {
      const auto row_idx = team.league_rank();

      const auto a_row_nnz_begin = A_row_map(row_idx);
      const auto a_row_nnz_end   = A_row_map(row_idx+1);

      const auto lu_row_nnz_begin = LU_row_map(row_idx);
      const auto lu_row_nnz_end   = LU_row_map(row_idx+1);

      size_type a_l_nnz = 0, a_u_nnz = 0, lu_l_nnz = 0, lu_u_nnz = 0, dup_l_nnz = 0, dup_u_nnz = 0;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, a_row_nnz_begin, a_row_nnz_end),
        [&](const size_type nnz, size_type& nnzL_inner) {
          const auto col_idx = A_entries(nnz);
          nnzL_inner += col_idx <= row_idx;
      }, a_l_nnz);

      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, a_row_nnz_begin, a_row_nnz_end),
        [&](const size_type nnz, size_type& nnzU_inner) {
          const auto col_idx = A_entries(nnz);
          nnzU_inner += col_idx >= row_idx;
      }, a_u_nnz);

      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, lu_row_nnz_begin, lu_row_nnz_end),
        [&](const size_type nnz, size_type& nnzL_inner) {
          const auto col_idx = LU_entries(nnz);
          nnzL_inner += col_idx <= row_idx;
      }, lu_l_nnz);

      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, lu_row_nnz_begin, lu_row_nnz_end),
        [&](const size_type nnz, size_type& nnzU_inner) {
          const auto col_idx = LU_entries(nnz);
          nnzU_inner += col_idx >= row_idx;
      }, lu_u_nnz);

      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, a_row_nnz_begin, a_row_nnz_end),
        [&](const size_type nnz, size_type& dupL_inner) {
          const auto a_col_idx = A_entries(nnz);
          if (a_col_idx <= row_idx) {
            for (size_type lu_i = lu_row_nnz_begin; lu_i < lu_row_nnz_end; ++lu_i) {
              const auto lu_col_idx = LU_entries(lu_i);
              if (a_col_idx == lu_col_idx) {
                ++dupL_inner;
                break;
              }
              else if (lu_col_idx > a_col_idx) {
                break;
              }
            }
          }
      }, dup_l_nnz);

      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, a_row_nnz_begin, a_row_nnz_end),
        [&](const size_type nnz, size_type& dupU_inner) {
          const auto a_col_idx = A_entries(nnz);
          if (a_col_idx >= row_idx) {
            for (size_type lu_i = lu_row_nnz_begin; lu_i < lu_row_nnz_end; ++lu_i) {
              const auto lu_col_idx = LU_entries(lu_i);
              if (a_col_idx == lu_col_idx) {
                ++dupU_inner;
                break;
              }
              else if (lu_col_idx > a_col_idx) {
                break;
              }
            }
          }
      }, dup_u_nnz);

      team.team_barrier();

      Kokkos::single(
        Kokkos::PerTeam(team), [&] () {
          const auto l_nnz = (a_l_nnz + lu_l_nnz - dup_l_nnz);
          const auto u_nnz = (a_u_nnz + lu_u_nnz - dup_u_nnz);

          L_new_row_map(row_idx) = l_nnz;
          U_new_row_map(row_idx) = u_nnz;
      });
  });

  // prefix sum
  const size_type l_new_nnz = prefix_sum(ih, L_new_row_map, prefix_sum_view);
  const size_type u_new_nnz = prefix_sum(ih, U_new_row_map, prefix_sum_view);

  Kokkos::resize(L_new_entries, l_new_nnz);
  Kokkos::resize(U_new_entries, u_new_nnz);
  Kokkos::resize(L_new_values,  l_new_nnz);
  Kokkos::resize(U_new_values,  u_new_nnz);

  constexpr auto sentinel = std::numeric_limits<size_type>::max();

  using scalar_t        = typename AValuesType::non_const_value_type;

  std::cout << "A:" << std::endl;
  print_matrix(decompress_matrix<scalar_t>(A_row_map, A_entries, A_values));

  std::cout << "LU:" << std::endl;
  print_matrix(decompress_matrix<scalar_t>(LU_row_map, LU_entries, LU_values));

  std::cout << "u_new_nnz: " << u_new_nnz << " extent(0) = " << U_new_entries.extent(0) << std::endl;

  // Now compute the actual candidate values
  Kokkos::parallel_for(
    "add_candidates",
    range_policy(0, nrows), // No team level parallelism in this alg
    KOKKOS_LAMBDA(const size_type row_idx) {
            auto a_row_nnz_begin = A_row_map(row_idx);
      const auto a_row_nnz_end   = A_row_map(row_idx+1);
      const auto a_tot           = a_row_nnz_end - a_row_nnz_begin;

            auto lu_row_nnz_begin = LU_row_map(row_idx);
      const auto lu_row_nnz_end   = LU_row_map(row_idx+1);
      const auto lu_tot           = lu_row_nnz_end - lu_row_nnz_begin;

      const auto tot = a_tot + lu_tot;

      size_type l_new_nnz   = L_new_row_map(row_idx);
      size_type u_new_nnz   = U_new_row_map(row_idx);
      size_type l_old_begin = L_row_map(row_idx);
      size_type l_old_end   = L_row_map(row_idx+1) - 1; // skip diagonal
      size_type u_old_begin = U_row_map(row_idx);
      size_type u_old_end   = U_row_map(row_idx+1);
      bool      finished_l  = false;
      bool      skip        = false;
      std::cout << "Begin row_idx: " << row_idx << " with tot=" << tot << std::endl;
      for (size_type i = 0; i < tot; ++i) {
        if (skip) {
          skip = false;
          continue;
        }

        const auto a_col  = a_row_nnz_begin  < a_row_nnz_end  ? A_entries(a_row_nnz_begin)   : sentinel;
              auto a_val  = a_row_nnz_begin  < a_row_nnz_end  ? A_values(a_row_nnz_begin)    : 0.;
        const auto lu_col = lu_row_nnz_begin < lu_row_nnz_end ? LU_entries(lu_row_nnz_begin) : sentinel;
              auto lu_val = lu_row_nnz_begin < lu_row_nnz_end ? LU_values(lu_row_nnz_begin)  : 0.;

        const auto col_idx = std::min(a_col, lu_col);

        const bool a_active  = col_idx == a_col;
        const bool lu_active = col_idx == lu_col;

        a_val  = a_active  ? a_val  : 0.;
        lu_val = lu_active ? lu_val : 0.;

        skip = a_active && lu_active;

        a_row_nnz_begin  += a_active;
        lu_row_nnz_begin += lu_active;

        const auto r_val = a_val - lu_val;
        // load matching entry of L + U
        const auto lpu_col = finished_l ? (u_old_begin < u_old_end ? U_entries(u_old_begin) : sentinel) : L_entries(l_old_begin);
        const auto lpu_val = finished_l ? (u_old_begin < u_old_end ?  U_values(u_old_begin) : 0.      ) : L_values(l_old_begin);
        // load diagonal entry of U for lower diagonal entries
        const auto diag = col_idx < row_idx ? U_values(U_row_map(col_idx)) : 1.;
        // if there is already an entry present, use that instead.
        const auto out_val = lpu_col == col_idx ? lpu_val : r_val / diag;
        // store output entries
        std::cout << "  row_idx=" << row_idx << ", col_idx=" << col_idx << ", l_new_nnz=" << l_new_nnz << ", u_new_nnz=" << u_new_nnz << ", out_val=" << out_val << ", a_col=" << a_col << ", lu_col=" << lu_col << " a_row_nnz_begin=" << a_row_nnz_begin << ", lu_row_nnz_begin=" << lu_row_nnz_begin << std::endl;
        if (row_idx >= col_idx) {
          L_new_entries(l_new_nnz) = col_idx;
          L_new_values(l_new_nnz)  = row_idx == col_idx ? 1. : out_val;
          ++l_new_nnz;
        }
        if (row_idx <= col_idx) {
          U_new_entries(u_new_nnz) = col_idx;
          U_new_values(u_new_nnz)  = out_val;
          std::cout << "    Setting U_new_entries(" << u_new_nnz << ") = " << col_idx << "( " << out_val << ")" << std::endl;
          ++u_new_nnz;
        }
        // advance entry of L + U if we used it
        if (finished_l) {
          u_old_begin += (lpu_col == col_idx);
        } else {
          l_old_begin += (lpu_col == col_idx);
          finished_l = (l_old_begin == l_old_end);
        }
      }

  });
}

template <class KHandle, class IlutHandle,
          class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType>
void ilut_numeric(KHandle& kh, IlutHandle &thandle, const ARowMapType &A_row_map,
                  const AEntriesType &A_entries, const AValuesType &A_values,
                  LRowMapType &L_row_map, LEntriesType &L_entries, LValuesType &L_values,
                  URowMapType &U_row_map, UEntriesType &U_entries, UValuesType &U_values) {
  using execution_space         = typename IlutHandle::execution_space;
  using memory_space            = typename IlutHandle::memory_space;
  using size_type               = typename IlutHandle::size_type;
  using nnz_lno_t               = typename IlutHandle::nnz_lno_t;
  using HandleDeviceEntriesType = typename IlutHandle::nnz_lno_view_t;
  using HandleDeviceRowMapType  = typename IlutHandle::nnz_row_view_t;
  using HandleDeviceValueType   = typename IlutHandle::nnz_value_view_t;

  const size_type nlevels = thandle.get_num_levels();
  const size_type nrows   = thandle.get_nrows();

  bool converged = false;

  std::string myalg("SPGEMM_KK_MEMORY");
  KokkosSparse::SPGEMMAlgorithm spgemm_algorithm =
    KokkosSparse::StringToSPGEMMAlgorithm(myalg);
  kh.create_spgemm_handle(spgemm_algorithm);

  const auto policy = thandle.get_default_team_policy();

  // temporary workspaces
  HandleDeviceRowMapType
    LU_row_map(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "LU_row_map"),
      nrows + 1),
    L_new_row_map(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "L_new_row_map"),
      nrows + 1),
    U_new_row_map(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "U_new_row_map"),
      nrows + 1),
    prefix_sum_view(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "prefix_sum_view"),
      policy.league_size());
  HandleDeviceEntriesType LU_entries, L_new_entries, U_new_entries;
  HandleDeviceValueType LU_values, L_new_values, U_new_values;

  while (!converged) {
    // LU = L*U
    KokkosSparse::Experimental::spgemm_symbolic(
      &kh, nrows, nrows, nrows,
      L_row_map, L_entries, false,
      U_row_map, U_entries, false,
      LU_row_map);

    const size_type lu_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
    Kokkos::resize(LU_entries, lu_nnz_size);
    Kokkos::resize(LU_values, lu_nnz_size);

    KokkosSparse::Experimental::spgemm_numeric(
      &kh, nrows, nrows, nrows,
      L_row_map, L_entries, L_values, false,
      U_row_map, U_entries, U_values, false,
      LU_row_map, LU_entries, LU_values);

    add_candidates(thandle,
      A_row_map, A_entries, A_values,
      L_row_map, L_entries, L_values,
      U_row_map, U_entries, U_values,
      LU_row_map, LU_entries, LU_values,
      L_new_row_map, L_new_entries, L_new_values,
      U_new_row_map, U_new_entries, U_new_values,
      prefix_sum_view);

    converged = true;
  }

  Kokkos::deep_copy(L_row_map, L_new_row_map);
  Kokkos::resize(L_entries,    L_new_entries.extent(0));
  Kokkos::deep_copy(L_entries, L_new_entries);
  Kokkos::resize(L_values,     L_new_values.extent(0));
  Kokkos::deep_copy(L_values,  L_new_values);

  Kokkos::deep_copy(U_row_map, U_new_row_map);
  std::cout << "HERE1: " << U_new_entries.extent(0) << std::endl;
  Kokkos::resize(U_entries,    U_new_entries.extent(0));
  std::cout << "HERE2: " << U_entries.extent(0) << std::endl;
  Kokkos::deep_copy(U_entries, U_new_entries);
  Kokkos::resize(U_values,     U_new_values.extent(0));
  Kokkos::deep_copy(U_values,  U_new_values);

}  // end ilut_numeric

}  // namespace Experimental
}  // namespace Impl
}  // namespace KokkosSparse

#endif
