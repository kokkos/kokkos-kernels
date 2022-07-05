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
#include <KokkosSparse_spgemm.hpp>
#include <KokkosKernels_SparseUtils.hpp>

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
      const size_type endi   = Kokkos::fmin(nrows, (team_rank + 1) * rows_per_team);
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
      const size_type endi   = Kokkos::fmin(nrows, (team_rank + 1) * rows_per_team);
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

  const size_type nrows = ih.get_nrows();

  //const auto policy = ih.get_default_team_policy();
  policy_type policy = policy_type(1, 1);

  Kokkos::parallel_for(
    "add_candidates sizing",
    policy,
    KOKKOS_LAMBDA(const member_type& team) {
      for (size_type row_idx = 0; row_idx < nrows; ++row_idx) {
        //const auto row_idx = team.league_rank();

      const auto a_row_nnz_begin = A_row_map(row_idx);
      const auto a_row_nnz_end   = A_row_map(row_idx+1);

      const auto lu_row_nnz_begin = LU_row_map(row_idx);
      const auto lu_row_nnz_end   = LU_row_map(row_idx+1);

      printf("For row_idx: %lu, nnz_begin=%lu, nnz_end=%lu\n", row_idx, lu_row_nnz_begin, lu_row_nnz_end);

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
          printf("Checking dups for row_idx: %lu, a_col: %lu\n", row_idx, a_col_idx);
          if (a_col_idx <= row_idx) {
            for (size_type lu_i = lu_row_nnz_begin; lu_i < lu_row_nnz_end; ++lu_i) {
              const auto lu_col_idx = LU_entries(lu_i);
              printf("  Checking lu_col_idx: %lu against a_col_idx: %lu \n", lu_col_idx, a_col_idx);
              if (a_col_idx == lu_col_idx) {
                printf("    Found!\n");
                ++dupL_inner;
                break;
              }
              else if (lu_col_idx > a_col_idx) {
                printf("    Break!\n");
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

      printf("a_l_nnz: %lu lu_l_nnz: %lu dup_l_nnz: %lu\n", a_l_nnz, lu_l_nnz, dup_l_nnz);

      Kokkos::single(
        Kokkos::PerTeam(team), [&] () {
          const auto l_nnz = ((a_l_nnz + lu_l_nnz) - dup_l_nnz);
          printf("l_nnz: %lu\n", l_nnz);
          const auto u_nnz = (a_u_nnz + lu_u_nnz - dup_u_nnz);

          L_new_row_map(row_idx) = l_nnz;
          U_new_row_map(row_idx) = u_nnz;
      });
      }
  });

  Kokkos::fence();

  auto lnrwh = Kokkos::create_mirror_view(L_new_row_map);
  auto unrwh = Kokkos::create_mirror_view(U_new_row_map);
  Kokkos::deep_copy(lnrwh, L_new_row_map);
  Kokkos::deep_copy(unrwh, U_new_row_map);

  std::cout << "L_new_row_map:" << std::endl;
  for (size_type i = 0; i < nrows+1; ++i) {
    std::cout << lnrwh(i) << " ";
  } std::cout << std::endl;

  std::cout << "U_new_row_map:" << std::endl;
  for (size_type i = 0; i < nrows+1; ++i) {
    std::cout << unrwh(i) << " ";
  } std::cout << std::endl;

  // prefix sum
  const size_type l_new_nnz = prefix_sum(ih, L_new_row_map, prefix_sum_view);
  const size_type u_new_nnz = prefix_sum(ih, U_new_row_map, prefix_sum_view);

  Kokkos::deep_copy(lnrwh, L_new_row_map);
  Kokkos::deep_copy(unrwh, U_new_row_map);

  std::cout << "L_new_row_map:" << std::endl;
  for (size_type i = 0; i < nrows+1; ++i) {
    std::cout << lnrwh(i) << " ";
  } std::cout << std::endl;

  std::cout << "U_new_row_map:" << std::endl;
  for (size_type i = 0; i < nrows+1; ++i) {
    std::cout << unrwh(i) << " ";
  } std::cout << std::endl;

  Kokkos::resize(L_new_entries, l_new_nnz);
  Kokkos::resize(U_new_entries, u_new_nnz);
  Kokkos::resize(L_new_values,  l_new_nnz);
  Kokkos::resize(U_new_values,  u_new_nnz);

  constexpr auto sentinel = std::numeric_limits<size_type>::max();

  using scalar_t        = typename AValuesType::non_const_value_type;

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
      bool      finished_l  = l_old_begin == l_old_end;
      bool      skip        = false;
      for (size_type i = 0; i < tot; ++i) {
        if (skip) {
          skip = false;
          continue;
        }

        const auto a_col  = a_row_nnz_begin  < a_row_nnz_end  ? A_entries(a_row_nnz_begin)   : sentinel;
              auto a_val  = a_row_nnz_begin  < a_row_nnz_end  ? A_values(a_row_nnz_begin)    : 0.;
        const auto lu_col = lu_row_nnz_begin < lu_row_nnz_end ? LU_entries(lu_row_nnz_begin) : sentinel;
              auto lu_val = lu_row_nnz_begin < lu_row_nnz_end ? LU_values(lu_row_nnz_begin)  : 0.;

        const size_type col_idx = Kokkos::fmin(a_col, lu_col);

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
        if (row_idx >= col_idx) {
          L_new_entries(l_new_nnz) = col_idx;
          L_new_values(l_new_nnz)  = row_idx == col_idx ? 1. : out_val;
          ++l_new_nnz;
        }
        if (row_idx <= col_idx) {
          U_new_entries(u_new_nnz) = col_idx;
          U_new_values(u_new_nnz)  = out_val;
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

template <class ForwardIterator, class T>
KOKKOS_FUNCTION
ForwardIterator kok_lower_bound(ForwardIterator first, ForwardIterator last, const T& val)
{
  ForwardIterator it;
  size_t count, step;
  count = last - first;
  while (count>0)
  {
    it = first; step=count/2; it += step;
    if (*it<val) {                 // or: if (comp(*it,val)), for version (2)
      first=++it;
      count-=step+1;
    }
    else count=step;
  }
  return first;
}

template <class IlutHandle,
          class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class UtRowMapType, class UtEntriesType, class UtValuesType>
KOKKOS_FUNCTION
Kokkos::pair<typename AValuesType::non_const_value_type, typename IlutHandle::size_type> compute_sum(
  typename IlutHandle::size_type row_idx, typename IlutHandle::size_type col_idx,
  const ARowMapType&  A_row_map,  const AEntriesType& A_entries,   const AValuesType& A_values,
  const LRowMapType&  L_row_map,  const LEntriesType& L_entries,   const LValuesType& L_values,
  const UtRowMapType& Ut_row_map, const UtEntriesType& Ut_entries, const UtValuesType& Ut_values)
{
  using scalar_t  = typename AValuesType::non_const_value_type;
  using size_type = typename IlutHandle::size_type;

  const auto a_row_nnz_begin = A_row_map(row_idx);
  const auto a_row_nnz_end   = A_row_map(row_idx+1);
  auto a_nnz_it =
    kok_lower_bound(A_entries.data() + a_row_nnz_begin, A_entries.data() + a_row_nnz_end, col_idx);
  auto a_nnz = a_nnz_it - A_entries.data();
  const bool has_a = a_nnz < a_row_nnz_end && A_entries(a_nnz) == col_idx;
  const auto a_val = has_a ? A_values(a_nnz) : 0.0;
  scalar_t sum = 0.0;
  size_type ut_nnz = 0;

        auto l_row_nnz     = L_row_map(row_idx);
  const auto l_row_nnz_end = L_row_map(row_idx+1);

        auto ut_row_nnz     = Ut_row_map(col_idx);
  const auto ut_row_nnz_end = Ut_row_map(col_idx+1);

  const size_type last_entry = Kokkos::fmin(row_idx, col_idx);
  while (l_row_nnz < l_row_nnz_end && ut_row_nnz < ut_row_nnz_end) {
    const auto l_col = L_entries(l_row_nnz);
    const auto u_row = Ut_entries(ut_row_nnz);
    if (l_col == u_row && l_col < last_entry) {
      sum += L_values(l_row_nnz) * Ut_values(ut_row_nnz);
    }
    if (u_row == row_idx) {
      ut_nnz = ut_row_nnz;
    }

    l_row_nnz  += l_col <= u_row ? 1 : 0;
    ut_row_nnz += u_row <= l_col ? 1 : 0;
  }

  return Kokkos::make_pair(a_val - sum, ut_nnz);
}

template <class IlutHandle, class RowMapType, class EntriesType, class ValuesType>
void cp_vector_to_device(
  const RowMapType& row_map, const EntriesType& entries, ValuesType& values,
  const std::vector<std::vector<typename ValuesType::non_const_value_type> >& v)
{
  using size_type = typename IlutHandle::size_type;

  const size_type nrows = row_map.extent(0) - 1;

  auto hrow_map = Kokkos::create_mirror_view(row_map);
  auto hentries = Kokkos::create_mirror_view(entries);
  auto hvalues  = Kokkos::create_mirror_view(values);

  for (size_type row_idx = 0; row_idx < nrows; ++row_idx) {
    const size_type row_nnz_begin = hrow_map(row_idx);
    const size_type row_nnz_end   = hrow_map(row_idx+1);
    for (size_type row_nnz = row_nnz_begin; row_nnz < row_nnz_end; ++row_nnz) {
      const auto col_idx = hentries(row_nnz);
      hvalues(row_nnz) = v[row_idx][col_idx];
    }
  }

  Kokkos::deep_copy(values, hvalues);
}

template <class IlutHandle,
          class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType,
          class UtRowMapType, class UtEntriesType, class UtValuesType>
void hardcode_compute_l_u_factors(
  IlutHandle& ih,
  const ARowMapType& A_row_map, const AEntriesType& A_entries, const AValuesType& A_values,
  LRowMapType& L_row_map,   LEntriesType& L_entries,   LValuesType& L_values,
  URowMapType& U_row_map,   UEntriesType& U_entries,   UValuesType& U_values,
  UtRowMapType& Ut_row_map, UtEntriesType& Ut_entries, UtValuesType& Ut_values)
{
  using scalar_t                = typename AValuesType::non_const_value_type;
  using size_type               = typename IlutHandle::size_type;
  using execution_space         = typename IlutHandle::execution_space;
  using range_policy            = typename IlutHandle::RangePolicy;
  using HandleDeviceEntriesType = typename IlutHandle::nnz_lno_view_t;
  using HandleDeviceRowMapType  = typename IlutHandle::nnz_row_view_t;
  using HandleDeviceValueType   = typename IlutHandle::nnz_value_view_t;

  const auto nrows = A_row_map.extent(0) - 1;

  std::vector<std::vector<scalar_t> > hardcoded_L = {
    {1.0, 0.0, 0.0, 0.0},
    {2.0, 1.0, 0.0, 0.0},
    {0.5, 0.35294117647058826, 1.0, 0.0},
    {0.2, 0.1, -1.3189655172413792, 1.0}
  };

  std::vector<std::vector<scalar_t> > hardcoded_U = {
    {1.0, 6.0, 4.0, 7.0},
    {0.0, -17.0, -8.0, -6.0},
    {0.0, 0.0, 6.8235294117647065, -1.3823529411764706},
    {0.0, 0.0, 0.0, -2.623275862068965}
  };

  cp_vector_to_device<IlutHandle>(L_row_map, L_entries, L_values, hardcoded_L);
  cp_vector_to_device<IlutHandle>(U_row_map, U_entries, U_values, hardcoded_U);

  Kokkos::parallel_for(
    range_policy(0, nrows+1),
    KOKKOS_LAMBDA(const size_type row_idx) {
      Ut_row_map(row_idx) = 0;
    });
  Kokkos::resize(Ut_entries, U_entries.extent(0));
  Kokkos::resize(Ut_values,  U_values.extent(0));

  Kokkos::fence();

  KokkosKernels::Impl::transpose_matrix<
    HandleDeviceRowMapType, HandleDeviceEntriesType, HandleDeviceValueType,
    HandleDeviceRowMapType, HandleDeviceEntriesType, HandleDeviceValueType,
    HandleDeviceRowMapType, execution_space>(
      nrows, nrows,
      U_row_map, U_entries, U_values,
      Ut_row_map, Ut_entries, Ut_values);
}

template <class IlutHandle,
          class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType,
          class UtRowMapType, class UtEntriesType, class UtValuesType>
void compute_l_u_factors(
  IlutHandle& ih,
  const ARowMapType& A_row_map, const AEntriesType& A_entries, const AValuesType& A_values,
  LRowMapType& L_row_map,   LEntriesType& L_entries,   LValuesType& L_values,
  URowMapType& U_row_map,   UEntriesType& U_entries,   UValuesType& U_values,
  UtRowMapType& Ut_row_map, UtEntriesType& Ut_entries, UtValuesType& Ut_values)
{
  using size_type       = typename IlutHandle::size_type;
  using policy_type     = typename IlutHandle::TeamPolicy;
  using member_type     = typename policy_type::member_type;
  using range_policy    = typename IlutHandle::RangePolicy;

  const auto policy = ih.get_default_team_policy();

  const size_type nrows = ih.get_nrows();

  Kokkos::parallel_for(
    "compute_l_u_factors",
    policy,
    KOKKOS_LAMBDA(const member_type& team) {
      const auto row_idx = team.league_rank();

      const auto l_row_nnz_begin = L_row_map(row_idx);
      const auto l_row_nnz_end   = L_row_map(row_idx+1);

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, l_row_nnz_begin, l_row_nnz_end-1),
        [&](const size_type l_nnz) {
          const auto col_idx = L_entries(l_nnz);
          const auto u_diag = Ut_values(Ut_row_map(col_idx+1) -1);
          if (u_diag != 0.0) {
            const auto new_val = compute_sum<IlutHandle>(
              row_idx, col_idx,
              A_row_map, A_entries, A_values,
              L_row_map, L_entries, L_values,
              Ut_row_map, Ut_entries, Ut_values).first / u_diag;
            L_values(l_nnz) = new_val;
          }
      });

      team.team_barrier();

      const auto u_row_nnz_begin = U_row_map(row_idx);
      const auto u_row_nnz_end   = U_row_map(row_idx+1);

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, u_row_nnz_begin, u_row_nnz_end),
        [&](const size_type u_nnz) {
          const auto col_idx = U_entries(u_nnz);
          const auto sum = compute_sum<IlutHandle>(
            row_idx, col_idx,
            A_row_map, A_entries, A_values,
            L_row_map, L_entries, L_values,
            Ut_row_map, Ut_entries, Ut_values);
          const auto new_val = sum.first;
          const auto ut_nnz = sum.second;
          U_values(u_nnz) = new_val;
          Ut_values(ut_nnz) = new_val;
      });
  });
}

template <class IlutHandle, class ValuesType, class ValuesCopyType>
typename IlutHandle::nnz_scalar_t threshold_select(
  IlutHandle& ih, ValuesType& values, const typename IlutHandle::nnz_lno_t rank, ValuesCopyType& values_copy)
{
  using index_type = typename IlutHandle::nnz_lno_t;
  using scalar_t   = typename IlutHandle::nnz_scalar_t;

  const index_type size = values.extent(0);

  Kokkos::resize(values_copy, size);
  Kokkos::deep_copy(values_copy, values);

  auto begin  = values_copy.data();
  auto target = begin + rank;
  auto end    = begin + size;
  std::nth_element(begin, target, end,
                   [](scalar_t a, scalar_t b) { return std::abs(a) < std::abs(b); });

  return std::abs(values_copy(rank));
}

template <class IlutHandle,
          class IRowMapType, class IEntriesType, class IValuesType,
          class ORowMapType, class OEntriesType, class OValuesType,
          class PrefixSumType>
void threshold_filter(
  IlutHandle& ih, const typename IlutHandle::nnz_scalar_t threshold,
  const IRowMapType& I_row_map, const IEntriesType& I_entries, const IValuesType& I_values,
  ORowMapType& O_row_map,             OEntriesType& O_entries,       OValuesType& O_values,
  PrefixSumType& prefix_sum_view)
{
  using size_type   = typename IlutHandle::size_type;
  using policy_type = typename IlutHandle::TeamPolicy;
  using member_type = typename policy_type::member_type;
  using RangePolicy = typename IlutHandle::RangePolicy;

  const auto policy = ih.get_default_team_policy();
  const size_type nrows  = ih.get_nrows();

  Kokkos::parallel_for(
    "threshold_filter count",
    policy,
    KOKKOS_LAMBDA(const member_type& team) {
      const auto row_idx = team.league_rank();

      const auto row_nnx_begin = I_row_map(row_idx);
      const auto row_nnx_end   = I_row_map(row_idx+1);

      size_type count = 0;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, row_nnx_begin, row_nnx_end),
        [&](const size_type nnz, size_type& count_inner) {
          if (std::abs(I_values(nnz)) >= threshold || I_entries(nnz) == row_idx) {
            count_inner += 1;
          }
      }, count);

      team.team_barrier();

      O_row_map(row_idx) = count;

      O_row_map(nrows+1) = 0; // harmless race
  });

  Kokkos::fence();
  const auto new_nnz = prefix_sum(ih, O_row_map, prefix_sum_view);

  Kokkos::resize(O_entries, new_nnz);
  Kokkos::resize(O_values, new_nnz);

  Kokkos::parallel_for(
    "threshold_filter assign",
    RangePolicy(0, nrows),
    KOKKOS_LAMBDA(const size_type row_idx) {
      const auto i_row_nnx_begin = I_row_map(row_idx);
      const auto i_row_nnx_end   = I_row_map(row_idx+1);

      auto onnz = O_row_map(row_idx);

      for (size_type innz = i_row_nnx_begin; innz < i_row_nnx_end; ++innz) {
        if (std::abs(I_values(innz)) >= threshold || I_entries(innz) == row_idx) {
          O_entries(onnz) = I_entries(innz);
          O_values(onnz)  = I_values(innz);
          ++onnz;
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
                  URowMapType &U_row_map, UEntriesType &U_entries, UValuesType &U_values)
{
  using execution_space         = typename IlutHandle::execution_space;
  using memory_space            = typename IlutHandle::memory_space;
  using index_type              = typename IlutHandle::nnz_lno_t;
  using size_type               = typename IlutHandle::size_type;
  using range_policy            = typename IlutHandle::RangePolicy;
  using HandleDeviceEntriesType = typename IlutHandle::nnz_lno_view_t;
  using HandleDeviceRowMapType  = typename IlutHandle::nnz_row_view_t;
  using HandleDeviceValueType   = typename IlutHandle::nnz_value_view_t;

  const size_type nrows    = thandle.get_nrows();
  const auto fill_in_limit = thandle.get_fill_in_limit();
  const auto l_nnz_limit = static_cast<index_type>(fill_in_limit * thandle.get_nnzL());
  const auto u_nnz_limit = static_cast<index_type>(fill_in_limit * thandle.get_nnzU());

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
      policy.league_size()),
    Ut_new_row_map(
      "Ut_new_row_map",
      nrows + 1);
  HandleDeviceEntriesType LU_entries, L_new_entries, U_new_entries, Ut_new_entries;
  HandleDeviceValueType LU_values, L_new_values, U_new_values, Ut_new_values, V_copy_d;
  auto V_copy = Kokkos::create_mirror_view(V_copy_d);

  while (!converged) {
    // LU = L*U
    KokkosSparse::Experimental::spgemm_symbolic(
      &kh, nrows, nrows, nrows,
      L_row_map, L_entries, false,
      U_row_map, U_entries, false,
      LU_row_map);

    Kokkos::fence();

    const size_type lu_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
    Kokkos::resize(LU_entries, lu_nnz_size);
    Kokkos::resize(LU_values, lu_nnz_size);

    KokkosSparse::Experimental::spgemm_numeric(
      &kh, nrows, nrows, nrows,
      L_row_map, L_entries, L_values, false,
      U_row_map, U_entries, U_values, false,
      LU_row_map, LU_entries, LU_values);

    Kokkos::fence();

    add_candidates(thandle,
      A_row_map, A_entries, A_values,
      L_row_map, L_entries, L_values,
      U_row_map, U_entries, U_values,
      LU_row_map, LU_entries, LU_values,
      L_new_row_map, L_new_entries, L_new_values,
      U_new_row_map, U_new_entries, U_new_values,
      prefix_sum_view);

    Kokkos::fence();

    Kokkos::deep_copy(L_row_map, L_new_row_map);
    Kokkos::resize(L_entries, L_new_entries.extent(0));
    Kokkos::deep_copy(L_entries, L_new_entries);
    Kokkos::resize(L_values, L_new_values.extent(0));
    Kokkos::deep_copy(L_values, L_new_values);

    Kokkos::deep_copy(U_row_map, U_new_row_map);
    Kokkos::resize(U_entries, U_new_entries.extent(0));
    Kokkos::deep_copy(U_entries, U_new_entries);
    Kokkos::resize(U_values, U_new_values.extent(0));
    Kokkos::deep_copy(U_values, U_new_values);

    Kokkos::fence();

#if 0
    Kokkos::parallel_for(
      range_policy(0, nrows+1),
      KOKKOS_LAMBDA(const size_type row_idx) {
        Ut_new_row_map(row_idx) = 0;
    });
    Kokkos::resize(Ut_new_entries, U_new_entries.extent(0));
    Kokkos::resize(Ut_new_values,  U_new_values.extent(0));

    Kokkos::fence();

    KokkosKernels::Impl::transpose_matrix<
      HandleDeviceRowMapType, HandleDeviceEntriesType, HandleDeviceValueType,
      HandleDeviceRowMapType, HandleDeviceEntriesType, HandleDeviceValueType,
      HandleDeviceRowMapType, execution_space>(
      nrows, nrows,
      U_new_row_map, U_new_entries, U_new_values,
      Ut_new_row_map, Ut_new_entries, Ut_new_values);

    Kokkos::fence();

    hardcode_compute_l_u_factors(thandle,
      A_row_map, A_entries, A_values,
      L_new_row_map, L_new_entries, L_new_values,
      U_new_row_map, U_new_entries, U_new_values,
      Ut_new_row_map, Ut_new_entries, Ut_new_values);

    Kokkos::fence();

    const index_type l_nnz = L_new_values.extent(0);
    const index_type u_nnz = U_new_values.extent(0);

    const auto l_filter_rank = std::max(0, l_nnz - l_nnz_limit - 1);
    const auto u_filter_rank = std::max(0, u_nnz - u_nnz_limit - 1);

    const auto l_threshold = threshold_select(thandle, L_new_values, l_filter_rank, V_copy);
    const auto u_threshold = threshold_select(thandle, U_new_values, u_filter_rank, V_copy);

    threshold_filter(
      thandle, l_threshold,
      L_new_row_map, L_new_entries, L_new_values,
      L_row_map, L_entries, L_values,
      prefix_sum_view);

    Kokkos::fence();

    threshold_filter(
      thandle, u_threshold,
      U_new_row_map, U_new_entries, U_new_values,
      U_row_map, U_entries, U_values,
      prefix_sum_view);

    Kokkos::fence();

    Kokkos::parallel_for(
      range_policy(0, nrows+1),
      KOKKOS_LAMBDA(const size_type row_idx) {
        Ut_new_row_map(row_idx) = 0;
    });
    Kokkos::resize(Ut_new_entries, U_entries.extent(0));
    Kokkos::resize(Ut_new_values,  U_values.extent(0));

    Kokkos::fence();

    KokkosKernels::Impl::transpose_matrix<
      HandleDeviceRowMapType, HandleDeviceEntriesType, HandleDeviceValueType,
      HandleDeviceRowMapType, HandleDeviceEntriesType, HandleDeviceValueType,
      HandleDeviceRowMapType, execution_space>(
      nrows, nrows,
      U_row_map, U_entries, U_values,
      Ut_new_row_map, Ut_new_entries, Ut_new_values);

    Kokkos::fence();

    compute_l_u_factors(thandle,
      A_row_map, A_entries, A_values,
      L_row_map, L_entries, L_values,
      U_row_map, U_entries, U_values,
      Ut_new_row_map, Ut_new_entries, Ut_new_values);
#endif
    converged = true;
  }

  Kokkos::fence();
}  // end ilut_numeric

}  // namespace Experimental
}  // namespace Impl
}  // namespace KokkosSparse

#endif
