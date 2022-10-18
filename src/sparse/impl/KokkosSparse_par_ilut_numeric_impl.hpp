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
#include <KokkosSparse_spadd.hpp>
#include <KokkosSparse_Utils.hpp>
#include <KokkosSparse_SortCrs.hpp>
#include <KokkosKernels_Utils.hpp>

#include <limits>

//#define NUMERIC_OUTPUT_INFO

namespace KokkosSparse {
namespace Impl {
namespace Experimental {

/**
 * prefix_sum: Take a row_map of counts and transform it to sums, and
 * return the total sum.
 */
template <class IlutHandle, class RowMapType>
typename IlutHandle::size_type prefix_sum(IlutHandle& ih, RowMapType& row_map)
{
  using size_type    = typename IlutHandle::size_type;

  KokkosKernels::Impl::exclusive_parallel_prefix_sum<RowMapType, typename IlutHandle::HandleExecSpace>
    (row_map.extent(0), row_map);

  // Grab last value
  size_type result = 0;
  Kokkos::parallel_reduce(1, KOKKOS_LAMBDA(const size_type i, size_type& inner) {
      inner = row_map(row_map.extent(0) - 1);
  }, Kokkos::Max<size_type>(result));

  return result;
}

/**
 * Just a convenience wrapper around spgemm
 */
template <class KHandle, class IlutHandle,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType,
          class LURowMapType, class LUEntriesType, class LUValuesType>
void multiply_matrices(
  KHandle& kh, IlutHandle& ih,
  const LRowMapType& L_row_map, const LEntriesType& L_entries, const LValuesType& L_values,
  const URowMapType& U_row_map, const UEntriesType& U_entries, const UValuesType& U_values,
  LURowMapType& LU_row_map,           LUEntriesType& LU_entries,     LUValuesType& LU_values)
{
  using execution_space = typename IlutHandle::execution_space;
  using size_type       = typename IlutHandle::size_type;

  const size_type nrows = ih.get_nrows();

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

  // Need to sort LU CRS if on CUDA!
  sort_crs_matrix<execution_space>(LU_row_map, LU_entries, LU_values);
}

/**
 * Just a convenience wrapper around transpose_matrix
 */
template <class IlutHandle,
          class RowMapType,  class EntriesType,  class ValuesType,
          class TRowMapType, class TEntriesType, class TValuesType>
void transpose_wrap(
  IlutHandle& ih,
  const RowMapType& row_map, const EntriesType& entries, const ValuesType& values,
  TRowMapType& t_row_map,          TEntriesType& t_entries,    TValuesType& t_values)
{
  using execution_space = typename IlutHandle::execution_space;
  using size_type       = typename IlutHandle::size_type;

  using HandleDeviceEntriesType = typename IlutHandle::nnz_lno_view_t;
  using HandleDeviceRowMapType  = typename IlutHandle::nnz_row_view_t;
  using HandleDeviceValueType   = typename IlutHandle::nnz_value_view_t;

  const size_type nrows = ih.get_nrows();

  // Need to reset t_row_map
  Kokkos::deep_copy(t_row_map, 0);

  Kokkos::resize(t_entries, entries.extent(0));
  Kokkos::resize(t_values,  values.extent(0));

  KokkosKernels::Impl::transpose_matrix<
    HandleDeviceRowMapType, HandleDeviceEntriesType, HandleDeviceValueType,
    HandleDeviceRowMapType, HandleDeviceEntriesType, HandleDeviceValueType,
    HandleDeviceRowMapType, execution_space>(
      nrows, nrows,
      row_map, entries, values,
      t_row_map, t_entries, t_values);

  // Need to ensure output is sorted
  sort_crs_matrix<execution_space>(t_row_map, t_entries, t_values);
}

/**
 * Adds new entries from the sparsity pattern of A - L * U
 * to L and U, where new values are chosen based on the residual
 * value divided by the corresponding diagonal entry.
 */
template <class IlutHandle,
          class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType,
          class LURowMapType, class LUEntriesType, class LUValuesType,
          class LNewRowMapType, class LNewEntriesType, class LNewValuesType,
          class UNewRowMapType, class UNewEntriesType, class UNewValuesType>
void add_candidates(
  IlutHandle& ih,
  const ARowMapType& A_row_map, const AEntriesType& A_entries, const AValuesType& A_values,
  const LRowMapType& L_row_map, const LEntriesType& L_entries, const LValuesType& L_values,
  const URowMapType& U_row_map, const UEntriesType& U_entries, const UValuesType& U_values,
  const LURowMapType& LU_row_map, const LUEntriesType& LU_entries, const LUValuesType& LU_values,
  LNewRowMapType& L_new_row_map, LNewEntriesType& L_new_entries, LNewValuesType& L_new_values,
  UNewRowMapType& U_new_row_map, UNewEntriesType& U_new_entries, UNewValuesType& U_new_values)
{
  using size_type       = typename IlutHandle::size_type;
  using policy_type     = typename IlutHandle::TeamPolicy;
  using member_type     = typename policy_type::member_type;
  using range_policy    = typename IlutHandle::RangePolicy;

  const size_type nrows = ih.get_nrows();

  const auto policy = ih.get_default_team_policy();

  // Sizing run for add_candidates. Count nnz's and remove dupes
  Kokkos::parallel_for(
    "add_candidates sizing",
    policy,
    KOKKOS_LAMBDA(const member_type& team) {
      const auto row_idx = team.league_rank();

      const auto a_row_nnz_begin = A_row_map(row_idx);
      const auto a_row_nnz_end   = A_row_map(row_idx+1);

      const auto lu_row_nnz_begin = LU_row_map(row_idx);
      const auto lu_row_nnz_end   = LU_row_map(row_idx+1);

      // Really wish kokkos could do a multi-reduce here
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
  const size_type l_new_nnz = prefix_sum(ih, L_new_row_map);
  const size_type u_new_nnz = prefix_sum(ih, U_new_row_map);

  Kokkos::resize(L_new_entries, l_new_nnz);
  Kokkos::resize(U_new_entries, u_new_nnz);
  Kokkos::resize(L_new_values,  l_new_nnz);
  Kokkos::resize(U_new_values,  u_new_nnz);

  constexpr auto sentinel = std::numeric_limits<size_type>::max();

  using scalar_t = typename AValuesType::non_const_value_type;

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

/**
 * A device-safe lower_bound impl
 */
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

/**
 * The compute_sum component of compute_l_u_factors
 */
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

template <class IlutHandle,
          class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType,
          class UtRowMapType, class UtEntriesType, class UtValuesType,
          class MemberType>
KOKKOS_FUNCTION
void compute_l_u_factors_impl(
  IlutHandle& ih,
  const ARowMapType& A_row_map, const AEntriesType& A_entries, const AValuesType& A_values,
  LRowMapType& L_row_map,   LEntriesType& L_entries,   LValuesType& L_values,
  URowMapType& U_row_map,   UEntriesType& U_entries,   UValuesType& U_values,
  UtRowMapType& Ut_row_map, UtEntriesType& Ut_entries, UtValuesType& Ut_values,
  MemberType& team)
{
  using size_type = typename IlutHandle::size_type;

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
      Ut_values(ut_nnz) = new_val; // ut_nnz is not guarateed to fail into range used exclusively by this team
  });
}

/**
 * Implements a single iteration/sweep of the fixed-point ILU algorithm.
 * The results of this function are non-deterministic due to concurrent
 * reading and writing of Ut values. deterministic can be set to true to
 * make this function determistic, but it will be run in Serial exe space
 * if so.
 */
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
  UtRowMapType& Ut_row_map, UtEntriesType& Ut_entries, UtValuesType& Ut_values,
  bool deterministic)
{
  using size_type = typename IlutHandle::size_type;

  if (deterministic) {
    std::cout << "Using deterministic version of compute_l_u_factors" << std::endl;
    using spolicy_type = Kokkos::TeamPolicy<Kokkos::Serial>;
    using smember_type = typename spolicy_type::member_type;

    const size_type nrows  = ih.get_nrows();
    spolicy_type policy(nrows, 1);

    auto A_row_map_h  = Kokkos::create_mirror_view(A_row_map);
    auto A_entries_h  = Kokkos::create_mirror_view(A_entries);
    auto A_values_h   = Kokkos::create_mirror_view(A_values);
    auto L_row_map_h  = Kokkos::create_mirror_view(L_row_map);
    auto L_entries_h  = Kokkos::create_mirror_view(L_entries);
    auto L_values_h   = Kokkos::create_mirror_view(L_values);
    auto U_row_map_h  = Kokkos::create_mirror_view(U_row_map);
    auto U_entries_h  = Kokkos::create_mirror_view(U_entries);
    auto U_values_h   = Kokkos::create_mirror_view(U_values);
    auto Ut_row_map_h = Kokkos::create_mirror_view(Ut_row_map);
    auto Ut_entries_h = Kokkos::create_mirror_view(Ut_entries);
    auto Ut_values_h  = Kokkos::create_mirror_view(Ut_values);

    Kokkos::deep_copy(A_row_map_h, A_row_map);
    Kokkos::deep_copy(A_entries_h, A_entries);
    Kokkos::deep_copy(A_values_h,  A_values);
    Kokkos::deep_copy(L_row_map_h, L_row_map);
    Kokkos::deep_copy(L_entries_h, L_entries);
    Kokkos::deep_copy(L_values_h,  L_values);
    Kokkos::deep_copy(U_row_map_h, U_row_map);
    Kokkos::deep_copy(U_entries_h, U_entries);
    Kokkos::deep_copy(U_values_h,  U_values);
    Kokkos::deep_copy(Ut_row_map_h, Ut_row_map);
    Kokkos::deep_copy(Ut_entries_h, Ut_entries);
    Kokkos::deep_copy(Ut_values_h,  Ut_values);

    Kokkos::parallel_for(
      "compute_l_u_factors",
      policy,
      KOKKOS_LAMBDA(const smember_type& team) {
        compute_l_u_factors_impl(ih,
          A_row_map_h, A_entries_h, A_values_h,
          L_row_map_h, L_entries_h, L_values_h,
          U_row_map_h, U_entries_h, U_values_h,
          Ut_row_map_h, Ut_entries_h, Ut_values_h, team);
    });

    Kokkos::deep_copy(L_values, L_values_h);
    Kokkos::deep_copy(U_values, U_values_h);
    Kokkos::deep_copy(Ut_values, Ut_values_h);
  }
  else {
    std::cout << "Using non-deterministic version of compute_l_u_factors" << std::endl;

    using policy_type     = typename IlutHandle::TeamPolicy;
    using member_type     = typename policy_type::member_type;

    const auto policy = ih.get_default_team_policy();

    Kokkos::parallel_for(
      "compute_l_u_factors",
      policy,
      KOKKOS_LAMBDA(const member_type& team) {
        compute_l_u_factors_impl(ih,
          A_row_map, A_entries, A_values,
          L_row_map, L_entries, L_values,
          U_row_map, U_entries, U_values,
          Ut_row_map, Ut_entries, Ut_values, team);
    });
  }
}

/**
 * Select threshold based on filter rank. Do all this on host
 */
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

/**
 * Remove non-diagnal elements that are below the threshold.
 */
template <class IlutHandle,
          class IRowMapType, class IEntriesType, class IValuesType,
          class ORowMapType, class OEntriesType, class OValuesType>
void threshold_filter(
  IlutHandle& ih, const typename IlutHandle::nnz_scalar_t threshold,
  const IRowMapType& I_row_map, const IEntriesType& I_entries, const IValuesType& I_values,
  ORowMapType& O_row_map,             OEntriesType& O_entries,       OValuesType& O_values)
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

  const auto new_nnz = prefix_sum(ih, O_row_map);

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

/**
 * Compute residual norm for R = A - LU
 */
template <class KHandle, class IlutHandle,
          class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType,
          class RRowMapType, class REntriesType, class RValuesType,
          class LURowMapType, class LUEntriesType, class LUValuesType>
typename IlutHandle::nnz_scalar_t compute_residual_norm(
  KHandle& kh, IlutHandle& ih,
  const ARowMapType& A_row_map, const AEntriesType& A_entries, const AValuesType& A_values,
  const LRowMapType& L_row_map, const LEntriesType& L_entries, const LValuesType& L_values,
  const URowMapType& U_row_map, const UEntriesType& U_entries, const UValuesType& U_values,
  RRowMapType& R_row_map,             REntriesType& R_entries,       RValuesType& R_values,
  LURowMapType& LU_row_map,           LUEntriesType& LU_entries,     LUValuesType& LU_values)
{
  using size_type   = typename IlutHandle::size_type;
  using scalar_t    = typename AValuesType::non_const_value_type;
  using idx_t       = typename AEntriesType::non_const_value_type;
  using policy_type = typename IlutHandle::TeamPolicy;
  using member_type = typename policy_type::member_type;

  multiply_matrices(kh, ih,
                    L_row_map, L_entries, L_values,
                    U_row_map, U_entries, U_values,
                    LU_row_map, LU_entries, LU_values);

  auto addHandle = kh.get_spadd_handle();
  KokkosSparse::Experimental::spadd_symbolic(
    &kh,
    A_row_map, A_entries,
    LU_row_map, LU_entries,
    R_row_map);

  const size_type r_nnz = addHandle->get_c_nnz();
  Kokkos::resize(R_entries, r_nnz);
  Kokkos::resize(R_values, r_nnz);

  KokkosSparse::Experimental::spadd_numeric
    (&kh,
     A_row_map, A_entries, A_values, 1.,
     LU_row_map, LU_entries, LU_values, -1.,
     R_row_map, R_entries, R_values);

  scalar_t result;

  auto policy = ih.get_default_team_policy();

  Kokkos::parallel_reduce(
    "compute_residual_norm",
    policy,
    KOKKOS_LAMBDA(const member_type& team, scalar_t& total_sum) {
      const auto row_idx = team.league_rank();

      const auto a_row_nnz_begin = A_row_map(row_idx);
      const auto a_row_nnz_end   = A_row_map(row_idx+1);

      const auto a_row_entries_begin = A_entries.data() + a_row_nnz_begin;
      const auto a_row_entries_end   = A_entries.data() + a_row_nnz_end;

      const auto r_row_nnz_begin = R_row_map(row_idx);
      const auto r_row_nnz_end   = R_row_map(row_idx+1);

      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, r_row_nnz_begin, r_row_nnz_end),
        [&](const size_type nnz, scalar_t& sum_inner) {
          const auto r_col_idx = R_entries(nnz);
          const idx_t* lb = kok_lower_bound(a_row_entries_begin, a_row_entries_end, r_col_idx);
          if (lb != a_row_entries_end && *lb == r_col_idx) {
            sum_inner += R_values(nnz) * R_values(nnz);
          }
      }, total_sum);

  }, result);

  return std::sqrt(result);
}

/**
 * The main par_ilut numeric function.
 */
template <class KHandle, class IlutHandle,
          class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType>
void ilut_numeric(KHandle& kh, IlutHandle &thandle, const ARowMapType &A_row_map,
                  const AEntriesType &A_entries, const AValuesType &A_values,
                  LRowMapType &L_row_map, LEntriesType &L_entries, LValuesType &L_values,
                  URowMapType &U_row_map, UEntriesType &U_entries, UValuesType &U_values,
                  bool deterministic)
{
  //
  // Useful types, constants, and handles
  //
  using execution_space         = typename IlutHandle::execution_space;
  using memory_space            = typename IlutHandle::memory_space;
  using index_type              = typename IlutHandle::nnz_lno_t;
  using size_type               = typename IlutHandle::size_type;
  using range_policy            = typename IlutHandle::RangePolicy;
  using HandleDeviceEntriesType = typename IlutHandle::nnz_lno_view_t;
  using HandleDeviceRowMapType  = typename IlutHandle::nnz_row_view_t;
  using HandleDeviceValueType   = typename IlutHandle::nnz_value_view_t;
  using scalar_t                = typename AValuesType::non_const_value_type;

  const size_type nrows    = thandle.get_nrows();
  const auto fill_in_limit = thandle.get_fill_in_limit();
  const auto l_nnz_limit = static_cast<index_type>(fill_in_limit * thandle.get_nnzL());
  const auto u_nnz_limit = static_cast<index_type>(fill_in_limit * thandle.get_nnzU());

  const scalar_t residual_norm_delta_stop = thandle.get_residual_norm_delta_stop();
  const size_type max_iter = thandle.get_max_iter();

  std::string myalg("SPGEMM_KK_MEMORY");
  KokkosSparse::SPGEMMAlgorithm spgemm_algorithm =
    KokkosSparse::StringToSPGEMMAlgorithm(myalg);
  kh.create_spgemm_handle(spgemm_algorithm);

  kh.create_spadd_handle(true /*we expect inputs to be sorted*/);

  const auto policy = thandle.get_default_team_policy();

  //
  // temporary workspaces and scalars
  //
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
    R_row_map(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "R_row_map"),
      nrows + 1),
    Ut_new_row_map(
      "Ut_new_row_map",
      nrows + 1);

  HandleDeviceEntriesType LU_entries, L_new_entries, U_new_entries, Ut_new_entries, R_entries;
  HandleDeviceValueType LU_values, L_new_values, U_new_values, Ut_new_values, V_copy_d, R_values;
  auto V_copy = Kokkos::create_mirror_view(V_copy_d);

  size_type itr = 0;
  scalar_t prev_residual = std::numeric_limits<scalar_t>::max();
  bool converged = false;

  //
  // main loop
  //
  while (!converged && itr < max_iter) {
    // LU = L*U
    if (prev_residual == std::numeric_limits<scalar_t>::max()) {
      multiply_matrices(kh, thandle,
                        L_row_map, L_entries, L_values,
                        U_row_map, U_entries, U_values,
                        LU_row_map, LU_entries, LU_values);
    }

    // Identify candidate locations and add them
    add_candidates(thandle,
      A_row_map, A_entries, A_values,
      L_row_map, L_entries, L_values,
      U_row_map, U_entries, U_values,
      LU_row_map, LU_entries, LU_values,
      L_new_row_map, L_new_entries, L_new_values,
      U_new_row_map, U_new_entries, U_new_values);

    // Get transpose of U_new, needed for compute_l_u_factors
    transpose_wrap(
      thandle,
      U_new_row_map, U_new_entries, U_new_values,
      Ut_new_row_map, Ut_new_entries, Ut_new_values);

    // Do one sweep of the fixed-point ILU algorithm
    compute_l_u_factors(thandle,
      A_row_map, A_entries, A_values,
      L_new_row_map, L_new_entries, L_new_values,
      U_new_row_map, U_new_entries, U_new_values,
      Ut_new_row_map, Ut_new_entries, Ut_new_values,
      deterministic);

    // Filter smallest elements from L_new and U_new. Store result back
    // in L and U.
    {
      const index_type l_nnz = L_new_values.extent(0);
      const index_type u_nnz = U_new_values.extent(0);

      const auto l_filter_rank = std::max(0, l_nnz - l_nnz_limit - 1);
      const auto u_filter_rank = std::max(0, u_nnz - u_nnz_limit - 1);

      const auto l_threshold = threshold_select(thandle, L_new_values, l_filter_rank, V_copy);
      const auto u_threshold = threshold_select(thandle, U_new_values, u_filter_rank, V_copy);

      threshold_filter(
        thandle, l_threshold,
        L_new_row_map, L_new_entries, L_new_values,
        L_row_map, L_entries, L_values);

      threshold_filter(
        thandle, u_threshold,
        U_new_row_map, U_new_entries, U_new_values,
        U_row_map, U_entries, U_values);
    }

    // Get transpose of U, needed for compute_l_u_factors. Store in Ut_new*
    // since we aren't using those temporaries anymore
    transpose_wrap(
      thandle,
      U_row_map, U_entries, U_values,
      Ut_new_row_map, Ut_new_entries, Ut_new_values);

    // Do one sweep of the fixed-point ILU algorithm
    compute_l_u_factors(thandle,
      A_row_map, A_entries, A_values,
      L_row_map, L_entries, L_values,
      U_row_map, U_entries, U_values,
      Ut_new_row_map, Ut_new_entries, Ut_new_values,
      deterministic);

    // Compute residual and terminate if converged
    {
      const scalar_t curr_residual = compute_residual_norm(
        kh, thandle,
        A_row_map,  A_entries,  A_values,
        L_row_map,  L_entries,  L_values,
        U_row_map,  U_entries,  U_values,
        R_row_map,  R_entries,  R_values,
        LU_row_map, LU_entries, LU_values);

      if (prev_residual - curr_residual <= residual_norm_delta_stop) {
        converged = true;
      }
      else {
        prev_residual = curr_residual;
      }
    }

    ++itr;
  }

  kh.destroy_spgemm_handle();
  kh.destroy_spadd_handle();
}  // end ilut_numeric

}  // namespace Experimental
}  // namespace Impl
}  // namespace KokkosSparse

#endif
