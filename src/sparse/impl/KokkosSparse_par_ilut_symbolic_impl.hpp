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

#ifndef KOKKOSSPARSE_IMPL_PAR_ILUT_SYMBOLIC_HPP_
#define KOKKOSSPARSE_IMPL_PAR_ILUT_SYMBOLIC_HPP_

/// \file KokkosSparse_par_ilut_symbolic_impl.hpp
/// \brief Implementation of the symbolic phase of sparse ILU(k).

#include <KokkosKernels_config.h>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_par_ilut_handle.hpp>
#include <KokkosSparse_par_ilut_numeric_impl.hpp>
#include <Kokkos_Sort.hpp>
#include <KokkosKernels_Error.hpp>

//#define SYMBOLIC_OUTPUT_INFO

namespace KokkosSparse {
namespace Impl {
namespace Experimental {

template <class IlutHandle,
          class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType>
void ilut_symbolic(IlutHandle& thandle,
                   const typename IlutHandle::const_nnz_lno_t& fill_lev,
                   const ARowMapType& A_row_map_d, const AEntriesType& A_entries_d, const AValuesType& A_values_d,
                   LRowMapType& L_row_map_d, LEntriesType& L_entries_d, LValuesType& L_values_d,
                   URowMapType& U_row_map_d, UEntriesType& U_entries_d, UValuesType& U_values_d)
{
  using execution_space        = typename ARowMapType::execution_space;
  using policy_type            = Kokkos::TeamPolicy<execution_space>;
  using member_type            = typename policy_type::member_type;
  using size_type              = typename IlutHandle::size_type;
  using nnz_lno_t              = typename IlutHandle::nnz_lno_t;
  using scalar_t               = typename AValuesType::non_const_value_type;
  using RangePolicy            = typename IlutHandle::RangePolicy;
  using HandleDeviceRowMapType = typename IlutHandle::nnz_row_view_t;

  const size_type nrows = thandle.get_nrows();

  const auto policy = thandle.get_default_team_policy();

  HandleDeviceRowMapType
    prefix_sum_view(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "prefix_sum_view"),
      policy.league_size());

  // Sizing
  Kokkos::parallel_for(
    "symbolic sizing",
    policy,
    KOKKOS_LAMBDA(const member_type& team) {
      const auto row_idx = team.league_rank();

      const auto row_nnz_begin = A_row_map_d(row_idx);
      const auto row_nnz_end   = A_row_map_d(row_idx+1);

      size_type nnzsL_temp = 0, nnzsU_temp = 0;
      // Multi-reductions are not supported at the TeamThread level
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, row_nnz_begin, row_nnz_end),
        [&](const size_type nnz, size_type& nnzsL_inner) {
          const auto col_idx = A_entries_d(nnz);
          nnzsL_inner += col_idx < row_idx;
      }, nnzsL_temp);

      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, row_nnz_begin, row_nnz_end),
        [&](const size_type nnz, size_type& nnzsU_inner) {
          const auto col_idx = A_entries_d(nnz);
          nnzsU_inner += col_idx > row_idx;
      }, nnzsU_temp);

      team.team_barrier();

      Kokkos::single(
        Kokkos::PerTeam(team), [&] () {
          L_row_map_d(row_idx) = nnzsL_temp+1;
          U_row_map_d(row_idx) = nnzsU_temp+1;
      });
  });

  Kokkos::fence();

  const size_type nnzsL = prefix_sum(thandle, L_row_map_d, prefix_sum_view);
  const size_type nnzsU = prefix_sum(thandle, U_row_map_d, prefix_sum_view);

  thandle.set_nnzL(nnzsL);
  thandle.set_nnzU(nnzsU);

  // Now set actual L/U values

  Kokkos::parallel_for(
    "symbolic values",
    RangePolicy(0, nrows), // No team level parallelism in this alg
    KOKKOS_LAMBDA(const size_type& row_idx) {
      const auto row_nnz_begin = A_row_map_d(row_idx);
      const auto row_nnz_end   = A_row_map_d(row_idx+1);

      size_type current_index_l = L_row_map_d(row_idx);
      size_type current_index_u = U_row_map_d(row_idx) + 1; // we treat the diagonal separately

      // if there is no diagonal value, set it to 1 by default
      scalar_t diag = 1.;

      for (size_type row_nnz = row_nnz_begin; row_nnz < row_nnz_end; ++row_nnz) {
        const auto val     = A_values_d(row_nnz);
        const auto col_idx = A_entries_d(row_nnz);

        if (col_idx < row_idx) {
          L_entries_d(current_index_l) = col_idx;
          L_values_d(current_index_l)  = val;
          ++current_index_l;
        }
        else if (col_idx == row_idx) {
          // save diagonal
          diag = val;
        }
        else {
          U_entries_d(current_index_u) = col_idx;
          U_values_d(current_index_u)  = val;
          ++current_index_u;
        }
      }

      // store diagonal values separately
      const auto l_diag_idx = L_row_map_d(row_idx + 1) - 1;
      const auto u_diag_idx = U_row_map_d(row_idx);
      L_entries_d(l_diag_idx) = row_idx;
      U_entries_d(u_diag_idx) = row_idx;
      L_values_d(l_diag_idx) = 1.;
      U_values_d(u_diag_idx) = diag;
  });

  Kokkos::fence();

}  // end ilut_symbolic

}  // namespace Experimental
}  // namespace Impl
}  // namespace KokkosSparse

#endif
