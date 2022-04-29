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
#include <Kokkos_Sort.hpp>
#include <KokkosKernels_Error.hpp>

//#define SYMBOLIC_OUTPUT_INFO

namespace KokkosSparse {
namespace Impl {
namespace Experimental {

template <class IlukHandle, class ARowMapType, class AEntriesType,
          class LRowMapType, class LEntriesType, class URowMapType,
          class UEntriesType>
void iluk_symbolic(IlukHandle& thandle,
                   const typename IlukHandle::const_nnz_lno_t& fill_lev,
                   const ARowMapType& A_row_map_d, const AEntriesType& A_entries_d,
                   LRowMapType& L_row_map_d, LEntriesType& L_entries_d,
                   URowMapType& U_row_map_d, UEntriesType& U_entries_d)
{
  // Symbolic phase currently compute on host - need host copy
  // of all views

  using execution_space = typename ARowMapType::execution_space;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;
  using member_type     = typename policy_type::member_type;

  typedef typename IlukHandle::size_type size_type;
  typedef typename IlukHandle::nnz_lno_t nnz_lno_t;

  typedef typename IlukHandle::nnz_lno_view_t HandleDeviceEntriesType;
  typedef typename IlukHandle::nnz_row_view_t HandleDeviceRowMapType;

  // typedef typename IlukHandle::signed_integral_t signed_integral_t;

  size_type nrows = thandle.get_nrows();

  // Sizing
  const auto policy = thandle.get_default_team_policy();
  auto nnzs = Kokkos::make_pair<size_type, size_type>(0, 0);
  Kokkos::parallel_reduce(
    "symbolic sizing",
    policy,
    KOKKOS_LAMBDA(const member_type& team, decltype(nnzs)& nnzs_outer) {
      const auto row_idx = team.league_rank();

      const auto row_nnz_begin = A_row_map_d(row_idx);
      const auto row_nnz_end   = A_row_map_d(row_idx+1);
      Kokkos::parallel_reduce(
        "symbolic sizing team",
        Kokkos::TeamThreadRange(team, row_nnz_begin, row_nnz_end),
        [&](const size_type nnz, decltype(nnzs)& nnzs_inner) {
          const auto col_idx = A_entries_d(nnz);
          nnzs_inner.first  += col_idx < row_idx;
          nnzs_inner.second += col_idx > row_idx;
      }, nnzs_outer);
  }, nnzs);
}  // end iluk_symbolic

}  // namespace Experimental
}  // namespace Impl
}  // namespace KokkosSparse

#endif
