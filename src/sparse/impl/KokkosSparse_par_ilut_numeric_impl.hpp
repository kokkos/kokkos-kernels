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

//#define NUMERIC_OUTPUT_INFO

namespace KokkosSparse {
namespace Impl {
namespace Experimental {

template <class ARowMapType, class AEntriesType, class AValuesType,
          class LRowMapType, class LEntriesType, class LValuesType,
          class URowMapType, class UEntriesType, class UValuesType,
          class LURowMapType, class LUEntriesType, class LUValuesType,
          class LNewRowMapType, class LNewEntriesType, class LNewValuesType,
          class UNewRowMapType, class UNewEntriesType, class UNewValuesType>
void add_candidates(
  const ARowMapType& A_row_map, const AEntriesType& A_entries, const AValuesType& A_values,
  const LRowMapType& L_row_map, const LEntriesType& L_entries, const LValuesType& L_values,
  const URowMapType& U_row_map, const UEntriesType& U_entries, const UValuesType& U_values,
  const LURowMapType& LU_row_map, const LUEntriesType& LU_entries, const LUValuesType& LU_values,
  LNewRowMapType& L_new_row_map, LNewEntriesType& L_new_entries, LNewValuesType& L_new_values,
  UNewRowMapType& U_new_row_map, UNewEntriesType& U_new_entries, UNewValuesType& U_new_values)
{
  
}

template <class KHandle, class IlukHandle, class ARowMapType, class AEntriesType,
          class AValuesType, class LRowMapType, class LEntriesType,
          class LValuesType, class URowMapType, class UEntriesType,
          class UValuesType>
void ilut_numeric(KHandle& kh, IlukHandle &thandle, const ARowMapType &A_row_map,
                  const AEntriesType &A_entries, const AValuesType &A_values,
                  const LRowMapType &L_row_map, const LEntriesType &L_entries,
                  LValuesType &L_values, const URowMapType &U_row_map,
                  const UEntriesType &U_entries, UValuesType &U_values) {
  using execution_space         = typename IlukHandle::execution_space;
  using memory_space            = typename IlukHandle::memory_space;
  using size_type               = typename IlukHandle::size_type;
  using nnz_lno_t               = typename IlukHandle::nnz_lno_t;
  using HandleDeviceEntriesType = typename IlukHandle::nnz_lno_view_t;
  using HandleDeviceRowMapType  = typename IlukHandle::nnz_row_view_t;
  using HandleDeviceValueType   = typename IlukHandle::nnz_value_view_t;

  size_type nlevels = thandle.get_num_levels();
  size_type nrows   = thandle.get_nrows();

  bool converged = false;

  std::string myalg("SPGEMM_KK_MEMORY");
  KokkosSparse::SPGEMMAlgorithm spgemm_algorithm =
    KokkosSparse::StringToSPGEMMAlgorithm(myalg);
  kh.create_spgemm_handle(spgemm_algorithm);

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
      nrows + 1);
  HandleDeviceEntriesType LU_entries, L_new_entries, U_new_entries;
  HandleDeviceValueType LU_values, L_new_values, U_new_values;

  while (!converged) {
    // LU = L*U
    KokkosSparse::Experimental::spgemm_symbolic(
      &kh, nrows, nrows, nrows,
      L_row_map, L_entries, false,
      U_row_map, U_entries, false,
      LU_row_map);

    const size_t lu_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
    Kokkos::resize(LU_entries, lu_nnz_size);
    Kokkos::resize(LU_values, lu_nnz_size);

    KokkosSparse::Experimental::spgemm_numeric(
      &kh, nrows, nrows, nrows,
      L_row_map, L_entries, L_values, false,
      U_row_map, U_entries, U_values, false,
      LU_row_map, LU_entries, LU_values);

    add_candidates(
      A_row_map, A_entries, A_values,
      L_row_map, L_entries, L_values,
      U_row_map, U_entries, U_values,
      LU_row_map, LU_entries, LU_values,
      L_new_row_map, L_new_entries, L_values,
      U_new_row_map, U_new_entries, U_values);

    converged = true;
  }

}  // end iluk_numeric

}  // namespace Experimental
}  // namespace Impl
}  // namespace KokkosSparse

#endif
