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

/// \file KokkosSparse_spiluk.hpp
/// \brief Parallel Minimum Discarded Fill method
/// \author Luc Berger-Vergiat
/// \date March 2022
///
/// This file provides KokkosSparse::mdf_symbolic, KokkosSparse::mdf_symbolic
/// and KokkosSparse::mdf_ordering.  These functions perform a
/// local (no MPI) sparse MDF(0) on matrices stored in
/// compressed row sparse ("Crs") format.

#ifndef KOKKOSSPARSE_MDF_HANDLE_HPP_
#define KOKKOSSPARSE_MDF_HANDLE_HPP_

#include "KokkosSparse_SortCrs.hpp"
#include "KokkosSparse_Utils.hpp"

namespace KokkosSparse {
namespace Experimental {

template <class matrix_type>
struct MDF_handle {
  using crs_matrix_type = matrix_type;
  using execution_space = typename matrix_type::execution_space;
  using row_map_type    = typename crs_matrix_type::StaticCrsGraphType::
      row_map_type::non_const_type;
  using col_ind_type = typename crs_matrix_type::StaticCrsGraphType::
      entries_type::non_const_type;
  using values_type  = typename crs_matrix_type::values_type::non_const_type;
  using size_type    = typename crs_matrix_type::size_type;
  using ordinal_type = typename crs_matrix_type::ordinal_type;

  ordinal_type numRows;

  // Views needed to construct L and U
  // at the end of the numerical phase.
  row_map_type row_mapL, row_mapU;
  col_ind_type entriesL, entriesU;
  values_type valuesL, valuesU;

  // Row permutation that defines
  // the MDF ordering or order of
  // elimination during the factorization.
  col_ind_type permutation, permutation_inv;

  int verbosity;

  MDF_handle(const crs_matrix_type A)
      : numRows(A.numRows()),
        permutation(col_ind_type("row permutation", A.numRows())),
        permutation_inv(col_ind_type("inverse row permutation", A.numRows())),
        verbosity(0){};

  void set_verbosity(const int verbosity_level) { verbosity = verbosity_level; }

  void allocate_data(const size_type nnzL, const size_type nnzU) {
    // Allocate L
    row_mapL = row_map_type("row map L", numRows + 1);
    entriesL = col_ind_type("entries L", nnzL);
    valuesL  = values_type("values L", nnzL);

    // Allocate U
    row_mapU = row_map_type("row map U", numRows + 1);
    entriesU = col_ind_type("entries U", nnzU);
    valuesU  = values_type("values U", nnzU);
  }

  col_ind_type get_permutation() { return permutation; }

  void sort_factors() {
    KokkosSparse::sort_crs_matrix<execution_space, row_map_type, col_ind_type,
                                  values_type>(row_mapL, entriesL, valuesL);
    KokkosSparse::sort_crs_matrix<execution_space, row_map_type, col_ind_type,
                                  values_type>(row_mapU, entriesU, valuesU);
  }

  crs_matrix_type getL() {
    return KokkosSparse::Impl::transpose_matrix<crs_matrix_type>(
        crs_matrix_type("L", numRows, numRows, entriesL.extent(0), valuesL,
                        row_mapL, entriesL));
  }

  crs_matrix_type getU() {
    return crs_matrix_type("U", numRows, numRows, entriesU.extent(0), valuesU,
                           row_mapU, entriesU);
  }
};

}  // namespace Experimental
}  // namespace KokkosSparse

#endif  // KOKKOSSPARSE_MDF_HANDLE_HPP_
