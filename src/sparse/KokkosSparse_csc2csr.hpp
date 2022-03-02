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

#ifndef _KOKKOSSPARSE_CSC2CSR_HPP
#define _KOKKOSSPARSE_CSC2CSR_HPP
namespace KokkosSparse {
template <class OrdinalType, class SizeType, class ValViewType,
          class RowIdViewType, class ColMapViewType>
auto csc2csr(OrdinalType nrows, OrdinalType ncols, SizeType nnz,
             ValViewType vals, RowIdViewType row_ids, ColMapViewType col_map) {
  using CrsST             = typename ValViewType::value_type;
  using CrsOT             = OrdinalType;
  using CrsDT             = typename ValViewType::execution_space;
  using CrsMT             = void;
  using CrsSzT            = SizeType;
  using CrsType           = CrsMatrix<CrsST, CrsOT, CrsDT, CrsMT, CrsSzT>;
  using CrsValsViewType   = typename CrsType::values_type;
  using CrsRowMapViewType = typename CrsType::row_map_type::non_const_type;
  using CrsColIdViewType  = typename CrsType::index_type;

  CrsValsViewType crs_vals(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "csc2csr vals"), nnz);
  CrsRowMapViewType crs_row_map(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "csc2csr row_map"),
      nrows + 1);
  CrsColIdViewType crs_col_ids(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "csc2csr col_ids"), nnz);

  // TODO: populate crs views

  return CrsType("csc2csr", nrows, ncols, nnz, crs_vals, crs_row_map,
                 crs_col_ids);
}
}  // namespace KokkosSparse
#endif  //  _KOKKOSSPARSE_CSC2CSR_HPP