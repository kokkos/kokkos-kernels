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

#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_Utils.hpp"
#include <Kokkos_StdAlgorithms.hpp>

#ifndef _KOKKOSSPARSE_COO2CSR_HPP
#define _KOKKOSSPARSE_COO2CSR_HPP
namespace KokkosSparse {
namespace Impl {
template <class RowViewType, class ColViewType, class DataViewType>
class Coo2Csr {
 private:
  using CrsST             = typename DataViewType::value_type;
  using CrsOT             = int64_t;
  using CrsET             = typename DataViewType::execution_space;
  using CrsMT             = void;
  using CrsSzT            = size_t;
  using CrsType           = CrsMatrix<CrsST, CrsOT, CrsET, CrsMT, CrsSzT>;
  using CrsValsViewType   = typename CrsType::values_type;
  using CrsRowMapViewType = typename CrsType::row_map_type::non_const_type;
  using CrsColIdViewType  = typename CrsType::index_type;
  using RowIdViewType     = RowViewType;
  using ColMapViewType    = ColViewType;
  using ValViewType       = DataViewType;
  using OrdinalType       = CrsOT;
  using SizeType          = CrsSzT;

  OrdinalType __nrows;
  OrdinalType __ncols;
  SizeType __nnz;
  ValViewType __vals;
  RowIdViewType __row_ids;
  ColMapViewType __col_map;

  RowIdViewType __crs_row_cnt;

  CrsValsViewType __crs_vals;
  CrsRowMapViewType __crs_row_map;
  CrsRowMapViewType __crs_row_map_scratch;
  CrsColIdViewType __crs_col_ids;

  bool __insert_mode;

 public:
  struct phase1Tags {
    struct s1RowCntDup {};
    struct s2MaxRowCnt {};
    struct s3UniqRows {};
    struct s4RowCnt {};
    struct s5MaxRowCnt {};
  };

  template <class MemberType>
  class __Phase1Functor {
   private:
    RowViewType __row;
    ColViewType __col;
    DataViewType __data;

   public:
    __Phase1Functor(RowViewType row, ColViewType col, DataViewType data)
        : __row(row), __col(col), __data(data){};
  };

  struct phase2Tags {
    struct s6RowMap {};
    struct s7Copy {};
  };

  template <class MemberType>
  class __Phase2Functor {
   private:
    OrdinalType __nrows;
    OrdinalType __ncols;
    SizeType __nnz;
    ValViewType __vals;
    CrsValsViewType __crs_vals;
    RowIdViewType __row_ids;
    CrsRowMapViewType __crs_row_map;
    CrsRowMapViewType __crs_row_map_scratch;
    ColMapViewType __col_map;
    CrsColIdViewType __crs_col_ids;
    RowIdViewType __crs_row_cnt;

   public:
    __Phase2Functor(OrdinalType nrows, OrdinalType ncols, SizeType nnz,
                    ValViewType vals, CrsValsViewType crs_vals,
                    RowIdViewType row_ids, CrsRowMapViewType crs_row_map,
                    CrsRowMapViewType crs_row_map_scratch,
                    ColMapViewType col_map, CrsColIdViewType crs_col_ids,
                    RowIdViewType crs_row_cnt)
        : __nrows(nrows),
          __ncols(ncols),
          __nnz(nnz),
          __vals(vals),
          __crs_vals(crs_vals),
          __row_ids(row_ids),
          __crs_row_map(crs_row_map),
          __crs_row_map_scratch(crs_row_map_scratch),
          __col_map(col_map),
          __crs_col_ids(crs_col_ids),
          __crs_row_cnt(crs_row_cnt){};
  };

 private:
  using TeamPolicyType = Kokkos::TeamPolicy<typename phase2Tags::s7Copy, CrsET>;

  int __suggested_team_size, __suggested_vec_size, __league_size;

  template <class MemberType>
  void __runPhase1(__Phase1Functor<MemberType> &functor) {
    /*     // s1RowCntTag
        {
          Kokkos::parallel_for("Coo2Csr",
                               Kokkos::RangePolicy<AlgoTags::s1RowCntDup,
       CrsET>(0, __coo_n), functor); CrsET().fence();
        } */
    return;
  }

  template <class MemberType>
  void __runPhase2(__Phase2Functor<MemberType> &functor) {
    return;
  }

 public:
  Coo2Csr(RowViewType row, ColViewType col, DataViewType data,
          bool insert_mode) {
    __insert_mode = insert_mode;
    __Phase1Functor<typename TeamPolicyType::member_type> phase1Functor(
        row, col, data);
    __runPhase1<typename TeamPolicyType::member_type>(phase1Functor);

    /*    __crs_vals = CrsValsViewType(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "__crs_vals"), nnz);
    __crs_row_map = CrsRowMapViewType(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "__crs_row_map"),
        nrows + 1);
    __crs_row_map_scratch =
        CrsRowMapViewType(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             "__crs_row_map_scratch"),
                          nrows + 1);
    __crs_col_ids = CrsColIdViewType(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "__crs_col_ids"), nnz);

    __crs_row_cnt = RowIdViewType("__crs_row_cnt", __nrows + 1);

        KokkosKernels::Impl::get_suggested_vector_size<int64_t, CrsET>(
        __suggested_vec_size, __nrows, __nnz);
    __suggested_team_size =
        KokkosKernels::Impl::get_suggested_team_size<TeamPolicyType>(
            functor, __suggested_vec_size);

    __Phase2Functor<typename TeamPolicyType::member_type> phase2Functor(__nrows,
    __ncols, __nnz, __vals, __crs_vals, __row_ids, __crs_row_map,
        __crs_row_map_scratch, __col_map, __crs_col_ids, __crs_row_cnt);
    __runPhase2(phase2Functor);
    */
  }

  CrsType get_csrMat() {
    return CrsType("coo2csr", __nrows, __ncols, __nnz, __crs_vals,
                   __crs_row_map, __crs_col_ids);
  }
};
}  // namespace Impl
///
/// \brief Converts a coo matrix to a CrsMatrix.
/// \tparam RowViewType The row array view type
/// \tparam ColViewType The column array view type
/// \tparam DataViewType The data array view type
/// \param row the array of row ids
/// \param col the array of col ids
/// \param data the array of data
/// \param insert_mode whether to insert values. By default, values are added.
/// \return A KokkosSparse::CrsMatrix.
template <class RowViewType, class ColViewType, class DataViewType>
auto coo2csr(RowViewType row, ColViewType col, DataViewType data,
             bool insert_mode = false) {
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  static_assert(Kokkos::is_view<RowViewType>::value,
                "RowViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<ColViewType>::value,
                "CalViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<DataViewType>::value,
                "DataViewType must be a Kokkos::View.");
  static_assert(static_cast<int>(RowViewType::rank) == 1,
                "RowViewType must have rank 1.");
  static_assert(static_cast<int>(ColViewType::rank) == 1,
                "ColViewType must have rank 1.");
  static_assert(static_cast<int>(DataViewType::rank) == 1,
                "DataViewType must have rank 1.");
#endif

  static_assert(std::is_integral<typename RowViewType::value_type>::value,
                "RowViewType::value_type must be an integral.");
  static_assert(std::is_integral<typename ColViewType::value_type>::value,
                "ColViewType::value_type must be an integral.");

  if (insert_mode) Kokkos::abort("insert_mode not supported yet.");

  using Coo2csrType = Impl::Coo2Csr<RowViewType, ColViewType, DataViewType>;
  Coo2csrType Coo2Csr(row, col, data, insert_mode);
  return Coo2Csr.get_csrMat();
}

#if 0
/// \brief Inserts new values into the given CrsMatrix.
/// \tparam RowViewType The row array view type
/// \tparam ColViewType The column array view type
/// \tparam DataViewType The data array view type
/// \param row the array of row ids
/// \param col the array of col ids
/// \param data the array of data
/// \param insert_mode whether to insert values. By default, values are added.
/// \return A KokkosSparse::CrsMatrix.
template <class RowViewType, class ColViewType, class DataViewType, class CrsMatrixType>
auto coo2csr(RowViewType row, ColViewType col, DataViewType data, CrsMatrixType crsMatrix) {
  // TODO: Run phase2 only.
}
#endif
}  // namespace KokkosSparse
#endif  //  _KOKKOSSPARSE_COO2CSR_HPP
