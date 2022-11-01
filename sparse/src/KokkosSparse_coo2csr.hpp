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
#include "KokkosKernels_HashmapAccumulator.hpp"
#include <Kokkos_StdAlgorithms.hpp>

#ifndef _KOKKOSSPARSE_COO2CSR_HPP
#define _KOKKOSSPARSE_COO2CSR_HPP
namespace KokkosSparse {
namespace Impl {
template <class DimType, class RowViewType, class ColViewType,
          class DataViewType>
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
  using AtomicRowIdViewType =
      Kokkos::View<typename RowIdViewType::value_type *,
                   typename RowIdViewType::array_layout,
                   typename RowIdViewType::execution_space,
                   Kokkos::MemoryTraits<Kokkos::Atomic>>;
  using ColMapViewType = ColViewType;
  using ValViewType    = DataViewType;
  using OrdinalType    = CrsOT;
  using SizeType       = CrsSzT;

  using KeyType          = uint32_t;
  using ValueType        = typename DataViewType::value_type;
  using ScratchSpaceType = typename CrsET::scratch_memory_space;
  using KeyViewScratch =
      Kokkos::View<KeyType *, Kokkos::LayoutRight, ScratchSpaceType>;
  /* using ValViewScratch =
      Kokkos::View<ValueType *, Kokkos::LayoutRight, ScratchSpaceType>; */
  using SizeViewScratch =
      Kokkos::View<SizeType *, Kokkos::LayoutRight, ScratchSpaceType>;

  OrdinalType __nrows;
  OrdinalType __ncols;
  SizeType __nnz;
  ValViewType __vals;
  RowIdViewType __row_ids;
  ColMapViewType __col_map;

  AtomicRowIdViewType __crs_row_cnt;
  using RowViewScalarType = typename AtomicRowIdViewType::value_type;

  CrsValsViewType __crs_vals;
  CrsRowMapViewType __crs_row_map;
  CrsRowMapViewType __crs_row_map_scratch;
  CrsColIdViewType __crs_col_ids;

  size_t __n_tuples;
  bool __insert_mode;

 public:
  struct phase1Tags {
    struct s1RowCntDup {};
    struct s2MaxRowCnt {};
    struct s3UniqRows {};
    struct s4RowCnt {};
    struct s5MaxRowCnt {};
  };

  using s3Policy = Kokkos::TeamPolicy<typename phase1Tags::s3UniqRows, CrsET>;

  class __Phase1Functor {
   private:
    using s3MemberType = typename s3Policy::member_type;
    AtomicRowIdViewType __crs_row_cnt;
    unsigned __n;

   public:
    RowViewType __row;
    ColViewType __col;
    DataViewType __data;
    unsigned teams_work, last_teams_work;
    RowViewScalarType max_row_cnt;
    RowViewScalarType pow2_max_row_cnt;
    Kokkos::View<unsigned *, CrsET> n_unique_rows_per_team;

    __Phase1Functor(RowViewType row, ColViewType col, DataViewType data,
                    AtomicRowIdViewType crs_row_cnt)
        : __crs_row_cnt(crs_row_cnt), __row(row), __col(col), __data(data) {
      __n = data.extent(0);
    };

    KOKKOS_INLINE_FUNCTION
    void operator()(const typename phase1Tags::s1RowCntDup &,
                    const int &thread_id) const {
      auto i = __row(thread_id);
      auto j = __col(thread_id);

      if (i >= 0 && j >= 0) __crs_row_cnt(i)++;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const typename phase1Tags::s2MaxRowCnt,
                    const unsigned long &thread_id,
                    RowViewScalarType &value) const {
      if (__crs_row_cnt(thread_id) > value) value = __crs_row_cnt(thread_id);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const typename phase1Tags::s3UniqRows &,
                    const s3MemberType &member) const {
      unsigned n = member.league_rank() == member.league_size() - 1
                       ? teams_work
                       : last_teams_work;
      unsigned start_n = teams_work * member.league_rank();
      unsigned stop_n  = start_n + n;

      KeyViewScratch keys(max_row_cnt);
      SizeViewScratch hash_ll(pow2_max_row_cnt * 2);
      volatile SizeType used_size;
      unsigned unique_row_cnt = 0;

      KokkosKernels::Experimental::HashmapAccumulator<
          SizeType, KeyType, ValueType,
          KokkosKernels::Experimental::HashOpType::bitwiseAnd>
          uset(max_row_cnt, pow2_max_row_cnt, hash_ll.data(),
               hash_ll.data() + pow2_max_row_cnt, keys.data(), nullptr);

      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, start_n, stop_n),
                           [&](const int &tid) {
                             KeyType i = __row(tid);
                             unique_row_cnt +=
                                 uset.vector_atomic_insert_into_hash_KeyCounter(
                                     i, &used_size);
                           });
      // All threads add their partial row counts
      Kokkos::atomic_add(n_unique_rows_per_team[member.league_rank()],
                         unique_row_cnt);
    }
  };

  struct phase2Tags {
    struct s6RowMap {};
    struct s7Copy {};
  };

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
    AtomicRowIdViewType __crs_row_cnt;

   public:
    __Phase2Functor(OrdinalType nrows, OrdinalType ncols, SizeType nnz,
                    ValViewType vals, CrsValsViewType crs_vals,
                    RowIdViewType row_ids, CrsRowMapViewType crs_row_map,
                    CrsRowMapViewType crs_row_map_scratch,
                    ColMapViewType col_map, CrsColIdViewType crs_col_ids,
                    AtomicRowIdViewType crs_row_cnt)
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
  unsigned int __suggested_team_size, __suggested_vec_size, __league_size;

  template <class FunctorType>
  void __runPhase1(FunctorType &functor) {
    {
#if 0
      for (size_t i = 0; i < __n_tuples; i++) {
        std::cout << "(" << functor.__row(i) << ", " << functor.__col(i) << ", " << functor.__data(i) << ")" << std::endl;
      }
#endif

      Kokkos::parallel_for(
          "Coo2Csr::phase1Tags::s1RowCntDup",
          Kokkos::RangePolicy<typename phase1Tags::s1RowCntDup, CrsET>(
              0, __n_tuples),
          functor);
      CrsET().fence();

      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<CrsET, typename phase1Tags::s2MaxRowCnt>(0,
                                                                       __nrows),
          functor, functor.max_row_cnt);
      CrsET().fence();

      functor.pow2_max_row_cnt = 1;
      while (functor.pow2_max_row_cnt < functor.max_row_cnt)
        functor.pow2_max_row_cnt *= 2;

      s3Policy s3p;
      __suggested_team_size =
          s3p.team_size_recommended(functor, Kokkos::ParallelForTag());
      __league_size      = __n_tuples / __suggested_team_size;
      functor.teams_work = __league_size;
      functor.last_teams_work =
          __n_tuples - (__league_size * __suggested_team_size);
      functor.last_teams_work = functor.last_teams_work == 0
                                    ? functor.teams_work
                                    : functor.last_teams_work;
      functor.n_unique_rows_per_team = Kokkos::View<unsigned *, CrsET>(
          "n_unique_rows_per_team", __league_size);

      int shmem_size =
          KeyViewScratch::shmem_size(functor.max_row_cnt) +
          SizeViewScratch::shmem_size(functor.pow2_max_row_cnt *
                                      2);  // * 2 for hash_begins & hash_nexts

      s3p = s3Policy(__league_size, __suggested_team_size);
      s3p.set_scratch_size(0, Kokkos::PerTeam(shmem_size));
    }
    return;
  }

  template <class FunctorType>
  void __runPhase2(FunctorType &functor) {
    return;
  }

 public:
  Coo2Csr(DimType m, DimType n, RowViewType row, ColViewType col,
          DataViewType data, bool insert_mode) {
    __insert_mode = insert_mode;
    __n_tuples    = data.extent(0);
    __nrows       = m;
    __ncols       = n;

    __crs_row_cnt = AtomicRowIdViewType("__crs_row_cnt", m);
    __Phase1Functor phase1Functor(row, col, data, __crs_row_cnt);
    __runPhase1(phase1Functor);

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
/// \tparam DimType the dimension type
/// \tparam RowViewType The row array view type
/// \tparam ColViewType The column array view type
/// \tparam DataViewType The data array view type
/// \param m the number of rows
/// \param n the number of columns
/// \param row the array of row ids
/// \param col the array of col ids
/// \param data the array of data
/// \param insert_mode whether to insert values. By default, values are added.
/// \return A KokkosSparse::CrsMatrix.
template <class DimType, class RowViewType, class ColViewType,
          class DataViewType>
auto coo2csr(DimType m, DimType n, RowViewType row, ColViewType col,
             DataViewType data, bool insert_mode = false) {
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

  if (row.extent(0) != col.extent(0) || row.extent(0) != data.extent(0))
    Kokkos::abort("row.extent(0) = col.extent(0) = data.extent(0) required.");

  using Coo2csrType =
      Impl::Coo2Csr<DimType, RowViewType, ColViewType, DataViewType>;
  Coo2csrType Coo2Csr(m, n, row, col, data, insert_mode);
  return Coo2Csr.get_csrMat();
}

#if 0
/// \brief Inserts new values into the given CrsMatrix.
/// \tparam DimType the dimension type
/// \tparam RowViewType The row array view type
/// \tparam ColViewType The column array view type
/// \tparam DataViewType The data array view type
/// \param m the number of rows
/// \param n the number of columns
/// \param row the array of row ids
/// \param col the array of col ids
/// \param data the array of data
/// \param insert_mode whether to insert values. By default, values are added.
/// \return A KokkosSparse::CrsMatrix.
template <class DimType, class RowViewType, class ColViewType, class DataViewType, class CrsMatrixType>
auto coo2csr(DimType m, DimType n, RowViewType row, ColViewType col, DataViewType data, CrsMatrixType crsMatrix) {
  // TODO: Run phase2 only.
}
#endif
}  // namespace KokkosSparse
#endif  //  _KOKKOSSPARSE_COO2CSR_HPP
