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

#ifndef _KOKKOS_TWOSTAGE_GS_IMP_HPP
#define _KOKKOS_TWOSTAGE_GS_IMP_HPP

#include <KokkosKernels_config.h>
//#include "KokkosKernels_Utils.hpp"
#include "Kokkos_Core.hpp"
#include "KokkosSparse_gauss_seidel_handle.hpp"
//#include <Kokkos_Atomic.hpp>
//#include <impl/Kokkos_Timer.hpp>
//#include <Kokkos_MemoryTraits.hpp>

#include "KokkosBlas1_scal.hpp"
#include "KokkosBlas1_mult.hpp"
#include "KokkosBlas1_axpby.hpp"
#include "KokkosSparse_spmv.hpp"
#define KOKKOSSPARSE_IMPL_CLASSICAL_GS
#ifdef KOKKOSSPARSE_IMPL_CLASSICAL_GS
 //#include "KokkosSparse_sptrsv_handle.hpp"
 #include "KokkosSparse_sptrsv.hpp"
#endif

//FOR DEBUGGING
#include "KokkosBlas1_nrm2.hpp"

#define IFPACK_GS_JR_MERGE_SPMV
#define IFPACK_GS_JR_EXPLICIT_RESIDUAL

namespace KokkosSparse{
  namespace Impl{

    template <typename HandleType, typename input_row_map_view_t, typename input_entries_view_t, typename input_values_view_t>
    class TwostageGaussSeidel{

    public:
      using TwoStageGaussSeidelHandleType = typename HandleType::TwoStageGaussSeidelHandleType;
      using execution_space = typename HandleType::HandleExecSpace;
      using memory_space    = typename TwoStageGaussSeidelHandleType::memory_space;

      using const_ordinal_t     = typename TwoStageGaussSeidelHandleType::const_ordinal_t;
      using       ordinal_t     = typename TwoStageGaussSeidelHandleType::ordinal_t;
      using       size_type     = typename TwoStageGaussSeidelHandleType::size_type;
      using        scalar_t     = typename TwoStageGaussSeidelHandleType::scalar_t;
      using  row_map_view_t     = typename TwoStageGaussSeidelHandleType::row_map_view_t;
      using  entries_view_t     = typename TwoStageGaussSeidelHandleType::entries_view_t;
      using   values_view_t     = typename TwoStageGaussSeidelHandleType::values_view_t;

      using  const_scalar_t     = typename TwoStageGaussSeidelHandleType::const_scalar_t;
      using  const_row_map_view_t = typename TwoStageGaussSeidelHandleType::const_row_map_view_t;

      using       crsmat_t      = typename TwoStageGaussSeidelHandleType::crsmat_t;
      using        graph_t      = typename TwoStageGaussSeidelHandleType::graph_t;

      using range_type = Kokkos::pair<int, int>;

      // to wrap input (rowmap, colind, values) into crsmat
      using input_device_t = Kokkos::Device<typename input_row_map_view_t::execution_space, typename input_row_map_view_t::memory_space>;
      using input_crsmat_t = KokkosSparse::CrsMatrix <typename input_values_view_t::value_type,
                                                      typename input_entries_view_t::value_type,
                                                      input_device_t,
                                                      typename input_values_view_t::memory_traits,
                                                      typename input_row_map_view_t::value_type>;
      using input_graph_t  = typename input_crsmat_t::StaticCrsGraphType;

    private:
      HandleType *handle;

      //Get the specialized TwostageGaussSeidel handle from the main handle
      TwoStageGaussSeidelHandleType* get_gs_handle()
      {
        auto gsHandle = dynamic_cast<TwoStageGaussSeidelHandleType*>(this->handle->get_gs_handle());
        if(!gsHandle)
        {
          throw std::runtime_error("TwostageGaussSeidel: GS handle has not been created, or is set up for Cluster GS.");
        }
        return gsHandle;
      }

      const_ordinal_t num_rows, num_cols;

      input_row_map_view_t rowmap_view;
      input_entries_view_t column_view;
      input_values_view_t  values_view;

    // --------------------------------------------------------- //
    public:
      // tag for counting nnz
      struct Tag_countNnzL{};
      struct Tag_countNnzU{};
      struct Tag_countNnzLU{};
      // tag for inserting entries
      struct Tag_entriesL{};
      struct Tag_entriesU{};
      struct Tag_entriesLU{};
      // tag for inserting values
      struct Tag_valuesL{};
      struct Tag_valuesU{};
      struct Tag_valuesLU{};
      // tag for computing map_row
      struct Tag_compPtr{};

      template <typename output_row_map_view_t, 
                typename output_entries_view_t, 
                typename output_values_view_t>
      struct TwostageGaussSeidel_functor {

        public:
        // input
        bool two_stage;
        const_ordinal_t num_rows;
        input_row_map_view_t rowmap_view;
        input_entries_view_t column_view;
        input_values_view_t values_view;
        // output
        output_row_map_view_t  row_map;
        output_entries_view_t  entries;
        output_values_view_t   values;
        output_values_view_t   diags;

        output_row_map_view_t  row_cnt;

        // output
        output_row_map_view_t  row_map2;
        output_entries_view_t  entries2;
        output_values_view_t   values2;

#define KOKKOSSPARSE_IMPL_TWOSTAGE_GS_SPLIT_NNZ_COUNT
#ifdef KOKKOSSPARSE_IMPL_TWOSTAGE_GS_SPLIT_NNZ_COUNT
        // for counting nnz
        TwostageGaussSeidel_functor (
                  bool two_stage_,
                  const_ordinal_t num_rows_,
                  input_row_map_view_t  rowmap_view_,
                  input_entries_view_t  column_view_,
                  output_row_map_view_t row_map_) :
          two_stage(two_stage_),
          num_rows(num_rows_),
          rowmap_view(rowmap_view_),
          column_view(column_view_),
          values_view(),
          row_map(row_map_)
        {}
#else
        // for counting nnzL & nnzU
        TwostageGaussSeidel_functor (
                  bool two_stage_,
                  const_ordinal_t num_rows_,
                  input_row_map_view_t  rowmap_view_,
                  input_entries_view_t  column_view_,
                  output_row_map_view_t row_map_,
                  output_row_map_view_t row_map2_) :
          two_stage(two_stage_),
          num_rows(num_rows_),
          rowmap_view(rowmap_view_),
          column_view(column_view_),
          values_view(),
          row_map(row_map_),
          entries(),
          values(),
          diags(),
          row_map2(row_map2_)
        {}
#endif

        // for storing entries
        TwostageGaussSeidel_functor (
                  bool two_stage_,
                  const_ordinal_t num_rows_,
                  input_row_map_view_t  rowmap_view_,
                  input_entries_view_t  column_view_,
                  output_row_map_view_t row_map_,
                  output_entries_view_t entries_) :
          two_stage(two_stage_),
          num_rows(num_rows_),
          rowmap_view(rowmap_view_),
          column_view(column_view_),
          values_view(),
          row_map(row_map_),
          entries(entries_)
        {}

        // for storing values
        TwostageGaussSeidel_functor (
                  bool two_stage_,
                  const_ordinal_t num_rows_,
                  input_row_map_view_t  rowmap_view_,
                  input_entries_view_t  column_view_,
                  input_values_view_t   values_view_,
                  output_row_map_view_t row_map_,
                  output_values_view_t  values_,
                  output_values_view_t  diags_) :
          two_stage(two_stage_),
          num_rows(num_rows_),
          rowmap_view(rowmap_view_),
          column_view(column_view_),
          values_view(values_view_),
          row_map(row_map_),
          entries(),
          values(values_),
          diags(diags_)
        {}

        // for storing booth L&U entries
        TwostageGaussSeidel_functor (
                  bool two_stage_,
                  const_ordinal_t num_rows_,
                  input_row_map_view_t  rowmap_view_,
                  input_entries_view_t  column_view_,
                  output_row_map_view_t row_map_,
                  output_entries_view_t entries_,
                  output_row_map_view_t row_map2_,
                  output_entries_view_t entries2_) :
          two_stage(two_stage_),
          num_rows(num_rows_),
          rowmap_view(rowmap_view_),
          column_view(column_view_),
          values_view(),
          row_map(row_map_),
          entries(entries_),
          values(),
          diags(),
          row_map2(row_map2_),
          entries2(entries2_)
        {}

        // for storing both L&U values 
        TwostageGaussSeidel_functor (
                  bool two_stage_,
                  const_ordinal_t num_rows_,
                  input_row_map_view_t  rowmap_view_,
                  input_entries_view_t  column_view_,
                  input_values_view_t   values_view_,
                  output_row_map_view_t row_map_,
                  output_values_view_t  values_,
                  output_values_view_t  diags_,
                  output_row_map_view_t row_map2_,
                  output_values_view_t  values2_):
          two_stage(two_stage_),
          num_rows(num_rows_),
          rowmap_view(rowmap_view_),
          column_view(column_view_),
          values_view(values_view_),
          row_map(row_map_),
          entries(),
          values(values_),
          diags(diags_),
          row_map2(row_map2_),
          entries2(),
          values2(values2_)
        {}

        // for generating ptr with parallel-scan
        TwostageGaussSeidel_functor (
                  const_ordinal_t num_rows_,
                  output_row_map_view_t row_map_,
                  output_row_map_view_t row_cnt_):
          two_stage(true),
          num_rows(num_rows_),
          rowmap_view(),
          column_view(),
          values_view(),
          row_map(row_map_),
          entries(),
          values(),
          diags(),
          row_cnt(row_cnt_)
        {}

        // ------------------------------------------------------- //
        // functor for counting nnzL (with parallel_reduce)
        KOKKOS_INLINE_FUNCTION
        void operator()(const Tag_countNnzL&, const ordinal_t i, ordinal_t &nnz) const
        {
          ordinal_t nnz_i = 0;
          for (size_type k = rowmap_view (i); k < rowmap_view (i+1); k++) {
            if (column_view (k) < i) {
              nnz_i ++;
            } else if(!two_stage && column_view (k) == i) {
              nnz_i ++;
            }
          }
          row_map (i+1) = nnz_i;
          if (i == 0) {
            row_map (0) = 0;
          }
          nnz +=  nnz_i;
        }

        // functor for storing entriesL (with parallel_for)
        KOKKOS_INLINE_FUNCTION
        void operator()(const Tag_entriesL&, const ordinal_t i) const
        {
          ordinal_t nnz = row_map (i);
          for (size_type k = rowmap_view (i); k < rowmap_view (i+1); k++) {
            if (column_view (k) < i) {
              entries (nnz) = column_view (k);
              nnz ++;
            } else if(!two_stage && column_view (k) == i) {
              entries (nnz) = column_view (k);
              nnz ++;
            }
          }
        }

        // functor for storing valuesL (with parallel_for)
        KOKKOS_INLINE_FUNCTION
        void operator()(const Tag_valuesL&, const ordinal_t i) const
        {
          const_scalar_t one (1.0);
          ordinal_t nnz = row_map (i);
          for (size_type k = rowmap_view (i); k < rowmap_view (i+1); k++) {
            if (column_view (k) < i) {
              values (nnz) = values_view (k);
              nnz ++;
            } else if (column_view (k) == i) {
              if (two_stage) {
                diags (i) = one / values_view (k);
              } else {
                values (nnz) = values_view (k);
                nnz ++;
              }
            }
          }
          #if defined(IFPACK_GS_JR_MERGE_SPMV)
          if (two_stage) {
            for (ordinal_t k = row_map (i); k < nnz; k++) {
              values (k) *= diags (i);
            }
          }
          #endif
        }


        // ------------------------------------------------------- //
        // functor for counting nnzU (with parallel_reduce)
        KOKKOS_INLINE_FUNCTION
        void operator()(const Tag_countNnzU&, const ordinal_t i, ordinal_t &nnz) const
        {
          ordinal_t nnz_i = 0;
          for (size_type k = rowmap_view (i); k < rowmap_view (i+1); k++) {
            if (column_view (k) > i && column_view (k) < num_rows) {
              nnz_i ++;
            } else if(!two_stage && column_view (k) == i) {
              nnz_i ++;
            }
          }
          row_map (i+1) = nnz_i;
          if (i == 0) {
            row_map (0) = 0;
          }
          nnz +=  nnz_i;
        }

        // functor for storing entriesU (with parallel_for)
        KOKKOS_INLINE_FUNCTION
        void operator()(const Tag_entriesU&, const ordinal_t i) const
        {
          ordinal_t nnz = row_map (i);
          for (size_type k = rowmap_view (i); k < rowmap_view (i+1); k++) {
            if (column_view (k) > i && column_view (k) < num_rows) {
              entries (nnz) = column_view (k);
              nnz ++;
            } else if(!two_stage && column_view (k) == i) {
              entries (nnz) = column_view (k);
              nnz ++;
            }
          }
        }

        // functor for storing valuesL (with parallel_for)
        KOKKOS_INLINE_FUNCTION
        void operator()(const Tag_valuesU&, const ordinal_t i) const
        {
          const_scalar_t one (1.0);
          ordinal_t nnz = row_map (i);
          for (size_type k = rowmap_view (i); k < rowmap_view (i+1); k++) {
            if (column_view (k) == i) {
              if (two_stage) {
                diags (i) = one / values_view (k);
              } else {
                values (nnz) = values_view (k);
                nnz ++;
              }
            } else if (column_view (k) > i && column_view (k) < num_rows) {
              values (nnz) = values_view (k);
              nnz ++;
            }
          }
          #if defined(IFPACK_GS_JR_MERGE_SPMV)
          if (two_stage) {
            for (ordinal_t k = row_map (i); k < nnz; k++) {
              values (k) *= diags (i);
            }
          }
          #endif
        }

        // ------------------------------------------------------- //
        // functor for counting nnzL & nnzU / row, and total nnzL (with parallel_reduce)
        KOKKOS_INLINE_FUNCTION
        void operator()(const Tag_countNnzLU&, const ordinal_t i, ordinal_t &nnz) const
        {
          ordinal_t nnzL = 0;
          ordinal_t nnzU = 0;
          for (size_type k = rowmap_view (i); k < rowmap_view (i+1); k++) {
            if (column_view (k) < i) {
              nnzL ++;
            } else if (column_view (k) == i && !two_stage) {
              nnzL ++;
              nnzU ++;
            } else if (column_view (k) > i && column_view (k) < num_rows) {
              nnzU ++;
            }
          }
          row_map (i+1) = nnzL;
          row_map2 (i+1) = nnzU;
          if (i == 0) {
            row_map (0) = 0;
            row_map2 (0) = 0;
          }
          nnz += nnzL;
        }

        // functor for storing entriesL and entriesU (with parallel_for)
        KOKKOS_INLINE_FUNCTION
        void operator()(const Tag_entriesLU&, const ordinal_t i) const
        {
          ordinal_t nnzL = row_map (i);
          ordinal_t nnzU = row_map2 (i);
          for (size_type k = rowmap_view (i); k < rowmap_view (i+1); k++) {
            if (column_view (k) < i) {
              entries (nnzL) = column_view (k);
              nnzL ++;
            } else if (column_view (k) == i && !two_stage) {
              entries (nnzL) = column_view (k);
              entries2 (nnzU) = column_view (k);
              nnzL ++;
              nnzU ++;
            } else if (column_view (k) > i && column_view (k) < num_rows) {
              entries2 (nnzU) = column_view (k);
              nnzU ++;
            }
          }
        }

        // functor for storing both valuesL & valuesU (with parallel_for)
        KOKKOS_INLINE_FUNCTION
        void operator()(const Tag_valuesLU&, const ordinal_t i) const
        {
          const_scalar_t one (1.0);
          ordinal_t nnzL = row_map (i);
          ordinal_t nnzU = row_map2 (i);
          for (size_type k = rowmap_view (i); k < rowmap_view (i+1); k++) {
            if (column_view (k) < i) {
              values (nnzL) = values_view (k);
              nnzL ++;
            } else if (column_view (k) == i) {
              if (two_stage) {
                diags (i) = one / values_view (k);
              } else {
                values (nnzL) = values_view (k);
                values2 (nnzU) = values_view (k);
                nnzL ++;
                nnzU ++;
              }
            } else if (column_view (k) < num_rows) {
              values2 (nnzU) = values_view (k);
              nnzU ++;
            }
          }
          #if defined(IFPACK_GS_JR_MERGE_SPMV)
          if (two_stage) {
            for (ordinal_t k = row_map (i); k < nnzL; k++) {
              values (k) *= diags (i);
            }
            for (ordinal_t k = row_map2 (i); k < nnzU; k++) {
              values2 (k) *= diags (i);
            }
          }
          #endif
        }


        // ------------------------------------------------------- //
        // functor for computinr row_map (with parallel_scan)
        KOKKOS_INLINE_FUNCTION
        void operator() (const Tag_compPtr&, const ordinal_t i, ordinal_t &update, const bool final) const
        {
          update += row_cnt (i);
          if (final) {
            row_map (i) = update;
          }
        }

        KOKKOS_INLINE_FUNCTION
        void init (unsigned & update) const { update = 0; }

        KOKKOS_INLINE_FUNCTION
        void join (volatile unsigned &update, const volatile unsigned &input) const { update += input; }
      };
    // --------------------------------------------------------- //


    public:
      /**
       * \brief constructor
       */
      // for symbolic (wihout values)
      TwostageGaussSeidel(HandleType *handle_,
                  const_ordinal_t num_rows_,
                  const_ordinal_t num_cols_,
                  input_row_map_view_t rowmap_view_,
                  input_entries_view_t column_view_) :
        handle(handle_),
        num_rows(num_rows_), num_cols(num_cols_),
        rowmap_view(rowmap_view_),
        column_view(column_view_),
        values_view() {}

      // for numeric/solve (with values)
      TwostageGaussSeidel (HandleType *handle_,
                           const_ordinal_t num_rows_,
                           const_ordinal_t num_cols_,
                           input_row_map_view_t rowmap_view_,
                           input_entries_view_t column_view_,
                           input_values_view_t values_view_) :
        handle(handle_),
        num_rows(num_rows_), num_cols(num_cols_),
        rowmap_view(rowmap_view_),
        column_view(column_view_),
        values_view(values_view_) {}


      /**
       * Symbolic setup
       */
      void initialize_symbolic ()
      {
#define KOKKOSSPARSE_IMPL_TIME_TWOSTAGE_GS
#ifdef KOKKOSSPARSE_IMPL_TIME_TWOSTAGE_GS
        double tic;
        Kokkos::Impl::Timer timer;
        Kokkos::fence();
        tic = timer.seconds ();
#endif
        auto *gsHandle = get_gs_handle();
        bool two_stage = gsHandle->isTwoStage ();
        GSDirection direction = gsHandle->getSweepDirection ();
        using GS_Functor_t = TwostageGaussSeidel_functor<row_map_view_t, entries_view_t, values_view_t>;
        // count nnz in local L & U matrices (rowmap_viewL/rowmap_viewU stores offsets for each row)
        ordinal_t nnzL = 0;
        row_map_view_t  rowmap_viewL ("row_mapL", num_rows+1);
        row_map_view_t  rowcnt_viewL ("row_cntL", num_rows+1);

        ordinal_t nnzU = 0;
        row_map_view_t  rowmap_viewU ("row_mapU", num_rows+1);
        row_map_view_t  rowcnt_viewU ("row_cntU", num_rows+1);
#ifdef KOKKOSSPARSE_IMPL_TWOSTAGE_GS_SPLIT_NNZ_COUNT
        if (direction == GS_FORWARD || direction == GS_SYMMETRIC) {
          using range_policy = Kokkos::RangePolicy <Tag_countNnzL, execution_space>;
          Kokkos::parallel_reduce ("nnzL", range_policy (0, num_rows),
                                   GS_Functor_t (two_stage, num_rows, rowmap_view, column_view,
                                                                      rowcnt_viewL),
                                   nnzL);
        }
        if (direction == GS_BACKWARD || direction == GS_SYMMETRIC) {
          using range_policy = Kokkos::RangePolicy <Tag_countNnzU, execution_space>;
          Kokkos::parallel_reduce ("nnzU", range_policy (0, num_rows),
                                   GS_Functor_t (two_stage, num_rows, rowmap_view, column_view,
                                                                      rowcnt_viewU),
                                   nnzU);
        }
#else
        {
          using range_policy = Kokkos::RangePolicy <Tag_countNnzLU, execution_space>;
          Kokkos::parallel_reduce ("nnzLU", range_policy (0, num_rows),
                                   GS_Functor_t (two_stage, num_rows, rowmap_view, column_view,
                                                                      rowcnt_viewL, rowcnt_viewU),
                                   nnzL);
          // NOTE: assuming nonzero diagonals
          nnzU = column_view.extent(0) - (nnzL + num_rows);
        }
#endif
#ifdef KOKKOSSPARSE_IMPL_TIME_TWOSTAGE_GS
        Kokkos::fence();
        tic = timer.seconds ();
        std::cout << std::endl << "TWO-STAGE GS::SYMBOLIC::COUNT-NNZ TIME : " << tic << std::endl;
        timer.reset();
#endif
        // shift ptr so that it now contains offsets (combine it with the previous functor calls?)
        if (direction == GS_FORWARD || direction == GS_SYMMETRIC) {
          using range_policy = Kokkos::RangePolicy <Tag_compPtr, execution_space>;
          Kokkos::parallel_scan ("ptrL", range_policy (0, 1+num_rows),
                                 GS_Functor_t (num_rows, rowmap_viewL, rowcnt_viewL));
        }
        if (direction == GS_BACKWARD || direction == GS_SYMMETRIC) {
          using range_policy = Kokkos::RangePolicy <Tag_compPtr, execution_space>;
          Kokkos::parallel_scan ("ptrU", range_policy (0, 1+num_rows),
                                 GS_Functor_t (num_rows, rowmap_viewU, rowcnt_viewU));
        }
#ifdef KOKKOSSPARSE_IMPL_TIME_TWOSTAGE_GS
        Kokkos::fence();
        tic = timer.seconds ();
        std::cout << "TWO-STAGE GS::SYMBOLIC::COMP-PTR TIME  : " << tic << std::endl;
        timer.reset();
#endif
        // allocate memory to store local D
        values_view_t viewD (Kokkos::ViewAllocateWithoutInitializing("diags"), num_rows);

        // allocate memory to store local L
        entries_view_t  column_viewL (Kokkos::ViewAllocateWithoutInitializing("entriesL"), nnzL);
        values_view_t   values_viewL (Kokkos::ViewAllocateWithoutInitializing("valuesL"),  nnzL);

        // allocate memory to store local U
        entries_view_t  column_viewU (Kokkos::ViewAllocateWithoutInitializing("entriesU"), nnzU);
        values_view_t   values_viewU (Kokkos::ViewAllocateWithoutInitializing("valuesU"),  nnzU);
#ifdef KOKKOSSPARSE_IMPL_TIME_TWOSTAGE_GS
        Kokkos::fence();
        tic = timer.seconds ();
        std::cout << "TWO-STAGE GS::SYMBOLIC::ALLOCATE TIME  : " << tic << std::endl;
        timer.reset();
#endif

        {
          // extract local L & U structures (for computing (L+D)^{-1} or (D+U)^{-1})
          using range_policy = Kokkos::RangePolicy <Tag_entriesLU, execution_space>;
          Kokkos::parallel_for ("entryLU", range_policy (0, num_rows),
                                GS_Functor_t (two_stage, num_rows, rowmap_view, column_view,
                                                                   rowmap_viewL, column_viewL,
                                                                   rowmap_viewU, column_viewU));
        }
#ifdef KOKKOSSPARSE_IMPL_TIME_TWOSTAGE_GS
        Kokkos::fence();
        tic = timer.seconds ();
        std::cout << "TWO-STAGE GS::SYMBOLIC::INSERT TIME    : " << tic << std::endl;
        timer.reset();
#endif

        // construct CrsMat with them
        graph_t graphL (column_viewL, rowmap_viewL);
        graph_t graphU (column_viewU, rowmap_viewU);
        crsmat_t crsmatL ("L", num_rows, values_viewL, graphL);
        crsmat_t crsmatU ("U", num_rows, values_viewU, graphU);

        // store them in handle
        gsHandle->setL (crsmatL);
        gsHandle->setU (crsmatU);
        gsHandle->setD (viewD);
#ifdef KOKKOSSPARSE_IMPL_CLASSICAL_GS
        if (!(gsHandle->isTwoStage ())) {
          using namespace KokkosSparse::Experimental;
          #if 1 // !defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE) || !defined(KOKKOS_ENABLE_CUDA)
          using namespace KokkosSparse::Experimental;
          sptrsv_symbolic (handle->get_gs_sptrsvL_handle(), rowmap_viewL, crsmatL.graph.entries);
          sptrsv_symbolic (handle->get_gs_sptrsvU_handle(), rowmap_viewL, crsmatU.graph.entries);
          #endif
        }
#endif
      }


      /**
       * Numerical setup
       */
      void initialize_numeric ()
      {
#ifdef KOKKOSSPARSE_IMPL_TIME_TWOSTAGE_GS
        double tic;
        Kokkos::Impl::Timer timer;
        Kokkos::fence();
        timer.reset();
#endif
        using GS_Functor_t = TwostageGaussSeidel_functor<const_row_map_view_t, entries_view_t, values_view_t>;

        auto *gsHandle = get_gs_handle();
        bool two_stage = gsHandle->isTwoStage ();

        // load local D from handle
        auto viewD = gsHandle->getD ();

        // load local L from handle
        auto crsmatL = gsHandle->getL ();
        auto values_viewL = crsmatL.values;
        auto rowmap_viewL = crsmatL.graph.row_map;

        // load local U from handle
        auto crsmatU = gsHandle->getU ();
        auto values_viewU = crsmatU.values;
        auto rowmap_viewU = crsmatU.graph.row_map;

        // extract local L, D & U matrices
        using range_policy = Kokkos::RangePolicy <Tag_valuesLU, execution_space>;
        Kokkos::parallel_for ("valueLU", range_policy (0, num_rows),
                              GS_Functor_t (two_stage, num_rows, rowmap_view, column_view, values_view, 
                                                                 rowmap_viewL, values_viewL, viewD,
                                                                 rowmap_viewU, values_viewU));
#ifdef KOKKOSSPARSE_IMPL_TIME_TWOSTAGE_GS
        Kokkos::fence();
        tic = timer.seconds ();
        std::cout << std::endl << "TWO-STAGE GS::NUMERIC::INSERT LU TIME : " << tic << std::endl;
        timer.reset();
#endif

#ifdef KOKKOSSPARSE_IMPL_CLASSICAL_GS
        if (!(gsHandle->isTwoStage ())) {
          #if 0 // defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE) && defined(KOKKOS_ENABLE_CUDA)
          using namespace KokkosSparse::Experimental;
          sptrsv_symbolic (handle->get_gs_sptrsvL_handle(), rowmap_viewL, crsmatL.graph.entries, values_viewL);
          sptrsv_symbolic (handle->get_gs_sptrsvU_handle(), rowmap_viewL, crsmatU.graph.entries, values_viewU);
          #endif
        }
#endif
      }


      /**
       * Apply solve
       */
      template <typename x_value_array_type, typename y_value_array_type>
      void apply (x_value_array_type localX, // in/out
                  y_value_array_type localB, // in
                  bool init_zero_x_vector = false,
                  int numIter = 1,
                  scalar_t omega = Kokkos::Details::ArithTraits<scalar_t>::one(),
                  bool apply_forward = true,
                  bool apply_backward = true,
                  bool update_y_vector = true)
      {
        const_scalar_t one (1.0);
        const_scalar_t zero (0.0);
#ifdef KOKKOSSPARSE_IMPL_TIME_TWOSTAGE_GS
        double tic;
        Kokkos::Impl::Timer timer;
        Kokkos::fence();
        tic = timer.seconds ();
#endif

        //
        auto *gsHandle = get_gs_handle();
        bool two_stage = gsHandle->isTwoStage ();
        GSDirection direction = gsHandle->getSweepDirection ();
        if (apply_forward && apply_backward) {
          direction = GS_SYMMETRIC;
        } else if (apply_forward) {
          direction = GS_FORWARD;
        } else if (apply_backward) {
          direction = GS_BACKWARD;
        } else {
          return;
        }

        // load auxiliary matrices from handle
        auto localD = gsHandle->getD ();
        auto crsmatL = gsHandle->getL ();
        auto crsmatU = gsHandle->getU ();

        // wratp A into crsmat
        input_graph_t graphA (column_view, rowmap_view);
        input_crsmat_t crsmatA ("A", num_rows, values_view, graphA);
#ifdef KOKKOSSPARSE_IMPL_TIME_TWOSTAGE_GS
        Kokkos::fence();
        tic = timer.seconds ();
        std::cout << std::endl << "TWO-STAGE GS::APPLY::CREATE CRS_A TIME : " << tic << std::endl;
        timer.reset();
#endif

        // load auxiliary vectors
        int nrows = num_rows;
        int nrhs = localX.extent (1);
        gsHandle->initVectors (nrows, nrhs);
        auto localR = gsHandle->getVectorR ();
        auto localT = gsHandle->getVectorT ();
        auto localZ = gsHandle->getVectorZ ();

        // outer Gauss-Seidel iteration
        int NumSweeps = numIter;
        int NumInnerSweeps = gsHandle->getNumInnerSweeps ();
        if (direction == GS_SYMMETRIC) {
          NumSweeps *= 2;
        }
        for (int sweep = 0; sweep < NumSweeps; ++sweep) {
          #if defined(IFPACK_GS_JR_EXPLICIT_RESIDUAL)
          // R = B - A*x
          KokkosBlas::scal (localR, one, localB);
          if (sweep > 0 || !init_zero_x_vector) {
            KokkosSparse::
            spmv ("N", -one, crsmatA,
                             localX,
                        one, localR);
          }
          #else // !defined(IFPACK_GS_JR_EXPLICIT_RESIDUAL)
          if (direction == GS_FORWARD ||
             (direction == GS_SYMMETRIC && sweep%2 == 0)) {
            // R = B - U*x
            for (int i = 0; i < nrows; i++) {
              for (int j = 0; j < nrhs; j++) {
                localR (i, j) = localB (i, j);
                for (ordinal_t k = rowmap_view (i); k < rowmap_view (i+1); k++) {
                  if (entries (k) > i) {
                    localR (i, j) -= values(k) * localX (entries (k), j);
                  }
                }
              }
            }
          }
          else {
            // R = B - L*x
            for (int i = 0; i < nrows; i++) {
              for (int j = 0; j < nrhs; j++) {
                localR (i, j) = localB (i, j);
                for (ordinal_t k = rowmap_view (i); k < rowmap_view (i+1); k++) {
                  if (entries (k) < i || entries (k) >= nrows) {
                    localR (i, j) -= values(k) * localX (entries (k), j);
                  }
                }
              }
            }
          }
          #endif

          if (!two_stage) { // ===== sparse-triangular solve =====
            if (direction == GS_FORWARD ||
               (direction == GS_SYMMETRIC && sweep%2 == 0)) {
              // T = (L+D)^{-1} * R
              #if 1
              using namespace KokkosSparse::Experimental;
              for (int j = 0; j < nrhs; j++) {
                using single_vector_view_t = Kokkos::View<scalar_t*, Kokkos::LayoutLeft, execution_space>;
                auto localRj = Kokkos::subview (localR, Kokkos::ALL (), range_type (j, j+1));
                auto localZj = Kokkos::subview (localZ, Kokkos::ALL (), range_type (j, j+1));
                single_vector_view_t Rj (localRj.data (), nrows);
                single_vector_view_t Zj (localZj.data (), nrows);
                sptrsv_solve (handle->get_gs_sptrsvL_handle(), crsmatL.graph.row_map, crsmatL.graph.entries, crsmatL.values, Rj, Zj);
              }
              #else
              for (int i = 0; i < nrows; i++) {
                for (int j = 0; j < nrhs; j++) {
                  scalar_t d = zero;
                  localZ (i, j) = localR (i, j);
                  for (int k = rowmap_view (i); k < rowmap_view (i+1); k++) {
                    if (column_view (k) == i) {
                      d = values_view (k);
                    } else if (column_view (k) < i) {
                      localZ (i, j) -= values_view(k) * localZ (column_view (k), j);
                    }
                  } 
                  localZ (i, j) /= d;
                }
              }
              #endif
            } else {
              // T = (D+U)^{-1} * R
              #if 1
              using namespace KokkosSparse::Experimental;
              for (int j = 0; j < nrhs; j++) {
                using single_vector_view_t = Kokkos::View<scalar_t*, Kokkos::LayoutLeft, execution_space>;
                auto localRj = Kokkos::subview (localR, Kokkos::ALL (), range_type (j, j+1));
                auto localZj = Kokkos::subview (localZ, Kokkos::ALL (), range_type (j, j+1));
                single_vector_view_t Rj (localRj.data (), nrows);
                single_vector_view_t Zj (localZj.data (), nrows);
                sptrsv_solve (handle->get_gs_sptrsvU_handle(), crsmatU.graph.row_map, crsmatU.graph.entries, crsmatU.values, Rj, Zj);
              }
              #else
              for (int i = nrows-1; i >= 0; i--) {
                for (int j = 0; j < nrhs; j++) {
                  scalar_t d = zero;
                  localZ (i, j) = localR (i, j);
                  for (int k = rowmap_view (i); k < rowmap_view (i+1); k++) {
                    if (column_view (k) == i) {
                      d = values_view (k);
                    } else if (column_view (k) > i && column_view (k) < nrows) {
                      localZ (i, j) -= values_view (k) * localZ (column_view (k), j);
                    }
                  }
                  localZ (i, j) /= d;
                }
              }
              #endif
            }
            #if !defined(IFPACK_GS_JR_MERGE_SPMV)
            #endif
          } else {// ====== inner Jacobi-Richardson =====
            // compute starting vector: T = D^{-1}*R
            #if defined(IFPACK_GS_JR_EXPLICIT_RESIDUAL) && defined(IFPACK_GS_JR_MERGE_SPMV)
            if (NumInnerSweeps == 0) {
              // this is Jacobi-Richardson X_{k+1} := X_{k} + D^{-1}(b-A*X_{k})
              // copy to localZ (output of JR iteration)
              KokkosBlas::mult (zero, localZ,
                                one,  localD, localR);
            } else {
              // copy to localT (temporary used in JR iteration)
              KokkosBlas::mult (zero, localT,
                                one,  localD, localR);
            }
            #else
            KokkosBlas::mult (zero, localT,
                              one,  localD, localR);
            #endif
            // inner Jacobi-Richardson:
            for (int ii = 0; ii < NumInnerSweeps; ii++) {
              // Z = R
              #if defined(IFPACK_GS_JR_MERGE_SPMV)
              // R = D^{-1}*R, and L = D^{-1}*L and U = D^{-1}*U
              KokkosBlas::scal (localZ, one, localT);
              #else
              KokkosBlas::scal (localZ, one, localR);
              #endif
              if (direction == GS_FORWARD ||
                 (direction == GS_SYMMETRIC && sweep%2 == 0)) {
                // Z = R - L*T
                KokkosSparse::
                spmv("N", -one, crsmatL,
                                localT,
                           one, localZ);
              }
              else {
                // Z = R - U*T
                KokkosSparse::
                spmv("N", -one, crsmatU,
                                localT,
                           one, localZ);
              }
              #if defined(IFPACK_GS_JR_MERGE_SPMV)
              if (ii+1 < NumInnerSweeps) {
                KokkosBlas::scal (localT, one, localZ);
              }
              #else
              // T = D^{-1}*Z
              KokkosBlas::mult (zero, localT,
                                one,  localD, localZ);
              #endif
            } // end of inner Jacobi Richardson
          }

          #if defined(IFPACK_GS_JR_EXPLICIT_RESIDUAL)
           // Y = X + T
           auto localY = Kokkos::subview (localX, range_type(0, nrows), Kokkos::ALL ());
           #if defined(IFPACK_GS_JR_MERGE_SPMV)
           KokkosBlas::axpy (one, localZ, localY);
           #else
           KokkosBlas::axpy (one, localT, localY);
           #endif
          #else
          // Y = T
          for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < nrhs; j++) {
              localX (i, j) = localT (i, j);
            }
          }
          #endif
        } // end of outer GS sweep
      }
    };
  }
}
#endif
