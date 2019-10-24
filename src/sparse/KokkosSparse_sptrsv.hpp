/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

/// \file KokkosSparse_sptrsv.hpp
/// \brief Parallel sparse triangular solve
///
/// This file provides KokkosSparse::sptrsv.  This function performs a
/// local (no MPI) sparse triangular solve on matrices stored in
/// compressed row sparse ("Crs") format.

#ifndef KOKKOSSPARSE_SPTRSV_HPP_
#define KOKKOSSPARSE_SPTRSV_HPP_

#include <type_traits>

//#include "KokkosSparse_sptrsv_handle.hpp"
#include "KokkosKernels_helpers.hpp"
#include "KokkosSparse_sptrsv_symbolic_spec.hpp"
#include "KokkosSparse_sptrsv_solve_spec.hpp"

#ifdef KOKKOSKERNELS_ENABLE_TPL_SUPERLU
 #include "slu_ddefs.h"
 #include "KokkosSparse_sptrsv_superlu.hpp"
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_CHOLMOD
 #include "cholmod.h"
 #include "KokkosSparse_sptrsv_cholmod.hpp"
#endif

namespace KokkosSparse {
namespace Experimental {

#define KOKKOSKERNELS_SPTRSV_SAME_TYPE(A, B) std::is_same<typename std::remove_const<A>::type, typename std::remove_const<B>::type>::value

  template <typename KernelHandle,
            typename lno_row_view_t_,
            typename lno_nnz_view_t_>
  void sptrsv_symbolic(
      KernelHandle *handle, 
      lno_row_view_t_ rowmap,
      lno_nnz_view_t_ entries)
  {
    typedef typename KernelHandle::size_type size_type;
    typedef typename KernelHandle::nnz_lno_t ordinal_type;

    static_assert(KOKKOSKERNELS_SPTRSV_SAME_TYPE(typename lno_row_view_t_::non_const_value_type, size_type),
        "sptrsv_symbolic: A size_type must match KernelHandle size_type (const doesn't matter)");

    static_assert(KOKKOSKERNELS_SPTRSV_SAME_TYPE(typename lno_nnz_view_t_::non_const_value_type, ordinal_type),
        "sptrsv_symbolic: A entry type must match KernelHandle entry type (aka nnz_lno_t, and const doesn't matter)");


    typedef typename KernelHandle::const_size_type c_size_t;
    typedef typename KernelHandle::const_nnz_lno_t c_lno_t;
    typedef typename KernelHandle::const_nnz_scalar_t c_scalar_t;

    typedef typename KernelHandle::HandleExecSpace c_exec_t;
    typedef typename KernelHandle::HandleTempMemorySpace c_temp_t;
    typedef typename KernelHandle::HandlePersistentMemorySpace c_persist_t;

    typedef typename  KokkosKernels::Experimental::KokkosKernelsHandle<c_size_t, c_lno_t, c_scalar_t, c_exec_t, c_temp_t, c_persist_t> const_handle_type;
    const_handle_type tmp_handle (*handle);

    typedef Kokkos::View<
          typename lno_row_view_t_::const_value_type*,
          typename KokkosKernels::Impl::GetUnifiedLayout<lno_row_view_t_>::array_layout,
          typename lno_row_view_t_::device_type,
          Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess> > RowMap_Internal;

    typedef Kokkos::View<
          typename lno_nnz_view_t_::const_value_type*,
          typename KokkosKernels::Impl::GetUnifiedLayout<lno_nnz_view_t_>::array_layout,
          typename lno_nnz_view_t_::device_type,
          Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess> > Entries_Internal;


    RowMap_Internal rowmap_i = rowmap;
    Entries_Internal entries_i = entries;

    KokkosSparse::Impl::SPTRSV_SYMBOLIC<const_handle_type, RowMap_Internal, Entries_Internal>::sptrsv_symbolic (&tmp_handle, rowmap_i, entries_i);

  } // sptrsv_symbolic


  template <typename KernelHandle,
            typename lno_row_view_t_,
            typename lno_nnz_view_t_,
            typename scalar_nnz_view_t_,
            class BType,
            class XType>
  void sptrsv_solve(
      KernelHandle *handle, 
      lno_row_view_t_ rowmap,
      lno_nnz_view_t_ entries,
      scalar_nnz_view_t_ values,
      BType b,
      XType x)
  {
    typedef typename KernelHandle::size_type size_type;
    typedef typename KernelHandle::nnz_lno_t ordinal_type;
    typedef typename KernelHandle::nnz_scalar_t scalar_type;
    
    static_assert(KOKKOSKERNELS_SPTRSV_SAME_TYPE(typename lno_row_view_t_::non_const_value_type, size_type),
        "sptrsv_solve: A size_type must match KernelHandle size_type (const doesn't matter)");
    static_assert(KOKKOSKERNELS_SPTRSV_SAME_TYPE(typename lno_nnz_view_t_::non_const_value_type, ordinal_type),
        "sptrsv_solve: A entry type must match KernelHandle entry type (aka nnz_lno_t, and const doesn't matter)");
    static_assert(KOKKOSKERNELS_SPTRSV_SAME_TYPE(typename scalar_nnz_view_t_::value_type, scalar_type),
        "sptrsv_solve: A scalar type must match KernelHandle entry type (aka nnz_lno_t, and const doesn't matter)");

    static_assert (Kokkos::Impl::is_view<BType>::value,
        "sptrsv: b is not a Kokkos::View.");
    static_assert (Kokkos::Impl::is_view<XType>::value,
        "sptrsv: x is not a Kokkos::View.");
    static_assert ((int) BType::rank == (int) XType::rank,
        "sptrsv: The ranks of b and x do not match.");
    static_assert (BType::rank == 1,
        "sptrsv: b and x must both either have rank 1.");
    static_assert (std::is_same<typename XType::value_type,
        typename XType::non_const_value_type>::value,
        "sptrsv: The output x must be nonconst.");
    static_assert (std::is_same<typename BType::device_type, typename XType::device_type>::value,
        "sptrsv: Views BType and XType have different device_types.");
    static_assert (std::is_same<typename BType::device_type::execution_space, typename KernelHandle::SPTRSVHandleType::execution_space>::value,
        "sptrsv: KernelHandle and Views have different execution spaces.");
    static_assert (std::is_same<typename lno_row_view_t_::device_type, typename lno_nnz_view_t_::device_type>::value,
        "sptrsv: rowmap and entries have different device types.");
    static_assert (std::is_same<typename lno_row_view_t_::device_type, typename scalar_nnz_view_t_::device_type>::value,
        "sptrsv: rowmap and values have different device types.");


    typedef typename KernelHandle::const_size_type c_size_t;
    typedef typename KernelHandle::const_nnz_lno_t c_lno_t;
    typedef typename KernelHandle::const_nnz_scalar_t c_scalar_t;

    typedef typename KernelHandle::HandleExecSpace c_exec_t;
    typedef typename KernelHandle::HandleTempMemorySpace c_temp_t;
    typedef typename KernelHandle::HandlePersistentMemorySpace c_persist_t;

    typedef typename  KokkosKernels::Experimental::KokkosKernelsHandle<c_size_t, c_lno_t, c_scalar_t, c_exec_t, c_temp_t, c_persist_t> const_handle_type;
    const_handle_type tmp_handle (*handle);

    typedef Kokkos::View<
          typename lno_row_view_t_::const_value_type*,
          typename KokkosKernels::Impl::GetUnifiedLayout<lno_row_view_t_>::array_layout,
          typename lno_row_view_t_::device_type,
          Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess> > RowMap_Internal;

    typedef Kokkos::View<
          typename lno_nnz_view_t_::const_value_type*,
          typename KokkosKernels::Impl::GetUnifiedLayout<lno_nnz_view_t_>::array_layout,
          typename lno_nnz_view_t_::device_type,
          Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess> > Entries_Internal;

    typedef Kokkos::View<
          typename scalar_nnz_view_t_::const_value_type*,
          typename KokkosKernels::Impl::GetUnifiedLayout<scalar_nnz_view_t_>::array_layout,
          typename scalar_nnz_view_t_::device_type,
          Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess> > Values_Internal;


    typedef Kokkos::View<
          typename BType::const_value_type*,
          typename KokkosKernels::Impl::GetUnifiedLayout<BType>::array_layout,
          typename BType::device_type,
          Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess> > BType_Internal;

    typedef Kokkos::View<
          typename XType::non_const_value_type*,
          typename KokkosKernels::Impl::GetUnifiedLayout<XType>::array_layout,
          typename XType::device_type,
          Kokkos::MemoryTraits<Kokkos::Unmanaged> > XType_Internal;


    RowMap_Internal rowmap_i = rowmap;
    Entries_Internal entries_i = entries;
    Values_Internal values_i = values;

    BType_Internal b_i = b;
    XType_Internal x_i = x;

    //printf( " calling sptrsv_solve from KokkosSparse_sptrsv.hpp\n" );
    KokkosSparse::Impl::SPTRSV_SOLVE<const_handle_type, RowMap_Internal, Entries_Internal, Values_Internal, BType_Internal, XType_Internal>::sptrsv_solve (&tmp_handle, rowmap_i, entries_i, values_i, b_i, x_i);

  } // sptrsv_solve


#ifdef KOKKOSKERNELS_ENABLE_TPL_SUPERLU
  // ---------------------------------------------------------------------
  template <typename KernelHandle,
            typename scalar_type,
            typename host_graph_t,
            typename graph_t>
  void sptrsv_symbolic(
      KernelHandle *handleL,
      KernelHandle *handleU,
      bool merge,
      SuperMatrix &L,
      SuperMatrix &U,
      int *etree)
  {
    // ===================================================================
    // read CrsGraph from SuperLU factor
    std::cout << " > Read SuperLU factor into KokkosSparse::CrsMatrix (invert diagonal and copy to device)" << std::endl;
    if (merge) {
      std::cout << " > Merge supernodes" << std::endl;
    }
    Kokkos::Timer timer;
    bool cusparse = false; // pad diagonal blocks with zeros
    host_graph_t graphL_host;
    host_graph_t graphU_host;
    graph_t graphL;
    graph_t graphU;

    timer.reset();
    int nrows = L.nrow;
    if (merge) {
      graphL_host = read_superlu_graphL<host_graph_t> (cusparse, merge, &L);
    } else {
      graphL = read_superlu_graphL<graph_t> (cusparse, merge, &L);
    }
    if (merge) {
      graphU_host = read_superlu_graphU<host_graph_t> (&L, &U); 
    } else {
      graphU = read_superlu_graphU<graph_t> (&L, &U); 
    }    

    std::cout << "   Conversion Time for U: " << timer.seconds() << std::endl << std::endl;
    //print_factor_superlu<scalar_t> (nrows, &L, &U, perm_r, perm_c);

    // ===================================================================
    // setup supnodal info
    SCformat *Lstore = (SCformat*)(L.Store);
    int nsuper = 1 + Lstore->nsuper;
    int *supercols = Lstore->sup_to_col;

    if (merge) {
      // =================================================================
      // merge supernodes
      timer.reset ();
      int nsuper_merged = nsuper;
      // > make a copy of etree
      int *etree_merged = new int[nsuper];
      for (int i = 0; i < nsuper; i++) {
        etree_merged[i] = etree[i];
      }
      // > make a copy of supercols
      int *supercols_merged = new int[1+nsuper];
      for (int i = 0; i <= nsuper; i++) {
        supercols_merged[i] = supercols[i];
      }
      check_supernode_sizes("Original structure", nrows, nsuper, supercols_merged, graphL_host);
      merge_supernodal_graph<host_graph_t> (nrows, &nsuper_merged, supercols_merged,
                                            graphL_host, graphU_host, etree_merged);

      // =================================================================
      // generate merged graph for L-solve
      int nnzL_merged;
      int nnzL = graphL_host.row_map (nrows);
      graphL = generate_merged_supernodal_graph<host_graph_t, graph_t> (nrows, nsuper, supercols,
                                                                        nsuper_merged, supercols_merged,
                                                                        graphL_host, &nnzL_merged);
      check_supernode_sizes("After Merge", nrows, nsuper_merged, supercols_merged, graphL);
      std::cout << " for L factor:" << std::endl;
      std::cout << "   Merge Supernodes Time: " << timer.seconds() << std::endl;
      std::cout << "   Number of nonzeros   : " << nnzL << " -> " << nnzL_merged
                << " : " << double(nnzL_merged) / double(nnzL) << "x" << std::endl;

      // =================================================================
      // generate merged graph for L-solve
      int nnzU_merged;
      int nnzU = graphU_host.row_map (nrows);
      graphU = generate_merged_supernodal_graph<host_graph_t, graph_t> (nrows, nsuper, supercols,
                                                                        nsuper_merged, supercols_merged,
                                                                        graphU_host, &nnzU_merged);
      check_supernode_sizes("After Merge", nrows, nsuper_merged, supercols_merged, graphU);
      std::cout << " for U factor:" << std::endl;
      std::cout << "   Merge Supernodes Time: " << timer.seconds() << std::endl;
      std::cout << "   Number of nonzeros   : " << nnzU << " -> " << nnzU_merged
                << " : " << double(nnzU_merged) / double(nnzU) << "x" << std::endl;

      // replace the supernodal info with the merged ones
      nsuper = nsuper_merged;
      supercols = supercols_merged;
      etree = etree_merged;
    }

    // ===================================================================
    // save the supernodal info in the handles for L/U solves
    handleL->set_supernodes (nsuper, supercols, etree);
    handleU->set_supernodes (nsuper, supercols, etree);

    if (handleL->get_sptrsv_handle ()->get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_DAG) {
      // generate supernodal graphs for DAG scheduling
      auto supL = generate_supernodal_graph<host_graph_t, graph_t> (merge, nrows, graphL, nsuper, supercols);
      auto supU = generate_supernodal_graph<host_graph_t, graph_t> (merge, nrows, graphU, nsuper, supercols);

      int **dagL = generate_supernodal_dag<host_graph_t> (nsuper, supL, supU);
      int **dagU = generate_supernodal_dag<host_graph_t> (nsuper, supU, supL);
      handleL->set_supernodal_dag (dagL);
      handleU->set_supernodal_dag (dagU);
    }

    // ===================================================================
    // do symbolic for L solve on the host
    auto row_mapL = graphL.row_map;
    auto entriesL = graphL.entries;
    timer.reset();
    std::cout << std::endl;
    sptrsv_symbolic (handleL, row_mapL, entriesL);
    std::cout << " > Lower-TRI: " << std::endl;
    std::cout << "   Symbolic Time: " << timer.seconds() << std::endl;

    // ===================================================================
    // do symbolic for U solve on the host
    auto row_mapU = graphU.row_map;
    auto entriesU = graphU.entries;
    timer.reset ();
    sptrsv_symbolic (handleU, row_mapU, entriesU);
    std::cout << " > Upper-TRI: " << std::endl;
    std::cout << "   Symbolic Time: " << timer.seconds() << std::endl;

    // ===================================================================
    // save options
    handleL->get_sptrsv_handle ()->set_merge_supernodes (merge);
    handleU->get_sptrsv_handle ()->set_merge_supernodes (merge);

    // ===================================================================
    // save graphs
    handleL->get_sptrsv_handle ()->set_graph (graphL);
    handleL->get_sptrsv_handle ()->set_graph_host (graphL_host);

    handleU->get_sptrsv_handle ()->set_graph (graphU);
    handleU->get_sptrsv_handle ()->set_graph_host (graphU_host);
  }


  // ---------------------------------------------------------------------
  template <typename KernelHandle,
            typename host_crsmat_t,
            typename crsmat_t>
  void sptrsv_compute(
      KernelHandle *handleL,
      KernelHandle *handleU,
      bool invert_offdiag,
      SuperMatrix &L,
      SuperMatrix &U)
  {
    typedef typename host_crsmat_t::StaticCrsGraphType host_graph_t;
    typedef typename      crsmat_t::StaticCrsGraphType      graph_t;

    // load options
    bool merge = handleL->get_sptrsv_handle ()->get_merge_supernodes ();

    // load graphs
    auto graphL = handleL->get_sptrsv_handle ()->get_graph ();
    auto graphL_host = handleL->get_sptrsv_handle ()->get_graph_host ();

    auto graphU = handleU->get_sptrsv_handle ()->get_graph ();
    auto graphU_host = handleU->get_sptrsv_handle ()->get_graph_host ();

    int nrows = L.nrow;
    int nsuper = handleL->get_sptrsv_handle ()->get_num_supernodes ();
    const int* supercols = handleL->get_sptrsv_handle ()->get_supercols_host ();
    crsmat_t superluL, superluU;
    host_crsmat_t superluL_host, superluU_host;
    if (merge) {
      // NOTE: we first load into CRS, and then merge (should be combined)

      // ==============================================
      // read in the numerical L-values into merged crs
      bool cusparse = false;
      bool invert_diag = false; // invert after merge
      superluL_host = read_superlu_valuesL<host_crsmat_t> (cusparse, merge, invert_diag, invert_offdiag, &L, graphL_host);
      invert_diag = true;       // TODO: diagonals are always inverted
      superluL = read_merged_supernodes<host_crsmat_t, graph_t, crsmat_t> (nrows, nsuper, supercols,
                                                                           true, invert_diag, invert_offdiag,
                                                                           superluL_host, graphL);

      // ==============================================
      // read in the numerical U-values into merged crs
      invert_diag = false;     // invert after merge
      superluU_host = read_superlu_valuesU<host_crsmat_t, host_graph_t> (invert_diag, &L, &U, graphU_host);
      invert_diag = true;      // TODO: diagonals are always inverted
      bool invert_offdiagU = false;  // TODO: offdiagonal iversion are not supported for U-solve
      superluU = read_merged_supernodes<host_crsmat_t, graph_t, crsmat_t> (nrows, nsuper, supercols,
                                                                           false, invert_diag, invert_offdiagU,
                                                                           superluU_host, graphU);
    } else {
      bool cusparse = false;
      bool invert_diag = true; // only, invert diag is supported for now
      superluL = read_superlu_valuesL<crsmat_t> (cusparse, merge, invert_diag, invert_offdiag, &L, graphL);
      superluU = read_superlu_valuesU<crsmat_t, graph_t> (invert_diag, &L, &U, graphU);
    }

    // ==============================================
    // save crsmat
    handleL->get_sptrsv_handle ()->set_invert_offdiagonal(invert_offdiag);
    handleU->get_sptrsv_handle ()->set_invert_offdiagonal(invert_offdiag);
    handleL->get_sptrsv_handle ()->set_crsmat (superluL);
    handleU->get_sptrsv_handle ()->set_crsmat (superluU);
  } // sptrsv_compute
#endif //KOKKOSKERNELS_ENABLE_TPL_SUPERLU


#ifdef KOKKOSKERNELS_ENABLE_TPL_CHOLMOD
  // ---------------------------------------------------------------------
  template <typename KernelHandle,
            typename scalar_type,
            typename host_graph_t,
            typename graph_t>
  void sptrsv_symbolic(
      KernelHandle *handleL,
      KernelHandle *handleU,
      cholmod_factor *L,
      cholmod_common *cm,
      int *etree)
  {
    Kokkos::Timer timer;

    // ==============================================
    // extract CrsGraph from Cholmod
    bool cusparse = false; // pad diagonal blocks with zeros
    auto graph = read_cholmod_graphL<graph_t>(cusparse, L, cm);
    auto row_map = graph.row_map;
    auto entries = graph.entries;

    // ==============================================
    // extract etree from Cholmod
    //int *etree;
    //compute_etree_cholmod(L, cm, &etree);

    // ==============================================
    // setup supnodal info 
    int nsuper = (int)(L->nsuper);
    int *supercols = (int*)(L->super);
    handleL->set_supernodes (nsuper, supercols, etree);
    handleU->set_supernodes (nsuper, supercols, etree);

    // ==============================================
    // symbolic for L-solve on the host
    timer.reset();
    sptrsv_symbolic (handleL, row_map, entries);
    std::cout << " > Lower-TRI: " << std::endl;
    std::cout << "   Symbolic Time: " << timer.seconds() << std::endl;

    // ==============================================
    // symbolic for L^T-solve on the host
    timer.reset ();
    std::cout << " > Upper-TRI: " << std::endl;
    sptrsv_symbolic (handleU, row_map, entries);
    std::cout << "   Symbolic Time: " << timer.seconds() << std::endl;

    // ==============================================
    // save graphs
    handleL->get_sptrsv_handle ()->set_graph (graph);
    handleU->get_sptrsv_handle ()->set_graph (graph);
  }


  // ---------------------------------------------------------------------
  template <typename KernelHandle,
            typename host_crsmat_t,
            typename crsmat_t>
  void sptrsv_compute(
      KernelHandle *handleL,
      KernelHandle *handleU,
      cholmod_factor *L,
      cholmod_common *cm)
  {
    // ==============================================
    // load crsGraph
    typedef typename crsmat_t::StaticCrsGraphType graph_t;
    auto graph = handleL->get_sptrsv_handle ()->get_graph ();

    // ==============================================
    // read numerical values of L from Cholmod
    bool invert_diag = true;
    bool cusparse = false; // pad diagonal blocks with zeros
    auto cholmodL = read_cholmod_factor<crsmat_t, host_crsmat_t, graph_t> (cusparse, invert_diag, L, cm, graph);

    // ==============================================
    // save crsmat
    bool invert_offdiag = false;
    handleL->get_sptrsv_handle ()->set_invert_offdiagonal(invert_offdiag);
    handleU->get_sptrsv_handle ()->set_invert_offdiagonal(invert_offdiag);
    handleL->get_sptrsv_handle ()->set_crsmat (cholmodL);
    handleU->get_sptrsv_handle ()->set_crsmat (cholmodL);
  }
#endif // KOKKOSKERNELS_ENABLE_TPL_CHOLMOD

#if defined(KOKKOSKERNELS_ENABLE_TPL_CHOLMOD) | defined(KOKKOSKERNELS_ENABLE_TPL_SUPERLU)
  // ---------------------------------------------------------------------
  template <typename KernelHandle,
            class XType>
  void sptrsv_solve(
      KernelHandle *handle, 
      XType x)
  {
    auto crsmat = handle->get_sptrsv_handle ()->get_crsmat ();
    auto values  = crsmat.values;
    auto graph   = crsmat.graph;
    auto row_map = graph.row_map;
    auto entries = graph.entries;

    // the fifth argument (i.e., first x) is not used
    sptrsv_solve(handle, row_map, entries, values, x, x);
  }
#endif

} // namespace Experimental
} // namespace KokkosSparse

#undef KOKKOSKERNELS_SPTRSV_SAME_TYPE

#endif // KOKKOSSPARSE_SPTRSV_HPP_

