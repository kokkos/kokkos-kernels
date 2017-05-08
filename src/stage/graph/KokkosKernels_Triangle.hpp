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
#ifndef _KOKKOS_TRIANGLE_HPP
#define _KOKKOS_TRIANGLE_HPP
#include "KokkosKernels_IOUtils.hpp"
#include "KokkosKernels_Handle.hpp"
#include "KokkosKernels_SPGEMM_impl.hpp"
namespace KokkosKernels{

namespace Experimental{

namespace Graph{
template <typename KernelHandle,
typename alno_row_view_t_,
typename alno_nnz_view_t_,
typename blno_row_view_t_,
typename blno_nnz_view_t_,
typename clno_row_view_t_>
void triangle_count(
    KernelHandle *handle,
    typename KernelHandle::nnz_lno_t m,
    typename KernelHandle::nnz_lno_t n,
    typename KernelHandle::nnz_lno_t k,
    alno_row_view_t_ row_mapA,
    alno_nnz_view_t_ entriesA,
    bool transposeA,
    blno_row_view_t_ row_mapB,
    blno_nnz_view_t_ entriesB,
    bool transposeB,
    clno_row_view_t_ row_mapC){

  typedef typename KernelHandle::SPGEMMHandleType spgemmHandleType;
  spgemmHandleType *sh = handle->get_spgemm_handle();
  switch (sh->get_algorithm_type()){
  case SPGEMM_KK_TRIANGLE_LL:
  {
    KokkosKernels::Experimental::Graph::Impl::KokkosSPGEMM
    <KernelHandle,
    alno_row_view_t_, alno_nnz_view_t_, typename KernelHandle::in_scalar_nnz_view_t,
    blno_row_view_t_, blno_nnz_view_t_, typename KernelHandle::in_scalar_nnz_view_t>
    kspgemm (handle,m,n,k,row_mapA, entriesA, transposeA, row_mapB, entriesB, transposeB);
    kspgemm.KokkosSPGEMM_symbolic_triangle(row_mapC);
  }
  break;

  case SPGEMM_KK_TRIANGLE_AI:
  case SPGEMM_KK_TRIANGLE_IA:
  case SPGEMM_KK_TRIANGLE_IA_UNION:
  default:
  {
    KokkosKernels::Experimental::Graph::Impl::KokkosSPGEMM
    <KernelHandle,
    alno_row_view_t_, alno_nnz_view_t_, typename KernelHandle::in_scalar_nnz_view_t,
    blno_row_view_t_, blno_nnz_view_t_, typename KernelHandle::in_scalar_nnz_view_t>
    kspgemm (handle,m,n,k,row_mapA, entriesA, transposeA, row_mapB, entriesB, transposeB);
    kspgemm.KokkosSPGEMM_symbolic_triangle(row_mapC);
  }
  break;


  }
  sh->set_call_symbolic();

}


template <typename KernelHandle,
typename alno_row_view_t_,
typename alno_nnz_view_t_,
typename blno_row_view_t_,
typename blno_nnz_view_t_,
typename clno_row_view_t_,
typename clno_nnz_view_t_>
void triangle_enumerate(
    KernelHandle *handle,
    typename KernelHandle::nnz_lno_t m,
    typename KernelHandle::nnz_lno_t n,
    typename KernelHandle::nnz_lno_t k,
    alno_row_view_t_ row_mapA,
    alno_nnz_view_t_ entriesA,

    bool transposeA,
    blno_row_view_t_ row_mapB,
    blno_nnz_view_t_ entriesB,
    bool transposeB,
    clno_row_view_t_ row_mapC,
    clno_nnz_view_t_ &entriesC1,
    clno_nnz_view_t_ &entriesC2 = NULL
){


  typedef typename KernelHandle::SPGEMMHandleType spgemmHandleType;
  spgemmHandleType *sh = handle->get_spgemm_handle();
  if (!sh->is_symbolic_called()){
    triangle_count<KernelHandle,
    alno_row_view_t_, alno_nnz_view_t_,
    blno_row_view_t_, blno_nnz_view_t_,
    clno_row_view_t_>(
        handle, m, n, k,
        row_mapA, entriesA, transposeA,
        row_mapB, entriesB, transposeB,
        row_mapC
    );

    typename clno_row_view_t_::value_type c_nnz_size = handle->get_spgemm_handle()->get_c_nnz();
    if (c_nnz_size){
      entriesC1 = clno_nnz_view_t_ (Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size);
      //entriesC2 = clno_nnz_view_t_ (Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size);
    }
  }


  switch (sh->get_algorithm_type()){
  default:

  case SPGEMM_KK_TRIANGLE_AI:
  case SPGEMM_KK_TRIANGLE_IA:
  case SPGEMM_KK_TRIANGLE_IA_UNION:
  {
    KokkosKernels::Experimental::Graph::Impl::KokkosSPGEMM
    <KernelHandle,
    alno_row_view_t_, alno_nnz_view_t_, typename KernelHandle::in_scalar_nnz_view_t,
    blno_row_view_t_, blno_nnz_view_t_,  typename KernelHandle::in_scalar_nnz_view_t>
    kspgemm (handle,m,n,k,row_mapA, entriesA, transposeA, row_mapB, entriesB, transposeB);
    kspgemm.KokkosSPGEMM_numeric_triangle(row_mapC, entriesC1, entriesC2);
  }
  break;
  }
}

template <typename KernelHandle,
typename alno_row_view_t_,
typename alno_nnz_view_t_,
typename blno_row_view_t_,
typename blno_nnz_view_t_,
typename visit_struct_t>
void triangle_generic(
    KernelHandle *handle,
    typename KernelHandle::nnz_lno_t m,
    typename KernelHandle::nnz_lno_t n,
    typename KernelHandle::nnz_lno_t k,
    alno_row_view_t_ row_mapA,
    alno_nnz_view_t_ entriesA,
    bool transposeA,
    blno_row_view_t_ row_mapB,
    blno_nnz_view_t_ entriesB,
    bool transposeB,
    visit_struct_t visit_struct){

  typedef typename KernelHandle::SPGEMMHandleType spgemmHandleType;
  spgemmHandleType *sh = handle->get_spgemm_handle();
  switch (sh->get_algorithm_type()){
  //case SPGEMM_KK_TRIANGLE_LL:
  case SPGEMM_KK_TRIANGLE_AI:
  case SPGEMM_KK_TRIANGLE_IA:
  case SPGEMM_KK_TRIANGLE_IA_UNION:
  default:
  {
    KokkosKernels::Experimental::Graph::Impl::KokkosSPGEMM
    <KernelHandle,
    alno_row_view_t_, alno_nnz_view_t_, typename KernelHandle::in_scalar_nnz_view_t,
    blno_row_view_t_, blno_nnz_view_t_, typename KernelHandle::in_scalar_nnz_view_t>
    kspgemm (handle,m,n,k,row_mapA, entriesA, transposeA, row_mapB, entriesB, transposeB);
    kspgemm.KokkosSPGEMM_generic_triangle(visit_struct);
  }
  break;


  }
}

template <typename KernelHandle,
typename alno_row_view_t_,
typename alno_nnz_view_t_,
typename visit_struct_t>
void triangle_generic(
    KernelHandle *handle,
    typename KernelHandle::nnz_lno_t m,
    alno_row_view_t_ row_mapA,
    alno_nnz_view_t_ entriesA,
    visit_struct_t visit_struct){

  typedef typename KernelHandle::nnz_lno_t nnz_lno_t;
  typedef typename KernelHandle::size_type size_type;

  typedef typename KernelHandle::SPGEMMHandleType spgemmHandleType;
  typedef typename KernelHandle::nnz_lno_persistent_work_view_t nnz_lno_persistent_work_view_t;
  typedef typename KernelHandle::row_lno_persistent_work_view_t row_lno_persistent_work_view_t;

  typedef typename KernelHandle::HandleExecSpace ExecutionSpace;


  spgemmHandleType *sh = handle->get_spgemm_handle();
  Kokkos::Impl::Timer timer1;

  //////SORT BASE ON THE SIZE OF ROWS/////
  bool sort_lower_triangle = sh->get_sort_lower_triangular();
  if (sort_lower_triangle){

    if(sh->get_lower_triangular_permutation().data() == NULL){
      nnz_lno_persistent_work_view_t new_indices(Kokkos::ViewAllocateWithoutInitializing("new_indices"), m);
      bool sort_decreasing_order = true;
      ////If true we place the largest row to top, so that largest row size will be minimized in lower triangle.
      if (sh->get_algorithm_type() == SPGEMM_KK_TRIANGLE_AI || sh->get_algorithm_type() == SPGEMM_KK_TRIANGLE_LU){
        sort_decreasing_order = false;
        //if false we place the largest row to bottom, so that largest column is minimizedin lower triangle.
      }
#if 0
      if (0)
      {
      typename alno_row_view_t_::non_const_type new_row_map("new", m + 1);

      for (nnz_lno_t i = 0; i < m; ++i){
        nnz_lno_t used_size = 0;
        size_type rowBegin = row_mapA(i);
        nnz_lno_t left_work = row_mapA(i + 1) - rowBegin;
        if (left_work > 0){
          const nnz_lno_t n = entriesA(rowBegin);
          nnz_lno_t prev_nset_ind = n / 32;
          for (nnz_lno_t i = 1; i < left_work; ++i){

            const size_type adjind = i + rowBegin;
            const nnz_lno_t nn = entriesA(adjind);
            nnz_lno_t n_set_index = nn / 32;
            //n_set = n_set << (nn & compression_bit_mask);
            if (prev_nset_ind != n_set_index){
              ++used_size;
              prev_nset_ind = n_set_index;
            }
          }
          ++used_size;

        }
        /*
        if (used_size * left_work < m)
        new_row_map(i) = used_size * left_work;
        else
          new_row_map(i) = m;
        */
        new_row_map(i) = used_size * left_work;

        std::cout << "row:" << i << " original_size:" << left_work << " used_size:" << used_size << std::endl;
      }

      /*
      KokkosKernels::Experimental::Util::exclusive_parallel_prefix_sum
      <typename alno_row_view_t_::non_const_type, ExecutionSpace> (m + 1, new_row_map);

      KokkosKernels::Experimental::Util::kk_sort_by_row_size<size_type, nnz_lno_t, ExecutionSpace>(
          m, new_row_map.data(), new_indices.data(), sort_decreasing_order);
      */
      std::vector<struct KokkosKernels::Experimental::Util::Edge<nnz_lno_t, nnz_lno_t> > to_sort (m);

      for (nnz_lno_t i = 0; i < m; ++i){
        to_sort[i].src = new_row_map(i);
        to_sort[i].dst = i;
      }
      std::sort (to_sort.begin(), to_sort.begin() + m);
      for (nnz_lno_t i = 0; i < m; ++i){
        new_indices[to_sort[i].dst] = m - i;
      }
      }
      else
#endif
      {
      KokkosKernels::Experimental::Util::kk_sort_by_row_size<size_type, nnz_lno_t, ExecutionSpace>(
          m, row_mapA.data(), new_indices.data(), sort_decreasing_order);
      }
      sh->set_lower_triangular_permutation(new_indices);
    }
  }
  if (handle->get_verbose()){
    std::cout << "Preprocess Sorting Time:" << timer1.seconds() << std::endl;
  }
  //////SORT BASE ON THE SIZE OF ROWS/////

  /////////CREATE LOWER TRIANGLE///////
  bool create_lower_triangular = sh->get_create_lower_triangular();
  row_lno_persistent_work_view_t lower_triangular_matrix_rowmap;
  nnz_lno_persistent_work_view_t lower_triangular_matrix_entries;
  timer1.reset();
  if (create_lower_triangular ||
      sh->get_algorithm_type() == SPGEMM_KK_TRIANGLE_LL ||
      sh->get_algorithm_type() == SPGEMM_KK_TRIANGLE_LU){
    sh->get_lower_triangular_matrix(lower_triangular_matrix_rowmap, lower_triangular_matrix_entries);
    if( lower_triangular_matrix_rowmap.data() == NULL ||
        lower_triangular_matrix_entries.data() == NULL){

      alno_nnz_view_t_ null_values;
      nnz_lno_persistent_work_view_t new_indices = sh->get_lower_triangular_permutation();

      KokkosKernels::Experimental::Util::kk_get_lower_triangle
      <alno_row_view_t_, alno_nnz_view_t_, alno_nnz_view_t_,
      row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, alno_nnz_view_t_,
      nnz_lno_persistent_work_view_t, ExecutionSpace> (
          m,
          row_mapA, entriesA, null_values,
          lower_triangular_matrix_rowmap, lower_triangular_matrix_entries, null_values,
          new_indices, handle->is_dynamic_scheduling()
      );
      sh->set_lower_triangular_matrix(lower_triangular_matrix_rowmap, lower_triangular_matrix_entries);
    }
  }
  if (handle->get_verbose()){
    std::cout << "Preprocess Create Lower Triangular Time:" << timer1.seconds() << std::endl;
  }
  timer1.reset();

  row_lno_persistent_work_view_t upper_triangular_matrix_rowmap;
  nnz_lno_persistent_work_view_t upper_triangular_matrix_entries;
  if (sh->get_algorithm_type() == SPGEMM_KK_TRIANGLE_LU){
    sh->get_lower_triangular_matrix(lower_triangular_matrix_rowmap, lower_triangular_matrix_entries);
    alno_nnz_view_t_ null_values;
    nnz_lno_persistent_work_view_t new_indices = sh->get_lower_triangular_permutation();

    KokkosKernels::Experimental::Util::kk_get_lower_triangle
    <alno_row_view_t_, alno_nnz_view_t_, alno_nnz_view_t_,
    row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, alno_nnz_view_t_,
    nnz_lno_persistent_work_view_t, ExecutionSpace> (
        m,
        row_mapA, entriesA, null_values,
        upper_triangular_matrix_rowmap, upper_triangular_matrix_entries, null_values,
        new_indices, handle->is_dynamic_scheduling(), 4, false
    );

  }
  if (handle->get_verbose()){
    std::cout << "Preprocess Create Upper Triangular Time:" << timer1.seconds() << std::endl;
  }



  /////////CREATE LOWER TRIANGLE///////

  ////
  ///CREATE INCIDENCE MATRIX
  ///
  timer1.reset();
  row_lno_persistent_work_view_t incidence_transpose_rowmap;
  nnz_lno_persistent_work_view_t incidence_transpose_entries;

  row_lno_persistent_work_view_t incidence_rowmap;
  nnz_lno_persistent_work_view_t incidence_entries;
  switch (sh->get_algorithm_type()){

  //IF it is one of below, we perform I^T x (A) or (L).
  //so create the transpose of I.
  case SPGEMM_KK_TRIANGLE_IA_UNION:
  case SPGEMM_KK_TRIANGLE_IA:
  {
    //these are the algorithms that requires transpose of the incidence matrix.
    sh->get_lower_triangular_matrix(lower_triangular_matrix_rowmap, lower_triangular_matrix_entries);

    if( lower_triangular_matrix_rowmap.data() == NULL ||
        lower_triangular_matrix_entries.data() == NULL){
      std::cout << "Creating lower triangular A" << std::endl;

      alno_nnz_view_t_ null_values;
      nnz_lno_persistent_work_view_t new_indices = sh->get_lower_triangular_permutation();

      KokkosKernels::Experimental::Util::kk_get_lower_triangle
      <alno_row_view_t_, alno_nnz_view_t_, alno_nnz_view_t_,
      row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, alno_nnz_view_t_,
      nnz_lno_persistent_work_view_t, ExecutionSpace> (
          m,
          row_mapA, entriesA, null_values,
          lower_triangular_matrix_rowmap, lower_triangular_matrix_entries, null_values,
          new_indices, handle->is_dynamic_scheduling()
      );
    }
    KokkosKernels::Experimental::Util::kk_create_incidence_tranpose_matrix_from_lower_triangle
    <alno_row_view_t_, alno_nnz_view_t_,
    row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t,
    ExecutionSpace>
        (m,
        lower_triangular_matrix_rowmap, lower_triangular_matrix_entries,
        incidence_transpose_rowmap, incidence_transpose_entries,
        handle->is_dynamic_scheduling());
  }
  break;



  //IF it is one of below, we perform (A) or (L) x I
  //so create I.
  case SPGEMM_KK_TRIANGLE_AI:
  {
    //these are the algorithms that requires the incidence matrix.

    KokkosKernels::Experimental::Util::kk_create_incidence_matrix_from_original_matrix
            < alno_row_view_t_, alno_nnz_view_t_,
              row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t,
              ExecutionSpace>
            (m,
                row_mapA, entriesA,
            incidence_rowmap, incidence_entries, sh->get_lower_triangular_permutation(),
            handle->is_dynamic_scheduling());
  }
  break;
  case SPGEMM_KK_TRIANGLE_LU:
  case SPGEMM_KK_TRIANGLE_LL:
  default:
  {
    break;
  }
  }

  if (handle->get_verbose()){
    std::cout << "Preprocess Incidence Matrix Create Time:" << timer1.seconds() << std::endl;
  }
  ////
  ///CREATE INCIDENCE MATRIX END
  ///


  switch (sh->get_algorithm_type()){
  default:
  case SPGEMM_KK_TRIANGLE_LL:
  {

    KokkosKernels::Experimental::Graph::Impl::KokkosSPGEMM
    <KernelHandle,
    row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, typename KernelHandle::in_scalar_nnz_view_t,
    row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, typename KernelHandle::in_scalar_nnz_view_t>
    kspgemm (handle,m,m,m,
        lower_triangular_matrix_rowmap, lower_triangular_matrix_entries,
        false,
        lower_triangular_matrix_rowmap, lower_triangular_matrix_entries,
        false);
    kspgemm.KokkosSPGEMM_generic_triangle(visit_struct);
  }
  break;
  case SPGEMM_KK_TRIANGLE_LU:
  {

    KokkosKernels::Experimental::Graph::Impl::KokkosSPGEMM
    <KernelHandle,
    row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, typename KernelHandle::in_scalar_nnz_view_t,
    row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, typename KernelHandle::in_scalar_nnz_view_t>
    kspgemm (handle,m,m,m,
        lower_triangular_matrix_rowmap, lower_triangular_matrix_entries,
        false,
        upper_triangular_matrix_rowmap, upper_triangular_matrix_entries,
        false);
    kspgemm.KokkosSPGEMM_generic_triangle(visit_struct);
  }
  break;
  case SPGEMM_KK_TRIANGLE_AI:
  {
    if (create_lower_triangular){
      KokkosKernels::Experimental::Graph::Impl::KokkosSPGEMM
      <KernelHandle,
      row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, typename KernelHandle::in_scalar_nnz_view_t,
      row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, typename KernelHandle::in_scalar_nnz_view_t>
      kspgemm (handle,m,m,incidence_entries.dimension_0() / 2,
          lower_triangular_matrix_rowmap, lower_triangular_matrix_entries,
          false, //transpose ignore.
          incidence_rowmap, incidence_entries,
          false);
      kspgemm.KokkosSPGEMM_generic_triangle(visit_struct);
    }
    else {
      KokkosKernels::Experimental::Graph::Impl::KokkosSPGEMM
      <KernelHandle,
      alno_row_view_t_, alno_nnz_view_t_, typename KernelHandle::in_scalar_nnz_view_t,
      row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, typename KernelHandle::in_scalar_nnz_view_t>
      kspgemm (handle,m,m,incidence_entries.dimension_0() / 2,
          row_mapA, entriesA,
          false, //transpose ignore.
          incidence_rowmap, incidence_entries,
          false);
      kspgemm.KokkosSPGEMM_generic_triangle(visit_struct);
    }
  }

  break;
  case SPGEMM_KK_TRIANGLE_IA_UNION:
  case SPGEMM_KK_TRIANGLE_IA:
  {
    if (create_lower_triangular){
      KokkosKernels::Experimental::Graph::Impl::KokkosSPGEMM
      <KernelHandle,
      row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, typename KernelHandle::in_scalar_nnz_view_t,
      row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, typename KernelHandle::in_scalar_nnz_view_t>
      kspgemm (handle,
          incidence_transpose_rowmap.dimension_0() - 1,m,m,
          incidence_transpose_rowmap, incidence_transpose_entries,
          false, //transpose ignore.
          lower_triangular_matrix_rowmap, lower_triangular_matrix_entries,
          false);
      kspgemm.KokkosSPGEMM_generic_triangle(visit_struct);
    }
    else {
      KokkosKernels::Experimental::Graph::Impl::KokkosSPGEMM
      <KernelHandle,
      row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, typename KernelHandle::in_scalar_nnz_view_t,
      alno_row_view_t_, alno_nnz_view_t_, typename KernelHandle::in_scalar_nnz_view_t>
      kspgemm (handle,
          incidence_transpose_rowmap.dimension_0() - 1,m,m,
          incidence_transpose_rowmap, incidence_transpose_entries,
          false, //transpose ignore.
          row_mapA, entriesA,
          false);
      kspgemm.KokkosSPGEMM_generic_triangle(visit_struct);
    }
  }
  break;


  }

}

}
}
}
#endif
