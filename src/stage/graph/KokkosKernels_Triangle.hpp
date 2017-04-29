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

    case SPGEMM_KK_TRIANGLE_DEFAULT:
    case SPGEMM_KK_TRIANGLE_DENSE:
    case SPGEMM_KK_TRIANGLE_MEM:
    case SPGEMM_KK_TRIANGLE_IA_DEFAULT:
    case SPGEMM_KK_TRIANGLE_IA_DENSE:
    case SPGEMM_KK_TRIANGLE_IA_MEM:
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

    case SPGEMM_KK_TRIANGLE_DEFAULT:
    case SPGEMM_KK_TRIANGLE_DENSE:
    case SPGEMM_KK_TRIANGLE_MEM:
    case SPGEMM_KK_TRIANGLE_IA_DEFAULT:
    case SPGEMM_KK_TRIANGLE_IA_DENSE:
    case SPGEMM_KK_TRIANGLE_IA_MEM:
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
    case SPGEMM_KK_TRIANGLE_LL:
    case SPGEMM_KK_TRIANGLE_DEFAULT:
    case SPGEMM_KK_TRIANGLE_DENSE:
    case SPGEMM_KK_TRIANGLE_MEM:
    case SPGEMM_KK_TRIANGLE_IA_DEFAULT:
    case SPGEMM_KK_TRIANGLE_IA_DENSE:
    case SPGEMM_KK_TRIANGLE_IA_MEM:
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
    sh->set_call_symbolic();

  }

}
}
}
#endif
