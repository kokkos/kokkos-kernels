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

#ifndef KOKKOSSPARSE_SPTRSV_SUPERLU_HPP_
#define KOKKOSSPARSE_SPTRSV_SUPERLU_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_SUPERLU
#include "slu_ddefs.h"

#include "KokkosSparse_sptrsv_supernode.hpp"

namespace KokkosSparse {
namespace Experimental {


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* For symbolic analysis                                                                     */

/* ========================================================================================= */
template <typename graph_t>
graph_t read_superlu_graphL(bool cusparse, bool merge, SuperMatrix *L) {

  /* ---------------------------------------------------------------------- */
  /* get inputs */
  /* ---------------------------------------------------------------------- */
  int n = L->nrow;
  SCformat *Lstore = (SCformat*)(L->Store);

  int nsuper = 1 + Lstore->nsuper;     // # of supernodal columns
  int * mb = Lstore->rowind_colptr;
  int * nb = Lstore->sup_to_col;
  int * colptr = Lstore->nzval_colptr;
  int * rowind = Lstore->rowind;

  bool ptr_by_column = true;
  return read_supernodal_graphL<graph_t> (cusparse, merge,
                                          n, nsuper, ptr_by_column, mb, nb, colptr, rowind);
}


/* ========================================================================================= */
template <typename graph_t>
graph_t read_superlu_graphU(SuperMatrix *L,  SuperMatrix *U) {

  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type   cols_view_t;

  SCformat *Lstore = (SCformat*)(L->Store);
  NCformat *Ustore = (NCformat*)(U->Store);

  /* create a map from row id to supernode id */
  int n = L->nrow;
  int nsuper = 1 + Lstore->nsuper;     // # of supernodal columns
  int *nb = Lstore->sup_to_col;
  int *colptrU = Ustore->colptr;
  int *rowindU = Ustore->rowind;

  int *map = new int[n];
  int supid = 0;
  for (int k = 0; k < nsuper; k++)
  {
    int j1 = nb[k];
    int j2 = nb[k+1];
    for (int j = j1; j < j2; j++) {
        map[j] = supid;
    }
    supid ++;
  }

  /* count number of nonzeros in each row */
  row_map_view_t rowmap_view ("rowmap_view", n+1);
  typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view (rowmap_view);
  for (int i = 0; i < n; i++) {
    hr (i) = 0;
  }

  int *check = new int[nsuper];
  for (int k = 0; k < nsuper; k++) {
    check[k] = 0;
  }
  for (int k = nsuper-1; k >= 0; k--) {
    int j1 = nb[k];
    int nscol = nb[k+1] - j1;

    /* the diagonal block */
    for (int i = 0; i < nscol; i++) {
      hr (j1+i + 1) += nscol;
    }

    /* the off-diagonal blocks */
    for (int jcol = j1; jcol < j1 + nscol; jcol++) {
      for (int i = colptrU[jcol]; i < colptrU[jcol+1]; i++ ){
        int irow = rowindU[i];
        supid = map[irow];
        if (check[supid] == 0) {
          for (int ii = nb[supid]; ii < nb[supid+1]; ii++) {
            hr (ii + 1) += nscol;
          }
          check[supid] = 1;
        }
      }
    }
    // reset check
    for (int i = 0; i < nsuper; i++ ) {
      check[i] = 0;
    }
  }

  // convert to the offset for each row
  for (int i = 1; i <= n; i++) {
    hr (i) += hr (i-1);
  }

  /* Upper-triangular matrix */
  int nnzA = hr (n);
  cols_view_t    column_view ("colmap_view", nnzA);
  typename cols_view_t::HostMirror   hc = Kokkos::create_mirror_view (column_view);

  int *sup = new int[nsuper];
  for (int k = 0; k < nsuper; k++) {
    int j1 = nb[k];
    int nscol = nb[k+1] - j1;

    /* the diagonal "dense" block */
    for (int i = 0; i < nscol; i++) {
      for (int j = 0; j < i; j++) {
        hc(hr(j1 + i) + j) = j1 + j;
      }

      for (int j = i; j < nscol; j++) {
        hc(hr(j1 + i) + j) = j1 + j;
      }
      hr (j1 + i) += nscol;
    }

    /* the off-diagonal "sparse" blocks */
    // let me first find off-diagonal supernodal blocks..
    int nsup = 0;
    for (int jcol = j1; jcol < j1 + nscol; jcol++) {
      for (int i = colptrU[jcol]; i < colptrU[jcol+1]; i++ ){
        int irow = rowindU[i];
        if (check[map[irow]] == 0) {
          check[map[irow]] = 1;
          sup[nsup] = map[irow];
          nsup ++;
        }
      }
    }
    for (int jcol = j1; jcol < j1 + nscol; jcol++) {
      // move up all the row pointers for all the supernodal blocks
      for (int i = 0; i < nsup; i++) {
        for (int ii = nb[sup[i]]; ii < nb[sup[i]+1]; ii++) {
          hc(hr(ii)) = jcol;
          hr(ii) ++;
        }
      }
    }
    // reset check
    for (int i = 0; i < nsup; i++) {
      check[sup[i]] = 0;
    }
  }
  delete[] sup;
  delete[] check;

  // fix hr
  for (int i = n; i >= 1; i--) {
    hr(i) = hr(i-1);
  }
  hr(0) = 0;
  std::cout << "    * Matrix size = " << n << std::endl;
  std::cout << "    * Total nnz   = " << hr (n) << std::endl;
  std::cout << "    * nnz / n     = " << hr (n)/n << std::endl;

  // deepcopy
  Kokkos::deep_copy (rowmap_view, hr);
  Kokkos::deep_copy (column_view, hc);

  // create crs
  graph_t static_graph (column_view, rowmap_view);
  return static_graph;
}



/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* For numeric computation                                                                   */

/* ========================================================================================= */
template <typename crsmat_t, typename graph_t>
crsmat_t read_superlu_valuesL(bool cusparse, bool merge, bool invert_diag, bool invert_offdiag, SuperMatrix *L, graph_t &static_graph) {

  typedef typename crsmat_t::values_type::non_const_type values_view_t;
  typedef typename values_view_t::value_type scalar_t;

  /* ---------------------------------------------------------------------- */
  /* get inputs */
  /* ---------------------------------------------------------------------- */
  SCformat *Lstore = (SCformat*)(L->Store);
  scalar_t *Lx = (scalar_t*)(Lstore->nzval);

  int n = L->nrow;
  int nsuper = 1 + Lstore->nsuper;     // # of supernodal columns
  int * mb = Lstore->rowind_colptr;
  int * nb = Lstore->sup_to_col;
  int * colptr = Lstore->nzval_colptr;
  int * rowind = Lstore->rowind;

  bool unit_diag = true;
  bool ptr_by_column = true;
  return read_supernodal_valuesL<crsmat_t, graph_t, scalar_t> (cusparse, merge, invert_diag, invert_offdiag,
                                                               unit_diag, n, nsuper, ptr_by_column, mb, nb, colptr, rowind, Lx, static_graph);
}


/* ========================================================================================= */
template <typename crsmat_t, typename graph_t>
crsmat_t read_superlu_valuesU(bool invert_diag, SuperMatrix *L,  SuperMatrix *U, graph_t &static_graph) {

  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type   cols_view_t;
  typedef typename crsmat_t::values_type::non_const_type values_view_t;

  typedef typename values_view_t::value_type scalar_t;
  typedef Kokkos::Details::ArithTraits<scalar_t> STS;

  SCformat *Lstore = (SCformat*)(L->Store);
  scalar_t *Lx = (scalar_t*)(Lstore->nzval);

  NCformat *Ustore = (NCformat*)(U->Store);
  double *Uval = (double*)(Ustore->nzval);

  int n = L->nrow;
  int nsuper = 1 + Lstore->nsuper;     // # of supernodal columns
  int *nb = Lstore->sup_to_col;
  int *mb = Lstore->rowind_colptr;
  int *colptrL = Lstore->nzval_colptr;
  int *colptrU = Ustore->colptr;
  int *rowindU = Ustore->rowind;

  /* create a map from row id to supernode id */
  int *map = new int[n];
  int supid = 0;
  for (int k = 0; k < nsuper; k++)
  {
    int j1 = nb[k];
    int j2 = nb[k+1];
    for (int j = j1; j < j2; j++) {
        map[j] = supid;
    }
    supid ++;
  }

  auto rowmap_view = static_graph.row_map;
  typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view (rowmap_view);
  Kokkos::deep_copy (hr, rowmap_view);

  /* Upper-triangular matrix */
  int nnzA = hr (n);
  values_view_t  values_view ("values_view", nnzA);
  typename values_view_t::HostMirror hv = Kokkos::create_mirror_view (values_view);

  int *sup = new int[nsuper];
  int *check = new int[nsuper];
  for (int k = 0; k < nsuper; k++) {
    check[k] = 0;
  }
  for (int k = 0; k < nsuper; k++) {
    int j1 = nb[k];
    int nscol = nb[k+1] - j1;

    int i1 = mb[j1];
    int nsrow = mb[j1+1] - i1;

    /* the diagonal "dense" block */
    int psx = colptrL[j1];
    if (invert_diag) {
      LAPACKE_dtrtri(LAPACK_COL_MAJOR,
                     'U', 'N', nscol, &Lx[psx], nsrow);
    }
    for (int i = 0; i < nscol; i++) {
      for (int j = 0; j < i; j++) {
        hv(hr(j1 + i) + j) = STS::zero ();
      }

      for (int j = i; j < nscol; j++) {
        hv(hr(j1 + i) + j) = Lx[psx + i + j*nsrow];
      }
      hr (j1 + i) += nscol;
    }

    /* the off-diagonal "sparse" blocks */
    // let me first find off-diagonal supernodal blocks..
    int nsup = 0;
    for (int jcol = j1; jcol < j1 + nscol; jcol++) {
      for (int i = colptrU[jcol]; i < colptrU[jcol+1]; i++ ){
        int irow = rowindU[i];
        if (check[map[irow]] == 0) {
          check[map[irow]] = 1;
          sup[nsup] = map[irow];
          nsup ++;
        }
      }
    }
    for (int jcol = j1; jcol < j1 + nscol; jcol++) {
      // add nonzeros in jcol-th column
      for (int i = colptrU[jcol]; i < colptrU[jcol+1]; i++ ){
        int irow = rowindU[i];
        hv(hr(irow)) = Uval[i];
      }
      // move up all the row pointers for all the supernodal blocks
      for (int i = 0; i < nsup; i++) {
        for (int ii = nb[sup[i]]; ii < nb[sup[i]+1]; ii++) {
          hr(ii) ++;
        }
      }
    }
    // reset check
    for (int i = 0; i < nsup; i++) {
      check[sup[i]] = 0;
    }
  }
  delete[] sup;
  delete[] check;

  // fix hr
  for (int i = n; i >= 1; i--) {
    hr(i) = hr(i-1);
  }
  hr(0) = 0;
  std::cout << "    * Matrix size = " << n << std::endl;
  std::cout << "    * Total nnz   = " << hr (n) << std::endl;
  std::cout << "    * nnz / n     = " << hr (n)/n << std::endl;

  // deepcopy
  Kokkos::deep_copy (values_view, hv);
  // create crs
  crsmat_t crsmat("CrsMatrix", n, values_view, static_graph);
  return crsmat;
}

} // namespace Experimental
} // namespace KokkosSparse

#endif // KOKKOSKERNELS_ENABLE_TPL_SUPERLU
#endif // KOKKOSSPARSE_SPTRSV_SUPERLU_HPP_

