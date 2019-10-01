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

#include <cstdio>

#include <ctime>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <limits.h>
#include <cmath>
#include <unordered_map>
#include <algorithm>

#include <Kokkos_Core.hpp>
#include <matrix_market.hpp>

#include "KokkosKernels_IOUtils.hpp"
#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosSparse_sptrsv.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

#if defined( KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA ) && (!defined(KOKKOS_ENABLE_CUDA) || ( 8000 <= CUDA_VERSION ))

#ifdef KOKKOSKERNELS_ENABLE_TPL_SUPERLU
#include "cblas.h"
#include "lapacke.h"
#include "slu_ddefs.h"

#ifdef KOKKOSKERNELS_ENABLE_TPL_METIS
#include "metis.h"
#endif

#include "KokkosSparse_sptrsv_aux.hpp"


/* ========================================================================================= */
template <typename scalar_t>
void print_factor_superlu(int n, SuperMatrix *L, SuperMatrix *U, int *perm_r, int *perm_c) {
  typedef Kokkos::Details::ArithTraits<scalar_t> STS;

  SCformat *Lstore = (SCformat*)(L->Store);
  scalar_t   *Lx = (scalar_t*)(Lstore->nzval);

  int *nb = Lstore->sup_to_col;
  int *mb = Lstore->rowind_colptr;
  int *colptr = Lstore->nzval_colptr;
  int *rowind = Lstore->rowind;

  for (int k = 0; k <= Lstore->nsuper; k++)
  {
    int j1 = nb[k];
    int j2 = nb[k+1];
    int nscol  = j2 - j1;
    printf( "%d %d %d\n",k,j1,nscol );
  }

  /* permutation vectors */
  int *iperm_r = new int[n];
  int *iperm_c = new int[n];
  for (int k = 0; k < n; k++) {
    iperm_r[perm_r[k]] = k;
    iperm_c[perm_c[k]] = k;
  }
  //printf( "P=[\n" );
  //for (int k = 0; k < n; k++) {
  //  printf( "%d, %d %d, %d %d\n",k, perm_r[k],iperm_r[k], perm_c[k],iperm_c[k] );
  //}
  //printf( "];\n" );

#if 1
  /* Lower-triangular matrix */
  printf( " L = [\n ");
  for (int k = 0; k <= Lstore->nsuper; k++)
  {
    int j1 = nb[k];
    int j2 = nb[k+1];
    int nscol = j2 - j1;

    int i1 = mb[j1];
    int i2 = mb[j1+1];
    int nsrow = i2 - i1;
    int nsrow2 = nsrow - nscol;
    int ps2    = i1 + nscol;

    int psx = colptr[j1];

    /* the diagonal block */
    for (int i = 0; i < nscol; i++) {
      for (int j = 0; j < i; j++) {
        if (Lx[psx + i + j*nsrow] != STS::zero())
        printf( "%d %d %.16e\n",1+j1+i, 1+j1+j, Lx[psx + i + j*nsrow] );
      }
      printf( "%d %d 1.0\n",1+j1+i, 1+j1+i );
    }

    /* the off-diagonal blocks */
    for (int ii = 0; ii < nsrow2; ii++) {
      int i = rowind [ps2 + ii];
      for (int j = 0; j < nscol; j++) printf( "%d %d %.16e\n",1+i, 1+j1+j, Lx[psx+nscol + ii + j*nsrow] );
    }
  }
  printf( "];\n ");
#endif

#if 0
  /* Upper-triangular matrix */
  NCformat *Ustore = (NCformat*)(U->Store);
  double *Uval = (double*)(Ustore->nzval);
  printf( " U = [\n ");
  for (int k = Lstore->nsuper; k >= 0; k--) {
    int j1 = nb[k];
    int nscol = nb[k+1] - j1;

    int i1 = mb[j1];
    int nsrow = mb[j1+1] - i1;

    int psx = colptr[j1];

    /* the diagonal block */
    for (int i = 0; i < nscol; i++) {
      for (int j = i; j < nscol; j++) {
        printf( "%d %d %.16e\n",j1+i, j1+j, Lx[psx + i + j*nsrow] );
        //std::cout << j1+i+1 << " " << j1+j+1 << " " << Lx[psx + i + j*nsrow] << std::endl;
      }
    }

    /* the off-diagonal blocks */
    for (int jcol = j1; jcol < j1 + nscol; jcol++) {
      for (int i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++ ){
        int irow = U_SUB(i);
        printf( "%d %d %.16e\n", irow, jcol, Uval[i] );
        //std::cout << irow+1 << " " << jcol+1 << " " << Uval[i] << std::endl;
      }
    }
  }
  printf( "];\n" );
#endif
}


/* ========================================================================================= */
template <typename crsmat_t>
crsmat_t read_superlu_Lfactor(bool cusparse, bool merge, bool invert_diag, bool invert_offdiag, int n, SuperMatrix *L) {

  typedef typename crsmat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type   cols_view_t;
  typedef typename crsmat_t::values_type::non_const_type values_view_t;

  typedef typename values_view_t::value_type scalar_t;
  typedef Kokkos::Details::ArithTraits<scalar_t> STS;

  /* ---------------------------------------------------------------------- */
  /* get inputs */
  /* ---------------------------------------------------------------------- */
  SCformat *Lstore = (SCformat*)(L->Store);
  scalar_t *Lx = (scalar_t*)(Lstore->nzval);

  int nsuper = 1 + Lstore->nsuper;     // # of supernodal columns
  int * mb = Lstore->rowind_colptr;
  int * nb = Lstore->sup_to_col;
  int * colptr = Lstore->nzval_colptr;
  int * rowind = Lstore->rowind;

  int nnzA = colptr[n] - colptr[0];
  row_map_view_t rowmap_view ("rowmap_view", n+1);
  cols_view_t    column_view ("colmap_view", nnzA);
  values_view_t  values_view ("values_view", nnzA);

  typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view (rowmap_view);
  typename cols_view_t::HostMirror    hc = Kokkos::create_mirror_view (column_view);
  typename values_view_t::HostMirror  hv = Kokkos::create_mirror_view (values_view);

  // compute offset for each row
  int j = 0;
  int max_nnz_per_row = 0;
  hr(j) = 0;
  for (int s = 0 ; s < nsuper ; s++) {
    int j1 = nb[s];
    int j2 = nb[s+1];
    int nscol = j2 - j1;      // number of columns in the s-th supernode column

    int i1 = mb[j1];
    int i2 = mb[j1+1];
    int nsrow = i2 - i1;      // "total" number of rows in all the supernodes (diagonal+off-diagonal)

    for (int jj = 0; jj < nscol; jj++) {
      hr(j+1) = hr(j) + nsrow;
      j++;
    }
    if (nsrow > max_nnz_per_row) {
      max_nnz_per_row = nsrow;
    }
  }

  int *sorted_rowind = new int[max_nnz_per_row];
  // store L in csr
  for (int s = 0 ; s < nsuper ; s++) {
    int j1 = nb[s];
    int j2 = nb[s+1];
    int nscol = j2 - j1;      // number of columns in the s-th supernode column

    int i1 = mb[j1];
    int i2 = mb[j1+1];
    int nsrow  = i2 - i1;    // "total" number of rows in all the supernodes (diagonal+off-diagonal)
    int nsrow2 = nsrow - nscol;  // "total" number of rows in all the off-diagonal supernodes
    int ps2    = i1 + nscol;     // offset into rowind

    int psx = colptr[j1] ;        // offset into data,   Lx[s][s]

    /* diagonal block */
    // for each column (or row due to symmetry), the diagonal supernodal block is stored (in ascending order of row indexes) first
    // so that we can do TRSM on the diagonal block
    if (invert_diag) {
      LAPACKE_dtrtri(LAPACK_COL_MAJOR,
                     'L', 'U', nscol, &Lx[psx], nsrow);
      if (nsrow2 > 0 && invert_offdiag) {
        cblas_dtrmm (CblasColMajor,
              CblasRight, CblasLower, CblasNoTrans, CblasUnit,
              nsrow2, nscol,
              1.0, &Lx[psx], nsrow,
                   &Lx[psx+nscol], nsrow);
      }
    }
    for (int ii = 0; ii < nscol; ii++) {
      // lower-triangular part
      for (int jj = 0; jj < ii; jj++) {
        hc(hr(j1+jj)) = j1+ii;
        hv(hr(j1+jj)) = Lx[psx + (ii + jj*nsrow)];
        hr(j1+jj) ++;
      }
      // diagonal
      hc(hr(j1+ii)) = j1+ii;
      hv(hr(j1+ii)) = STS::one ();
      hr(j1+ii) ++;
      if (!cusparse) {
        // explicitly store zeros in upper-part
        for (int jj = ii+1; jj < nscol; jj++) {
          hc(hr(j1+jj)) = j1+ii;
          hv(hr(j1+jj)) = STS::zero ();
          hr(j1+jj) ++;
        }
      }
    }

    /* off-diagonal blocks */
    if (merge) {
      // sort rowind (to merge supernodes)
      for (int ii = 0; ii < nsrow2; ii++) {
        sorted_rowind[ii] = ii;
      }
      std::sort(&(sorted_rowind[0]), &(sorted_rowind[nsrow2]), sort_indices(&rowind[ps2]));
    }
    for (int kk = 0; kk < nsrow2; kk++) {
      int ii = (merge ? sorted_rowind[kk] : kk); // sorted rowind
      int i = rowind[ps2 + ii];
      for (int jj = 0; jj < nscol; jj++) {
        hc(hr(j1+jj)) = i;
        hv(hr(j1+jj)) = Lx[psx + (nscol+ii + jj*nsrow)];
        hr(j1+jj) ++;
      }
    }
  }
  delete [] sorted_rowind;

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
  Kokkos::deep_copy (values_view, hv);

  // create crs
  graph_t static_graph (column_view, rowmap_view);
  crsmat_t crsmat ("CrsMatrix", n, values_view, static_graph);
  return crsmat;
}


/* ========================================================================================= */
template <typename crsmat_t>
crsmat_t read_superlu_Ufactor(bool invert_diag, int n, SuperMatrix *L,  SuperMatrix *U) {

  typedef typename crsmat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type   cols_view_t;
  typedef typename crsmat_t::values_type::non_const_type values_view_t;

  typedef typename values_view_t::value_type scalar_t;
  typedef Kokkos::Details::ArithTraits<scalar_t> STS;

  SCformat *Lstore = (SCformat*)(L->Store);
  scalar_t *Lx = (scalar_t*)(Lstore->nzval);

  NCformat *Ustore = (NCformat*)(U->Store);
  double *Uval = (double*)(Ustore->nzval);

  /* create a map from row id to supernode id */
  int nsuper = 1 + Lstore->nsuper;     // # of supernodal columns
  int *nb = Lstore->sup_to_col;
  int *mb = Lstore->rowind_colptr;
  int *colptr = Lstore->nzval_colptr;

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
      for (int i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++ ){
        int irow = U_SUB(i);

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
  values_view_t  values_view ("values_view", nnzA);

  typename cols_view_t::HostMirror   hc = Kokkos::create_mirror_view (column_view);
  typename values_view_t::HostMirror hv = Kokkos::create_mirror_view (values_view);

  int *sup = new int[nsuper];
  for (int k = 0; k < nsuper; k++) {
    int j1 = nb[k];
    int nscol = nb[k+1] - j1;

    int i1 = mb[j1];
    int nsrow = mb[j1+1] - i1;

    /* the diagonal block */
    int psx = colptr[j1];
    if (invert_diag) {
      LAPACKE_dtrtri(LAPACK_COL_MAJOR,
                     'U', 'N', nscol, &Lx[psx], nsrow);
    }
    for (int i = 0; i < nscol; i++) {
      for (int j = 0; j < i; j++) {
        hc(hr(j1 + i) + j) = j1 + j;
        hv(hr(j1 + i) + j) = STS::zero ();
      }

      for (int j = i; j < nscol; j++) {
        hc(hr(j1 + i) + j) = j1 + j;
        hv(hr(j1 + i) + j) = Lx[psx + i + j*nsrow];
      }
      hr (j1 + i) += nscol;
    }

    /* the off-diagonal blocks */
    // let me first find off-diagonal supernodal blocks..
    int nsup = 0;
    for (int jcol = j1; jcol < j1 + nscol; jcol++) {
      for (int i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++ ){
        int irow = U_SUB(i);
        if (check[map[irow]] == 0) {
          check[map[irow]] = 1;

          sup[nsup] = map[irow];
          nsup ++;
        }
      }
    }
    for (int jcol = j1; jcol < j1 + nscol; jcol++) {
      // add nonzeros in jcol-th column
      for (int i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++ ){
        int irow = U_SUB(i);
        hv(hr(irow)) = Uval[i];
      }
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
  Kokkos::deep_copy (values_view, hv);

  // create crs
  graph_t static_graph (column_view, rowmap_view);
  crsmat_t crsmat("CrsMatrix", n, values_view, static_graph);
  return crsmat;
}


/* ========================================================================================= */
template<typename scalar_t>
void factor_superlu(bool metis, const int nrow, scalar_t *nzvals, int *rowptr, int *colind,
                    int panel_size, int relax_size, SuperMatrix &L, SuperMatrix &U,
                    int **perm_r, int **perm_c, int **parents) {

  // allocate permutation vectors for SuperLU
  *perm_c = new int[nrow];
  *perm_r = new int[nrow];

  #ifdef KOKKOSKERNELS_ENABLE_TPL_METIS
  if (metis) {
    idx_t n = nrow;
    idx_t nnz = rowptr[n];
      
    // remove diagonal elements (and casting to METIS idx_t)
    idx_t *metis_rowptr = new idx_t[n+1];
    idx_t *metis_colind = new idx_t[nnz];

    nnz = 0;
    metis_rowptr[0] = 0;
    for (int i = 0; i < n; i++) {
      for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
        if (colind[k] != i) {
          metis_colind[nnz] = colind[k];
          nnz ++;
        }
      }
      metis_rowptr[i+1] = nnz;
    }

    // call METIS
    idx_t *metis_perm = new idx_t[n];
    idx_t *metis_iperm = new idx_t[n];
    std::cout << "  + calling METIS_NodeND: (n=" << n << ", nnz=" << nnz << ") " << std::endl;
    if (METIS_OK != METIS_NodeND(&n, metis_rowptr, metis_colind, NULL, NULL, metis_perm, metis_iperm)) {
      std::cout << std::endl << "METIS_NodeND failed" << std::endl << std::endl;
    }

    // copy permutation to SuperLU
    for (idx_t i = 0; i < n; i++) {
      (*perm_r)[i] = metis_iperm[i];
      (*perm_c)[i] = metis_iperm[i];
    }

    delete [] metis_perm;
    delete [] metis_iperm;
    delete [] metis_rowptr;
    delete [] metis_colind;
  }
  #endif

  SuperMatrix A;
  NCformat *Astore;
  int      info;
  superlu_options_t options;
  SuperLUStat_t stat;

  set_default_options(&options);
  options.SymmetricMode = YES;
  #ifdef KOKKOSKERNELS_ENABLE_TPL_METIS
  if (metis) {
    options.ColPerm = MY_PERMC;
    options.RowPerm = MY_PERMR;
    //options.SymmetricMode = NO;
  }
  #endif

  int nnz = rowptr[nrow];
  dCreate_CompCol_Matrix(&A, nrow, nrow, nnz, nzvals, colind, rowptr, SLU_NC, SLU_D, SLU_GE);
  Astore = (NCformat*)(A.Store);

  /* Initialize the statistics variables. */
  StatInit(&stat);
  int w1 = (sp_ienv(1) > sp_ienv(2) ? sp_ienv(1) : sp_ienv(2));
  int w2 = (panel_size > relax_size ? panel_size : relax_size);
  if (w2 > w1) {
    SUPERLU_FREE(stat.panel_histo);
    stat.panel_histo = intCalloc(w2+1);
  }

  /* Call SuperLU to solve the problem. */
  int *etree = new int[A.ncol];
  if (options.ColPerm != MY_PERMC) {
    get_perm_c(options.ColPerm, &A, *perm_c);
  }
  SuperMatrix AC;
  sp_preorder(&options, &A, *perm_c, etree, &AC);

  GlobalLU_t Glu;
  int lwork = 0;
  printf( "  + calling SuperLU dgstrf with panel_size=%d, relax_size=%d..\n",panel_size,relax_size );
  printf( "   * Dimension %dx%d; # nonzeros %d\n", A.nrow, A.ncol, Astore->nnz);
  dgstrf(&options, &AC, relax_size, panel_size, etree,
         NULL, lwork, *perm_c, *perm_r, &L, &U, &Glu, &stat, &info);

  StatFree(&stat);
  Destroy_SuperMatrix_Store(&A);
  Destroy_CompCol_Permuted(&AC);

  /* convert etree to parents */
  SCformat *Lstore = (SCformat*)(L.Store);
  int nsuper = 1 + Lstore->nsuper;     // # of supernodal columns
  *parents = new int[nsuper];
  for (int s = 0; s < nsuper; s++) {
    int j = Lstore->sup_to_col[s+1]-1; // the last column index of this supernode
    if (etree[j] == nrow) {
        (*parents)[s] = -1;
    } else {
        (*parents)[s] = Lstore->col_to_sup[etree[j]];
    }
  }

  return;
}

#endif // KOKKOSKERNELS_ENABLE_TPL_SUPERLU


using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

enum {CUSPARSE, SUPERNODAL_NAIVE, SUPERNODAL_ETREE, SUPERNODAL_DAG};


#ifdef KOKKOSKERNELS_ENABLE_TPL_SUPERLU
template<typename scalar_t>
int test_sptrsv_perf(std::vector<int> tests, std::string& filename, bool metis, bool merge, bool invert_offdiag,
                     int panel_size, int relax_size, int sup_size_unblocked, int sup_size_blocked, int loop) {

  typedef Kokkos::Details::ArithTraits<scalar_t> STS;
  typedef int lno_t;
  typedef int size_type;

  // Default spaces
  //typedef Kokkos::OpenMP execution_space;
  typedef Kokkos::DefaultExecutionSpace execution_space;
  typedef typename execution_space::memory_space memory_space;

  // Host spaces
  typedef Kokkos::DefaultHostExecutionSpace host_execution_space;
  typedef typename host_execution_space::memory_space host_memory_space;

  //
  typedef KokkosSparse::CrsMatrix<scalar_t, lno_t, host_execution_space, void, size_type> host_crsmat_t;
  typedef KokkosSparse::CrsMatrix<scalar_t, lno_t,      execution_space, void, size_type> crsmat_t;

  //
  typedef typename host_crsmat_t::StaticCrsGraphType host_graph_t;
  typedef typename      crsmat_t::StaticCrsGraphType      graph_t;

  //
  typedef Kokkos::View< scalar_t*, host_memory_space > host_scalar_view_t;
  typedef Kokkos::View< scalar_t*,      memory_space > scalar_view_t;

  //
  typedef KokkosKernels::Experimental::KokkosKernelsHandle <size_type, lno_t, scalar_t,
    execution_space, memory_space, memory_space > KernelHandle;

  scalar_t ZERO = scalar_t(0);
  scalar_t ONE = scalar_t(1);

  // tolerance
  scalar_t tol = STS::epsilon();

  int num_failed = 0;
  std::cout << std::endl;
  std::cout << "Execution space: " << execution_space::name () << std::endl;
  std::cout << "Memory space   : " << memory_space::name () << std::endl;
  std::cout << std::endl;
  if (!filename.empty())
  {
    // ==============================================
    // read the matrix ** on host **
    std::cout << " SuperLU Tester Begin: Read matrix filename " << filename << std::endl;
    host_crsmat_t Mtx = KokkosKernels::Impl::read_kokkos_crst_matrix<host_crsmat_t>(filename.c_str()); //in_matrix

    const size_type nrows = Mtx.graph.numRows();

    auto  graph_host  = Mtx.graph; // in_graph
    auto row_map_host = graph_host.row_map;
    auto entries_host = graph_host.entries;
    auto values_host  = Mtx.values;
    //print_crsmat<host_crsmat_t> (nrows, Mtx);

    // ==============================================
    // call SuperLU on the host    
    // > data for SuperLU
    int *etree;
    int *perm_r, *perm_c;
    SuperMatrix L;
    SuperMatrix U;
    // > call SuperLU
    Kokkos::Timer timer;
    std::cout << " > call SuperLU for factorization" << std::endl;
    factor_superlu<scalar_t> (metis, nrows, values_host.data(), const_cast<int*> (row_map_host.data()), entries_host.data(),
                            panel_size, relax_size, L, U, &perm_r, &perm_c, &etree);
    std::cout << "   Factorization Time: " << timer.seconds() << std::endl << std::endl;

    // ==============================================
    // Run all requested algorithms
    for ( auto test : tests ) {
      std::cout << "\ntest = " << test << std::endl;

      KernelHandle khL, khU;
      switch(test) {
        case SUPERNODAL_NAIVE:
        case SUPERNODAL_ETREE:
        case SUPERNODAL_DAG:
        {
          // ==============================================
          // read SuperLU factor int crsMatrix on the host (superluMat_host) and copy to default host/device (superluL)
          host_crsmat_t superluL_host, superluU_host;
          crsmat_t superluL, superluU;
          std::cout << " > Read SuperLU factor into KokkosSparse::CrsMatrix (invert diagonal and copy to device)" << std::endl;
          if (merge) {
            std::cout << " > Merge supernodes" << std::endl;
          }
          timer.reset();
          bool cusparse = false; // pad diagonal blocks with zeros
          if (merge) {
            bool invert_diag = false; // invert after merge
            superluL_host = read_superlu_Lfactor<host_crsmat_t> (cusparse, merge, invert_diag, invert_offdiag, nrows, &L);
          } else {
            bool invert_diag = true; // only, invert diag is supported for now
            superluL = read_superlu_Lfactor<crsmat_t> (cusparse, merge, invert_diag, invert_offdiag, nrows, &L);
          }
          std::cout << "   Conversion Time for L: " << timer.seconds() << std::endl;

          timer.reset();
          if (merge) {
            bool invert_diag = false; // invert after merge
            superluU_host = read_superlu_Ufactor<host_crsmat_t> (invert_diag, nrows, &L, &U);
          } else {
            bool invert_diag = true; // only, invert diag is supported for now
            superluU = read_superlu_Ufactor<crsmat_t> (invert_diag, nrows, &L, &U);
          }
          std::cout << "   Conversion Time for U: " << timer.seconds() << std::endl << std::endl;
          //print_factor_superlu<scalar_t> (nrows, &L, &U, perm_r, perm_c);


          // ==============================================
          // create an handle
          if (test == SUPERNODAL_NAIVE) {
            std::cout << " > create handle for SUPERNODAL_NAIVE" << std::endl << std::endl;
            khL.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_NAIVE, nrows, true);
            khU.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_NAIVE, nrows, false);
          } else if (test == SUPERNODAL_ETREE) {
            std::cout << " > create handle for SUPERNODAL_ETREE" << std::endl << std::endl;
            khL.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_ETREE, nrows, true);
            khU.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_ETREE, nrows, false);
          } else {
            std::cout << " > create handle for SUPERNODAL_DAG" << std::endl << std::endl;
            khL.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_DAG, nrows, true);
            khU.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_DAG, nrows, false);
          }
          khL.set_diag_supernode_sizes (sup_size_unblocked, sup_size_blocked);
          khU.set_diag_supernode_sizes (sup_size_unblocked, sup_size_blocked);

          // ==============================================
          // setup supnodal info
          SCformat *Lstore = (SCformat*)(L.Store);
          int nsuper = 1 + Lstore->nsuper;
          int *supercols = Lstore->sup_to_col;
          int *superrows = Lstore->rowind_colptr;

          int nsuperL = nsuper;
          int *supercolsL = supercols;
          int *superrowsL = superrows;
          if (merge) {
            timer.reset ();
            // make a copy of etree
            int *etreeL = new int[nsuper];
            for (int i = 0; i < nsuper; i++) {
              etreeL[i] = etree[i];
            }
            // make a copy of supercols
            supercolsL = new int[1+nsuper];
            for (int i = 0; i <= nsuper; i++) {
              supercolsL[i] = supercols[i];
            }
            // make a copy of superrows
            superrowsL = new int[1+nrows];
            for (int i = 0; i <= nrows; i++) {
              superrowsL[i] = superrows[i];
            }

            // merge L-factor
            bool invert_diag = true;
            int nnzL = superluL_host.nnz ();
            superluL = merge_supernodes<host_crsmat_t, crsmat_t> (nrows, &nsuperL, supercolsL, superrowsL,
                                                                  true, invert_diag, invert_offdiag,
                                                                  superluL_host, superluU_host, etreeL);

            // save the supernodal info in the handle for U-solve
            khL.set_supernodes (nsuperL, supercolsL, etreeL);
            std::cout << "   L factor:" << std::endl;
            std::cout << "   Merge Supernodes Time: " << timer.seconds() << std::endl;
            std::cout << "   Number of nonzeros   : " << nnzL << " -> " << superluL.nnz () 
                      << " : " << double(superluL.nnz ()) / double(nnzL) << "x" << std::endl;
          } else {
            khL.set_supernodes (nsuper, supercols, etree);
          }
          int nsuperU = nsuper;
          if (merge) {
            // NOTE: starting over for U-factor
            timer.reset ();
            // make a copy of etree
            int *etreeU = new int[nsuper];
            for (int i = 0; i < nsuper; i++) {
              etreeU[i] = etree[i];
            }
            // make a copy of supercols
            int *supercolsU = new int[1+nsuper];
            for (int i = 0; i <= nsuper; i++) {
              supercolsU[i] = supercols[i];
            }
            // make a copy of superrows
            int *superrowsU = new int[1+nrows];
            for (int i = 0; i <= nrows; i++) {
              superrowsU[i] = superrows[i];
            }
            // merge U-factor
            bool invert_diag = true;
            // TODO: invert offdiagonal for U-solve
            bool invert_offdiagU = false;
            int nnzU = superluU_host.nnz ();
            superluU = merge_supernodes<host_crsmat_t, crsmat_t> (nrows, &nsuperU, supercolsU, superrowsU,
                                                                  false, invert_diag, invert_offdiagU,
                                                                  superluU_host, superluL_host, etreeU);
            // save the supernodal info in the handle for U-solve
            khU.set_supernodes (nsuperU, supercolsU, etreeU);
            std::cout << "   U factor:" << std::endl;
            std::cout << "   Merge Supernodes Time: " << timer.seconds() << std::endl;
            std::cout << "   Number of nonzeros   : " << nnzU << " -> " << superluU.nnz () 
                      << " : " << double(superluU.nnz ()) / double(nnzU) << "x" << std::endl;
          } else {
            khU.set_supernodes (nsuper, supercols, etree);
          }

          {
            // generate supernodal graphs for DAG scheduling
            auto supL = generate_supernodal_graph<host_graph_t, graph_t> (merge, nrows, superluL.graph, nsuperL, superrowsL, supercolsL);
            auto supU = generate_supernodal_graph<host_graph_t, graph_t> (merge, nrows, superluU.graph, nsuperU, superrowsL, supercolsL);
            //print_graph<host_graph_t> (nsuper2, supL);
            //print_graph<host_graph_t> (nsuper2, supU);

            int **dagL = generate_supernodal_dag<host_graph_t> (nsuperL, supL, supU);
            int **dagU = generate_supernodal_dag<host_graph_t> (nsuperU, supU, supL);
            khL.set_supernodal_dag (dagL);
            khU.set_supernodal_dag (dagU);
          }
          //print_crsmat<crsmat_t> (nrows, superluL);
          //print_crsmat<crsmat_t> (nrows, superluU);
 
          // ==============================================
          // do symbolic for L solve on the host
          auto graphL = superluL.graph; // in_graph
          auto row_mapL = graphL.row_map;
          auto entriesL = graphL.entries;
          auto valuesL  = superluL.values;
          timer.reset();
          std::cout << std::endl;
          sptrsv_symbolic (&khL, row_mapL, entriesL);
          std::cout << " > Lower-TRI: " << std::endl;
          std::cout << "   Symbolic Time: " << timer.seconds() << std::endl;

          // ==============================================
          // do symbolic for U solve on the host
          auto graphU = superluU.graph; // in_graph
          auto row_mapU = graphU.row_map;
          auto entriesU = graphU.entries;
          auto valuesU  = superluU.values;
          timer.reset ();
          sptrsv_symbolic (&khU, row_mapU, entriesU);
          std::cout << " > Upper-TRI: " << std::endl;
          std::cout << "   Symbolic Time: " << timer.seconds() << std::endl;

          // ==============================================
          // setup some solver options
          // NOTE: not sure how to set this?
          if (invert_offdiag) {
            std::cout << " > Invert off-diagonal blocks of L-factor" << std::endl;
          }
          khL.set_invert_offdiagonal(invert_offdiag);
          khU.set_invert_offdiagonal(invert_offdiag);

          // ==============================================
          // Preaparing for the first solve
          //> create the known solution and set to all 1's ** on host **
          host_scalar_view_t sol_host("sol_host", nrows);
          Kokkos::deep_copy(sol_host, ONE);

          // > create the rhs ** on host **
          // A*sol generates rhs: rhs is dense, use spmv
          host_scalar_view_t rhs_host("rhs_host", nrows);
          KokkosSparse::spmv( "N", ONE, Mtx, sol_host, ZERO, rhs_host);
          //for (int i = 0; i < nrows; i++) printf( "%.16e\n",rhs_host(i) );

          // ==============================================
          // apply forward-pivot to rhs on the host
          host_scalar_view_t tmp_host ("temp", nrows);
          forwardP_superlu<scalar_t> (nrows, perm_r, 1, rhs_host.data(), nrows, tmp_host.data(), nrows);

          // copy rhs to the default host/device
          scalar_view_t rhs ("rhs", nrows);
          scalar_view_t sol ("sol", nrows);
          Kokkos::deep_copy (rhs, tmp_host);

          // ==============================================
          // do L solve
          timer.reset();
          sptrsv_solve (&khL, row_mapL, entriesL, valuesL, sol, rhs);
          Kokkos::fence();
          std::cout << " > Lower-TRI: " << std::endl;
          std::cout << "   Solve Time   : " << timer.seconds() << std::endl;
          //Kokkos::deep_copy (tmp_host, rhs);
          //printf( "y=[" );
          //for (int ii=0; ii<nrows; ii++) printf( " %d %.16e\n",ii,tmp_host(ii) );
          //printf( "];\n" );

          // ==============================================
          // do U solve
          sptrsv_solve (&khU, row_mapU, entriesU, valuesU, sol, rhs);
          Kokkos::fence ();
          std::cout << " > Upper-TRI: " << std::endl;
          std::cout << "   Solve Time   : " << timer.seconds() << std::endl;
 
          // copy solution to host
          Kokkos::deep_copy(tmp_host, rhs);
          // apply backward-pivot
          backwardP_superlu<scalar_t>(nrows, perm_c, 1, tmp_host.data(), nrows, sol_host.data(), nrows);
          //printf( "x=[" );
          //for (int ii=0; ii<nrows; ii++) printf( " %d %.16e\n",ii,tmp_host(ii) );
          //printf( "];\n" );


          // ==============================================
          // Error Check ** on host **
          Kokkos::fence();
          std::cout << std::endl;
          if (!check_errors(tol, Mtx, rhs_host, sol_host)) {
            num_failed ++;
          }

          // try again?
          {
            Kokkos::deep_copy(sol_host, ONE);
            KokkosSparse::spmv( "N", ONE, Mtx, sol_host, ZERO, rhs_host);
            forwardP_superlu<scalar_t> (nrows, perm_r, 1, rhs_host.data(), nrows, tmp_host.data(), nrows);
            Kokkos::deep_copy (rhs, tmp_host);

            sptrsv_solve (&khL, row_mapL, entriesL, valuesL, sol, rhs);
            sptrsv_solve (&khU, row_mapU, entriesU, valuesU, sol, rhs);

            Kokkos::fence();
            Kokkos::deep_copy(tmp_host, rhs);
            backwardP_superlu<scalar_t>(nrows, perm_c, 1, tmp_host.data(), nrows, sol_host.data(), nrows);

            if (!check_errors(tol, Mtx, rhs_host, sol_host)) {
              num_failed ++;
            }
          }
          std::cout << std::endl;

          // Benchmark
          // L-solve
          double min_time = 1.0e32;
          double max_time = 0.0;
          double ave_time = 0.0;
          Kokkos::fence();
          for(int i = 0; i < loop; i++) {
            timer.reset();
            sptrsv_solve (&khL, row_mapL, entriesL, valuesL, sol, rhs);
            Kokkos::fence();
            double time = timer.seconds();
            ave_time += time;
            if(time>max_time) max_time = time;
            if(time<min_time) min_time = time;
            //std::cout << time << std::endl;
          }
          std::cout << " L-solve: loop = " << loop << std::endl;
          std::cout << "  LOOP_AVG_TIME:  " << ave_time/loop << std::endl;
          std::cout << "  LOOP_MAX_TIME:  " << max_time << std::endl;
          std::cout << "  LOOP_MIN_TIME:  " << min_time << std::endl << std::endl;

          // U-solve
          min_time = 1.0e32;
          max_time = 0.0;
          ave_time = 0.0;
          Kokkos::fence();
          for(int i = 0; i < loop; i++) {
            timer.reset();
            sptrsv_solve (&khU, row_mapU, entriesU, valuesU, sol, rhs);
            Kokkos::fence();
            double time = timer.seconds();
            ave_time += time;
            if(time>max_time) max_time = time;
            if(time<min_time) min_time = time;
            //std::cout << time << std::endl;
          }
          std::cout << " U-solve: loop = " << loop << std::endl;
          std::cout << "  LOOP_AVG_TIME:  " << ave_time/loop << std::endl;
          std::cout << "  LOOP_MAX_TIME:  " << max_time << std::endl;
          std::cout << "  LOOP_MIN_TIME:  " << min_time << std::endl << std::endl;
        }
        break;

        case CUSPARSE:
        {

#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE) 
          // ==============================================
          // read SuperLU factor int crsMatrix on the host (superluMat_host) and copy to default host/device (superluL)
          crsmat_t superluL, superluU;
          std::cout << " > Read SuperLU factor into KokkosSparse::CrsMatrix (invert diagonal and copy to device)" << std::endl;
          timer.reset();
          bool cusparse = true; // pad diagonal blocks with zeros
          bool invert_diag = false;
          superluL = read_superlu_Lfactor<crsmat_t> (cusparse, merge, invert_diag, invert_offdiag, nrows, &L);
          std::cout << "   Conversion Time for L: " << timer.seconds() << std::endl;

          timer.reset();
          superluU = read_superlu_Ufactor<crsmat_t> (invert_diag, nrows, &L, &U);
          std::cout << "   Conversion Time for U: " << timer.seconds() << std::endl << std::endl;
          //print_factor_superlu<scalar_t> (nrows, &L, &U, perm_r, perm_c);

          // ==============================================
          // > create a handle
          cusparseStatus_t status;
          cusparseHandle_t handle = 0;
          status = cusparseCreate(&handle);
          if (CUSPARSE_STATUS_SUCCESS != status)
            std::cout << "handle create status error name " << (status) << std::endl;
          cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST); // scalars are passed by reference on host

          // > create a empty info structure
          //std::cout << "  cusparse: create csrsv2info" << std::endl;
          csrsv2Info_t info = 0;
          status = cusparseCreateCsrsv2Info(&info);
          if (CUSPARSE_STATUS_SUCCESS != status)
            std::cout << "csrsv2info create status error name " << (status) << std::endl;

          // ==============================================
          // Preparing for L-solve
          // step 1: create a descriptor
          int nnzL = superluL.nnz();
          auto graphL = superluL.graph; // in_graph
          auto row_mapL = graphL.row_map;
          auto entriesL = graphL.entries;
          auto valuesL  = superluL.values;

          // NOTE: it is stored in CSC = UPPER + TRANSPOSE
          cusparseMatDescr_t descrL = 0;
          status = cusparseCreateMatDescr(&descrL);
          if (CUSPARSE_STATUS_SUCCESS != status)
            std::cout << "matdescr create status error name " << (status) << std::endl;
          cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
          cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_UPPER);
          cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT);
          cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_GENERAL);

          // ==============================================
          // step 2: query how much memory used in csrsv2, and allocate the buffer
          // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
          int pBufferSize;
          void *pBufferL = 0;
          cusparseOperation_t trans = CUSPARSE_OPERATION_TRANSPOSE;
          cusparseDcsrsv2_bufferSize(handle, trans, nrows, nnzL, descrL,
                                     valuesL.data(), row_mapL.data(), entriesL.data(), info,
                                     &pBufferSize);
          cudaMalloc((void**)&pBufferL, pBufferSize);

          // ==============================================
          // step 3: analysis
          std::cout << "  Lower-Triangular" << std::endl;
          timer.reset();
          const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
          status = cusparseDcsrsv2_analysis(handle, trans, nrows, nnzL, descrL,
                                            valuesL.data(), row_mapL.data(), entriesL.data(),
                                            info, policy, pBufferL);
          std::cout << "  Cusparse Symbolic Time: " << timer.seconds() << std::endl;
          if (CUSPARSE_STATUS_SUCCESS != status)
            std::cout << "analysis status error name " << (status) << std::endl;
          // L has unit diagonal, so no structural zero is reported.

          //std::cout << "  cusparse path: analysis" << std::endl;
          int structural_zero;
          status = cusparseXcsrsv2_zeroPivot(handle, info, &structural_zero);
          if (CUSPARSE_STATUS_ZERO_PIVOT == status){
             printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
          }

          // ==============================================
          // Preaparing for the first solve
          //> create the known solution and set to all 1's ** on host **
          host_scalar_view_t sol_host("sol_host", nrows);
          Kokkos::deep_copy(sol_host, ONE);

          // > create the rhs ** on host **
          // A*sol generates rhs: rhs is dense, use spmv
          host_scalar_view_t rhs_host("rhs_host", nrows);
          KokkosSparse::spmv( "N", ONE, Mtx, sol_host, ZERO, rhs_host);
          //for (int i = 0; i < nrows; i++) printf( "%.16e\n",rhs_host(i) );

          // ==============================================
          // step 1: apply forward-pivot to rhs on the host
          host_scalar_view_t tmp_host ("temp", nrows);
          forwardP_superlu<scalar_t> (nrows, perm_r, 1, rhs_host.data(), nrows, tmp_host.data(), nrows);

          // copy rhs to the default host/device
          scalar_view_t rhs ("rhs", nrows);
          scalar_view_t sol ("sol", nrows);
          Kokkos::deep_copy (rhs, tmp_host);

          // ==============================================
          // step 2: solve L*y = x
          timer.reset();
          const double alpha = 1.;
          status = cusparseDcsrsv2_solve(handle, trans, nrows, nnzL, &alpha, descrL, 
                                         valuesL.data(), row_mapL.data(), entriesL.data(), info, 
                                         rhs.data(), sol.data(), policy, pBufferL);
          Kokkos::fence();
          std::cout << "  Cusparse Solve Time   : " << timer.seconds() << std::endl;
          if (CUSPARSE_STATUS_SUCCESS != status)
            std::cout << "solve status error name " << (status) << std::endl;
          // L has unit diagonal, so no numerical zero is reported.
          int numerical_zero;
          status = cusparseXcsrsv2_zeroPivot(handle, info, &numerical_zero);
          if (CUSPARSE_STATUS_ZERO_PIVOT == status){
            printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
          }
          //Kokkos::deep_copy (tmp_host, sol);
          //printf( "y=[" );
          //for (int ii=0; ii<nrows; ii++) printf( " %d %.16e\n",ii,tmp_host(ii) );
          //printf( "];\n" );


          // ==============================================
          // Preparing for U-solve
          int nnzU = superluU.nnz();
          auto graphU = superluU.graph; // in_graph
          auto row_mapU = graphU.row_map;
          auto entriesU = graphU.entries;
          auto valuesU  = superluU.values;

          // ==============================================
          // step 1: create a descriptor
          // NOTE: it is stored in CSR = UPPER + NO-TRANSPOSE
          cusparseMatDescr_t descrU = 0;
          status = cusparseCreateMatDescr(&descrU);
          if (CUSPARSE_STATUS_SUCCESS != status)
            std::cout << "matdescr create status error name " << (status) << std::endl;
          cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO);
          cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
          cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
          cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_GENERAL);

          // ==============================================
          // step 2: query how much memory used in csrsv2, and allocate the buffer
          // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
          void *pBufferU = 0;
          trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
          cusparseDcsrsv2_bufferSize(handle, trans, nrows, nnzU, descrU,
                                     valuesU.data(), row_mapU.data(), entriesU.data(),
                                     info, &pBufferSize);
          cudaMalloc((void**)&pBufferU, pBufferSize);

          // ==============================================
          // step 3: analysis
          std::cout << std::endl << "  Upper-Triangular" << std::endl;
          timer.reset();
          status = cusparseDcsrsv2_analysis(handle, trans, nrows, nnzU, descrU,
                                            valuesU.data(), row_mapU.data(), entriesU.data(),
                                            info, policy, pBufferU);
          std::cout << "  Cusparse Symbolic Time: " << timer.seconds() << std::endl;
          if (CUSPARSE_STATUS_SUCCESS != status)
            std::cout << "analysis status error name " << (status) << std::endl;

          status = cusparseXcsrsv2_zeroPivot(handle, info, &structural_zero);
          if (CUSPARSE_STATUS_ZERO_PIVOT == status){
            printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
          }

          // ==============================================
          // step 1: solve U*y = x
          timer.reset();
          status = cusparseDcsrsv2_solve(handle, trans, nrows, nnzU, &alpha, descrU,
                                         valuesU.data(), row_mapU.data(), entriesU.data(), info,
                                         sol.data(), rhs.data(), policy, pBufferU);
          Kokkos::fence();
          std::cout << "  Cusparse Solve Time   : " << timer.seconds() << std::endl;
          if (CUSPARSE_STATUS_SUCCESS != status)
            std::cout << "solve status error name " << (status) << std::endl;
          status = cusparseXcsrsv2_zeroPivot(handle, info, &numerical_zero);
          if (CUSPARSE_STATUS_ZERO_PIVOT == status){
            printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
          }

          // ==============================================
          // copy solution to host
          Kokkos::deep_copy(tmp_host, rhs);
          // apply backward-pivot
          backwardP_superlu<scalar_t>(nrows, perm_c, 1, tmp_host.data(), nrows, sol_host.data(), nrows);
          //printf( "x=[" );
          //for (int ii=0; ii<nrows; ii++) printf( " %d %.16e\n",ii,tmp_host(ii) );
          //printf( "];\n" );

          // ==============================================
          // Error Check ** on host **
          Kokkos::fence();
          std::cout << std::endl;
          if (!check_errors(tol, Mtx, rhs_host, sol_host)) {
            num_failed ++;
          }
          std::cout << std::endl;

          // ==============================================
          // Benchmark
          // L-solve
          double min_time = 1.0e32;
          double max_time = 0.0;
          double ave_time = 0.0;
          Kokkos::fence();
          for(int i = 0; i < loop; i++) {
            timer.reset();
            cusparseDcsrsv2_solve(handle, trans, nrows, nnzL, &alpha, descrL, 
                                  valuesL.data(), row_mapL.data(), entriesL.data(), info, 
                                  rhs.data(), sol.data(), policy, pBufferL);
            Kokkos::fence();
            double time = timer.seconds();
            ave_time += time;
            if(time>max_time) max_time = time;
            if(time<min_time) min_time = time;
            //std::cout << time << std::endl;
          }
          std::cout << " L-solve: loop = " << loop << std::endl;
          std::cout << "  LOOP_AVG_TIME:  " << ave_time/loop << std::endl;
          std::cout << "  LOOP_MAX_TIME:  " << max_time << std::endl;
          std::cout << "  LOOP_MIN_TIME:  " << min_time << std::endl << std::endl;

          // U-solve
          min_time = 1.0e32;
          max_time = 0.0;
          ave_time = 0.0;
          Kokkos::fence();
          for(int i = 0; i < loop; i++) {
            timer.reset();
            cusparseDcsrsv2_solve(handle, trans, nrows, nnzU, &alpha, descrU,
                                  valuesU.data(), row_mapU.data(), entriesU.data(), info,
                                  sol.data(), rhs.data(), policy, pBufferU);
            Kokkos::fence();
            double time = timer.seconds();
            ave_time += time;
            if(time>max_time) max_time = time;
            if(time<min_time) min_time = time;
            //std::cout << time << std::endl;
          }
          std::cout << " U-solve: loop = " << loop << std::endl;
          std::cout << "  LOOP_AVG_TIME:  " << ave_time/loop << std::endl;
          std::cout << "  LOOP_MAX_TIME:  " << max_time << std::endl;
          std::cout << "  LOOP_MIN_TIME:  " << min_time << std::endl << std::endl;
#endif
        }
        break;

        default:
          std::cout << " > Invalid test ID < " << std::endl;
          exit(0);
      }
    }
  }
  std::cout << std::endl << std::endl;

  return num_failed;
}
#endif //KOKKOSKERNELS_ENABLE_TPL_SUPERLU


void print_help_sptrsv() {
  printf("Options:\n");
  printf("  --test [OPTION] : Use different kernel implementations\n");
  printf("                    Options:\n");
  printf("                    superlu-naive, superlu-etree, superlu-dag\n\n");
  printf("  -f [file]       : Read in Matrix Market formatted text file 'file'.\n");
  printf("  --loop [LOOP]   : How many spmv to run to aggregate average time. \n");
}


int main(int argc, char **argv)
{
#ifdef KOKKOSKERNELS_ENABLE_TPL_SUPERLU
  std::vector<int> tests;
  std::string filename;

  int loop = 1;
  // parameters for sparse-triangular solve
  int sup_size_unblocked = 100;
  int sup_size_blocked = 200;
  // use METIS before SuperLU
  bool metis = false;
  // merge supernodes
  bool merge = false;
  // invert off-diagonal of L-factor
  bool invert_offdiag = false;
  // parameters for SuperLU (only affects factorization)
  int panel_size = sp_ienv(1);
  int relax_size = sp_ienv(2);

  if(argc == 1)
  {
    print_help_sptrsv();
    return 0;
  }

  for(int i = 0; i < argc; i++) {
    if((strcmp(argv[i],"--test")==0)) {
      i++;
      if((strcmp(argv[i],"superlu-naive")==0)) {
        tests.push_back( SUPERNODAL_NAIVE );
      }
      if((strcmp(argv[i],"superlu-etree")==0)) {
        tests.push_back( SUPERNODAL_ETREE );
      }
      if((strcmp(argv[i],"superlu-dag")==0)) {
        tests.push_back( SUPERNODAL_DAG );
      }
      if((strcmp(argv[i],"cusparse")==0)) {
        tests.push_back( CUSPARSE );
      }
      continue;
    }
    if((strcmp(argv[i],"-f")==0)) {
      filename = argv[++i];
      continue;
    }
    if((strcmp(argv[i],"--loop")==0)) {
      loop = atoi(argv[++i]);
      continue;
    }
    if((strcmp(argv[i],"--sup-size-unblocked")==0)) {
      sup_size_unblocked = atoi(argv[++i]);
      continue;
    }
    if((strcmp(argv[i],"--sup-size-blocked")==0)) {
      sup_size_blocked = atoi(argv[++i]);
      continue;
    }
    if((strcmp(argv[i],"--metis")==0)) {
      metis = true;
      continue;
    }
    if((strcmp(argv[i],"--merge")==0)) {
      merge = true;
      continue;
    }
    if((strcmp(argv[i],"--invert-offdiag")==0)) {
      invert_offdiag = true;
      continue;
    }
    if((strcmp(argv[i],"--panel-size")==0)) {
      panel_size = atoi(argv[++i]);
      continue;
    }
    if((strcmp(argv[i],"--relax-size")==0)) {
      relax_size = atoi(argv[++i]);
      continue;
    }
    if((strcmp(argv[i],"--help")==0) || (strcmp(argv[i],"-h")==0)) {
      print_help_sptrsv();
      return 0;
    }
  }

  std::cout << std::endl;
  for (size_t i = 0; i < tests.size(); ++i) {
    std::cout << "tests[" << i << "] = " << tests[i] << std::endl;
  }

  Kokkos::initialize(argc,argv);
  {
    std::cout << " > supernode_size_unblocked: " << sup_size_unblocked << std::endl;
    std::cout << " > supernode_size_blocked:   " << sup_size_blocked << std::endl;
    int total_errors = test_sptrsv_perf<double>(tests, filename, metis, merge, invert_offdiag, panel_size,
                                                relax_size, sup_size_unblocked, sup_size_blocked, loop);
    if(total_errors == 0)
      printf("Kokkos::SPTRSV Test: Passed\n\n");
    else
      printf("Kokkos::SPTRSV Test: Failed\n\n");
  }
  Kokkos::finalize();
#else
  std::cout << "SUPERLU NOT ENABLED:" << std::endl;
  exit(0);
#endif
  return 0;
}
#else // defined( KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA ) && (!defined(KOKKOS_ENABLE_CUDA) || ( 8000 <= CUDA_VERSION ))
int main() {
#if !defined( KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA )
  printf( " KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA **not** defined\n" );
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  printf( " KOKKOS_ENABLE_CUDA defined\n" );
  #if !defined(KOKKOS_ENABLE_CUDA_LAMBDA)
  printf( " KOKKOS_ENABLE_CUDA_LAMBDA **not** defined\n" );
  #endif
#endif
  printf( " CUDA_VERSION = %d\n", CUDA_VERSION );
  return 0;
}
#endif
