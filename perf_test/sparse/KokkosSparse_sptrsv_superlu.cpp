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

#include <Kokkos_Core.hpp>
#include <matrix_market.hpp>

#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosSparse_sptrsv.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include <KokkosKernels_IOUtils.hpp>

#if defined( KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA ) && (!defined(KOKKOS_ENABLE_CUDA) || ( 8000 <= CUDA_VERSION ))

#ifdef KOKKOSKERNELS_ENABLE_TPL_SUPERLU
#include "cblas.h"
#include "lapacke.h"
#include "slu_ddefs.h"


template <typename Scalar>
void print_factor_superlu(int n, SuperMatrix *L, SuperMatrix *U, int *perm_r, int *perm_c) {
  SCformat *Lstore = (SCformat*)(L->Store);
  Scalar   *Lx = (Scalar*)(Lstore->nzval);

  int * nb = Lstore->sup_to_col;
  int * mb = Lstore->rowind_colptr;
  int * colptr = Lstore->nzval_colptr;
  int * rowind = Lstore->rowind;

  for (int k = 0; k <= Lstore->nsuper; k++)
  {
    int j1 = nb [k];
    int j2 = nb [k+1];
    int nscol  = j2 - j1;
    printf( "%d %d\n",k,nscol );
  }

  /* permutation vectors */
  int *iperm_r = (int*)malloc(n * sizeof(int));
  int *iperm_c = (int*)malloc(n * sizeof(int));
  for (int k = 0; k < n; k++) {
    iperm_r[perm_r[k]] = k;
    iperm_c[perm_c[k]] = k;
  }
  printf( "P=[\n" );
  for (int k = 0; k < n; k++) {
    printf( "%d, %d %d, %d %d\n",k, perm_r[k],iperm_r[k], perm_c[k],iperm_c[k] );
  }
  printf( "];\n" );
  free(iperm_r); free(iperm_c);

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
      for (int j = 0; j < i; j++) printf( "%d %d %.16e\n",j1+i, j1+j, Lx[psx + i + j*nsrow] );
      printf( "%d %d 1.0\n",j1+i, j1+i );
    }

    /* the off-diagonal blocks */
    for (int ii = 0; ii < nsrow2; ii++) {
      int i = rowind [ps2 + ii];
      for (int j = 0; j < nscol; j++) printf( "%d %d %.16e\n",i, j1+j, Lx[psx+nscol + ii + j*nsrow] );
    }
  }
  printf( "];\n ");

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
      for (int j = i; j < nscol; j++) printf( "%d %d %.16e\n",j1+i, j1+j, Lx[psx + i + j*nsrow] );
    }

    /* the off-diagonal blocks */
    for (int jcol = j1; jcol < j1 + nscol; jcol++) {
      for (int i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++ ){
        int irow = U_SUB(i);
        printf( "%d %d %.16e\n", irow, jcol, Uval[i] );
      }
    }
  }
  printf( "];\n" );
}


template <typename crsMat_t>
void print_crsmat(int n, crsMat_t &A) {
  auto graph = A.graph; // in_graph
  auto row_map = graph.row_map;
  auto entries = graph.entries;
  auto values  = A.values;

  for (int i = 0; i < n; i++) {
    for (int k = row_map[i]; k < row_map[i+1]; k++) {
      printf( "%d %d %.16e %d\n",i,entries[k],values[k],k );
    }
  }
}


/* ========================================================================================= */
template <typename crsMat_t, typename hostMat_t>
crsMat_t read_superlu_Lfactor(int n, SuperMatrix *L, hostMat_t *hostMat) {

  typedef typename hostMat_t::StaticCrsGraphType host_graph_t;
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type   cols_view_t;
  typedef typename crsMat_t::values_type::non_const_type values_view_t;

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

  int j1, j2, i1, i2, psx, nsrow, nscol, i, j, ii, jj, s, nsrow2, ps2;

  // compute offset for each row
  //printf( "\n -- store superlu factor (n=%d, nsuper=%d) --\n",n,nsuper );
  j = 0;
  hr(j) = 0;
  for (s = 0 ; s < nsuper ; s++) {
    j1 = nb[s];
    j2 = nb[s+1];
    nscol = j2 - j1;      // number of columns in the s-th supernode column

    i1 = mb[j1];
    i2 = mb[j1+1];
    nsrow = i2 - i1;    // "total" number of rows in all the supernodes (diagonal+off-diagonal)

    for (jj = 0; jj < nscol; jj++) {
      hr(j+1) = hr(j) + nsrow;
      //printf( " hr(%d) = %d\n",j+1,hr(j+1) );
      j++;
    }
  }

  // store L in csr
  for (s = 0 ; s < nsuper ; s++) {
    j1 = nb[s];
    j2 = nb[s+1];
    nscol = j2 - j1;      // number of columns in the s-th supernode column

    i1 = mb[j1] ;
    i2 = mb[j1+1] ;
    nsrow  = i2 - i1;    // "total" number of rows in all the supernodes (diagonal+off-diagonal)
    nsrow2 = nsrow - nscol;  // "total" number of rows in all the off-diagonal supernodes
    ps2    = i1 + nscol;     // offset into rowind

    psx = colptr[j1] ;        // offset into data,   Lx[s][s]
    //printf( "%d: nscol=%d, nsrow=%d\n",s,nscol,nsrow );

    /* diagonal block */
    // for each column (or row due to symmetry), the diagonal supernodal block is stored (in ascending order of row indexes) first
    // so that we can do TRSM on the diagonal block
    #define SUPERLU_INVERT_DIAG
    #ifdef SUPERLU_INVERT_DIAG
    LAPACKE_dtrtri(LAPACK_COL_MAJOR,
                   'L', 'U', nscol, &Lx[psx], nsrow);
    #endif
    for (ii = 0; ii < nscol; ii++) {
      // lower-triangular part
      for (jj = 0; jj < ii; jj++) {
        //printf("1 %d %d %d %.16e %d\n",hr(j1+jj), j1+ii, j1+jj, Lx[psx + (ii + jj*nsrow)], hr(j1+ii));
        hc(hr(j1+jj)) = j1+ii;
        hv(hr(j1+jj)) = Lx[psx + (ii + jj*nsrow)];
        hr(j1+jj) ++;
      }
      // diagonal
      hc(hr(j1+jj)) = j1+ii;
      hv(hr(j1+jj)) = STS::one ();
      hr(j1+jj) ++;
      // explicitly store zeros in upper-part
      for (jj = ii+1; jj < nscol; jj++) {
        //printf("2 %d %d %d %.16e %d\n",hr(j1+jj), j1+ii, j1+jj, STS::zero (), hr(j1+ii));
        hc(hr(j1+jj)) = j1+ii;
        hv(hr(j1+jj)) = STS::zero ();
        hr(j1+jj) ++;
      }
    }

    /* off-diagonal blocks */
    for (ii = 0; ii < nsrow2; ii++) {
      i = rowind [ps2 + ii] ;
      for (jj = 0; jj < nscol; jj++) {
        //printf("0 %d %d %d %.16e %d\n",hr(j1+jj), i, j1+jj, Lx[psx + (nscol+ii + jj*nsrow)], hr(i));
        hc(hr(j1+jj)) = i;
        hv(hr(j1+jj)) = Lx[psx + (nscol+ii + jj*nsrow)];
        hr(j1+jj) ++;
      }
    }
  }

  // fix hr
  for (i = n; i >= 1; i--) {
    hr(i) = hr(i-1);
    //printf( "hr[%d] = %d\n",i-1,hr(i) );
  }
  hr(0) = 0;

  // deepcopy
  Kokkos::deep_copy (rowmap_view, hr);
  Kokkos::deep_copy (column_view, hc);
  Kokkos::deep_copy (values_view, hv);


  // create crs
  host_graph_t host_graph (hc, hr);
  hostMat = new hostMat_t("HostMatrix", n, hv, host_graph);

  // create crs
  graph_t static_graph (column_view, rowmap_view);
  crsMat_t crsmat("CrsMatrix", n, values_view, static_graph);
  return crsmat;
}


/* ========================================================================================= */
template <typename crsMat_t, typename hostMat_t>
crsMat_t read_superlu_Ufactor(int n, SuperMatrix *L,  SuperMatrix *U, hostMat_t *hostMat) {

  typedef typename hostMat_t::StaticCrsGraphType host_graph_t;
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type   cols_view_t;
  typedef typename crsMat_t::values_type::non_const_type values_view_t;

  typedef typename values_view_t::value_type scalar_t;
  typedef Kokkos::Details::ArithTraits<scalar_t> STS;

  SCformat *Lstore = (SCformat*)(L->Store);
  scalar_t *Lx = (scalar_t*)(Lstore->nzval);

  NCformat *Ustore = (NCformat*)(U->Store);
  double *Uval = (double*)(Ustore->nzval);

  /* create a map from row id to supernode id */
  int * nb = Lstore->sup_to_col;
  int * mb = Lstore->rowind_colptr;
  int * colptr = Lstore->nzval_colptr;

  int supid = 0;
  int * map = (int*)malloc(n * sizeof(int));
  for (int k = 0; k <= Lstore->nsuper; k++)
  {
    int j1 = nb [k];
    int j2 = nb [k+1];
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

  int * check = (int*)calloc((Lstore->nsuper + 1), sizeof(int));
  int * sup   = (int*)calloc((Lstore->nsuper + 1), sizeof(int));
  for (int k = Lstore->nsuper; k >= 0; k--) {
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
    for (int i = 0; i < Lstore->nsuper + 1; i++ ) {
      check[i] = 0;
    }
  }

  // convert to the offset for each row
  //printf( " hr(%d) = %d\n",0,hr(0) );
  for (int i = 1; i <= n; i++) {
    //printf( " hr(%d) = %d",i,hr(i) );
    hr (i) += hr (i-1);
    //printf( " => %d\n",hr(i) );
  }

  /* Upper-triangular matrix */
  int nnzA = hr (n);
  cols_view_t    column_view ("colmap_view", nnzA);
  values_view_t  values_view ("values_view", nnzA);
  //printf( " nnzA = %d\n",nnzA );

  typename cols_view_t::HostMirror    hc = Kokkos::create_mirror_view (column_view);
  typename values_view_t::HostMirror  hv = Kokkos::create_mirror_view (values_view);

  for (int k = 0 ; k <= Lstore->nsuper; k++) {
    int j1 = nb[k];
    int nscol = nb[k+1] - j1;

    int i1 = mb[j1];
    int nsrow  = mb[j1+1] - i1;

    /* the diagonal block */
    int psx = colptr[j1];
    #ifdef SUPERLU_INVERT_DIAG
    LAPACKE_dtrtri(LAPACK_COL_MAJOR,
                   'U', 'N', nscol, &Lx[psx], nsrow);
    #endif
    for (int i = 0; i < nscol; i++) {
      for (int j = 0; j < i; j++) {
        hc(hr(j1 + i) + j) = j1 + j;
        hv(hr(j1 + i) + j) = STS::zero ();
        //printf( " > %d %d %e (%d)\n",j1 + i,j1 + j,STS::zero (), hr(j1 + i) + j );
      }

      for (int j = i; j < nscol; j++) {
        hc(hr(j1 + i) + j) = j1 + j;
        hv(hr(j1 + i) + j) = Lx[psx + i + j*nsrow];
        //printf( " > %d %d %e (%d)\n",j1 + i,j1 + j,Lx[psx + i + j*nsrow], hr(j1 + i) + j );
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
  free(check); free(sup);
  free(map);

  // fix hr
  for (int i = n; i >= 1; i--) {
    hr(i) = hr(i-1);
    //printf( "hr[%d] = %d\n",i-1,hr(i) );
  }
  hr(0) = 0;

  // deepcopy
  Kokkos::deep_copy (rowmap_view, hr);
  Kokkos::deep_copy (column_view, hc);
  Kokkos::deep_copy (values_view, hv);


  // create crs
  host_graph_t host_graph (hc, hr);
  hostMat = new hostMat_t("HostMatrix", n, hv, host_graph);

  // create crs
  graph_t static_graph (column_view, rowmap_view);
  crsMat_t crsmat("CrsMatrix", n, values_view, static_graph);
  return crsmat;
}


/* ========================================================================================= */
template<typename Scalar>
void factor_superlu(const int nrow, const int nnz, Scalar *nzvals, int *rowptr, int *colind,
                    SuperMatrix &L, SuperMatrix &U, int **perm_r, int **perm_c, int **parents) {
  SuperMatrix A;
  NCformat *Astore;
  //int      *perm_c; /* column permutation vector */
  //int      *perm_r; /* row permutations from partial pivoting */
  int      info;
  superlu_options_t options;
  SuperLUStat_t stat;

  set_default_options(&options);

  dCreate_CompCol_Matrix(&A, nrow, nrow, nnz, nzvals, colind, rowptr, SLU_NC, SLU_D, SLU_GE);
  Astore = (NCformat*)(A.Store);
  printf("Dimension %dx%d; # nonzeros %d\n", A.nrow, A.ncol, Astore->nnz);

  if ( !(*perm_c = intMalloc(nrow)) ) ABORT("Malloc fails for perm_c[].");
  if ( !(*perm_r = intMalloc(nrow)) ) ABORT("Malloc fails for perm_r[].");

  /* Initialize the statistics variables. */
  StatInit(&stat);

  /* Call SuperLU to solve the problem. */
  //for (int i=0; i<m; i++ ) printf( " B[%d]=%e\n",i,rhs[i] );
  int *etree = intMalloc(A.ncol);
  get_perm_c(options.ColPerm, &A, *perm_c);

  SuperMatrix AC;
  sp_preorder(&options, &A, *perm_c, etree, &AC);

  GlobalLU_t Glu;
  int lwork = 0;
  int panel_size = sp_ienv(1);
  int relax = sp_ienv(2);
  dgstrf(&options, &AC, relax, panel_size, etree,
         NULL, lwork, *perm_c, *perm_r, &L, &U, &Glu, &stat, &info);

  Destroy_CompCol_Permuted(&AC);

  /* convert etree to parents */
  SCformat *Lstore = (SCformat*)(L.Store);
  int nsuper = 1 + Lstore->nsuper;     // # of supernodal columns
  *parents = intMalloc(A.ncol);
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

/* ========================================================================================= */
template<typename Scalar>
void forwardP_superlu(int n, int *perm_r, int nrhs, Scalar *B, int ldb, Scalar *X, int ldx) {

  /* Permute right hand sides to form Pr*B */
  for (int j = 0; j < nrhs; j++) {
      double *rhs_work = &B[j*ldb];
      double *sol_work = &X[j*ldx];
      for (int k = 0; k < n; k++) sol_work[perm_r[k]] = rhs_work[k];
  }
}

template<typename Scalar>
void solveL_superlu (SuperMatrix *L,
                     int nrhs, double *X, int ldx) {

  double   one  = 1.0;
  double   zero = 0.0;

  SCformat *Lstore = (SCformat*)(L->Store);
  double   *Lx = (double*)(Lstore->nzval);

  /* allocate workspace */
  int n = L->nrow;
  int ldw = n;
  double *work = (double*)malloc(ldw * nrhs * sizeof(double));
  if ( !work ) ABORT("Malloc fails for local work[].");

  int * nb = Lstore->sup_to_col;
  int * mb = Lstore->rowind_colptr;
  int * colptr = Lstore->nzval_colptr;
  int * rowind = Lstore->rowind;

  /* Forward solve */
  for (int k = 0; k <= Lstore->nsuper; k++)
  {
    int j1 = nb[k];
    int j2 = nb[k+1];
    int nscol  = j2 - j1;

    int i1 = mb[j1];
    int i2 = mb[j1+1];
    int nsrow  = i2 - i1;
    int nsrow2 = nsrow - nscol;
    int ps2    = i1 + nscol;

    int psx = colptr[j1];

    /* do TRSM with the diagonal blocks */
    #ifdef SUPERLU_INVERT_DIAG
    cblas_dtrmm (CblasColMajor,
        CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
        nscol, nrhs,
        one,  &Lx[psx], nsrow,
              &X[j1],   ldx);
    #else
    cblas_dtrsm (CblasColMajor,
        CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
        nscol, nrhs,
        one,  &Lx[psx], nsrow,
              &X[j1],   ldx);
    #endif

    /* do GEMM update with the off-diagonal blocks */
    cblas_dgemm (CblasColMajor,
          CblasNoTrans, CblasNoTrans,
          nsrow2, nrhs, nscol,
          one,   &Lx[psx + nscol], nsrow,
                 &X[j1], ldx,
          zero,  work, ldw);

    /* scatter/accumulate updates into vectors */
    for (int ii = 0; ii < nsrow2; ii++)
    {
      int i = rowind [ps2 + ii];
      for (int j = 0; j < nrhs; j++)
      {
        X[i + j*ldx] -= work[ii + j*ldw];
      }
    }
  } /* for L-solve */
  free(work);
}

template<typename Scalar>
void solveU_superlu (SuperMatrix *L, SuperMatrix *U,
                     int nrhs, double *Bmat, int ldb) {

  double   one = 1.0;

  SCformat *Lstore = (SCformat*)(L->Store);
  NCformat *Ustore = (NCformat*)(U->Store);

  double *Lval = (double*)(Lstore->nzval);
  double *Uval = (double*)(Ustore->nzval);

  int * nb = Lstore->sup_to_col;
  int * mb = Lstore->rowind_colptr;
  int * colptr = Lstore->nzval_colptr;

  /*
   * Back solve.
   */
  for (int k = Lstore->nsuper; k >= 0; k--) {
    int j1 = nb[k];
    int nscol = nb[k+1] - j1;

    int i1 = mb[j1];
    int nsrow = mb[j1+1] - i1;

    int psx = colptr[j1];

    /* do TRSM with the diagonal block */
    #ifdef SUPERLU_INVERT_DIAG
    cblas_dtrmm (CblasColMajor,
        CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        nscol, nrhs,
        one,  &Lval[psx], nsrow,
              &Bmat[j1], ldb);
    #else
    cblas_dtrsm (CblasColMajor,
        CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        nscol, nrhs,
        one, &Lval[psx], nsrow,
             &Bmat[j1], ldb);
    #endif

    /* update with the off-diagonal blocks */
    for (int j = 0; j < nrhs; ++j) {
      double *rhs_work = &Bmat[j*ldb];
      for (int jcol = j1; jcol < j1 + nscol; jcol++) {
        for (int i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++ ){
          int irow = U_SUB(i);
          rhs_work[irow] -= rhs_work[jcol] * Uval[i];
        }
      }
    }
  } /* for U-solve */
}

template<typename Scalar>
void backwardP_superlu(int n, int *perm_c, int nrhs, Scalar *B, int ldb, Scalar *X, int ldx) {

    /* Compute the final solution X := Pc*X. */
    for (int j = 0; j < nrhs; j++) {
        double *rhs_work = &B[j*ldb];
        double *sol_work = &X[j*ldx];
        for (int k = 0; k < n; k++) sol_work[k] = rhs_work[perm_c[k]];
    }
}
#endif // KOKKOSKERNELS_ENABLE_TPL_SUPERLU

using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

enum {DEFAULT, CUSPARSE, LVLSCHED_RP, LVLSCHED_TP1/*, LVLSCHED_TP2*/, SUPERNODAL_NAIVE, SUPERNODAL_ETREE};


#ifdef KOKKOSKERNELS_ENABLE_TPL_SUPERLU
template<typename Scalar>
int test_sptrsv_perf(std::vector<int> tests, std::string& filename, int team_size, int vector_length, int idx_offset, int loop) {

  typedef Scalar scalar_t;
  typedef int lno_t;
  typedef int size_type;

  // Default spaces
  typedef Kokkos::DefaultExecutionSpace execution_space;
  typedef typename execution_space::memory_space memory_space;

  // Host spaces
  typedef Kokkos::DefaultHostExecutionSpace host_execution_space;
  typedef typename host_execution_space::memory_space host_memory_space;

  //
  typedef KokkosSparse::CrsMatrix<scalar_t, lno_t, host_execution_space, void, size_type> host_crsmat_t;
  typedef KokkosSparse::CrsMatrix<scalar_t, lno_t,      execution_space, void, size_type> crsmat_t;

  //
  typedef Kokkos::View< scalar_t*, host_memory_space > HostValuesType;
  typedef Kokkos::View< scalar_t*,      memory_space > ValuesType;

  //
  typedef KokkosKernels::Experimental::KokkosKernelsHandle <size_type, lno_t, scalar_t,
    execution_space, memory_space, memory_space > KernelHandle;

  scalar_t ZERO = scalar_t(0);
  scalar_t ONE = scalar_t(1);


// Read mtx
// Run all requested algorithms

  std::cout << std::endl;
  std::cout << "Execution space: " << execution_space::name () << std::endl;
  std::cout << "Memory space   : " << memory_space::name () << std::endl;
  std::cout << std::endl;
  if (!filename.empty())
  {
    // read the matrix ** on host **
    std::cout << " SuperLU Tester Begin: Read matrix filename " << filename << std::endl;
    host_crsmat_t Mtx = KokkosKernels::Impl::read_kokkos_crst_matrix<host_crsmat_t>(filename.c_str()); //in_matrix
    auto  graph_host  = Mtx.graph; // in_graph
    const size_type nrows = graph_host.numRows();

    // Create the known solution and set to all 1's ** on host **
    HostValuesType sol_host("sol_host", nrows);
    Kokkos::deep_copy(sol_host, ONE);

    // Create the rhs ** on host **
    // A*known_sol generates rhs: rhs is dense, use spmv
    HostValuesType rhs_host("rhs_host", nrows);
    KokkosSparse::spmv( "N", ONE, Mtx, sol_host, ZERO, rhs_host);

    // normB
    scalar_t normB = 0.0;
    Kokkos::parallel_reduce( Kokkos::RangePolicy<host_execution_space>(0, rhs_host.extent(0)), 
      KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
        tsum += rhs_host(i)*rhs_host(i);
      }, normB);
    normB = sqrt(normB);

    // normA
    scalar_t normA = 0.0;
    auto row_map_host = graph_host.row_map;
    auto entries_host = graph_host.entries;
    auto values_host  = Mtx.values;
    Kokkos::parallel_reduce( Kokkos::RangePolicy<host_execution_space>(0, Mtx.nnz()), 
      KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
        tsum += values_host(i)*values_host(i);
      }, normA);
    normA = sqrt(normA);

    // Solution to find
    for ( auto test : tests ) {
      std::cout << "\ntest = " << test << std::endl;

      SuperMatrix L;
      SuperMatrix U;
      crsmat_t superluL, superluU;
      KernelHandle khL, khU;
      switch(test) {
        case SUPERNODAL_NAIVE:
        case SUPERNODAL_ETREE:
        {
          Kokkos::Timer timer;
          // callSuperLU on the host    
          int *etree, *perm_r, *perm_c;
          std::cout << " > call SuperLU for factorization" << std::endl;
          factor_superlu<Scalar> (nrows, Mtx.nnz(), values_host.data(), const_cast<int*> (row_map_host.data()), entries_host.data(),
                                  L, U, &perm_r, &perm_c, &etree);
          std::cout << "   Factorization Time: " << timer.seconds() << std::endl << std::endl;

          // read SuperLU factor int crsMatrix on the host (superluMat_host) and copy to default host/device (superluL)
          std::cout << " > Read SuperLU factor into KokkosSparse::CrsMatrix (invert diagonal and copy to device)" << std::endl;
          host_crsmat_t *superluL_host = nullptr;
          host_crsmat_t *superluU_host = nullptr;
          timer.reset();
          superluL = read_superlu_Lfactor<crsmat_t, host_crsmat_t> (nrows, &L, superluL_host);
          std::cout << "   Conversion Time for L: " << timer.seconds() << std::endl;

          timer.reset();
          superluU = read_superlu_Ufactor<crsmat_t, host_crsmat_t> (nrows, &L, &U, superluU_host);
          std::cout << "   Conversion Time for U: " << timer.seconds() << std::endl << std::endl;

          //print_factor_superlu<Scalar> (nrows, &L, &U, perm_r, perm_c);
          //print_crsmat<crsmat_t> (nrows, superluL);
          //print_crsmat<crsmat_t> (nrows, superluU);

          // crsMatrix (storing L-factor) on the default host/device
          auto graphL = superluL.graph; // in_graph
          auto row_mapL = graphL.row_map;
          auto entriesL = graphL.entries;
          auto valuesL  = superluL.values;

          auto graphU = superluU.graph; // in_graph
          auto row_mapU = graphU.row_map;
          auto entriesU = graphU.entries;
          auto valuesU  = superluU.values;

          // create an handle
          if (test == SUPERNODAL_NAIVE) {
            std::cout << " > create handle for SUPERNODAL_NAIVE" << std::endl << std::endl;
            khL.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_NAIVE, nrows, true);
            khU.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_NAIVE, nrows, false);
          } else {
            std::cout << " > create handle for SUPERNODAL_ETREE" << std::endl << std::endl;
            khL.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_ETREE, nrows, true);
            khU.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_ETREE, nrows, false);
          }
          //khL.get_sptrsv_handle ()->print_algorithm ();

          // setup supnodal info
          SCformat *Lstore = (SCformat*)(L.Store);
          int nsuper = 1 + Lstore->nsuper;
          int *supercols = Lstore->sup_to_col;
          khL.set_supernodes (nsuper, supercols, etree);
          khU.set_supernodes (nsuper, supercols, etree);
 
          // Init run to check the error, and also to clear the cache
          // apply forward-pivot on the host
          HostValuesType tmp_host ("temp", nrows);
          forwardP_superlu<Scalar> (nrows, perm_r, 1, rhs_host.data(), nrows, tmp_host.data(), nrows);

          // copy rhs to the default host/device
          ValuesType rhs ("rhs", nrows);
          ValuesType sol ("sol", nrows);
          Kokkos::deep_copy (rhs, tmp_host);

          // ==============================================
          // do L solve
          #if 1
           // symbolic on the host
           timer.reset();
           std::cout << " > Lower-TRI: " << std::endl;
           sptrsv_symbolic (&khL, row_mapL, entriesL);
           std::cout << "   Symbolic Time: " << timer.seconds() << std::endl;

           timer.reset();
           // numeric (only rhs is modified) on the default device/host space
           sptrsv_solve (&khL, row_mapL, entriesL, valuesL, sol, rhs);
          #else
           timer.reset();
           // solveL with Cholmod data structure, L
           solveL_superlu<Scalar>(&L, 1, rhs.data(), nrows);
          #endif
          Kokkos::fence();
          std::cout << "   Solve Time   : " << timer.seconds() << std::endl;
          //Kokkos::deep_copy (tmp_host, rhs);
          //for (int ii=0; ii<nrows; ii++) printf( " y[%d] = %e\n",ii,tmp_host(ii) );
          //printf( "\n" );

          // ==============================================
          // do L^T solve
          #if 1
           // symbolic on the host
           timer.reset ();
           std::cout << " > Upper-TRI: " << std::endl;
           sptrsv_symbolic (&khU, row_mapU, entriesU);
           std::cout << "   Symbolic Time: " << timer.seconds() << std::endl;

           // numeric (only rhs is modified) on the default device/host space
           sptrsv_solve (&khU, row_mapU, entriesU, valuesU, sol, rhs);
          #else
          solveU_superlu<Scalar>(&L, &U, 1, rhs.data(), nrows);
          #endif
          Kokkos::fence ();
          std::cout << "   Solve Time   : " << timer.seconds() << std::endl;
 
          // copy solution to host
          Kokkos::deep_copy(tmp_host, rhs);

          // apply backward-pivot
          backwardP_superlu<Scalar>(nrows, perm_c, 1, tmp_host.data(), nrows, sol_host.data(), nrows);
          //for (int ii=0; ii<nrows; ii++) printf( " x[%d] = %e\n",ii,tmp_host(ii) );


          // ==============================================
          // Error Check ** on host **
          Kokkos::fence();
          // normX
          scalar_t normR = 0.0;
          scalar_t normX = 0.0;
          Kokkos::parallel_reduce( Kokkos::RangePolicy<host_execution_space>(0, sol_host.extent(0)), 
            KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
              tsum += sol_host(i)*sol_host(i);
            }, normX);
          normX = sqrt(normX);

          // normR = ||B - AX||
          KokkosSparse::spmv( "N", -ONE, Mtx, sol_host, ONE, rhs_host);
          Kokkos::parallel_reduce( Kokkos::RangePolicy<host_execution_space>(0, sol_host.extent(0)), 
            KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
              tsum += rhs_host(i) * rhs_host(i);
            }, normR);
          normR = sqrt(normR);

          std::cout << std::endl;
          std::cout << " > check : ||B - AX||/(||B|| + ||A||*||X||) = "
                    << normR << "/(" << normB << " + " << normA << " * " << normX << ") = "
                    << normR/(normB + normA * normX) << std::endl;

          // try again?
          {
            Kokkos::deep_copy(sol_host, ONE);
            KokkosSparse::spmv( "N", ONE, Mtx, sol_host, ZERO, rhs_host);
            forwardP_superlu<Scalar> (nrows, perm_r, 1, rhs_host.data(), nrows, tmp_host.data(), nrows);
            Kokkos::deep_copy (rhs, tmp_host);
             #if 1
             sptrsv_solve (&khL, row_mapL, entriesL, valuesL, sol, rhs);
             sptrsv_solve (&khU, row_mapU, entriesU, valuesU, sol, rhs);
             #else
             solveL_superlu<Scalar>(&L, 1, rhs.data(), nrows);
             solveU_superlu<Scalar>(&L, &U, 1, rhs.data(), nrows);
             #endif
            Kokkos::fence();
            Kokkos::deep_copy(tmp_host, rhs);
            backwardP_superlu<Scalar>(nrows, perm_c, 1, tmp_host.data(), nrows, sol_host.data(), nrows);

            // normX
            Kokkos::parallel_reduce( Kokkos::RangePolicy<host_execution_space>(0, sol_host.extent(0)), 
              KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
                tsum += sol_host(i)*sol_host(i);
              }, normX);
            normX = sqrt(normX);

            // normR = ||B - AX||
            KokkosSparse::spmv( "N", -ONE, Mtx, sol_host, ONE, rhs_host);
            Kokkos::parallel_reduce( Kokkos::RangePolicy<host_execution_space>(0, sol_host.extent(0)), 
              KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
                tsum += rhs_host(i) * rhs_host(i);
              }, normR);
            normR = sqrt(normR);

            std::cout << " > check : ||B - AX||/(||B|| + ||A||*||X||) = "
                      << normR << "/(" << normB << " + " << normA << " * " << normX << ") = "
                      << normR/(normB + normA * normX) << std::endl << std::endl;
          }
          std::cout << std::endl;

          // Benchmark
          // L-solve
          double min_time = 1.0e32;
          double max_time = 0.0;
          double ave_time = 0.0;
          Kokkos::fence();
          for(int i=0;i<loop;i++) {
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
          for(int i=0;i<loop;i++) {
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

        default:
          std::cout << " > Testing only Cholmod < " << std::endl;
          exit(0);
      }
    }
  }
  std::cout << std::endl << std::endl;

  return 0;
}
#endif //KOKKOSKERNELS_ENABLE_TPL_SUPERLU


void print_help_sptrsv() {
  printf("Options:\n");
  printf("  --test [OPTION] : Use different kernel implementations\n");
  printf("                    Options:\n");
  printf("                    superlu-naive, superlu-etree\n\n");
  printf("  -f [file]       : Read in Matrix Market formatted text file 'file'.\n");
  printf("  --offset [O]    : Subtract O from every index.\n");
  printf("                    Useful in case the matrix market file is not 0 based.\n\n");
  printf("  -rpt [K]        : Number of Rows assigned to a thread.\n");
  printf("  -ts [T]         : Number of threads per team.\n");
  printf("  -vl [V]         : Vector-length (i.e. how many Cuda threads are a Kokkos 'thread').\n");
  printf("  --loop [LOOP]   : How many spmv to run to aggregate average time. \n");
}


int main(int argc, char **argv)
{
#ifdef KOKKOSKERNELS_ENABLE_TPL_SUPERLU
  std::vector<int> tests;
  std::string filename;

  int vector_length = -1;
  int team_size = -1;
  int idx_offset = 0;
  int loop = 1;

  if(argc == 1)
  {
    print_help_sptrsv();
    return 0;
  }

  for(int i=0;i<argc;i++)
  {
    if((strcmp(argv[i],"--test")==0)) {
      i++;
      if((strcmp(argv[i],"superlu-naive")==0)) {
        tests.push_back( SUPERNODAL_NAIVE );
      }
      if((strcmp(argv[i],"superlu-etree")==0)) {
        tests.push_back( SUPERNODAL_ETREE );
      }
      continue;
    }
    if((strcmp(argv[i],"-f")==0)) {filename = argv[++i]; continue;}
    if((strcmp(argv[i],"-ts")==0)) {team_size=atoi(argv[++i]); continue;}
    if((strcmp(argv[i],"-vl")==0)) {vector_length=atoi(argv[++i]); continue;}
    if((strcmp(argv[i],"--offset")==0)) {idx_offset=atoi(argv[++i]); continue;}
    if((strcmp(argv[i],"--loop")==0)) {loop=atoi(argv[++i]); continue;}
    if((strcmp(argv[i],"--help")==0) || (strcmp(argv[i],"-h")==0)) {
      print_help_sptrsv();
      return 0;
    }
  }

  if (tests.size() == 0) {
    tests.push_back(DEFAULT);
  }
  std::cout << std::endl;
  for (size_t i = 0; i < tests.size(); ++i) {
    std::cout << "tests[" << i << "] = " << tests[i] << std::endl;
  }

  Kokkos::initialize(argc,argv);
  {
    // Cholmod may not support single, yet
    //int total_errors = test_sptrsv_perf<float>(tests,filename,team_size,vector_length,idx_offset,loop);
    // Kokkos::IO may not read complex?
    //int total_errors = test_sptrsv_perf<Kokkos::complex<double>>(tests,filename,team_size,vector_length,idx_offset,loop);

    int total_errors = test_sptrsv_perf<double>(tests,filename,team_size,vector_length,idx_offset,loop);
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
