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

#ifdef KOKKOSKERNELS_ENABLE_TPL_CHOLMOD
#include "cblas.h"
#include "lapacke.h"
#include "cholmod.h"


template<typename Scalar>
void print_factor_cholmod(cholmod_factor *L, cholmod_common *cm) {

  Scalar *Lx;
  int *mb, *colptr, *rowind, *nb;
  int nsuper, j1, j2, i1, i2, psx, nsrow, nscol, i, ii, jj, s,
      nsrow2, ps2;

  /* ---------------------------------------------------------------------- */
  /* get inputs */
  /* ---------------------------------------------------------------------- */

  nsuper = L->nsuper;      // # of supernodal columns
  mb = (int*)(L->pi);      // mb[s+1] - mb[s] = total number of rows in all the s-th supernodes (diagonal+off-diagonal)
  nb = (int*)(L->super);
  colptr = (int*)(L->px);
  rowind = (int*)(L->s);               // rowind
  Lx = (Scalar*)(L->x);                // data

  printf( " >> print factor(n=%ld, nsuper=%d) <<\n",L->n,nsuper );
  for (s = 0 ; s < nsuper ; s++) {
    j1 = nb [s];
    j2 = nb [s+1];
    nscol = j2 - j1;      // number of columns in the s-th supernode column
    printf( " nb[%d] = %d\n",s,nscol );
  }
  for (s = 0 ; s < nsuper ; s++) {
    j1 = nb [s];
    j2 = nb [s+1];
    nscol = j2 - j1;      // number of columns in the s-th supernode column

    i1 = mb [s];
    i2 = mb [s+1];
    nsrow  = i2 - i1;    // "total" number of rows in all the supernodes (diagonal+off-diagonal)
    nsrow2 = nsrow - nscol;  // "total" number of rows in all the off-diagonal supernodes
    ps2    = i1 + nscol;    // offset into rowind

    psx = colptr [s];           // offset into data,   Lx[s][s]

    /* print diagonal block */
    for (ii = 0; ii < nscol; ii++) {
      for (jj = 0; jj <= ii; jj++)
        std::cout << j1+ii+1 << " " << j1+jj+1 << " " <<  Lx[psx + (ii + jj*nsrow)] << std::endl;
    }

    /* print off-diagonal blocks */
    for (ii = 0; ii < nsrow2; ii++) {
      i = rowind [ps2 + ii] ;
      for (jj = 0; jj < nscol; jj++)
        std::cout << i+1 << " " << j1+jj+1 << " " << Lx[psx + (nscol+ii + jj*nsrow)] << std::endl;
    }
  }
}


template <typename crsMat_t>
void print_factor_cholmod(crsMat_t *L) {
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename crsMat_t::values_type::non_const_type values_view_t;
  typedef typename values_view_t::value_type scalar_t;

  graph_t  graph = L->graph;
  const int      *colptr = graph.row_map.data ();
  const int      *rowind = graph.entries.data ();
  const scalar_t *Lx     = L->values.data ();

  printf( "\n -- print cholmod factor in crs (numCols = %d) --\n",L->numCols () );
  for (int j = 0; j < L->numCols (); j++) {
    for (int k = colptr[j]; k < colptr[j+1]; k++) {
      std::cout << rowind[k] << " " <<  j << " " << Lx[k] << std::endl;
    }
  }
}

/* ========================================================================================= */
template <typename crsMat_t, typename hostMat_t>
crsMat_t read_cholmod_factor(cholmod_factor *L, cholmod_common *cm) {

  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type   cols_view_t;
  typedef typename crsMat_t::values_type::non_const_type values_view_t;

  typedef typename values_view_t::value_type scalar_t;
  typedef Kokkos::Details::ArithTraits<scalar_t> STS;

  /* ---------------------------------------------------------------------- */
  /* get inputs */
  /* ---------------------------------------------------------------------- */
  int nsuper = L->nsuper;     // # of supernodal columns
  int *mb = (int*)(L->pi);    // mb[s+1] - mb[s] = total number of rows in all the s-th supernodes (diagonal+off-diagonal)
  int *nb = (int*)(L->super);
  int *colptr = (int*)(L->px);      // colptr
  int *rowind = (int*)(L->s);       // rowind
  scalar_t *Lx = (scalar_t*)(L->x); // data

  int n = L->n;
  int nnzA = colptr[nsuper] - colptr[0];
  row_map_view_t rowmap_view ("rowmap_view", n+1);
  cols_view_t    column_view ("colmap_view", nnzA);
  values_view_t  values_view ("values_view", nnzA);

  typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view (rowmap_view);
  typename cols_view_t::HostMirror    hc = Kokkos::create_mirror_view (column_view);
  typename values_view_t::HostMirror  hv = Kokkos::create_mirror_view (values_view);

  int j1, j2, i1, i2, psx, nsrow, nscol, i, j, ii, jj, s, nsrow2, ps2;

  // compute offset for each row
  //printf( "\n -- store cholmod factor --\n" );
  j = 0;
  hr(j) = 0;
  for (s = 0 ; s < nsuper ; s++) {
    j1 = nb[s];
    j2 = nb[s+1];
    nscol = j2 - j1;      // number of columns in the s-th supernode column

    i1 = mb[s];
    i2 = mb[s+1];
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

    i1 = mb[s] ;
    i2 = mb[s+1] ;
    nsrow  = i2 - i1;    // "total" number of rows in all the supernodes (diagonal+off-diagonal)
    nsrow2 = nsrow - nscol;  // "total" number of rows in all the off-diagonal supernodes
    ps2    = i1 + nscol;     // offset into rowind

    psx = colptr[s] ;        // offset into data,   Lx[s][s]

    /* diagonal block */
    // for each column (or row due to symmetry), the diagonal supernodal block is stored (in ascending order of row indexes) first
    // so that we can do TRSM on the diagonal block
    #define CHOLMOD_INVERT_DIAG
    #ifdef CHOLMOD_INVERT_DIAG
    LAPACKE_dtrtri(LAPACK_COL_MAJOR,
                   'L', 'N', nscol, &Lx[psx], nsrow);
    #endif
    //printf( "nscol=%d, nsrow=%d\n",nscol,nsrow );
    for (ii = 0; ii < nscol; ii++) {
      // lower-triangular part
      for (jj = 0; jj <= ii; jj++) {
        //printf("1 %d %d %d %.16e %d\n",hr(j1+jj), j1+ii, j1+jj, Lx[psx + (ii + jj*nsrow)], hr(j1+ii));
        hc(hr(j1+jj)) = j1+ii;
        hv(hr(j1+jj)) = Lx[psx + (ii + jj*nsrow)];
        hr(j1+jj) ++;
      }
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
  std::cout << "    * Matrix size = " << n << std::endl;
  std::cout << "    * Total nnz   = " << hr (n) << std::endl;
  std::cout << "    * nnz / n     = " << hr (n)/n << std::endl;

  // deepcopy
  Kokkos::deep_copy (rowmap_view, hr);
  Kokkos::deep_copy (column_view, hc);
  Kokkos::deep_copy (values_view, hv);

  // create crs
  graph_t static_graph (column_view, rowmap_view);
  crsMat_t crsmat("CrsMatrix", n, values_view, static_graph);
  return crsmat;
}

template <typename crsMat_t, typename Scalar>
void solveL_cholmod(int nsuper, int *supptr, crsMat_t *L, int nrhs, Scalar *X, int ldx) {

  typedef typename crsMat_t::StaticCrsGraphType graph_t;

  Scalar *work;
  Scalar zero, one;
  int j1, j2, i1, i2, psx, nsrow, nscol, ii, s,
      nsrow2, ps2, j, i, ldw;

  /* ---------------------------------------------------------------------- */
  /* get inputs */
  /* ---------------------------------------------------------------------- */

  ldw = L->numCols ();
  work = new Scalar [nrhs*ldw];

  graph_t  graph = L->graph;
  const int    *colptr = graph.row_map.data ();
  const int    *rowind = graph.entries.data ();
  const Scalar *Lx     = L->values.data ();

  zero = 0.0;
  one  = 1.0;

  for (s = 0 ; s < nsuper ; s++) {
    j1 = supptr [s];
    j2 = supptr [s+1];
    nscol = j2 - j1;      // number of columns in the s-th supernode column

    i1  = colptr [j1];
    i2  = colptr [j1+1];
    nsrow  = i2 - i1;        // "total" number of rows in all the supernodes (diagonal+off-diagonal)
    nsrow2 = nsrow - nscol;  // "total" number of rows in all the off-diagonal supernodes
    ps2    = i1 + nscol;     // offset into rowind 

    psx = colptr [j1];        // offset into data,   Lx[s][s]

    printf( " nscol=%d, nsrows=%d (j1=%d, j2=%d), (i1=%d, i2=%d), psx=%d\n",nscol,nsrow, j1,j2, i1,i2, psx );
    /* TRSM with diagonal block */
    #ifdef CHOLMOD_INVERT_DIAG
    cblas_dtrmm (CblasColMajor,
        CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
        nscol, nrhs,
        one,  &Lx[psx], nsrow,
              &X[j1],   ldx);
    #else
    cblas_dtrsm (CblasColMajor,
        CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
        nscol, nrhs,
        one,  &Lx[psx], nsrow,
              &X[j1],   ldx);
    #endif
    for (ii=0; ii < nscol; ii++) printf( "%d %e\n",j1+ii,X[j1+ii]);

    /* GEMM to update with off diagonal blocks */
    if (nsrow2 > 0) {
      cblas_dgemm (CblasColMajor,
          CblasNoTrans, CblasNoTrans,
          nsrow2, nrhs, nscol,
          one,   &Lx[psx + nscol], nsrow,
                 &X[j1], ldx,
          zero,  work, ldw);
    }

    /* scatter vectors back into X */
    for (ii = 0 ; ii < nsrow2 ; ii++) {
      i = rowind [ps2 + ii];
      for (j = 0 ; j < nrhs ; j++) {
        printf( " X(%d) = %e - %e = %e\n",i,X[i],work[ii],X[i]-work[ii] );
        X [i + j*ldx] -= work [ii + j*ldw];
      }
    }
  }
  delete[] work;
}

/* ========================================================================================= */
void compute_etree_cholmod(cholmod_sparse *A, cholmod_common *cm, int **etree) {
  cholmod_factor *L;
  L = cholmod_analyze (A, cm);

  int n = L->n;
  int nsuper = L->nsuper;      // # of supernodal columns
  int *Iwork = (int*)(cm->Iwork);
  int *Parent = Iwork + (2*((size_t) n)); /* size nfsuper <= n [ */

  *etree = new int [nsuper];
  for (int ii = 0 ; ii < nsuper; ii++) (*etree)[ii] = Parent[ii];
}

/* ========================================================================================= */
template<typename Scalar>
cholmod_factor* factor_cholmod(const int nrow, const int nnz, Scalar *nzvals, int *rowptr, int *colind, cholmod_common *Comm, int **etree) {

  // Start Cholmod
  cholmod_common *cm = Comm;
  cholmod_start (cm);
  cm->supernodal = CHOLMOD_SUPERNODAL;

  // Manually, initialize the matrix
  cholmod_sparse A;
  A.stype = 1;   // symmetric
  A.sorted = 0;
  A.packed = 1;
  A.itype = CHOLMOD_INT;
  A.xtype = CHOLMOD_REAL;
  A.dtype = CHOLMOD_DOUBLE;

  A.nrow = nrow;
  A.ncol = nrow;
  A.nzmax = nnz;

  A.p = rowptr;
  A.x = nzvals;
  A.i = colind;

  // Symbolic factorization
  cholmod_factor *L;
  L = cholmod_analyze (&A, cm);
  if (cm->status != CHOLMOD_OK) {
    printf( " ** cholmod_analyze returned with status = %d **",cm->status );
  }

  // Numerical factorization
  if (!cholmod_factorize (&A, L, cm)) {
    printf( " ** cholmod_factorize returned FALSE **\n" );
  }
  if (cm->status != CHOLMOD_OK) {
    printf( " ** cholmod_factorize returned with status = %d, minor = %ld **",cm->status,L->minor );
    int i;
    int *Perm = (int*)(L->Perm);
    for (i = 0; i < (int)(L->n); i++) printf( "%d %d\n",i,Perm[i] );
  }
  switch (cm->selected) {
    case CHOLMOD_NATURAL: printf( "  > NATURAL ordering (%d)\n", CHOLMOD_NATURAL ); break;
    case CHOLMOD_AMD:     printf( "  > AMD ordering (%d)\n",     CHOLMOD_AMD     ); break;
    case CHOLMOD_METIS:   printf( "  > METIS ordering (%d)\n",   CHOLMOD_METIS   ); break;
    case CHOLMOD_NESDIS:  printf( "  > NESDIS ordering (%d)\n",  CHOLMOD_NESDIS  ); break;
  }
  //int *Perm = (int*)(L->Perm);
  //for (int i = 0; i < (int)(L->n); i++) printf( "%d\n",Perm[i] );
  //print_factor_cholmod<Scalar>(L, cm);
  compute_etree_cholmod(&A, cm, etree);

  return L;
}

/* ========================================================================================= */
template<typename Scalar>
void forwardP_cholmod(cholmod_factor *L, int nrhs, Scalar *B, int ldb, Scalar *X, int ldx) {

  int i;
  int *Perm = (int*)(L->Perm);
  for (i = 0; i < (int)(L->n); i++) {
    X[i] = B[Perm[i]];
  }
}

template<typename Scalar>
void solveL_cholmod(cholmod_factor *L, int nrhs, Scalar *X, int ldx) {

  Scalar *Lx, *work;
  Scalar zero, one;
  int *mb, *colptr, *rowind, *nb;
  int nsuper, j1, j2, i1, i2, psx, nsrow, nscol, ii, s,
      nsrow2, ps2, j, i, ldw;

  /* ---------------------------------------------------------------------- */
  /* get inputs */
  /* ---------------------------------------------------------------------- */

  ldw = L->maxesize;
  work = new Scalar [nrhs*ldw];

  nsuper = L->nsuper;      // # of supernodal columns
  mb = (int*)(L->pi);      // mb[s+1] - mb[s] = total number of rows in all the s-th supernodes (diagonal+off-diagonal)
  nb = (int*)(L->super);   // nb[s+1] - nb[s] = total number of columns in the s-th supernodes

  colptr = (int*)(L->px);
  rowind = (int*)(L->s);   // rowind
  Lx = (Scalar*)(L->x);    // data

  zero = 0.0;
  one  = 1.0;

  for (s = 0 ; s < nsuper ; s++) {
    j1 = nb [s];
    j2 = nb [s+1];
    nscol = j2 - j1;      // number of columns in the s-th supernode column

    i1  = mb [s];
    i2  = mb [s+1];
    nsrow  = i2 - i1;        // "total" number of rows in all the supernodes (diagonal+off-diagonal)
    nsrow2 = nsrow - nscol;  // "total" number of rows in all the off-diagonal supernodes
    ps2    = i1 + nscol;     // offset into rowind 

    psx = colptr [s];        // offset into data,   Lx[s][s]

    /* TRSM with diagonal block */
    #ifdef CHOLMOD_INVERT_DIAG
    cblas_dtrmm (CblasColMajor,
        CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
        nscol, nrhs,
        one,   &Lx[psx], nsrow,
               &X[j1],   ldx);
    #else
    cblas_dtrsm (CblasColMajor,
        CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
        nscol, nrhs,
        one,   &Lx[psx], nsrow,
               &X[j1],   ldx);
    #endif

    /* GEMM to update with off diagonal blocks */
    if (nsrow2 > 0) {
      cblas_dgemm (CblasColMajor,
          CblasNoTrans, CblasNoTrans,
          nsrow2, nrhs, nscol,
          one,   &Lx[psx + nscol], nsrow,
                 &X[j1], ldx,
          zero,  work, ldw);
    }

    /* scatter vectors back into X */
    for (ii = 0 ; ii < nsrow2 ; ii++) {
      i = rowind [ps2 + ii];
      for (j = 0 ; j < nrhs ; j++) {
        X [i + j*ldx] -= work [ii + j*ldw];
      }
    }
  }
  delete[] work;
}

template<typename Scalar>
void solveU_cholmod(cholmod_factor *L, int nrhs, Scalar *B, int ldb) {

  Scalar *Lx, *work ;
  Scalar one, mone ;
  int *mb, *colptr, *rowind, *nb ;
  int nsuper, j1, j2, i1, i2, psx, nsrow, nscol, ii, s,
      nsrow2, ps2, j, i, ldw;

  /* ---------------------------------------------------------------------- */
  /* get inputs */
  /* ---------------------------------------------------------------------- */

  ldw = L->maxesize;
  work = new Scalar [nrhs*ldw];

  nsuper = L->nsuper ;      // # of supernodal columns
  mb = (int*)(L->pi) ;      // mb[s+1] - mb[s] = total number of rows in all the s-th supernodes (diagonal+off-diagonal)
  nb = (int*)(L->super) ;   // nb[s+1] - nb[s] = total number of columns in the s-th supernodes

  colptr = (int*)(L->px) ;
  rowind = (int*)(L->s) ;           // rowind
  Lx = (Scalar*)(L->x) ;            // data

  one  =  1.0 ;
  mone = -1.0 ;

  for (s = nsuper-1 ; s >= 0 ; s--)
  {
    j1 = nb [s] ;
    j2 = nb [s+1] ;
    nscol = j2 - j1 ;      // number of columns in the s-th supernode column

    i1  = mb [s] ;
    i2  = mb [s+1] ;
    nsrow  = i2 - i1 ;        // "total" number of rows in all the supernodes (diagonal+off-diagonal)
    nsrow2 = nsrow - nscol ;  // "total" number of rows in all the off-diagonal supernodes
    ps2    = i1 + nscol ;     // offset into rowind 

    psx = colptr [s] ;           // offset into data,   Lx[s][s]

    /* gather X into workspace */
    for (ii = 0 ; ii < nsrow2 ; ii++)
    {   
      i = rowind [ps2 + ii] ;
      for (j = 0 ; j < nrhs ; j++)
      {   
        work [ii + j*ldw] = B [i + j*ldb];
      }   
    }   

    /* GEMM to update with off diagonal blocks */
    if (nsrow2 > 0)
    {   
      cblas_dgemm (CblasColMajor,
          CblasTrans, CblasNoTrans,
          nscol, nrhs, nsrow2,
          mone, &Lx[psx + nscol], nsrow,
                 work, ldw,
          one,  &B[j1], ldb);
    }   

    /* TRSM with diagonal block */
    #ifdef CHOLMOD_INVERT_DIAG
    cblas_dtrmm (CblasColMajor,
        CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
        nscol, nrhs,
        one,   &Lx[psx], nsrow,
               &B[j1],   ldb) ;
    #else
    cblas_dtrsm (CblasColMajor,
        CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
        nscol, nrhs,
        one,   &Lx[psx], nsrow,
               &B[j1],   ldb) ;
    #endif
  }
  delete[] work;
}

template<typename Scalar>
void backwardP_cholmod(cholmod_factor *L, int nrhs, Scalar *B, int ldb, Scalar *X, int ldx) {

    int i;
    int *Perm = (int*)(L->Perm) ;
    for (i = 0; i < (int)(L->n); i++)
    {
        X[Perm[i]] = B[i];
    }
}
#endif //  KOKKOSKERNELS_ENABLE_TPL_CHOLMOD

using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

enum {CUSPARSE, SUPERNODAL_NAIVE, SUPERNODAL_ETREE};


#ifdef KOKKOSKERNELS_ENABLE_TPL_CHOLMOD
template<typename Scalar>
int test_sptrsv_perf(std::vector<int> tests, std::string& filename, int loop) {

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
    std::cout << " CHOLMOD Tester Begin: Read matrix filename " << filename << std::endl;
    host_crsmat_t Mtx = KokkosKernels::Impl::read_kokkos_crst_matrix<host_crsmat_t>(filename.c_str()); //in_matrix
    auto  graph_host  = Mtx.graph; // in_graph
    const size_type nrows = graph_host.numRows();
    //print_factor_cholmod(&Mtx);

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

      cholmod_common cm;
      cholmod_factor *L = NULL;
      crsmat_t cholmodMtx;
      KernelHandle khL, khU;
      switch(test) {
        case SUPERNODAL_NAIVE:
        case SUPERNODAL_ETREE:
        {
          Kokkos::Timer timer;
          // call CHOLMOD on the host    
          int *etree;
          timer.reset();
          std::cout << " > call CHOLMOD for factorization" << std::endl;
          L = factor_cholmod<Scalar> (nrows, Mtx.nnz(), values_host.data(), const_cast<int*> (row_map_host.data()), entries_host.data(),
                                      &cm, &etree);
          std::cout << "   Factorization Time: " << timer.seconds() << std::endl << std::endl;

          // read CHOLMOD factor int crsMatrix on the host (cholmodMat_host) and copy to default host/device (cholmodMtx)
          timer.reset();
          std::cout << " > Read Cholmod factor into KokkosSparse::CrsMatrix (invert diagonabl, and copy to device) " << std::endl;
          cholmodMtx = read_cholmod_factor<crsmat_t, host_crsmat_t> (L, &cm);
          std::cout << "   Conversion Time: " << timer.seconds() << std::endl << std::endl;
          //print_factor_cholmod (&cholmodMtx_host);

          // crsMatrix (storing L-factor) on the default host/device
          auto graph = cholmodMtx.graph; // in_graph
          auto row_map = graph.row_map;
          auto entries = graph.entries;
          auto values  = cholmodMtx.values;

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
          int nsuper = (int)(L->nsuper);
          int *supercols = (int*)(L->super);
          khL.set_supernodes (nsuper, supercols, etree);
          khU.set_supernodes (nsuper, supercols, etree);
 
          // Init run to check the error, and also to clear the cache
          // apply forward-pivot on the host
          HostValuesType tmp_host ("temp", nrows);
          forwardP_cholmod<Scalar> (L, 1, rhs_host.data(), nrows, tmp_host.data(), nrows);

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
           sptrsv_symbolic (&khL, row_map, entries);
           std::cout << "   Symbolic Time: " << timer.seconds() << std::endl;

           timer.reset();
           // numeric (only rhs is modified) on the default device/host space
           sptrsv_solve (&khL, row_map, entries, values, sol, rhs);
          #else
           timer.reset();
           // solveL with Kokkos' csr version (read from Cholmod structure)
           solveL_cholmod<crsmat_t, Scalar>((int)(L->nsuper), (int*)(L->super), &cholmodMtx, 1, rhs.data(), nrows);
           // solveL with Cholmod data structure, L
           //solveL_cholmod<Scalar>(L, 1, rhs.data(), nrows);
          #endif
          Kokkos::fence();
          std::cout << "   Solve Time   : " << timer.seconds() << std::endl;
          //Kokkos::deep_copy (tmp_host, rhs);
          //for (int ii=0; ii<nrows; ii++) printf( " %d %e\n",ii,tmp_host(ii) );
          //printf( "\n" );

          // ==============================================
          // do L^T solve
          #if 1
           // symbolic on the host
           timer.reset ();
           std::cout << " > Upper-TRI: " << std::endl;
           sptrsv_symbolic (&khU, row_map, entries);
           std::cout << "   Symbolic Time: " << timer.seconds() << std::endl;

           // numeric (only rhs is modified) on the default device/host space
           sptrsv_solve (&khU, row_map, entries, values, sol, rhs);
          #else
          solveU_cholmod<Scalar>(L, 1, rhs.data(), nrows);
          #endif
          Kokkos::fence ();
          std::cout << "   Solve Time   : " << timer.seconds() << std::endl;
 
          // copy solution to host
          Kokkos::deep_copy(tmp_host, rhs);

          // apply backward-pivot
          backwardP_cholmod<Scalar>(L, 1, tmp_host.data(), nrows, sol_host.data(), nrows);
          //for (int ii=0; ii<nrows; ii++) printf( " %d %e\n",ii,tmp_host(ii) );


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
            forwardP_cholmod<Scalar> (L, 1, rhs_host.data(), nrows, tmp_host.data(), nrows);
            Kokkos::deep_copy (rhs, tmp_host);
             sptrsv_solve (&khL, row_map, entries, values, sol, rhs);
             sptrsv_solve (&khU, row_map, entries, values, sol, rhs);
            Kokkos::fence();
            Kokkos::deep_copy(tmp_host, rhs);
            backwardP_cholmod<Scalar>(L, 1, tmp_host.data(), nrows, sol_host.data(), nrows);

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
            sptrsv_solve (&khL, row_map, entries, values, sol, rhs);
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
            sptrsv_solve (&khU, row_map, entries, values, sol, rhs);
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
#endif //  KOKKOSKERNELS_ENABLE_TPL_CHOLMOD


void print_help_sptrsv() {
  printf("Options:\n");
  printf("  --test [OPTION] : Use different kernel implementations\n");
  printf("                    Options:\n");
  printf("                    cholmod_naive, cholmod_etree\n\n");
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
#ifdef KOKKOSKERNELS_ENABLE_TPL_CHOLMOD
  std::vector<int> tests;
  std::string filename;

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
      if((strcmp(argv[i],"cholmod-naive")==0)) {
        tests.push_back( SUPERNODAL_NAIVE );
      }
      if((strcmp(argv[i],"cholmod-etree")==0)) {
        tests.push_back( SUPERNODAL_ETREE );
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
    // Cholmod may not support single, yet
    //int total_errors = test_sptrsv_perf<float>(tests, filename, loop);
    // Kokkos::IO may not read complex?
    //int total_errors = test_sptrsv_perf<Kokkos::complex<double>>(tests, filename, loop);

    int total_errors = test_sptrsv_perf<double>(tests, filename, loop);
    if(total_errors == 0)
      printf("Kokkos::SPTRSV Test: Passed\n\n");
    else
      printf("Kokkos::SPTRSV Test: Failed\n\n");
  }
  Kokkos::finalize();
#else
  std::cout << "CHOLMOD NOT ENABLED:" << std::endl;
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
