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

#ifdef KOKKOSKERNELS_ENABLE_TPL_METIS
#include "metis.h"
#endif

#define SUPERLU_INVERT_DIAG
#define SUPERLU_MERGE_SUPERNODES

#ifdef SUPERLU_MERGE_SUPERNODES
class sort_indices {
   public:
     sort_indices(int* rowinds) : rowinds_(rowinds){}
     bool operator()(int i, int j) const { return rowinds_[i] < rowinds_[j]; }
   private:
     int* rowinds_; // rowindices
};


template <typename crsMat_host_t, typename crsMat_t>
crsMat_t merge_supernodes(int n, int *p_nsuper, int *nb, int *mb,
                          bool unit_diag, crsMat_host_t &superluL, crsMat_host_t &superluU,
                          int *etree) {

  auto graphL = superluL.graph; // in_graph
  auto row_mapL = graphL.row_map;
  auto entriesL = graphL.entries;
  auto valuesL  = superluL.values;

  auto graphU = superluU.graph; // in_graph
  auto row_mapU = graphU.row_map;
  auto entriesU = graphU.entries;
  auto valuesU  = superluU.values;

  // ---------------------------------------------------------------
  int nsuper = *p_nsuper;
  int min_nsrow = 0, max_nsrow = 0, tot_nsrow = 0;
  int min_nscol = 0, max_nscol = 0, tot_nscol = 0;
  for (int s = 0; s <nsuper; s++) {
    //printf( " mb[%d]=%d, nb[%d]=%d, etree[%d]=%d (nrow=%d, ncol=%d)\n",s,mb[s],s,nb[s],s,etree[s],mb[s+1]-mb[s],nb[s+1]-nb[s] );
    int j1 = nb[s];
    int j2 = nb[s+1];

    int nscol = j2 - j1;
    int nsrow = mb[j1+1] - mb[j1];

    if (s == 0) {
      min_nscol = max_nscol = tot_nscol = nscol;
      min_nsrow = max_nsrow = tot_nsrow = nsrow;
    } else {
      if (min_nsrow > nsrow) {
        min_nsrow = nsrow;
      }
      if (max_nsrow < nsrow) {
        max_nsrow = nsrow;
      }
      tot_nsrow += nsrow;

      if (min_nscol > nscol) {
        min_nscol = nscol;
      }
      if (max_nscol < nscol) {
        max_nscol = nscol;
      }
      tot_nscol += nscol;
    }
  }
  printf( "\n ------------------------------------- \n" );
  printf( " Original structure:\n" );
  printf( "  + nsuper = %d\n",nsuper );
  printf( "  > nsrow: min = %d, max = %d, avg = %d\n",min_nsrow,max_nsrow,tot_nsrow/nsuper );
  printf( "  > nscol: min = %d, max = %d, avg = %d\n",min_nscol,max_nscol,tot_nscol/nsuper );

  // ---------------------------------------------------------------
  // looking for supernodes to merge (i.e., dense diagonal blocks)
  double tol = 0.8;
  int nsuper2 = 0;
  // map the first supernode
  int *map = new int[nsuper]; // map old to new supernodes
  map[0] = 0;
  for (int s = 0; s < nsuper-1; s++) {
    //printf( " -- s = %d->%d --\n",s,nsuper2 );
    // checking if the next supernode can be merged
    int s2 = s;
    bool merged = false;
    do {
      int nnzL = 0;
      int nnzU = 0;

      int nscol1 = nb[s2+1] - nb[s];    // size of the current merged supernode
      int nscol2 = nb[s2+2] - nb[s2+1]; // size of the current supernode
      int nssize = nscol1 * nscol2;
      for (int t = s; t <= s2; t++ ) {
        //int nscol = nb[s2+1] - nb[t];   // size of the current supernode
        for (int j = nb[t]; j < nb[t+1]; j++) {
          // counting nnzL
          for (int k = row_mapL[j]; k < row_mapL[j+1]; k++) {
            if (entriesL[k] >= nb[s2+1]) {
              if (entriesL[k] < nb[s2+2]) {
                nnzL ++;
              } else {
                break;
              }
            }
          }
          // counting nnzU
          for (int k = row_mapU[j]; k < row_mapU[j+1]; k++) {
            if (entriesU[k] >= nb[s2+1]) {
              if (entriesU[k] < nb[s2+2]) {
                nnzU ++;
              } else {
                break;
              }
            }
          }
        }
      }
      merged = (nnzL > tol*nssize && nnzU > tol*nssize);
      if (merged) {
        //printf( "  >> merge s2+1=%d(%dx%d, row=%d:%d) with s=%d(%dx%d) (%dx%d: %d,%d, %.2e,%.2e) <<\n",
        //              s2+1,nb[s2+2]-nb[s2+1],nb[s2+2]-nb[s2+1],nb[s2+1],nb[s2+2]-1, s,nb[s+1]-nb[s],nb[s+1]-nb[s],
        //              nscol1,nscol2, nnzL,nnzU,((double)nnzL)/((double)nssize),((double)nnzU)/((double)nssize) );
        map[s2+1] = nsuper2;
        s2 ++;
      } else {
        //printf( "  -- not merge s2+1=%d(%dx%d, row=%d:%d) with s=%d(%dx%d) (%dx%d: %d,%d, %.2e,%.2e) --\n",
        //           s2+1,nb[s2+2]-nb[s2+1],nb[s2+2]-nb[s2+1],nb[s2+1],nb[s2+2]-1, s,nb[s+1]-nb[s],nb[s+1]-nb[s],
        //           nscol1,nscol2, nnzL,nnzU,((double)nnzL)/((double)nssize),((double)nnzU)/((double)nssize) );
        map[s2+1] = nsuper2+1;
      }
    } while (merged && s2 < nsuper-1);
    s = s2;
    nsuper2 ++;
  }
  nsuper2 = map[nsuper-1]+1;
  //printf( " nsuper2 = %d\n",nsuper2 );
  //printf( " map:\n" );
  //for (int s = 0; s < nsuper; s++) printf( "   %d %d\n",s,map[s] );

  // ----------------------------------------------------------
  // make sure each of the merged supernodes has the same parent in the etree
  int nsuper3 = 0;
  int *map2 = new int[nsuper]; // map old to new supernodes
  for (int s2 = 0, s = 0; s2 < nsuper2; s2++) {
    // look for parent of the first supernode
    int s3 = s;
    while (etree[s3] != -1 && map[etree[s3]] == map[s3]) {
      s3 ++;
    }
    map2[s] = nsuper3;
    int p = (etree[s3] == -1 ? -1 : map[etree[s3]]);

    // go through the rest of the supernode in this merged supernode
    s++;
    while (s < nsuper && map[s] == s2) {
      int q = (etree[s3] == -1 ? -1 : map[etree[s3]]);
      while (etree[s3] != -1 && map[etree[s3]] == map[s3]) {
        s3 ++;
        q = (etree[s3] == -1 ? -1 : map[etree[s3]]);
      }
       
      if (q != p) {
        p = q;
        nsuper3 ++;
      }
      map2[s] = nsuper3;
      s ++;
    }
    nsuper3 ++;
  }
  delete[] map;
  //printf( " nsuper3 = %d\n",nsuper3 );
  //printf( " map:\n" );
  //for (int s = 0; s < nsuper; s++) printf( "   %d %d\n",s,map2[s] );

  // ----------------------------------------------------------
  // now let me find nsrow for the merged supernode
  int nnzA = 0;
  int *nb2 = new int[nsuper3];
  int *mb2 = new int[nsuper3];
  int *work1 = new int[n];
  int *work2 = new int[n];
  int **rowind = new int*[nsuper3];
  for (int i = 0; i < n; i++) {
    work1[i] = 0;
  }
  for (int s2 = 0, s = 0; s2 < nsuper3; s2++) {
    nb2[s2] = 0;
    mb2[s2] = 0;
    // merging supernodal rows
    while(s < nsuper && map2[s] == s2) {
      int j1 = nb[s];
      for (int k = row_mapL[j1]; k < row_mapL[j1+1]; k++) {
        if(work1[entriesL[k]] == 0) {
          work1[entriesL[k]] = 1;
          work2[mb2[s2]] = entriesL[k];
          mb2[s2] ++;
        }
      }
      nb2[s2] += (nb[s+1]-nb[s]);
      s ++;
    }
    // sort such that diagonal come on the top
    std::sort(work2, work2+mb2[s2]);

    // save nonzero row ids
    rowind[s2] = new int [mb2[s2]];
    for (int k = 0; k < mb2[s2]; k++) {
      rowind[s2][k] = work2[k];
      work1[work2[k]] = 0;
    }
    nnzA += nb2[s2] * mb2[s2];
  }
  delete[] work1;
  delete[] work2;

  // ----------------------------------------------------------
  // construct new etree
  int *etree2 = new int[nsuper3];
  for (int s = 0; s < nsuper; s++) {
    // etree
    int s2 = map2[s];
    int p = (etree[s] == -1 ? -1 : map2[etree[s]]);
    if (p != s2) {
      etree2[s2] = p;
    }
  }

  // ----------------------------------------------------------
  // now let's create crs
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type   cols_view_t;
  typedef typename crsMat_t::values_type::non_const_type values_view_t;

  typedef typename values_view_t::value_type scalar_t;
  typedef Kokkos::Details::ArithTraits<scalar_t> STS;

  row_map_view_t rowmap_view ("rowmap_view", n+1);
  cols_view_t    column_view ("colmap_view", nnzA);
  values_view_t  values_view ("values_view", nnzA);

  typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view (rowmap_view);
  typename cols_view_t::HostMirror    hc = Kokkos::create_mirror_view (column_view);
  typename values_view_t::HostMirror  hv = Kokkos::create_mirror_view (values_view);
  double *dwork = new double[n];
  for (int i = 0; i < n; i++) {
    dwork[i] = 0.0;
  }

  nnzA = 0;
  hr(0) = 0;
  for (int s2 = 0, s = 0; s2 < nsuper3; s2++) {
    #if defined(SUPERLU_INVERT_DIAG)
    int nnzD = nnzA;
    #endif
    while(s < nsuper && map2[s] == s2) {
      for (int j = nb[s]; j < nb[s+1]; j++) {
        for (int k = row_mapL[j]; k < row_mapL[j+1]; k++) {
          dwork[entriesL[k]] = valuesL[k];
        }
        for (int k = 0; k < mb2[s2]; k++) {
          hc(nnzA) = rowind[s2][k];
          hv(nnzA) = dwork[rowind[s2][k]];
          nnzA ++;
        }
        for (int k = row_mapL[j]; k < row_mapL[j+1]; k++) {
          dwork[entriesL[k]] = STS::zero ();
        }
        hr(j+1) = nnzA;
      }
      s++;
    }
    delete[] rowind[s2];

    #if defined(SUPERLU_INVERT_DIAG)
    int nscol = nb2[s2];
    int nsrow = mb2[s2];
    if (unit_diag) {
      LAPACKE_dtrtri(LAPACK_COL_MAJOR,
                     'L', 'U', nscol, &hv(nnzD), nsrow);
    } else {
      LAPACKE_dtrtri(LAPACK_COL_MAJOR,
                     'L', 'N', nscol, &hv(nnzD), nsrow);
    }
    #endif
  }
  delete[] map2;
  delete[] dwork;
  delete[] rowind;
  /*for (int j = 0; j < n; j++) {
    for (int k = hr(j); k < hr(j+1); k++) {
      printf( "%d %d %.16e\n",hc(k),j,hv(k) );
    }
  }*/
  //for (int s = 0; s <nsuper3; s++) {
  //  printf( " mb[%d]=%d, nb[%d]=%d, etree[%d]=%d\n",s,mb2[s],s,nb2[s],s,etree2[s] );
  //}
  for (int s = 0; s <nsuper3; s++) {
    // copy supernode id to column id
    nb[s+1] = nb[s]+nb2[s];

    int j1 = nb[s];
    int j2 = nb[s+1];
    for (int j = j1; j < j2; j++) {
      mb[j+1] = mb[j]+mb2[s];
      //printf( " -> mb[%d]=%d, mb[%d]=%d\n",j,mb[j],j+1,mb[j+1]);
    }

    // copy etree
    etree[s] = etree2[s];
    //printf( " -> mb[%d]=%d, nb[%d]=%d, etree[%d]=%d\n",j2,mb[j2],s+1,nb[s+1],s,etree[s] );
  }
  *p_nsuper = nsuper3;
  delete[] nb2;
  delete[] mb2;
  delete[] etree2;

  for (int s = 0; s <nsuper3; s++) {
    //printf( " mb[%d]=%d, nb[%d]=%d, etree[%d]=%d (nrow=%d, ncol=%d)\n",s,mb[s],s,nb[s],s,etree[s],mb[s+1]-mb[s],nb[s+1]-nb[s] );
    int j1 = nb[s];
    int j2 = nb[s+1];

    int nscol = j2 - j1;
    int nsrow = mb[j1+1] - mb[j1];

    if (s == 0) {
      min_nscol = max_nscol = tot_nscol = nscol;
      min_nsrow = max_nsrow = tot_nsrow = nsrow;
    } else {
      if (min_nsrow > nsrow) {
        min_nsrow = nsrow;
      }
      if (max_nsrow < nsrow) {
        max_nsrow = nsrow;
      }
      tot_nsrow += nsrow;

      if (min_nscol > nscol) {
        min_nscol = nscol;
      }
      if (max_nscol < nscol) {
        max_nscol = nscol;
      }
      tot_nscol += nscol;
    }
  }
  std::cout << "    + Supernode   = " << nsuper3 << std::endl;
  std::cout << "      > nsrow: min = " << min_nsrow << ", max = " << max_nsrow << ", avg = " << tot_nsrow/nsuper3 << std::endl;
  std::cout << "      > nscol: min = " << min_nscol << ", max = " << max_nscol << ", avg = " << tot_nscol/nsuper3 << std::endl;
  std::cout << "    + Matrix size = " << n << std::endl;
  std::cout << "    + Total nnz   = " << hr (n) << std::endl;
  std::cout << "    + nnz / n     = " << hr (n)/n << std::endl;

  // deepcopy
  Kokkos::deep_copy (rowmap_view, hr);
  Kokkos::deep_copy (column_view, hc);
  Kokkos::deep_copy (values_view, hv);

  // create crs
  graph_t static_graph (column_view, rowmap_view);
  crsMat_t crsmat("CrsMatrix", n, values_view, static_graph);
  return crsmat;
}
#endif



/* ========================================================================================= */
template <typename Scalar>
void print_factor_superlu(int n, SuperMatrix *L, SuperMatrix *U, int *perm_r, int *perm_c) {
  SCformat *Lstore = (SCformat*)(L->Store);
  Scalar   *Lx = (Scalar*)(Lstore->nzval);

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

#if 0
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
      for (int j = 0; j < i; j++) printf( "%d %d %.16e\n",1+j1+i, 1+j1+j, Lx[psx + i + j*nsrow] );
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


template <typename crsMat_t>
void print_crsmat(int n, crsMat_t &A) {
  auto graph = A.graph; // in_graph
  auto row_map = graph.row_map;
  auto entries = graph.entries;
  auto values  = A.values;

  std::cout << "[";
  for (int i = 0; i < n; i++) {
    for (int k = row_map[i]; k < row_map[i+1]; k++) {
      std::cout << i << " " << entries[k] << " " << values[k] << " " << k << std::endl;
    }
  }
  std::cout << "];" << std::endl;
}


template <typename graph_t>
void print_graph(int n, graph_t &graph) {
  auto row_map = graph.row_map;
  auto entries = graph.entries;

  std::cout << "[";
  for (int i = 0; i < n; i++) {
    for (int k = row_map[i]; k < row_map[i+1]; k++) {
      std::cout << i << " " << entries[k] << " " << std::endl;
    }
  }
  std::cout << "];" << std::endl;
}


/* ========================================================================================= */
template <typename crsMat_t>
crsMat_t read_superlu_Lfactor(int n, SuperMatrix *L) {

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

  // compute offset for each row
  //printf( "\n -- store superlu factor (n=%d, nsuper=%d) --\n",n,nsuper );
  int j = 0;
  #ifdef SUPERLU_MERGE_SUPERNODES
  int nnz_per_row = 0;
  #endif
  hr(j) = 0;
  for (int s = 0 ; s < nsuper ; s++) {
    int j1 = nb[s];
    int j2 = nb[s+1];
    int nscol = j2 - j1;      // number of columns in the s-th supernode column

    int i1 = mb[j1];
    int i2 = mb[j1+1];
    int nsrow = i2 - i1;    // "total" number of rows in all the supernodes (diagonal+off-diagonal)

    for (int jj = 0; jj < nscol; jj++) {
      hr(j+1) = hr(j) + nsrow;
      //printf( " hr(%d) = %d\n",j+1,hr(j+1) );
      j++;
    }
    #ifdef SUPERLU_MERGE_SUPERNODES
    if (nsrow > nnz_per_row) {
      nnz_per_row = nsrow;
    }
    #endif
  }

  #ifdef SUPERLU_MERGE_SUPERNODES
  int rowids[nnz_per_row];
  #endif
  // store L in csr
  for (int s = 0 ; s < nsuper ; s++) {
    int j1 = nb[s];
    int j2 = nb[s+1];
    int nscol = j2 - j1;      // number of columns in the s-th supernode column

    int i1 = mb[j1] ;
    int i2 = mb[j1+1] ;
    int nsrow  = i2 - i1;    // "total" number of rows in all the supernodes (diagonal+off-diagonal)
    int nsrow2 = nsrow - nscol;  // "total" number of rows in all the off-diagonal supernodes
    int ps2    = i1 + nscol;     // offset into rowind

    int psx = colptr[j1] ;        // offset into data,   Lx[s][s]
    //printf( "%d: nscol=%d, nsrow=%d\n",s,nscol,nsrow );

    /* diagonal block */
    // for each column (or row due to symmetry), the diagonal supernodal block is stored (in ascending order of row indexes) first
    // so that we can do TRSM on the diagonal block
    #if defined(SUPERLU_INVERT_DIAG) & !defined(SUPERLU_MERGE_SUPERNODES) 
    LAPACKE_dtrtri(LAPACK_COL_MAJOR,
                   'L', 'U', nscol, &Lx[psx], nsrow);
    #endif
    for (int ii = 0; ii < nscol; ii++) {
      // lower-triangular part
      for (int jj = 0; jj < ii; jj++) {
        //printf("1 %d %d %d %.16e %d\n",hr(j1+jj), j1+ii, j1+jj, Lx[psx + (ii + jj*nsrow)], hr(j1+ii));
        hc(hr(j1+jj)) = j1+ii;
        hv(hr(j1+jj)) = Lx[psx + (ii + jj*nsrow)];
        hr(j1+jj) ++;
      }
      // diagonal
      hc(hr(j1+ii)) = j1+ii;
      hv(hr(j1+ii)) = STS::one ();
      hr(j1+ii) ++;
      // explicitly store zeros in upper-part
      for (int jj = ii+1; jj < nscol; jj++) {
        //printf("2 %d %d %d %.16e %d\n",hr(j1+jj), j1+ii, j1+jj, STS::zero (), hr(j1+ii));
        hc(hr(j1+jj)) = j1+ii;
        hv(hr(j1+jj)) = STS::zero ();
        hr(j1+jj) ++;
      }
    }

    /* off-diagonal blocks */
    #ifdef SUPERLU_MERGE_SUPERNODES
    for (int ii = 0; ii < nsrow2; ii++) {
      rowids[ii] = ii;
    }
    std::sort(rowids, rowids+nsrow2, sort_indices(&rowind[ps2]));
    #endif
    for (int kk = 0; kk < nsrow2; kk++) {
      #ifdef SUPERLU_MERGE_SUPERNODES
      int ii = rowids[kk];
      #else
      int ii = kk;
      #endif
      int i = rowind[ps2 + ii];
      for (int jj = 0; jj < nscol; jj++) {
        //printf("0 %d %d %d %.16e %d\n",hr(j1+jj), i, j1+jj, Lx[psx + (nscol+ii + jj*nsrow)], hr(i));
        hc(hr(j1+jj)) = i;
        hv(hr(j1+jj)) = Lx[psx + (nscol+ii + jj*nsrow)];
        hr(j1+jj) ++;
      }
    }
  }

  // fix hr
  for (int i = n; i >= 1; i--) {
    hr(i) = hr(i-1);
    //printf( "hr[%d] = %d\n",i-1,hr(i) );
  }
  hr(0) = 0;
  
  std::cout << "    * Matrix size = " << n << std::endl;
  std::cout << "    * Total nnz   = " << hr (n) << std::endl;
  std::cout << "    * nnz / n     = " << hr (n)/n << std::endl;
  /*std::cout << " [" << std::endl;
  for (int ii = 0; ii < n; ii++ ) {
    for (int jj = hr (ii) ; jj < hr (ii+1); jj ++ ) {
      //if (hv(jj) != STS:: zero()) {
        std::cout << ii << " " << hc (jj) << " " << hv (jj) << std::endl;
      ///}
    }
  }
  std::cout << " ];" << std::endl;*/

  // deepcopy
  Kokkos::deep_copy (rowmap_view, hr);
  Kokkos::deep_copy (column_view, hc);
  Kokkos::deep_copy (values_view, hv);

  // create crs
  graph_t static_graph (column_view, rowmap_view);
  crsMat_t crsmat("CrsMatrix", n, values_view, static_graph);
  return crsmat;
}


/* ========================================================================================= */
template <typename crsMat_t>
crsMat_t read_superlu_Ufactor(int n, SuperMatrix *L,  SuperMatrix *U) {

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

  int *sup = new int[nsuper];
  for (int k = 0; k < nsuper; k++) {
    int j1 = nb[k];
    int nscol = nb[k+1] - j1;

    int i1 = mb[j1];
    int nsrow = mb[j1+1] - i1;

    /* the diagonal block */
    int psx = colptr[j1];
    #if defined(SUPERLU_INVERT_DIAG) & !defined(SUPERLU_MERGE_SUPERNODES)
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
  delete[] sup;
  delete[] check;

  // fix hr
  for (int i = n; i >= 1; i--) {
    hr(i) = hr(i-1);
    //printf( "hr[%d] = %d\n",i-1,hr(i) );
  }
  hr(0) = 0;
  std::cout << "    * Matrix size = " << n << std::endl;
  std::cout << "    * Total nnz   = " << hr (n) << std::endl;
  std::cout << "    * nnz / n     = " << hr (n)/n << std::endl;
  /*std::cout << " [" << std::endl;
  for (int i = 0; i < n; i++ ) {
    for (int j = hr (i) ; j < hr (i+1); j ++ ) {
      if (hv(j) != STS:: zero()) {
        std::cout << i << " " << hc (j) << " " << hv (j) << std::endl;
      }
    }
  }
  std::cout << " ];" << std::endl;*/

  // deepcopy
  Kokkos::deep_copy (rowmap_view, hr);
  Kokkos::deep_copy (column_view, hc);
  Kokkos::deep_copy (values_view, hv);

  // create crs
  graph_t static_graph (column_view, rowmap_view);
  crsMat_t crsmat("CrsMatrix", n, values_view, static_graph);
  return crsmat;
}


/* ========================================================================================= */
template <typename HostGraph_t, typename Graph_t>
HostGraph_t generate_supernodal_graph(int n, Graph_t &graph, int nsuper, int *mb, int *nb) {

  typedef typename Graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename Graph_t::entries_type::non_const_type    cols_view_t;


  auto row_map = graph.row_map;
  auto entries = graph.entries;

  typename row_map_view_t::HostMirror row_map_host = Kokkos::create_mirror_view (row_map);
  typename cols_view_t::HostMirror    entries_host = Kokkos::create_mirror_view (entries);
  Kokkos::deep_copy (row_map_host, row_map);
  Kokkos::deep_copy (entries_host, entries);

  // map col/row to supernode
  int *map = new int[n];
  for (int s = 0; s < nsuper; s++) {
    for (int j = nb[s]; j < nb[s+1]; j++) {
      map[j] = s;
    }
  }

  // count non-empty supernodal blocks
  int nblocks = 0;
  for (int s = 0; s < nsuper; s++) {
    int j1 = nb[s];
    for (int i = row_map_host (j1); i < row_map_host (j1+1);) {
      int s2 = map[entries_host (i)];
      nblocks ++;
      i += (nb[s2+1]-nb[s2]);
    }
  }

  typedef typename HostGraph_t::row_map_type::non_const_type row_map_view_host_t;
  typedef typename HostGraph_t::entries_type::non_const_type    cols_view_host_t;
  row_map_view_host_t hr ("rowmap_view", nsuper+1);
  cols_view_host_t    hc ("colmap_view", nblocks);

  nblocks = 0;
  hr (0) = 0;
  for (int s = 0; s < nsuper; s++) {
    int j1 = nb[s];
    for (int i = row_map_host (j1); i < row_map_host (j1+1);) {
      int s2 = map[entries_host (i)];

      hc (nblocks) = s2;
      nblocks ++;
      i += (nb[s2+1]-nb[s2]);
    }
    hr (s+1) = nblocks;
    std::sort(&(hc (hr (s))), &(hc (hr (s+1))));
  }

  HostGraph_t static_graph (hc, hr);
  return static_graph;
}

template <typename Graph_t>
int** generate_supernodal_dag(int nsuper, Graph_t &supL, Graph_t &supU) {

  auto row_mapL = supL.row_map;
  auto entriesL = supL.entries;
  auto row_mapU = supU.row_map;
  auto entriesU = supU.entries;

  int *edges = new int[nsuper];
  int **dag = (int**)malloc(nsuper * sizeof(int*));
  for (int s = 0; s < nsuper; s ++) {
    // count # of edges (search for first matching nonzero)
    int nedges = 0;
    int k1 = 1 + row_mapL (s); // skip diagonal
    int k2 = 1 + row_mapU (s); // skip diagonal
    for (; k1 < row_mapL (s+1); k1++) {
       // look for match
       while (entriesL (k1) > entriesU (k2) && k2 < row_mapU (s+1)) {
         k2 ++;
       }
       if (entriesL (k1) <= entriesU (k2)) {
         edges[nedges] = entriesL (k1);
         nedges ++;
         if (entriesL (k1) == entriesU (k2)) {
           break;
         }
      }
    }
    // store the edges
    dag[s] = new int [1+nedges];
    dag[s][0] = nedges;
    for (int k = 0; k < nedges; k++) {
      dag[s][1+k] = edges[k];
    }
  }
  delete[] edges;

  return dag;
}


/* ========================================================================================= */
template<typename Scalar>
void factor_superlu(bool metis, const int nrow, const int nnz, Scalar *nzvals, int *rowptr, int *colind,
                    int panel_size, int relax_size, SuperMatrix &L, SuperMatrix &U,
                    int **perm_r, int **perm_c, int **parents) {
  SuperMatrix A;
  NCformat *Astore;
  //int      *perm_c; /* column permutation vector */
  //int      *perm_r; /* row permutations from partial pivoting */
  int      info;
  superlu_options_t options;
  SuperLUStat_t stat;

  set_default_options(&options);
  options.SymmetricMode = YES;
  #ifdef KOKKOSKERNELS_ENABLE_TPL_METIS
  if (metis) {
    options.ColPerm = NATURAL;
    //options.SymmetricMode = NO;
  }
  #endif

  dCreate_CompCol_Matrix(&A, nrow, nrow, nnz, nzvals, colind, rowptr, SLU_NC, SLU_D, SLU_GE);
  Astore = (NCformat*)(A.Store);
  printf("  Dimension %dx%d; # nonzeros %d\n", A.nrow, A.ncol, Astore->nnz);

  if ( !(*perm_c = (int*)malloc(nrow * sizeof(int))) ) ABORT("Malloc fails for perm_c[].");
  if ( !(*perm_r = (int*)malloc(nrow * sizeof(int))) ) ABORT("Malloc fails for perm_r[].");

  /* Initialize the statistics variables. */
  StatInit(&stat);
  int w1 = (sp_ienv(1) > sp_ienv(2) ? sp_ienv(1) : sp_ienv(2));
  int w2 = (panel_size > relax_size ? panel_size : relax_size);
  if (w2 > w1) {
    SUPERLU_FREE(stat.panel_histo);
    stat.panel_histo = intCalloc(w2+1);
  }

  /* Call SuperLU to solve the problem. */
  //for (int i=0; i<m; i++ ) printf( " B[%d]=%e\n",i,rhs[i] );
  int *etree = new int[A.ncol];
  get_perm_c(options.ColPerm, &A, *perm_c);

  SuperMatrix AC;
  sp_preorder(&options, &A, *perm_c, etree, &AC);

  GlobalLU_t Glu;
  int lwork = 0;
  printf( "  Calling SuperLU dgstrf with panel_size=%d, relax_size=%d..\n",panel_size,relax_size );
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

  double one  = 1.0;
  double zero = 0.0;

  SCformat *Lstore = (SCformat*)(L->Store);
  double   *Lx = (double*)(Lstore->nzval);

  /* allocate workspace */
  int n = L->nrow;
  int ldw = n;
  double *work = new double[ldw];

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
      int i = rowind[ps2 + ii];
      for (int j = 0; j < nrhs; j++)
      {
        X[i + j*ldx] -= work[ii + j*ldw];
      }
    }
  } /* for L-solve */
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

enum {CUSPARSE, SUPERNODAL_NAIVE, SUPERNODAL_ETREE, SUPERNODAL_DAG};


#ifdef KOKKOSKERNELS_ENABLE_TPL_SUPERLU
template<typename Scalar>
int test_sptrsv_perf(std::vector<int> tests, std::string& filename, bool metis, int panel_size, int relax_size,
                     int sup_size_unblocked, int sup_size_blocked, int loop) {

  typedef Scalar scalar_t;
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
  typedef Kokkos::View< scalar_t*, host_memory_space > HostValuesType;
  typedef Kokkos::View< scalar_t*,      memory_space > ValuesType;

  //
  typedef KokkosKernels::Experimental::KokkosKernelsHandle <size_type, lno_t, scalar_t,
    execution_space, memory_space, memory_space > KernelHandle;

  scalar_t ZERO = scalar_t(0);
  scalar_t ONE = scalar_t(1);


  int num_failed = 0;
  std::cout << std::endl;
  std::cout << "Execution space: " << execution_space::name () << std::endl;
  std::cout << "Memory space   : " << memory_space::name () << std::endl;
  std::cout << std::endl;
  if (!filename.empty())
  {
    // read the matrix ** on host **
    std::cout << " SuperLU Tester Begin: Read matrix filename " << filename << std::endl;
    host_crsmat_t Mtx = KokkosKernels::Impl::read_kokkos_crst_matrix<host_crsmat_t>(filename.c_str()); //in_matrix
    #ifdef KOKKOSKERNELS_ENABLE_TPL_METIS
    if (metis) {
      auto original_graph   = Mtx.graph; // in_graph
      auto original_row_map = original_graph.row_map;
      auto original_entries = original_graph.entries;
      auto original_values  = Mtx.values;

      typedef typename host_graph_t::row_map_type::non_const_type host_row_map_view_t;
      typedef typename host_graph_t::entries_type::non_const_type host_cols_view_t;
      typedef typename host_crsmat_t::values_type::non_const_type host_values_view_t;

      idx_t n = original_graph.numRows();
      idx_t nnz = original_row_map(n);
      const idx_t nnzA = original_row_map(n);
      
      // removing diagonals (and casting to METIS idx_t)
      idx_t *metis_rowptr = new idx_t[n+1];
      idx_t *metis_colind = new idx_t[nnz];

      nnz = 0;
      metis_rowptr[0] = 0;
      for (int i = 0; i < n; i++) {
        for (int k = original_row_map(i); k < original_row_map(i+1); k++) {
          if (original_entries(k) != i) {
            metis_colind[nnz] = original_entries(k);
            nnz ++;
          }
        }
        metis_rowptr[i+1] = nnz;
      }

      idx_t *perm = new idx_t[n];
      idx_t *iperm = new idx_t[n];

      std::cout << " calling METIS_NodeND: (n=" << n << ", nnz=" << nnzA << "->" << nnz << ") " << std::endl;
      if (METIS_OK != METIS_NodeND(&n, metis_rowptr, metis_colind, NULL, NULL, perm, iperm)) {
        std::cout << std::endl << "METIS_NodeND failed" << std::endl << std::endl;
      }
      //for (idx_t i = 0; i < n; i++) printf("%ld %ld %ld\n",i, perm[i], iperm[i]);
      //for (idx_t i = 0; i < n; i++) perm[i] = iperm[i] = i;

      host_row_map_view_t hr ("rowmap_view", n+1);
      host_cols_view_t    hc ("colmap_view", nnzA);
      host_values_view_t  hv ("values_view", nnzA);

      nnz = 0; hr (0) = 0;
      for (idx_t i = 0; i < n; i++) {
        for (idx_t j=original_row_map(perm[i]); j < original_row_map(perm[i]+1); j++) {
          hc(nnz) = iperm[ original_entries(j) ];
          hv(nnz) = original_values(j);
          nnz ++;
        }
        hr(1+i) = nnz;
      }
      host_graph_t host_static_graph(hc, hr);
      Mtx = host_crsmat_t("CrsMatrix", n, hv, host_static_graph);

      delete [] perm;
      delete [] iperm;
      delete [] metis_rowptr;
      delete [] metis_colind;
    }
    #endif

    auto  graph_host  = Mtx.graph; // in_graph
    auto row_map_host = graph_host.row_map;
    auto entries_host = graph_host.entries;
    auto values_host  = Mtx.values;
    const size_type nrows = graph_host.numRows();
    //print_crsmat<host_crsmat_t> (nrows, Mtx);

    // Create the known solution and set to all 1's ** on host **
    HostValuesType sol_host("sol_host", nrows);
    Kokkos::deep_copy(sol_host, ONE);

    // Create the rhs ** on host **
    // A*sol generates rhs: rhs is dense, use spmv
    HostValuesType rhs_host("rhs_host", nrows);
    KokkosSparse::spmv( "N", ONE, Mtx, sol_host, ZERO, rhs_host);
    //for (int i = 0; i < nrows; i++) printf( "%.16e\n",rhs_host(i) );

    // tolerance
    scalar_t tol = STS::epsilon();

    // normB
    scalar_t normB = 0.0;
    Kokkos::parallel_reduce( Kokkos::RangePolicy<host_execution_space>(0, rhs_host.extent(0)), 
      KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
        tsum += rhs_host(i)*rhs_host(i);
      }, normB);
    normB = sqrt(normB);

    // normA
    scalar_t normA = 0.0;
    Kokkos::parallel_reduce( Kokkos::RangePolicy<host_execution_space>(0, Mtx.nnz()), 
      KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
        tsum += values_host(i)*values_host(i);
      }, normA);
    normA = sqrt(normA);

    // Run all requested algorithms
    for ( auto test : tests ) {
      std::cout << "\ntest = " << test << std::endl;

      SuperMatrix L;
      SuperMatrix U;
      #ifdef SUPERLU_MERGE_SUPERNODES
      host_crsmat_t superluL_host, superluU_host;
      #endif
      crsmat_t superluL, superluU;
      KernelHandle khL, khU;
      switch(test) {
        case SUPERNODAL_NAIVE:
        case SUPERNODAL_ETREE:
        case SUPERNODAL_DAG:
        {
          Kokkos::Timer timer;
          // callSuperLU on the host    
          int *etree, *perm_r, *perm_c;
          std::cout << " > call SuperLU for factorization" << std::endl;
          factor_superlu<Scalar> (metis, nrows, Mtx.nnz(), values_host.data(), const_cast<int*> (row_map_host.data()), entries_host.data(),
                                  panel_size, relax_size, L, U, &perm_r, &perm_c, &etree);
          std::cout << "   Factorization Time: " << timer.seconds() << std::endl << std::endl;

          // read SuperLU factor int crsMatrix on the host (superluMat_host) and copy to default host/device (superluL)
          std::cout << " > Read SuperLU factor into KokkosSparse::CrsMatrix (invert diagonal and copy to device)" << std::endl;
          timer.reset();
          #ifdef SUPERLU_MERGE_SUPERNODES
          superluL_host = read_superlu_Lfactor<host_crsmat_t> (nrows, &L);
          #else
          superluL = read_superlu_Lfactor<crsmat_t> (nrows, &L);
          #endif
          std::cout << "   Conversion Time for L: " << timer.seconds() << std::endl;

          timer.reset();
          #ifdef SUPERLU_MERGE_SUPERNODES
          superluU_host = read_superlu_Ufactor<host_crsmat_t> (nrows, &L, &U);
          #else
          superluU = read_superlu_Ufactor<crsmat_t> (nrows, &L, &U);
          #endif
          std::cout << "   Conversion Time for U: " << timer.seconds() << std::endl << std::endl;
          //print_factor_superlu<Scalar> (nrows, &L, &U, perm_r, perm_c);

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
          //khL.get_sptrsv_handle ()->print_algorithm ();

          // setup supnodal info
          SCformat *Lstore = (SCformat*)(L.Store);
          int nsuper = 1 + Lstore->nsuper;
          int *supercols = Lstore->sup_to_col;
          #ifdef SUPERLU_MERGE_SUPERNODES
          {
            timer.reset ();
            // make a copy of etree
            int *etree2 = new int[nsuper];
            for (int i = 0; i < nsuper; i++) {
              etree2[i] = etree[i];
            }
            // make a copy of supercols
            int *supercols2 = new int[1+nsuper];
            for (int i = 0; i <= nsuper; i++) {
              supercols2[i] = supercols[i];
            }
            // make a copy of superrows
            int *superrows = Lstore->rowind_colptr;
            int *superrows2 = new int[1+nrows];
            for (int i = 0; i <= nrows; i++) {
              superrows2[i] = superrows[i];
            }

            // merge L-factor
            int nnzL = superluL_host.nnz ();
            int nsuper2 = nsuper;
            superluL = merge_supernodes<host_crsmat_t, crsmat_t> (nrows, &nsuper2, supercols2, superrows2,
                                                                  true, superluL_host, superluU_host, etree2);
            // save the supernodal info in the handle for U-solve
            khL.set_supernodes (nsuper2, supercols2, etree2);
            if (sup_size_unblocked > 0 && sup_size_blocked > 0) {
              khL.set_diag_supernode_sizes (sup_size_unblocked, sup_size_blocked);
            }
            std::cout << "   L factor:" << std::endl;
            std::cout << "   Merge Supernodes Time: " << timer.seconds() << std::endl;
            std::cout << "   Number of nonzeros   : " << nnzL << " -> " << superluL.nnz () 
                      << " : " << double(superluL.nnz ()) / double(nnzL) << "x" << std::endl;
          }
          {
            // NOTE: starting over for U-factor
            timer.reset ();
            // make a copy of etree
            int *etree2 = new int[nsuper];
            for (int i = 0; i < nsuper; i++) {
              etree2[i] = etree[i];
            }
            // make a copy of supercols
            int *supercols2 = new int[1+nsuper];
            for (int i = 0; i <= nsuper; i++) {
              supercols2[i] = supercols[i];
            }
            // make a copy of superrows
            int *superrows = Lstore->rowind_colptr;
            int *superrows2 = new int[1+nrows];
            for (int i = 0; i <= nrows; i++) {
              superrows2[i] = superrows[i];
            }
            // merge U-factor
            int nnzU = superluU_host.nnz ();
            int nsuper2 = nsuper;
            superluU = merge_supernodes<host_crsmat_t, crsmat_t> (nrows, &nsuper2, supercols2, superrows2,
                                                                  false, superluU_host, superluL_host, etree2);
            // save the supernodal info in the handle for U-solve
            khU.set_supernodes (nsuper2, supercols2, etree2);
            if (sup_size_unblocked > 0 && sup_size_blocked > 0) {
              khU.set_diag_supernode_sizes (sup_size_unblocked, sup_size_blocked);
            }
            std::cout << "   U factor:" << std::endl;
            std::cout << "   Merge Supernodes Time: " << timer.seconds() << std::endl;
            std::cout << "   Number of nonzeros   : " << nnzU << " -> " << superluU.nnz () 
                      << " : " << double(superluU.nnz ()) / double(nnzU) << "x" << std::endl;

            // generate supernodal graphs for DAG scheduling
            auto supL = generate_supernodal_graph<host_graph_t, graph_t> (nrows, superluL.graph, nsuper2, superrows2, supercols2);
            auto supU = generate_supernodal_graph<host_graph_t, graph_t> (nrows, superluU.graph, nsuper2, superrows2, supercols2);
            //print_graph<host_graph_t> (nsuper2, supL);
            //print_graph<host_graph_t> (nsuper2, supU);

            int **dagL = generate_supernodal_dag<host_graph_t> (nsuper2, supL, supU);
            int **dagU = generate_supernodal_dag<host_graph_t> (nsuper2, supU, supL);
            /*for (int s = 0; s < nsuper2; s++) {
              printf( " %d : ",s );
              for (int e = 0; e < dagL[s][0]; e++) {
                printf( "%d ",dagL[s][1+e] );
              }
              printf( "\n" );
            }*/
            khL.set_supernodal_dag (dagL);
            khU.set_supernodal_dag (dagU);
          }
          #else
          khL.set_supernodes (nsuper, supercols, etree);
          khU.set_supernodes (nsuper, supercols, etree);
          #endif
          //print_crsmat<crsmat_t> (nrows, superluL);
          //print_crsmat<crsmat_t> (nrows, superluU);
 
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
          auto graphL = superluL.graph; // in_graph
          auto row_mapL = graphL.row_map;
          auto entriesL = graphL.entries;
          auto valuesL  = superluL.values;
          #if 1
           // symbolic on the host
           timer.reset();
           std::cout << std::endl;
           std::cout << " > Lower-TRI: " << std::endl;
           sptrsv_symbolic (&khL, row_mapL, entriesL);
           std::cout << "   Symbolic Time: " << timer.seconds() << std::endl;

           timer.reset();
           // numeric (only rhs is modified) on the default device/host space
           sptrsv_solve (&khL, row_mapL, entriesL, valuesL, sol, rhs);
          #else
           timer.reset();
           // solveL with SuperLU data structure, L
           //solveL_superlu<Scalar>(&L, 1, rhs.data(), nrows);
           solveL_superlu<Scalar>(&L, 1, tmp_host.data(), nrows);
           Kokkos::deep_copy (rhs, tmp_host);
          #endif
          Kokkos::fence();
          std::cout << "   Solve Time   : " << timer.seconds() << std::endl;
          //Kokkos::deep_copy (tmp_host, rhs);
          //printf( "y=[" );
          //for (int ii=0; ii<nrows; ii++) printf( " %d %.16e\n",ii,tmp_host(ii) );
          //printf( "];\n" );

          // ==============================================
          // do L^T solve
          auto graphU = superluU.graph; // in_graph
          auto row_mapU = graphU.row_map;
          auto entriesU = graphU.entries;
          auto valuesU  = superluU.values;
          #if 1
           // symbolic on the host
           timer.reset ();
           std::cout << " > Upper-TRI: " << std::endl;
           sptrsv_symbolic (&khU, row_mapU, entriesU);
           std::cout << "   Symbolic Time: " << timer.seconds() << std::endl;

           // numeric (only rhs is modified) on the default device/host space
           sptrsv_solve (&khU, row_mapU, entriesU, valuesU, sol, rhs);
          #else
           //solveU_superlu<Scalar>(&L, &U, 1, rhs.data(), nrows);
           solveU_superlu<Scalar>(&L, &U, 1, tmp_host.data(), nrows);
           Kokkos::deep_copy (rhs, tmp_host);
          #endif
          Kokkos::fence ();
          std::cout << "   Solve Time   : " << timer.seconds() << std::endl;
 
          // copy solution to host
          Kokkos::deep_copy(tmp_host, rhs);

          // apply backward-pivot
          backwardP_superlu<Scalar>(nrows, perm_c, 1, tmp_host.data(), nrows, sol_host.data(), nrows);
          //printf( "x=[" );
          //for (int ii=0; ii<nrows; ii++) printf( " %d %.16e\n",ii,tmp_host(ii) );
          //printf( "];\n" );


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
          if (normR/(scalar_t(nrows) * (normB + normA * normX)) > tol) {
            num_failed ++;
          }

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
            if (normR/(scalar_t(nrows) * (normB + normA * normX)) > tol) {
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
    int total_errors = test_sptrsv_perf<double>(tests, filename, metis, panel_size,
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
