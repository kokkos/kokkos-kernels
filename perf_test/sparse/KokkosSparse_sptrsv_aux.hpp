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

#ifndef KOKKOSSPARSE_SPTRSV_AUX
#define KOKKOSSPARSE_SPTRSV_AUX

class sort_indices {
   public:
     sort_indices(int* rowinds) : rowinds_(rowinds){}
     bool operator()(int i, int j) const { return rowinds_[i] < rowinds_[j]; }
   private:
     int* rowinds_; // rowindices
};


/* ========================================================================================= */
template <typename host_crsmat_t, typename crsmat_t>
crsmat_t merge_supernodes(int n, int *p_nsuper, int *nb, int *mb,
                          bool unit_diag, bool invert_diag, bool invert_offdiag,
                          host_crsmat_t &superluL, host_crsmat_t &superluU,
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
  #if defined(fill_supernodal_blocks) // force zeros to fill supernodes
  int *work1 = new int[nsuper];
  #else
  int *work1 = new int[n];
  #endif
  int *work2 = new int[n];
  int *work3 = new int[n]; // map row to supernode
  int **rowind = new int*[nsuper3];
  for (int s = 0; s < nsuper; s++) {
    for (int i = nb[s]; i < nb[s+1]; i++) {
      work3[i] = s;
    }
  }
  #if defined(fill_supernodal_blocks)
  for (int i = 0; i < nsuper; i++) {
    work1[i] = 0;
  }
  #else
  for (int i = 0; i < n; i++) {
    work1[i] = 0;
  }
  #endif
  for (int s2 = 0, s = 0; s2 < nsuper3; s2++) {
    nb2[s2] = 0;
    mb2[s2] = 0;
    // merging supernodal rows
    // NOTE: SuperLU may not fill zeros to fill the supernodes
    //       So, these rows may be just subset of the supernodal rows
    while(s < nsuper && map2[s] == s2) {
      int j1 = nb[s];
      for (int k = row_mapL[j1]; k < row_mapL[j1+1]; k++) {
        #if defined(fill_supernodal_blocks)
        // forcing zeros to fill supernodal blocks
        int s3 = work3[entriesL[k]];
        if (work1[s3] == 0) {
          for (int i = nb[s3]; i < nb[s3+1]; i++) {
            work2[mb2[s2]] = i;
            mb2[s2] ++;
          }
          work1[s3] = 1;
        }
        #else
        // just taking union of rows
        if(work1[entriesL[k]] == 0) {
          work1[entriesL[k]] = 1;
          work2[mb2[s2]] = entriesL[k];
          mb2[s2] ++;
        }
        #endif
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
      #if !defined(fill_supernodal_blocks)
      work1[work2[k]] = 0;
      #endif
    }
    #if defined(fill_supernodal_blocks)
    for (int s3 = 0; s3 < nsuper; s3++) {
      work1[s3] = 0;
    }
    #endif
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
  typedef typename crsmat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type   cols_view_t;
  typedef typename crsmat_t::values_type::non_const_type values_view_t;

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
    int nnzD = nnzA;
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

    int nscol = nb2[s2];
    int nsrow = mb2[s2];
    if (invert_diag) {
      if (unit_diag) {
        LAPACKE_dtrtri(LAPACK_COL_MAJOR,
                       'L', 'U', nscol, &hv(nnzD), nsrow);
        if (nsrow > nscol && invert_offdiag) {
          cblas_dtrmm (CblasColMajor,
                CblasRight, CblasLower, CblasNoTrans, CblasUnit,
                nsrow-nscol, nscol,
                STS::one (), &hv(nnzD), nsrow,
                             &hv(nnzD+nscol), nsrow);
        }
      } else {
        LAPACKE_dtrtri(LAPACK_COL_MAJOR,
                       'L', 'N', nscol, &hv(nnzD), nsrow);
        if (nsrow > nscol && invert_offdiag) {
          cblas_dtrmm (CblasColMajor,
                CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
                nsrow-nscol, nscol,
                STS::one (), &hv(nnzD), nsrow,
                             &hv(nnzD+nscol), nsrow);
        }
      }
    }
  }
  delete[] map2;
  delete[] dwork;
  delete[] rowind;
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
    //printf( " %d: %d %d, %d\n",s,mb[s+1]-mb[s],nb[s+1]-nb[s],nb[s] );
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
  crsmat_t crsmat("CrsMatrix", n, values_view, static_graph);
  return crsmat;
}


/* ========================================================================================= */
template <typename host_graph_t, typename graph_t>
host_graph_t generate_supernodal_graph(bool merged, int n, graph_t &graph, int nsuper, int *mb, int *nb) {

  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type    cols_view_t;

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
  typedef typename host_graph_t::row_map_type::non_const_type row_map_view_host_t;
  row_map_view_host_t hr ("rowmap_view", nsuper+1);

  int *check = new int[nsuper];
  for (int s = 0; s < nsuper; s++) {
    check[s] = 0;
  }
  int nblocks = 0;
  for (int s = 0; s < nsuper; s++) {
    int j1 = nb[s];
    for (int i = row_map_host (j1); i < row_map_host (j1+1);) {
      int s2 = map[entries_host (i)];
      #if defined(fill_supernodal_blocks)
      // forced zeros to fill supernodal blocks
      if (merged) {
        nblocks ++;
        i += (nb[s2+1]-nb[s2]);
      } else
      #endif
      // may not have filled supernodal blocks with zeros
      {
        // rowids are not sorted
        if (check[s2] == 0) {
          check[s2] = 1;
          nblocks ++;
        }
        i ++;
      }
    }
    #if defined(fill_supernodal_blocks)
    if (!merged)
    #endif
    {
      for (int s2 = 0; s2 < nsuper; s2++) {
        check[s2] = 0;
      }
    }
  }

  typedef typename host_graph_t::entries_type::non_const_type cols_view_host_t;
  cols_view_host_t hc ("colmap_view", nblocks);

  nblocks = 0;
  hr (0) = 0;
  for (int s = 0; s < nsuper; s++) {
    int j1 = nb[s];
    for (int i = row_map_host (j1); i < row_map_host (j1+1);) {
      int s2 = map[entries_host (i)];
      #if defined(fill_supernodal_blocks)
      if (merged) {
        hc (nblocks) = s2;
        nblocks ++;
        i += (nb[s2+1]-nb[s2]);
      } else
      #endif
      {
        // rowids are not sorted
        if (check[s2] == 0) {
          check[s2] = 1;
          hc (nblocks) = s2;
          nblocks ++;
        }
        i ++;
      }
    }
    hr (s+1) = nblocks;
    std::sort(&(hc (hr (s))), &(hc (hr (s+1))));
    #if defined(fill_supernodal_blocks)
    if (!merged)
    #endif
    {
      for (int s2 = hr(s); s2 < hr(s+1); s2++) {
        check[hc(s2)] = 0;
      }
    }
  }
  delete [] check;

  //printf( " > supernodal graph:\n" );
  //for (int s = 0; s < nsuper; s++) {
  //  for (int i = hr(s); i < hr(s+1); i++) printf( "%d %d\n",s,hc(i) );
  //}
  //printf( "\n" );
  host_graph_t static_graph (hc, hr);
  return static_graph;
}

template <typename graph_t>
int** generate_supernodal_dag(int nsuper, graph_t &supL, graph_t &supU) {

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
template<typename scalar_t>
void forwardP_superlu(int n, int *perm_r, int nrhs, scalar_t *B, int ldb, scalar_t *X, int ldx) {

  /* Permute right hand sides to form Pr*B */
  for (int j = 0; j < nrhs; j++) {
      double *rhs_work = &B[j*ldb];
      double *sol_work = &X[j*ldx];
      for (int k = 0; k < n; k++) sol_work[perm_r[k]] = rhs_work[k];
  }
}

template<typename scalar_t>
void backwardP_superlu(int n, int *perm_c, int nrhs, scalar_t *B, int ldb, scalar_t *X, int ldx) {

    /* Compute the final solution X := Pc*X. */
    for (int j = 0; j < nrhs; j++) {
        double *rhs_work = &B[j*ldb];
        double *sol_work = &X[j*ldx];
        for (int k = 0; k < n; k++) sol_work[k] = rhs_work[perm_c[k]];
    }
}


/* ========================================================================================= */
template <typename scalar_t, typename crsmat_t, typename scalar_view_t>
bool check_errors(scalar_t tol, crsmat_t &Mtx, scalar_view_t rhs, scalar_view_t sol) {

  typedef typename crsmat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type entries_view_t;
  typedef typename crsmat_t::values_type::non_const_type values_view_t;

  typedef typename entries_view_t::non_const_value_type lno_t;
  //typedef typename values_view_t::value_type scalar_t;

  typedef typename scalar_view_t::execution_space execution_space;
  typedef typename execution_space::memory_space memory_space;

  scalar_t ZERO = scalar_t(0);
  scalar_t ONE = scalar_t(1);

  // normB
  scalar_t normB = ZERO;
  Kokkos::parallel_reduce( Kokkos::RangePolicy<execution_space>(0, rhs.extent(0)), 
    KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
      tsum += rhs(i)*rhs(i);
    }, normB);
  normB = sqrt(normB);

  // normA
  scalar_t normA = ZERO;
  Kokkos::parallel_reduce( Kokkos::RangePolicy<execution_space>(0, Mtx.nnz()), 
    KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
      tsum += Mtx.values(i)*Mtx.values(i);
    }, normA);
  normA = sqrt(normA);

  // normX
  scalar_t normX = ZERO;
  Kokkos::parallel_reduce( Kokkos::RangePolicy<execution_space>(0, sol.extent(0)), 
    KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
      tsum += sol(i)*sol(i);
    }, normX);
  normX = sqrt(normX);

  // normR = ||B - AX||
  scalar_t normR = ZERO;
  KokkosSparse::spmv( "N", -ONE, Mtx, sol, ONE, rhs);
  Kokkos::parallel_reduce( Kokkos::RangePolicy<execution_space>(0, sol.extent(0)), 
    KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
      tsum += rhs(i) * rhs(i);
    }, normR);
  normR = sqrt(normR);

  std::cout << " > check : ||B - AX||/(||B|| + ||A||*||X||) = "
            << normR << "/(" << normB << " + " << normA << " * " << normX << ") = "
            << normR/(normB + normA * normX) << std::endl;

  const int nrows = Mtx.graph.numRows();
  return (normR/(scalar_t(nrows) * (normB + normA * normX)) <= tol);
}


/* ========================================================================================= */
template <typename crsmat_t>
void print_crsmat(int n, crsmat_t &A) {
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

#endif
