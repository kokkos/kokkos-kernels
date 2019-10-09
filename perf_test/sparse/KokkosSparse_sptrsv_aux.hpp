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

#ifdef KOKKOSSPARSE_SPTRSV_CHOLMOD
class sort_indices {
   public:
     sort_indices(int* rowinds) : rowinds_(rowinds){}
     bool operator()(int i, int j) const { return rowinds_[i] < rowinds_[j]; }
   private:
     int* rowinds_; // rowindices
};



/* ========================================================================================= */
template <typename graph_t>
graph_t
read_supernodal_graphL(bool cusparse, bool merge,
                       int n, int nsuper, bool ptr_by_column, int *mb,
                       int *nb, int *colptr, int *rowind) {

  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type   cols_view_t;

  int nnzA;
  if (ptr_by_column) {
    nnzA = colptr[n] - colptr[0];
  } else {
    nnzA = colptr[nsuper] - colptr[0];
  }
  row_map_view_t rowmap_view ("rowmap_view", n+1);
  cols_view_t    column_view ("colmap_view", nnzA);

  typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view (rowmap_view);
  typename cols_view_t::HostMirror    hc = Kokkos::create_mirror_view (column_view);

  // compute offset for each row
  int j = 0;
  int max_nnz_per_row = 0;
  hr(j) = 0;
  for (int s = 0 ; s < nsuper ; s++) {
    int j1 = nb[s];
    int j2 = nb[s+1];
    int nscol = j2 - j1;      // number of columns in the s-th supernode column

    int i1, i2;
    if (ptr_by_column) {
      i1 = mb[j1];
      i2 = mb[j1+1];
    } else {
      i1 = mb[s];
      i2 = mb[s+1];
    }
    int nsrow = i2 - i1;      // "total" number of rows in all the supernodes (diagonal+off-diagonal)

    for (int jj = 0; jj < nscol; jj++) {
      if (cusparse) {
        hr(j+1) = hr(j) + nsrow - jj;
      } else {
        hr(j+1) = hr(j) + nsrow;
      }
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

    int i1, i2;
    if (ptr_by_column) {
      i1 = mb[j1];
      i2 = mb[j1+1];
    } else {
      i1 = mb[s];
      i2 = mb[s+1];
    }
    int nsrow  = i2 - i1;    // "total" number of rows in all the supernodes (diagonal+off-diagonal)
    int nsrow2 = nsrow - nscol;  // "total" number of rows in all the off-diagonal supernodes
    int ps2    = i1 + nscol;     // offset into rowind

    /* diagonal block */
    for (int ii = 0; ii < nscol; ii++) {
      // lower-triangular part
      for (int jj = 0; jj < ii; jj++) {
        hc(hr(j1+jj)) = j1+ii;
        hr(j1+jj) ++;
      }
      // diagonal
      hc(hr(j1+ii)) = j1+ii;
      hr(j1+ii) ++;
      if (!cusparse) {
        // explicitly store zeros in upper-part
        for (int jj = ii+1; jj < nscol; jj++) {
          hc(hr(j1+jj)) = j1+ii;
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

  // create crs
  graph_t static_graph (column_view, rowmap_view);
  return static_graph;
}

/* ========================================================================================= */
template <typename crsmat_t, typename graph_t, typename scalar_t>
crsmat_t
read_supernodal_valuesL(bool cusparse, bool merge, bool invert_diag, bool invert_offdiag,
                        bool unit_diag, int n, int nsuper, bool ptr_by_column, int *mb,
                        int *nb, int *colptr, int *rowind, scalar_t *Lx,
                        graph_t &static_graph) {

  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type   cols_view_t;
  typedef typename crsmat_t::values_type::non_const_type values_view_t;

  typedef Kokkos::Details::ArithTraits<scalar_t> STS;

  int nnzA;
  if (ptr_by_column) {
    nnzA = colptr[n] - colptr[0];
  } else {
    nnzA = colptr[nsuper] - colptr[0];
  }
  auto rowmap_view = static_graph.row_map;
  auto column_view = static_graph.entries;
  values_view_t  values_view ("values_view", nnzA);

  typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view (rowmap_view);
  typename cols_view_t::HostMirror    hc = Kokkos::create_mirror_view (column_view);
  typename values_view_t::HostMirror  hv = Kokkos::create_mirror_view (values_view);
  Kokkos::deep_copy (hr, rowmap_view);
  Kokkos::deep_copy (hc, column_view);

  // compute max nnz per row
  int j = 0;
  int max_nnz_per_row = 0;
  hr(j) = 0;
  for (int s = 0 ; s < nsuper ; s++) {

    int i1, i2;
    if (ptr_by_column) {
      int j1 = nb[s];

      i1 = mb[j1];
      i2 = mb[j1+1];
    } else {
      i1 = mb[s];
      i2 = mb[s+1];
    }
    int nsrow = i2 - i1;      // "total" number of rows in all the supernodes (diagonal+off-diagonal)
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

    int i1, i2;
    if (ptr_by_column) {
      i1 = mb[j1];
      i2 = mb[j1+1];
    } else {
      i1 = mb[s];
      i2 = mb[s+1];
    }
    int nsrow  = i2 - i1;    // "total" number of rows in all the supernodes (diagonal+off-diagonal)
    int nsrow2 = nsrow - nscol;  // "total" number of rows in all the off-diagonal supernodes
    int ps2    = i1 + nscol;     // offset into rowind

    int psx;                 // offset into data,   Lx[s][s]
    if (ptr_by_column) {
      psx = colptr[j1];
    } else {
      psx = colptr[s];
    }

    /* diagonal block */
    // for each column (or row due to symmetry), the diagonal supernodal block is stored (in ascending order of row indexes) first
    // so that we can do TRSM on the diagonal block
    if (invert_diag) {
      char diag_char = (unit_diag ? 'U' : 'N');
      LAPACKE_dtrtri(LAPACK_COL_MAJOR,
                     'L', diag_char, nscol, &Lx[psx], nsrow);
      if (nsrow2 > 0 && invert_offdiag) {
        CBLAS_DIAG diag_int = (unit_diag ? CblasUnit : CblasNonUnit);
        cblas_dtrmm (CblasColMajor,
              CblasRight, CblasLower, CblasNoTrans, diag_int,
              nsrow2, nscol,
              1.0, &Lx[psx], nsrow,
                   &Lx[psx+nscol], nsrow);
      }
    }
    for (int ii = 0; ii < nscol; ii++) {
      // lower-triangular part
      for (int jj = 0; jj < ii; jj++) {
        hv(hr(j1+jj)) = Lx[psx + (ii + jj*nsrow)];
        hr(j1+jj) ++;
      }
      // diagonal
      if (unit_diag) {
        hv(hr(j1+ii)) = STS::one ();
      } else {
        hv(hr(j1+ii)) = Lx[psx + (ii + ii*nsrow)];
      }
      hr(j1+ii) ++;
      if (!cusparse) {
        // explicitly store zeros in upper-part
        for (int jj = ii+1; jj < nscol; jj++) {
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
      for (int jj = 0; jj < nscol; jj++) {
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
  Kokkos::deep_copy (values_view, hv);
  // create crs
  crsmat_t crsmat ("CrsMatrix", n, values_view, static_graph);
  return crsmat;
}

/* ========================================================================================= */
template <typename input_graph_t>
void check_supernode_sizes(const char *title, int n, int nsuper, int *nb, input_graph_t &graph) {

  auto rowmap_view = graph.row_map;

  typedef typename input_graph_t::row_map_type::non_const_type row_map_view_t;
  typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view (rowmap_view);

  Kokkos::deep_copy (hr, rowmap_view);

  int min_nsrow = 0, max_nsrow = 0, tot_nsrow = 0;
  int min_nscol = 0, max_nscol = 0, tot_nscol = 0;
  for (int s = 0; s <nsuper; s++) {
    //printf( " mb[%d]=%d, nb[%d]=%d, etree[%d]=%d (nrow=%d, ncol=%d)\n",s,mb[s],s,nb[s],s,etree[s],mb[s+1]-mb[s],nb[s+1]-nb[s] );
    int j1 = nb[s];
    int j2 = nb[s+1];

    int nscol = j2 - j1;
    int nsrow = hr(j1+1) - hr(j1);

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
  printf( " %s:\n",title );
  printf( "  + nsuper = %d\n",nsuper );
  printf( "  > nsrow: min = %d, max = %d, avg = %d\n",min_nsrow,max_nsrow,tot_nsrow/nsuper );
  printf( "  > nscol: min = %d, max = %d, avg = %d\n",min_nscol,max_nscol,tot_nscol/nsuper );
  std::cout << "    + Matrix size = " << n << std::endl;
  std::cout << "    + Total nnz   = " << hr(n) << std::endl;
  std::cout << "    + nnz / n     = " << hr(n)/n << std::endl;
}



/* ========================================================================================= */
template <typename input_graph_t>
void merge_supernodal_graph(int n, int *p_nsuper, int *nb,
                            input_graph_t &graphL, input_graph_t &graphU,
                            int *etree) {

  //auto graphL = L.graph; // in_graph
  auto row_mapL = graphL.row_map;
  auto entriesL = graphL.entries;

  //auto graphU = U.graph; // in_graph
  auto row_mapU = graphU.row_map;
  auto entriesU = graphU.entries;

  int nsuper = *p_nsuper;

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
  // construct new supernodes
  int *nb2 = new int[1+nsuper3];
  for (int s2 = 0, s = 0; s2 < nsuper3; s2++) {
    nb2[1+s2] = 0;
    // merging supernodal rows
    while(s < nsuper && map2[s] == s2) {
      nb2[1+s2] += (nb[s+1]-nb[s]);
      s ++;
    }
  }
  // copy back the new supernodes "offsets"
  nb2[0] = 0;
  for (int s = 0; s < nsuper3; s++) {
    nb2[s+1] = nb2[s]+nb2[s+1];
  }

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
  // convert/copy nb, mb, and etree
  for (int s = 0; s <nsuper3; s++) {
    // copy supernode id to column id
    nb[s+1] = nb2[s+1];
    // copy etree
    etree[s] = etree2[s];
  }
  *p_nsuper = nsuper3;
  delete[] nb2;
  delete[] etree2;
}


/* ========================================================================================= */
template <typename input_graph_t, typename output_graph_t>
output_graph_t 
generate_merged_supernodal_graph(int n,
                                 int nsuper, int *nb,
                                 int nsuper2, int *nb2,
                                 input_graph_t &graph, int *nnz) {

  //auto graphL = L.graph; // in_graph
  auto row_map = graph.row_map;
  auto entries = graph.entries;

  // ----------------------------------------------------------
  // now let me find nsrow for the merged supernode
  int nnzA = 0;
  int *mb2 = new int[nsuper2];
  int *work1 = new int[n];
  int *work2 = new int[n];
  int **rowind = new int*[nsuper2];
  for (int i = 0; i < n; i++) {
    work1[i] = 0;
  }
  for (int s2 = 0, s = 0; s2 < nsuper2; s2++) {
    mb2[s2] = 0;
    // merging supernodal rows
    // NOTE: SuperLU may not fill zeros to fill the supernodes
    //       So, these rows may be just subset of the supernodal rows
    while (s < nsuper && nb[s+1] <= nb2[s2+1]) {
      int j1 = nb[s];
      for (int k = row_map[j1]; k < row_map[j1+1]; k++) {
        // just taking union of rows
        if(work1[entries[k]] == 0) {
          work1[entries[k]] = 1;
          work2[mb2[s2]] = entries[k];
          mb2[s2] ++;
        }
      }
      s++;
    }
    // sort such that diagonal come on the top
    std::sort(work2, work2+mb2[s2]);

    // save nonzero row ids
    rowind[s2] = new int [mb2[s2]];
    for (int k = 0; k < mb2[s2]; k++) {
      rowind[s2][k] = work2[k];
      work1[work2[k]] = 0;
    }
    nnzA += (nb2[s2+1]-nb2[s2]) * mb2[s2];
  }
  delete[] work1;
  delete[] work2;

  // ----------------------------------------------------------
  // now let's create crs graph
  typedef typename output_graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename output_graph_t::entries_type::non_const_type    cols_view_t;

  row_map_view_t rowmap_view ("rowmap_view", n+1);
  cols_view_t    column_view ("colmap_view", nnzA);

  typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view (rowmap_view);
  typename cols_view_t::HostMirror    hc = Kokkos::create_mirror_view (column_view);

  nnzA = 0;
  hr(0) = 0;
  for (int s2 = 0; s2 < nsuper2; s2++) {
    int nsrow = mb2[s2];
    for (int j = nb2[s2]; j < nb2[s2+1]; j++) {
      for (int k = 0; k < nsrow; k++) {
        hc(nnzA) = rowind[s2][k];
        nnzA ++;
      }
      hr(j+1) = nnzA;
    }
    delete[] rowind[s2];
  }
  delete[] mb2;
  delete[] rowind;
  *nnz = nnzA; 

  // deepcopy
  Kokkos::deep_copy (rowmap_view, hr);
  Kokkos::deep_copy (column_view, hc);

  // create crs
  output_graph_t static_graph (column_view, rowmap_view);
  return static_graph;
}


/* ========================================================================================= */
template <typename host_crsmat_t, typename graph_t, typename crsmat_t>
crsmat_t
read_merged_supernodes(int n, int nsuper, int *nb,
                       bool unit_diag, bool invert_diag, bool invert_offdiag,
                       host_crsmat_t &L, graph_t &static_graph) {

  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type    cols_view_t;
  typedef typename crsmat_t::values_type::non_const_type  values_view_t;

  typedef typename values_view_t::value_type scalar_t;
  typedef Kokkos::Details::ArithTraits<scalar_t> STS;

  // original matrix
  auto graphL = L.graph; // in_graph
  auto row_mapL = graphL.row_map;
  auto entriesL = graphL.entries;
  auto valuesL  = L.values;

  // merged graph
  auto rowmap_view = static_graph.row_map;
  auto column_view = static_graph.entries;

  typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view (rowmap_view);
  typename cols_view_t::HostMirror    hc = Kokkos::create_mirror_view (column_view);
  Kokkos::deep_copy (hr, rowmap_view);
  Kokkos::deep_copy (hc, column_view);

  // ----------------------------------------------------------
  // now let's copy numerical values
  double *dwork = new double[n];
  for (int i = 0; i < n; i++) {
    dwork[i] = 0.0;
  }

  int nnzA = hr (n);
  typedef typename crsmat_t::values_type::non_const_type values_view_t;
  values_view_t values_view ("values_view", nnzA);

  typename values_view_t::HostMirror hv = Kokkos::create_mirror_view (values_view);

  for (int s2 = 0; s2 < nsuper; s2++) {
    for (int j = nb[s2]; j < nb[s2+1]; j++) {
      for (int k = row_mapL[j]; k < row_mapL[j+1]; k++) {
        dwork[entriesL[k]] = valuesL[k];
      }
      for (int k = hr (j); k < hr (j+1); k++) {
        hv(k) = dwork[hc(k)];
      }
      for (int k = row_mapL[j]; k < row_mapL[j+1]; k++) {
        dwork[entriesL[k]] = STS::zero ();
      }
    }

    int j1 = nb[s2];
    int nsrow = hr(j1+1) - hr(j1);
    int nscol = nb[s2+1]-nb[s2];
    if (invert_diag) {
      int nnzD = hr (j1);
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
  delete[] dwork;

  // deepcopy
  Kokkos::deep_copy (values_view, hv);

  // create crs
  crsmat_t crsmat("CrsMatrix", n, values_view, static_graph);
  return crsmat;
}
#endif


/* ========================================================================================= */
template<typename scalar_t>
void forwardP_supernode(int n, int *perm_r, int nrhs, scalar_t *B, int ldb, scalar_t *X, int ldx) {

  /* Permute right hand sides to form Pr*B */
  for (int j = 0; j < nrhs; j++) {
      double *rhs_work = &B[j*ldb];
      double *sol_work = &X[j*ldx];
      for (int k = 0; k < n; k++) sol_work[perm_r[k]] = rhs_work[k];
  }
}

template<typename scalar_t>
void backwardP_supernode(int n, int *perm_c, int nrhs, scalar_t *B, int ldb, scalar_t *X, int ldx) {

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


template <typename crsmat_t, typename host_crsmat_t>
bool check_cusparse(host_crsmat_t &Mtx, crsmat_t &L, crsmat_t &U, int *perm_r, int *perm_c, double tol, int loop) {
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE) 
  typedef typename crsmat_t::values_type::non_const_type values_view_t;
  typedef typename values_view_t::value_type scalar_t;

  typedef typename host_crsmat_t::values_type::non_const_type host_values_view_t;
  typedef typename host_values_view_t::value_type host_scalar_t;

  typedef typename values_view_t::execution_space execution_space;
  typedef typename execution_space::memory_space memory_space;

  typedef typename host_values_view_t::execution_space host_execution_space;
  typedef typename host_execution_space::memory_space host_memory_space;

  typedef Kokkos::View< scalar_t*, host_memory_space > host_scalar_view_t;
  typedef Kokkos::View< scalar_t*,      memory_space > scalar_view_t;

  scalar_t ZERO = scalar_t(0);
  scalar_t ONE = scalar_t(1);

  Kokkos::Timer timer;
  const int nrows = Mtx.graph.numRows();

  // ==============================================
  // > create a handle
  cusparseStatus_t status;
  cusparseHandle_t handle = 0; 
  status = cusparseCreate(&handle);
  if (CUSPARSE_STATUS_SUCCESS != status) {
    std::cout << "handle create status error name " << (status) << std::endl;
  }
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST); // scalars are passed by reference on host

  // > create a empty info structure
  csrsv2Info_t info = 0; 
  status = cusparseCreateCsrsv2Info(&info);
  if (CUSPARSE_STATUS_SUCCESS != status) {
    std::cout << "csrsv2info create status error name " << (status) << std::endl;
  }

  // ==============================================
  // Preparing for L-solve
  // step 1: create a descriptor
  int nnzL = L.nnz();
  auto graphL = L.graph; // in_graph
  auto row_mapL = graphL.row_map;
  auto entriesL = graphL.entries;
  auto valuesL  = L.values;

  // NOTE: it is stored in CSC = UPPER + TRANSPOSE
  cusparseMatDescr_t descrL = 0;
  status = cusparseCreateMatDescr(&descrL);
  if (CUSPARSE_STATUS_SUCCESS != status) {
    std::cout << "matdescr create status error name " << (status) << std::endl;
  }
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
  if (CUSPARSE_STATUS_SUCCESS != status) {
    std::cout << "analysis status error name " << (status) << std::endl;
  }
  // L has unit diagonal, so no structural zero is reported.

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

  // ==============================================
  // step 1: apply forward-pivot to rhs on the host
  host_scalar_view_t tmp_host ("temp", nrows);
  forwardP_supernode<scalar_t> (nrows, perm_r, 1, rhs_host.data(), nrows, tmp_host.data(), nrows);

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
  if (CUSPARSE_STATUS_SUCCESS != status) {
    std::cout << "solve status error name " << (status) << std::endl;
  }
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
  int nnzU = U.nnz();
  auto graphU = U.graph; // in_graph
  auto row_mapU = graphU.row_map;
  auto entriesU = graphU.entries;
  auto valuesU  = U.values;

  // ==============================================
  // step 1: create a descriptor
  // NOTE: it is stored in CSR = UPPER + NO-TRANSPOSE
  cusparseMatDescr_t descrU = 0;
  status = cusparseCreateMatDescr(&descrU);
  if (CUSPARSE_STATUS_SUCCESS != status) {
    std::cout << "matdescr create status error name " << (status) << std::endl;
  }
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
  if (CUSPARSE_STATUS_SUCCESS != status) {
    std::cout << "analysis status error name " << (status) << std::endl;
  }
  status = cusparseXcsrsv2_zeroPivot(handle, info, &structural_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status){
    printf("U(%d,%d) is missing\n", structural_zero, structural_zero);
  }

  // ==============================================
  // step 1: solve U*y = x
  timer.reset();
  status = cusparseDcsrsv2_solve(handle, trans, nrows, nnzU, &alpha, descrU,
                                 valuesU.data(), row_mapU.data(), entriesU.data(), info,
                                 sol.data(), rhs.data(), policy, pBufferU);
  Kokkos::fence();
  std::cout << "  Cusparse Solve Time   : " << timer.seconds() << std::endl;
  if (CUSPARSE_STATUS_SUCCESS != status) {
    std::cout << "solve status error name " << (status) << std::endl;
  }
  status = cusparseXcsrsv2_zeroPivot(handle, info, &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status){
    printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
  }

  // ==============================================
  // copy solution to host
  Kokkos::deep_copy(tmp_host, rhs);
  // apply backward-pivot
  backwardP_supernode<scalar_t>(nrows, perm_c, 1, tmp_host.data(), nrows, sol_host.data(), nrows);
  //printf( "x=[" );
  //for (int ii=0; ii<nrows; ii++) printf( " %d %.16e\n",ii,tmp_host(ii) );
  //printf( "];\n" );

  // ==============================================
  // Error Check ** on host **
  Kokkos::fence();
  std::cout << std::endl;
  bool success = check_errors(tol, Mtx, rhs_host, sol_host);
  std::cout << std::endl;

  if (success) {
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
      if(time > max_time) max_time = time;
      if(time < min_time) min_time = time;
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
      if(time > max_time) max_time = time;
      if(time < min_time) min_time = time;
    }
    std::cout << " U-solve: loop = " << loop << std::endl;
    std::cout << "  LOOP_AVG_TIME:  " << ave_time/loop << std::endl;
    std::cout << "  LOOP_MAX_TIME:  " << max_time << std::endl;
    std::cout << "  LOOP_MIN_TIME:  " << min_time << std::endl << std::endl;
  }
  return success;
#else
  printf( " KOKKOSKERNELS_ENABLE_TPL_CUSPARSE **not** enabled\n" );
  return false;
#endif
}

#endif
