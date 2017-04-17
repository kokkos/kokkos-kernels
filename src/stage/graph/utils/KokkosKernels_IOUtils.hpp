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
#include <fstream>
#include <string>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <vector>
#include <stdexcept>
#ifndef _KOKKOSKERNELSIOUTILS_HPP
#define _KOKKOSKERNELSIOUTILS_HPP


#include <Kokkos_Core.hpp>
#include "KokkosKernels_SimpleUtils.hpp"

namespace KokkosKernels{
namespace Experimental{

namespace Util{

//TODO: need to fix the size_type. All over the reading inputs are lno_t.

template <typename stype>
void md_malloc(stype **arr, size_t n, std::string alloc_str = ""){
  *arr = new stype[n];
  if (*arr == NULL){
    throw std::runtime_error ("Memory Allocation Problem\n");
  }
}

template <typename idx_array_type>
inline void kk_write_1Dview_to_file(idx_array_type view, const char *filename){

  typedef typename idx_array_type::HostMirror host_type;
  typedef typename idx_array_type::size_type idx;
  host_type host_view = Kokkos::create_mirror_view (view);
  Kokkos::deep_copy (host_view , view);
  Kokkos::fence();
  std::ofstream myFile (filename, std::ios::out );
  for (size_t i = 0; i < view.dimension_0(); ++i){
	  myFile << host_view(i) << std::endl;
  }
  myFile.close();
}

template <typename idx_array_type>
inline void kk_read_1Dview_from_file(idx_array_type &view, const char *filename){

  typedef typename idx_array_type::HostMirror host_type;
  typedef typename idx_array_type::size_type idx;
  host_type host_view = Kokkos::create_mirror_view (view);
  std::ifstream myFile (filename, std::ios::in );

  for (size_t i = 0; i < view.dimension_0(); ++i){
	  myFile >> host_view(i);
  }
  myFile.close();
  Kokkos::deep_copy (view, host_view);
  Kokkos::fence();
}

template <typename idx, typename wt>
struct Edge{
  idx src;
  idx dst;
  wt ew;
  bool operator<(const Edge <idx,wt> & a) const
  {
    //return !((this->src < a.src) || (this->src == a.src && this->dst < a.dst));
    return (this->src < a.src) || (this->src == a.src && this->dst < a.dst);
  }
};


template <typename idx>
void convert_crs_to_lower_triangle_edge_list(idx nv, idx *xadj, idx *adj, idx *lower_triangle_srcs, idx *lower_triangle_dests){
  idx ind = 0;
  for (idx i = 0; i < nv; ++i){
    idx xb = xadj[i];
    idx xe = xadj[i+1];
    for (idx j = xb; j < xe; ++j){
      idx dst = adj[j];
      if (i < dst){
        lower_triangle_srcs[ind] = i;
        lower_triangle_dests[ind++] = dst;
      }
    }
  }
}

template <typename idx>
void convert_crs_to_edge_list(idx nv, idx *xadj, idx *srcs){
  for (idx i = 0; i < nv; ++i){
    idx xb = xadj[i];
    idx xe = xadj[i+1];
    for (idx j = xb; j < xe; ++j){
      srcs[j] = i;
    }
  }
}

template <typename idx, typename wt>
void convert_edge_list_to_csr (idx nv, idx ne, idx *srcs, idx *dests, wt *ew, idx *xadj, idx *adj, wt *crs_ew){

  std::vector <struct Edge<idx, wt> > edges (ne);
  for(idx i = 0; i < ne; ++i){
    edges[i].src = srcs[i];
    edges[i].dst = dests[i];
    edges[i].ew = ew[i];
  }
  std::sort (edges.begin(), edges.begin() + ne);

  idx eind = 0;
  for (idx i = 0; i < nv; ++i){
    (xadj)[i] = eind;
    while (edges[eind].src == i){
      (adj)[eind] = edges[eind].dst;
      (*crs_ew)[eind] = edges[eind].ew;
      ++eind;
    }
  }
  xadj[nv] = eind;

}


template <typename idx, typename wt>
void write_edgelist_bin(
    size_t ne,
    const idx *edge_begins,
    const  idx *edge_ends,
    const  wt *ew,
    const  char *filename){
  std::ofstream myFile (filename, std::ios::out | std::ios::binary);
  myFile.write((char *) &ne, sizeof(idx));
  myFile.write((char *) edge_begins, sizeof(idx) * (ne));
  myFile.write((char *) edge_ends, sizeof(idx) * (ne));
  myFile.write((char *) ew, sizeof(wt) * (ne));
  myFile.close();
}

template <typename idx, typename wt>
void read_edgelist_bin(
    idx *ne,
    idx **edge_begins,
    idx **edge_ends,
    wt **ew,
    const  char *filename){

  std::ifstream myFile (filename, std::ios::in | std::ios::binary);


  myFile.read((char *) ne, sizeof(idx));
  md_malloc<idx>(edge_begins, *ne);
  md_malloc<idx>(edge_ends, *ne);
  md_malloc<wt> (ew, *ne);
  myFile.read((char *) *edge_begins, sizeof(idx) * (*ne));
  myFile.read((char *) *edge_ends, sizeof(idx) * (*ne));
  myFile.read((char *) *ew, sizeof(wt) * (*ne));
  myFile.close();
}



template <typename lno_t, typename size_type, typename scalar_t>
void write_graph_bin(lno_t nv, size_type ne,const size_type *xadj,const  lno_t *adj,const  scalar_t *ew,const  char *filename){
  std::ofstream myFile (filename, std::ios::out | std::ios::binary);
  myFile.write((char *) &nv, sizeof(lno_t));
  myFile.write((char *) &ne, sizeof(size_type));
  myFile.write((char *) xadj, sizeof(size_type) * (nv + 1));
  myFile.write((char *) adj, sizeof(lno_t) * (ne));
  myFile.write((char *) ew, sizeof(scalar_t) * (ne));
  myFile.close();
}



template <typename lno_t, typename size_type, typename scalar_t>
void read_graph_bin(lno_t *nv, size_type *ne,size_type **xadj, lno_t **adj, scalar_t **ew, const char *filename){

  std::ifstream myFile (filename, std::ios::in | std::ios::binary);

  myFile.read((char *) nv, sizeof(lno_t));
  myFile.read((char *) ne, sizeof(size_type));
  md_malloc<size_type>(xadj, *nv+1);
  md_malloc<lno_t>(adj, *ne);
  md_malloc<scalar_t> (ew, *ne);
  myFile.read((char *) *xadj, sizeof(size_type) * (*nv + 1));
  myFile.read((char *) *adj, sizeof(lno_t) * (*ne));
  myFile.read((char *) *ew, sizeof(scalar_t) * (*ne));
  myFile.close();
}



inline bool endswith (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}


template <typename lno_t, typename size_type, typename scalar_t>
int read_mtx (
    const char *fileName,
    lno_t *nv, size_type *ne,
    size_type **xadj, lno_t **adj, scalar_t **ew,
    bool symmetrize = false, bool remove_diagonal = true,
    bool transpose = false){

  std::ifstream mmf (fileName, std::ifstream::in);
  if (!mmf.is_open()) {
    throw std::runtime_error ("File cannot be opened\n");
  }

  std::string fline = "";
  getline(mmf, fline);

  if (fline.size() < 2 || fline[0] != '%' || fline[1] != '%'){
    throw std::runtime_error ("Invalid MM file. Line-1\n");
  }


  int mtx_object = 0; // 0- matrix, 1-vector
  int mtx_format = 0; // 0- coordinate; 1- array
  //int mtx_field = 0; //0-real, 1-double, 2-complex, 3- integer, 4-pattern
  int mtx_sym = 0; //0-general, 1-symmetric, 2-skew-symmetric, 3-hermitian

  if (fline.find("matrix") != std::string::npos){
    mtx_object = 0;
  } else if (fline.find("vector") != std::string::npos){
    mtx_object = 1;
  }

  if (fline.find("coordinate") != std::string::npos){
    mtx_format = 0;
  }
  else if (fline.find("array") == std::string::npos){
    mtx_format = 1;
  }

  if (fline.find("real") != std::string::npos){
    //mtx_field = 0;
  }
  else if (fline.find("double") != std::string::npos){
    //mtx_field = 1;
  }
  else if (fline.find("complex") != std::string::npos){
    //mtx_field = 2;
  }
  else if (fline.find("integer") != std::string::npos){
    //mtx_field = 3;
  }
  else if (fline.find("pattern") != std::string::npos){
    //mtx_field = 4;
  }

  if (fline.find("skew-symmetric") != std::string::npos){
    mtx_sym = 2;
  }
  else if (fline.find("symmetric") != std::string::npos){
    mtx_sym = 1;
  }
  else if (fline.find("hermitian") != std::string::npos){
    mtx_sym = 3;
  }
  else if (fline.find("general") != std::string::npos){
    mtx_sym = 0;
  }

  if (mtx_object == 1) {
    throw std::runtime_error ("VECTOR TYPE NOT HANDLED YET\n");
  }
  if (mtx_format == 1) {
    throw std::runtime_error ("ARRAY TYPE NOT HANDLED YET\n");
  }
  if (!symmetrize && (mtx_sym == 2 || mtx_sym == 3)) {
    throw std::runtime_error ("SKEW-SYMMETRIC and HERMITIAN TYPE NOT HANDLED YET\n");
  }


  while(1){
    getline(mmf, fline);
    if(fline[0] != '%') break;
  }
  std::stringstream ss (fline);
  lno_t nr = 0, nc = 0;
  size_type nnz = 0;

  ss >> nr >> nc >> nnz;


  //if (nr != nc) {std::cerr << "NON-SQUARE MATRIX TYPE NOT HANDLED YET"<< std::endl; return (1); }
  size_type noEdges = nnz;
  if (mtx_sym == 1 || symmetrize) noEdges = 2 * nnz;

  std::vector <struct Edge<lno_t, scalar_t> > edges (noEdges);
  size_type nE = 0;
  lno_t noDiagonal = 0;
  for (size_type i = 0; i < nnz; ++i){
    getline(mmf, fline);
    std::stringstream ss2 (fline);
    struct Edge<lno_t, scalar_t> tmp;
    lno_t s,d;
    scalar_t w;
    ss2 >> s >> d >> w;
    if (!transpose){
      tmp.src = s - 1;
      tmp.dst = d - 1;
      tmp.ew = w;
    }
    else {
      tmp.src = d - 1;
      tmp.dst = s - 1;
      tmp.ew = w;
    }

    if (tmp.src == tmp.dst){
      noDiagonal++;
      if (!remove_diagonal){
        edges[nE++] = tmp;
      }
      continue;
    }
    edges[nE++] = tmp;
    if (mtx_sym == 1 || symmetrize){
      struct Edge<lno_t, scalar_t> tmp2;
      tmp2.src = tmp.dst;
      tmp2.dst = tmp.src;
      tmp2.ew = tmp.ew;
      edges[nE++] = tmp2;
    }
  }

  mmf.close();

  std::sort (edges.begin(), edges.begin() + nE);

  if (transpose){
    lno_t tmp = nr;
    nr = nc;
    nc = tmp;
  }

  //idx *nv, idx *ne, idx **xadj, idx **adj, wt **wt

  *nv = nr;

  *ne = nE;
  //*xadj = new idx[nr + 1];
  md_malloc<lno_t>(xadj, nr+1);
  //*adj = new idx[nE];
  md_malloc<lno_t>(adj, nE);
  //*ew = new wt[nE];
  md_malloc<scalar_t>(ew, nE);

  size_type eind = 0;
  size_type actual = 0;
  for (lno_t i = 0; i < nr; ++i){
    (*xadj)[i] = actual;
    bool is_first = true;
    while (edges[eind].src == i){
      if (is_first || !symmetrize || eind == 0 || (eind > 0 && edges[eind - 1].dst != edges[eind].dst)){

        (*adj)[actual] = edges[eind].dst;
        (*ew)[actual] = edges[eind].ew;
        ++actual;
      }
      is_first = false;
      ++eind;
    }
  }
  (*xadj)[nr] = actual;
  *ne = actual;
  return 0;
}

template <typename lno_t, typename size_type, typename scalar_t>
void read_matrix(lno_t *nv, size_type *ne,size_type **xadj, lno_t **adj, scalar_t **ew, const char *filename){

  std::string strfilename(filename);
  if (endswith(strfilename, ".mtx")){
    read_mtx (
        filename,
        nv, ne,
        xadj, adj, ew,false,false,false);
  }

  else if (endswith(strfilename, ".bin")){
    read_graph_bin(nv, ne,xadj, adj, ew, filename);
  }
  else {
    throw std::runtime_error ("Reader is not available\n");
  }
}

template <typename crsMat_t>
crsMat_t read_kokkos_crst_matrix(const char * filename_){

  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type   cols_view_t;
  typedef typename crsMat_t::values_type::non_const_type values_view_t;

  typedef typename row_map_view_t::value_type size_type;
  typedef typename cols_view_t::value_type   lno_t;
  typedef typename values_view_t::value_type scalar_t;


  lno_t nv, *adj;
  size_type *xadj, nnzA;
  scalar_t *values;
  read_matrix<lno_t, size_type, scalar_t>(
      &nv, &nnzA, &xadj, &adj, &values, filename_);

  row_map_view_t rowmap_view("rowmap_view", nv+1);
  cols_view_t columns_view("colsmap_view", nnzA);
  values_view_t values_view("values_view", nnzA);


  {
    typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view (rowmap_view);
    typename cols_view_t::HostMirror hc = Kokkos::create_mirror_view (columns_view);
    typename values_view_t::HostMirror hv = Kokkos::create_mirror_view (values_view);

    for (lno_t i = 0; i <= nv; ++i){
      hr(i) = xadj[i];
    }

    for (size_type i = 0; i < nnzA; ++i){
      hc(i) = adj[i];
      hv(i) = values[i];
    }
    Kokkos::deep_copy (rowmap_view , hr);
    Kokkos::deep_copy (columns_view , hc);
    Kokkos::deep_copy (values_view , hv);
  }

  lno_t ncols = 0;
  KokkosKernels::Experimental::Util::kk_view_reduce_max
      <cols_view_t, typename crsMat_t::execution_space>(nnzA, columns_view, ncols);
  ncols += 1;

  graph_t static_graph (columns_view, rowmap_view);
  crsMat_t crsmat("CrsMatrix", ncols, values_view, static_graph);
  delete [] xadj; delete [] adj; delete [] values;
  return crsmat;
}



template <typename size_type, typename nnz_lno_t>
inline void kk_sequential_create_incidence_matrix(
    nnz_lno_t num_rows,
    size_type *xadj,
    nnz_lno_t *adj,
    size_type *i_adj //output. preallocated
  ){

  std::vector<size_type> c_xadj(num_rows);
  for (nnz_lno_t i = 0; i < num_rows; i++){
    c_xadj[i] = xadj[i];
  }
  int eCnt=0;
  for (nnz_lno_t i = 0; i < num_rows; i++){
    size_type begin = xadj[i];
    size_type end = xadj[i + 1];
    nnz_lno_t adjsize = end - begin;

    for (nnz_lno_t j = 0; j < adjsize; j++){
      size_type aind = j + begin;
      nnz_lno_t col = adj[aind];
      if (i < col){
        i_adj[c_xadj[i]++] = eCnt;
        i_adj[c_xadj[col]++] = eCnt++;
      }
    }
  }

  for (nnz_lno_t i = 0; i < num_rows; i++){
    if (c_xadj[i] != xadj[i+1]){
      std::cout << "i:" << i << " c_xadj[i]:" << c_xadj[i] << " xadj[i+1]:" << xadj[i+1] << std::endl;
    }
  }
}

}
}

}
#endif
