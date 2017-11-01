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

#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <vector>
#include "KokkosKernels_Handle.hpp"

#ifndef _KOKKOSCOLORINGD2IMP_HPP
#define _KOKKOSCOLORINGD2IMP_HPP


// #define EBCOLORING_HIGHER_QUALITY        //suggested
namespace KokkosGraph{

namespace Impl{

// #define VB_COLORING_FORBIDDEN_SIZE 64
// #define VBBIT_COLORING_FORBIDDEN_SIZE 64
/*! \brief Base class for graph coloring purposes.
 *  Each color represents the set of the vertices that are independent,
 *  e.g. no vertex having same color shares an edge.
 *  General aim is to find the minimum number of colors, minimum number of independent sets.
 */
template <typename HandleType, typename lno_row_view_t_, typename lno_nnz_view_t_>
class GraphColorD2 {
public:

  typedef lno_row_view_t_ in_lno_row_view_t;
  typedef lno_nnz_view_t_ in_lno_nnz_view_t;

  typedef typename HandleType::color_t color_t;
  typedef typename HandleType::color_view_t color_view_t;

  typedef typename HandleType::size_type size_type;
  typedef typename HandleType::nnz_lno_t nnz_lno_t;

  typedef typename HandleType::HandleExecSpace MyExecSpace;
  typedef typename HandleType::HandleTempMemorySpace MyTempMemorySpace;
  // typedef typename HandleType::HandlePersistentMemorySpace MyPersistentMemorySpace;

  typedef typename HandleType::const_size_type const_size_type;
  typedef typename lno_row_view_t_::const_type const_lno_row_view_t;

  typedef typename lno_nnz_view_t_::const_type const_lno_nnz_view_t;
  typedef typename lno_nnz_view_t_::non_const_type non_const_lno_nnz_view_t;

  typedef typename HandleType::size_type_temp_work_view_t size_type_temp_work_view_t;
  typedef typename HandleType::scalar_temp_work_view_t scalar_temp_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_view_t nnz_lno_persistent_work_view_t;


protected:
  nnz_lno_t nr, nc; // num rows, num cols
  size_type ne; //# edges
  const_lno_row_view_t xadj, t_xadj; //rowmap, transpose of rowmap
  const_lno_nnz_view_t adj, t_adj; // entries, transpose of entries
  HandleType *cp;

public:
  /**
   * \brief GraphColor constructor.
   * \param nv_: number of vertices in the graph
   * \param ne_: number of edges in the graph
   * \param row_map: the xadj array of the graph. Its size is nv_ +1
   * \param entries: adjacency array of the graph. Its size is ne_
   * \param coloring_handle: GraphColoringHandle object that holds the specification about the graph coloring,
   *    including parameters.
   */
  GraphColor(
      nnz_lno_t nr_,
      nnz_lno_t nc_,
      size_type ne_,
      const_lno_row_view_t row_map,
      const_lno_nnz_view_t entries,
      const_lno_row_view_t t_row_map,
      const_lno_nnz_view_t t_entries,
      HandleType *coloring_handle):
        nr (nr_), nc (nc_), ne(ne_), xadj(row_map), adj(entries), t_xadj(t_row_map), t_adj(t_entries),
        cp(coloring_handle){}

  /** \brief GraphColor destructor.
   */
  virtual ~GraphColor(){}


  /** \brief Function to color the vertices of the graphs. This is the base class,
   * therefore, it only performs sequential coloring on the host device, ignoring the execution space.
   * \param colors is the output array corresponding the color of each vertex.Size is this->nv.
   *   Attn: Color array must be nonnegative numbers. If there is no initial colors,
   *   it should be all initialized with zeros. Any positive value in the given array, will make the
   *   algorithm to assume that the color is fixed for the corresponding vertex.
   * \param num_phases: The number of iterations (phases) that algorithm takes to converge.
   */
  virtual void color_graph(){

    // WCMCLEN: Brian's Code

    std::string algName = "SPGEMM_KK_MEMSPEED";
    cp->create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(algName));

    //typedef typename rowptrs_view::const_type const_rowptrs_view;
    //typedef typename colinds_view::const_type const_colinds_view;
    size_type_temp_work_view_t cRowptrs("cRowptrs", nr+1);
// typedef typename HandleType::size_type_temp_work_view_t size_type_temp_work_view_t;
// typedef typename HandleType::scalar_temp_work_view_t scalar_temp_work_view_t;
// typedef typename HandleType::nnz_lno_persistent_work_view_t nnz_lno_persistent_work_view_t;


    // Call symbolic multiplication of graph with itself (no transposes, and A and B are the same)
    KokkosSparse::Experimental::spgemm_symbolic(cp,
        nr, nc, nr,
        xadj, adj, false,
        t_xadj, t_adj, false, cRowptrs); 
    
    // Get num nz in C
    auto Cnnz = cp->get_spgemm_handle()->get_c_nnz();

    // Must create placeholder value views for A and C (values are meaningless)
    // Said above that the scalar view type is the same as the colinds view type
    scalar_temp_work_view_t aFakeValues("A/B placeholder values (meaningless)", adj.size());

    // Allocate C entries array, and placeholder values
    nnz_lno_persistent_work_view_t cColinds("C colinds", Cnnz);
    scalar_temp_work_view_t cFakeValues("C placeholder values (meaningless)", Cnnz);

    // Run the numeric kernel
    KokkosSparse::Experimental::spgemm_numeric(cp,
        nr, nc, nr,
        xadj, adj, aFakeValues, false, 
        t_xadj, t_adj, aFakeValues, false,
        cRowptrs, cColinds, cFakeValues);

    // done with spgemm 
    cp->destroy_spgemm_handle();

    // Now run distance-1 graph coloring on C
    // Use LocalOrdinal for storing colors
    
    // cp->create_graph_coloring_handle();

    KokkosGraph::Experimental::graph_color(
        cp,
        numRows, numRows,
        (const_rowptrs_view) cRowptrs, (const_colinds_view) cColinds);

    // extract the colors
    //auto coloringHandle = cp->get_graph_coloring_handle();
    //color_view_t colorsDevice = coloringHandle->get_vertex_colors();

    //clean up coloring handle
    // cp->destroy_graph_coloring_handle();

  }

};


} // end Impl namespace 
} // end namespace

#endif
