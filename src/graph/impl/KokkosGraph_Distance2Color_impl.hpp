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
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <Kokkos_MemoryTraits.hpp>

#include "KokkosKernels_Handle.hpp"
#include "KokkosGraph_GraphColorHandle.hpp"
#include "KokkosGraph_GraphColor.hpp"

#ifndef _KOKKOSCOLORINGD2IMP_HPP
#define _KOKKOSCOLORINGD2IMP_HPP


// #define EBCOLORING_HIGHER_QUALITY        //suggested
//
namespace KokkosGraph { 

namespace Impl {

#define VB_COLORING_FORBIDDEN_SIZE 64
// #define VBBIT_COLORING_FORBIDDEN_SIZE 64



/*! \brief Base class for graph coloring purposes.
 *  Each color represents the set of the vertices that are independent,
 *  e.g. no vertex having same color shares an edge.
 *  General aim is to find the minimum number of colors, minimum number of independent sets.
 */
// TODO: This routine has a lot of extra typedefs, members, etc. that should be cleaned.
template <typename HandleType, typename lno_row_view_t_, typename lno_nnz_view_t_, typename clno_row_view_t_, typename clno_nnz_view_t_ >
class GraphColorD2_MatrixSquared
{
public:

  typedef lno_row_view_t_ in_lno_row_view_t;
  typedef lno_nnz_view_t_ in_lno_nnz_view_t;

  typedef typename HandleType::GraphColoringHandleType::color_t      color_t;
  typedef typename HandleType::GraphColoringHandleType::color_view_t color_view_type;

  typedef typename HandleType::size_type size_type;
  typedef typename HandleType::nnz_lno_t nnz_lno_t;

  typedef typename HandleType::HandleExecSpace        MyExecSpace;
  typedef typename HandleType::HandleTempMemorySpace  MyTempMemorySpace;
  // typedef typename HandleType::HandlePersistentMemorySpace MyPersistentMemorySpace;
  typedef typename HandleType::const_size_type        const_size_type;

  typedef typename lno_row_view_t_::device_type    row_lno_view_device_t;
  typedef typename lno_row_view_t_::const_type     const_lno_row_view_t;
  typedef typename lno_nnz_view_t_::const_type     const_lno_nnz_view_t;
  typedef typename lno_nnz_view_t_::non_const_type non_const_lno_nnz_view_t;

  typedef typename clno_row_view_t_::const_type     const_clno_row_view_t;
  typedef typename clno_nnz_view_t_::const_type     const_clno_nnz_view_t;
  typedef typename clno_nnz_view_t_::non_const_type non_const_clno_nnz_view_t;

  typedef typename HandleType::size_type_temp_work_view_t     size_type_temp_work_view_t;
  typedef typename HandleType::scalar_temp_work_view_t        scalar_temp_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_view_t nnz_lno_persistent_work_view_t;

  typedef typename HandleType::nnz_lno_temp_work_view_t nnz_lno_temp_work_view_t;
  typedef typename Kokkos::View<nnz_lno_t, row_lno_view_device_t> single_dim_index_view_type;

  typedef Kokkos::RangePolicy<MyExecSpace> my_exec_space;


protected:
  nnz_lno_t             nr;       // num_rows  (# verts)
  nnz_lno_t             nc;       // num cols
  size_type             ne;       // # edges
  const_lno_row_view_t  xadj;     // rowmap, transpose of rowmap
  const_lno_nnz_view_t  adj;      // entries, transpose of entries   (size = # edges)
  const_clno_row_view_t t_xadj;   // rowmap, transpose of rowmap
  const_clno_nnz_view_t t_adj;    // entries, transpose of entries
  nnz_lno_t             nv;       // num vertices
  HandleType*           cp;       // the handle.

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
  GraphColorD2_MatrixSquared (nnz_lno_t             nr_,
                nnz_lno_t             nc_,
                size_type             ne_,
                const_lno_row_view_t  row_map,
                const_lno_nnz_view_t  entries,
                const_clno_row_view_t t_row_map,
                const_clno_nnz_view_t t_entries,
                HandleType*           coloring_handle):
        nr (nr_), 
        nc (nc_), 
        ne(ne_), 
        xadj(row_map), 
        adj(entries), 
        t_xadj(t_row_map), 
        t_adj(t_entries),
        nv (nr_), 
        cp(coloring_handle)
        //_chunkSize(coloring_handle->get_vb_chunk_size()),
        // _max_num_iterations(1000)
  {}


  /** \brief GraphColor destructor.
   */
  virtual ~GraphColorD2_MatrixSquared () 
  {}


  /** \brief Function to color the vertices of the graphs. This is the base class,
   * therefore, it only performs sequential coloring on the host device, ignoring the execution space.
   * \param colors is the output array corresponding the color of each vertex.Size is this->nv.
   *   Attn: Color array must be nonnegative numbers. If there is no initial colors,
   *   it should be all initialized with zeros. Any positive value in the given array, will make the
   *   algorithm to assume that the color is fixed for the corresponding vertex.
   * \param num_phases: The number of iterations (phases) that algorithm takes to converge.
   */
  virtual void color_graph_d2_matrix_squared()
  {
    // WCMCLEN: Brian's Code
    std::cout << ">>> WCMCLEN color_graph_d2_matrix_squared (KokkosGraph_Distance2Color_impl.hpp)" << std::endl;

    std::string algName = "SPGEMM_KK_MEMSPEED";
    cp->create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(algName));

    size_type_temp_work_view_t cRowptrs("cRowptrs", nr+1);

    // Call symbolic multiplication of graph with itself (no transposes, and A and B are the same)
    KokkosSparse::Experimental::spgemm_symbolic(cp, nr, nc, nr, xadj, adj, false, t_xadj, t_adj, false, cRowptrs);
    
    // Get num nz in C
    auto Cnnz = cp->get_spgemm_handle()->get_c_nnz();

    // Must create placeholder value views for A and C (values are meaningless)
    // Said above that the scalar view type is the same as the colinds view type
    scalar_temp_work_view_t aFakeValues("A/B placeholder values (meaningless)", adj.size());

    // Allocate C entries array, and placeholder values
    nnz_lno_persistent_work_view_t cColinds("C colinds", Cnnz);
    scalar_temp_work_view_t cFakeValues("C placeholder values (meaningless)", Cnnz);

    // Run the numeric kernel
    KokkosSparse::Experimental::spgemm_numeric(cp, nr, nc, nr, xadj, adj, aFakeValues, false, t_xadj, t_adj, 
                                               aFakeValues, false, cRowptrs, cColinds, cFakeValues);

    // done with spgemm 
    cp->destroy_spgemm_handle();

    // Now run distance-1 graph coloring on C
    // Use LocalOrdinal for storing colors
    KokkosGraph::Experimental::graph_color(cp, nr, nr, /*(const_rowptrs_view)*/ cRowptrs, /*(const_colinds_view)*/ cColinds);

    // WCMCLEN: Is this actually returning the colors anywhere???

    // extract the colors
    //auto coloringHandle = cp->get_graph_coloring_handle();
    //color_view_type colorsDevice = coloringHandle->get_vertex_colors();

    //clean up coloring handle
    // cp->destroy_graph_coloring_handle();
  }

};  // GraphColorD2_MatrixSquared (end)






/*! \brief Base class for graph coloring purposes.
 *  Each color represents the set of the vertices that are independent,
 *  e.g. no vertex having same color shares an edge.
 *  General aim is to find the minimum number of colors, minimum number of independent sets.
 */
template <typename HandleType, typename lno_row_view_t_, typename lno_nnz_view_t_, typename clno_row_view_t_, typename clno_nnz_view_t_ >
class GraphColorD2 
{
public:

  typedef lno_row_view_t_ in_lno_row_view_t;
  typedef lno_nnz_view_t_ in_lno_nnz_view_t;

  // typedef typename HandleType::GraphColoringHandleType::color_t      color_t;
  // typedef typename HandleType::GraphColoringHandleType::color_view_t color_view_type;
  typedef typename HandleType::color_view_t color_view_type;
  typedef typename HandleType::color_t      color_t;

  typedef typename HandleType::size_type size_type;
  typedef typename HandleType::nnz_lno_t nnz_lno_t;

  typedef typename HandleType::HandleExecSpace        MyExecSpace;
  typedef typename HandleType::HandleTempMemorySpace  MyTempMemorySpace;
  // typedef typename HandleType::HandlePersistentMemorySpace MyPersistentMemorySpace;
  typedef typename HandleType::const_size_type        const_size_type;

  typedef typename lno_row_view_t_::device_type    row_lno_view_device_t;
  typedef typename lno_row_view_t_::const_type     const_lno_row_view_t;
  typedef typename lno_nnz_view_t_::const_type     const_lno_nnz_view_t;
  typedef typename lno_nnz_view_t_::non_const_type non_const_lno_nnz_view_t;

  typedef typename clno_row_view_t_::const_type     const_clno_row_view_t;
  typedef typename clno_nnz_view_t_::const_type     const_clno_nnz_view_t;
  typedef typename clno_nnz_view_t_::non_const_type non_const_clno_nnz_view_t;

  //typedef typename HandleType::size_type_temp_work_view_t     size_type_temp_work_view_t;
  //typedef typename HandleType::scalar_temp_work_view_t        scalar_temp_work_view_t;
  //typedef typename HandleType::nnz_lno_persistent_work_view_t nnz_lno_persistent_work_view_t;

  typedef typename HandleType::nnz_lno_temp_work_view_t nnz_lno_temp_work_view_t;
  typedef typename Kokkos::View<nnz_lno_t, row_lno_view_device_t> single_dim_index_view_type;

  typedef Kokkos::RangePolicy<MyExecSpace> my_exec_space;


protected:
  nnz_lno_t             nr;       // num_rows  (# verts)
  nnz_lno_t             nc;       // num cols
  size_type             ne;       // # edges
  const_lno_row_view_t  xadj;     // rowmap, transpose of rowmap
  const_lno_nnz_view_t  adj;      // entries, transpose of entries   (size = # edges)
  const_clno_row_view_t t_xadj;   // rowmap, transpose of rowmap
  const_clno_nnz_view_t t_adj;    // entries, transpose of entries
  nnz_lno_t             nv;       // num vertices
  HandleType*           cp;       // pointer to the handle

private:

  int _chunkSize;                 // the size of the minimum work unit assigned to threads.  Changes the convergence on GPUs
  int _max_num_iterations;

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
  GraphColorD2 (nnz_lno_t             nr_,
                nnz_lno_t             nc_,
                size_type             ne_,
                const_lno_row_view_t  row_map,
                const_lno_nnz_view_t  entries,
                const_clno_row_view_t t_row_map,
                const_clno_nnz_view_t t_entries,
                HandleType*           coloring_handle):
        nr (nr_), 
        nc (nc_), 
        ne(ne_), 
        xadj(row_map), 
        adj(entries), 
        t_xadj(t_row_map), 
        t_adj(t_entries),
        nv (nr_), 
        cp(coloring_handle),
        _chunkSize(coloring_handle->get_vb_chunk_size()),
        _max_num_iterations(1000)
  {}


  /** \brief GraphColor destructor.
   */
  virtual ~GraphColorD2 () 
  {}


  // -----------------------------------------------------------------
  //
  // GraphColorD2::color_graph_d2()
  //
  // -----------------------------------------------------------------
  virtual void color_graph_d2(color_view_type colors_out)
  {
    std::cout << ">>> WCMCLEN color_graph_d2_wcmclen (KokkosGraph_Distance2Color_impl.hpp) <<<" << std::endl;

    // Data:
    // cp   = coloring_handle  
    // nr   = num_rows  (scalar)
    // nc   = num_cols  (scalar)
    // xadj = row_map   (view 1 dimension - [num_verts+1] - entries index into adj )
    // adj  = entries   (view 1 dimension - [num_edges]   - adjacency list )

    std::cout << ">>> WCMCLEN num_rows  = " << this->nr << std::endl;
    std::cout << ">>> WCMCLEN num_cols  = " << this->nc << std::endl;
    std::cout << ">>> WCMCLEN nv        = " << this->nv << std::endl;
    std::cout << ">>> WCMCLEN ne        = " << this->ne << std::endl;
    std::cout << ">>> WCMCLEN num_edges = " << this->adj.dimension_0() << std::endl;
    // std::cout << ">>> WCMCLEN " << std::endl;

    prettyPrint1DView(this->xadj, ">>> WCMCLEN xadj     ");
    prettyPrint1DView(this->adj, ">>> WCMCLEN adj      ");



    // conflictlist - store conflicts that can happen when we're coloring in parallel.
    nnz_lno_temp_work_view_t current_vertexList = nnz_lno_temp_work_view_t(Kokkos::ViewAllocateWithoutInitializing("vertexList"), this->nv);

    // init conflictlist sequentially.
    Kokkos::parallel_for(my_exec_space(0, this->nv), functorInitList<nnz_lno_temp_work_view_t>(current_vertexList));

    // Next iteratons's conflictlist
    nnz_lno_temp_work_view_t next_iteration_recolorList;

    // Size the next iteration conflictlist
    single_dim_index_view_type next_iteration_recolorListLength;

    // Vertices to recolor.  Will swap with vertexList
    next_iteration_recolorList = nnz_lno_temp_work_view_t(Kokkos::ViewAllocateWithoutInitializing("recolorList"), this->nv);
    next_iteration_recolorListLength = single_dim_index_view_type("recolorListLength");

    nnz_lno_t numUncolored = this->nv;
    nnz_lno_t current_vertexListLength = this->nv;

  // int _max_num_iterations = 2000;   // WCMCLEN - SCAFFOLDING 

  int iter=0; 
  for (; (iter < _max_num_iterations) && (numUncolored>0); iter++)
  {
    // Do greedy color
    std::cout << "--------------------------------------------------" << std::endl;
    this->colorGreedy(this->xadj, this->adj, colors_out, current_vertexList, current_vertexListLength);

    MyExecSpace::fence();

    // Find conflicts
    std::cout << "--------------------------------------------------" << std::endl;
    bool swap_work_arrays = true;

    // NOTE: not using colorset algorithm in this so we don't include colorset data
    numUncolored = this->findConflicts(swap_work_arrays,
                                       this->xadj,
                                       this->adj,
                                       colors_out,
                                       current_vertexList,
                                       current_vertexListLength,
                                       next_iteration_recolorList,
                                       next_iteration_recolorListLength); 



    MyExecSpace::fence();

    // break after first iteration if serial
    break;      // DEBUGGING

    // Swap Work Arrays
    if(iter+1 < _max_num_iterations)
    {
      nnz_lno_temp_work_view_t temp = current_vertexList;
      current_vertexList = next_iteration_recolorList;
      next_iteration_recolorList = temp;

      current_vertexListLength = numUncolored;
      next_iteration_recolorListLength = single_dim_index_view_type("recolorListLength");
    }

  }



    std::cout << std::endl;
    std::cout << std::endl; 
    std::cout << std::endl;

    std::ostringstream os;
    os << "GraphColorD2::color_graph_d2_wcmclen() not implemented -- [STUB CODE -X-]";
    Kokkos::Impl::throw_runtime_exception(os.str());
  }



private:


  // -----------------------------------------------------------------
  //
  // GraphColorD2::colorGreedy()
  //
  // -----------------------------------------------------------------
  // WCMCLEN (SCAFFOLDING) TODO: Find out why we're passing xadj_ and adj_ in as parameters to the method when they're class members...
  // WCMCLEN (SCAFFOLDING) TODO: Copying the D1 colorGreedy (for now).  Need to adapt this to the D2 version.
  void colorGreedy(const_lno_row_view_t     xadj_,
                   const_lno_nnz_view_t     adj_,
                   color_view_type          vertex_colors_,
                   nnz_lno_temp_work_view_t current_vertexList_,
                   nnz_lno_t                current_vertexListLength_)
  {
    std::cout << ">>> WCMCLEN colorGreedy (KokkosGraph_Distance2Color_impl.hpp) <<<" << std::endl;

    nnz_lno_t chunkSize_ = this->_chunkSize;

    if (current_vertexListLength_ < 100*chunkSize_)
    {
      chunkSize_ = 1;
    }

    functorGreedyColor gc(this->nv,
                          xadj_, 
                          adj_,
                          vertex_colors_,
                          current_vertexList_,
                          current_vertexListLength_,
                          chunkSize_
                          );

    Kokkos::parallel_for(my_exec_space(0, current_vertexListLength_ / chunkSize_+1), gc);

  }  // colorGreedy (end)



  // -----------------------------------------------------------------
  //
  // GraphColorD2::findConflicts()
  //
  // -----------------------------------------------------------------
  // NOTE: not using colorset algorithm in this so we don't include colorset data
  template <typename adj_view_t>
  nnz_lno_t findConflicts(bool&                      swap_work_arrays,
                          const_lno_row_view_t       xadj_,
                          adj_view_t                 adj_,
                          color_view_type            vertex_colors_,
                          nnz_lno_temp_work_view_t   current_vertexList_,
                          nnz_lno_t                  current_vertexListLength_,
                          nnz_lno_temp_work_view_t   next_iteration_recolorList_,
                          single_dim_index_view_type next_iteration_recolorListLength_
                          )
  {
    // WCMCLEN TODO: Fill this in.
    std::cout << ">>> WCMCLEN findConflicts (KokkosGraph_Distance2Color_impl.hpp) <<<" << std::endl;

    nnz_lno_t output_numUncolored=0;

    return output_numUncolored;
  }



  // ------------------------------------------------------
  // Helper Functors
  // ------------------------------------------------------

  // pretty-print a 1D View with label
  template<typename kokkos_view_t>
  void prettyPrint1DView(kokkos_view_t & view, const char* label)
  {
    std::cout << label << " = [ ";
    for(size_t i=0; i<view.dimension_0(); i++)
    {
      std::cout << view(i) << " ";
    }
    std::cout << " ]" << std::endl;
  }


  /**
   * Functor to init a list sequentialy, that is list[i] = i
   */
  template <typename view_type>
  struct functorInitList 
  {
    view_type _vertexList;
    functorInitList (view_type vertexList) : _vertexList(vertexList) {  }

    KOKKOS_INLINE_FUNCTION
    void operator()(const nnz_lno_t i) const 
    {
      // Natural order
      _vertexList(i) = i;
    }
  };



  /**
   * Functor for VB algorithm speculative coloring without edge filtering.
   */
  struct functorGreedyColor 
  {
    nnz_lno_t                nv;
    const_lno_row_view_t     _idx;
    const_lno_nnz_view_t     _adj;
    color_view_type          _colors;
    nnz_lno_temp_work_view_t _vertexList;
    nnz_lno_t                _vertexListLength;
    nnz_lno_t                _chunkSize;

    functorGreedyColor (nnz_lno_t                nv_,
                        const_lno_row_view_t     xadj_,
                        const_lno_nnz_view_t     adj_,
                        color_view_type          colors,
                        nnz_lno_temp_work_view_t vertexList,
                        nnz_lno_t                vertexListLength,
                        nnz_lno_t                chunkSize) 
          : nv(nv_),
            _idx(xadj_),
            _adj(adj_),
            _colors(colors),
            _vertexList(vertexList),
            _vertexListLength(vertexListLength),
            _chunkSize(chunkSize)
    {
    }


    KOKKOS_INLINE_FUNCTION
    void operator()(const nnz_lno_t ii) const
    {

      std::cout << ">>> WCMCLEN functorGreedyColor (KokkosGraph_Distance2Color_impl.hpp)" << std::endl;

      // TODO: Implement this (copy in the Distance-1 version initially and then adapt it.)

    }  // operator() (end)


  };  // functorGreedyColor (end)






};  // end class GraphColorD2


}  // end Impl namespace 
}  // end KokkosGraph namespace





#endif  // _KOKKOSCOLORINGD2IMP_HPP
