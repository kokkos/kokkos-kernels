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

  int  _chunkSize;                 // the size of the minimum work unit assigned to threads.  Changes the convergence on GPUs
  int  _max_num_iterations;
  char _conflictList;              // 0: none, 1: atomic (default), 2: parallel prefix sums (0, 2 not implemented)
  bool _serialConflictResolution;  // true if using serial conflict resolution, false otherwise (default)
  char _use_color_set;             // The VB Algorithm Type: 0: VB,  1: VBCS,  2: VBBIT  (1, 2 not implemented).

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
        _max_num_iterations(1000),
        _conflictList(1),
        _serialConflictResolution(false),
        _use_color_set(0)
  {
    std::cout << ">>> WCMCLEN GraphColorD2() (KokkosGraph_Distance2Color_impl.hpp)" << std::endl;
    //std::cout << ">>> WCMCLEN coloring_algo_type = " << coloring_handle->get_coloring_algo_type() << std::endl;
    //std::cout << ">>> WCMCLEN conflict_list_type = " << coloring_handle->get_conflict_list_type() << std::endl;
  }


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

    // Next iteratons's conflictList
    nnz_lno_temp_work_view_t next_iteration_recolorList;

    // Size the next iteration conflictList
    single_dim_index_view_type next_iteration_recolorListLength;

    // if we're using a conflictList
    if(this->_conflictList > 0)
    {
      // Vertices to recolor.  Will swap with vertexList
      next_iteration_recolorList = nnz_lno_temp_work_view_t(Kokkos::ViewAllocateWithoutInitializing("recolorList"), this->nv);
      next_iteration_recolorListLength = single_dim_index_view_type("recolorListLength");
    }

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
    prettyPrint1DView(colors_out, ">>> WCMCLEN colors_out");

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

    // Break after first iteration if using Serial Conflict Resolution
    //if(this->_serialConflictResolution) break;

    // If conflictList is used and we need to swap the work arrays
    if(this->_conflictList && swap_work_arrays) 
    {
      // Swap Work Arrays
      if(iter+1 < this->_max_num_iterations)
      {
        nnz_lno_temp_work_view_t temp = current_vertexList;
        current_vertexList = next_iteration_recolorList;
        next_iteration_recolorList = temp;

        current_vertexListLength = numUncolored;
        next_iteration_recolorListLength = single_dim_index_view_type("recolorListLength");
      }
    }
    break;      // DEBUGGING (STOP HERE)

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

    Kokkos::parallel_for(my_exec_space(0, current_vertexListLength_ / chunkSize_ + 1), gc);

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

    swap_work_arrays = true;
    nnz_lno_t output_numUncolored = 0;

    // conflictList mode: 
    if(0 == this->_conflictList)
    {
      // Throw an error -- we aren't using this mode (yet).
    }

    // conflictList mode: Parallel Prefix Sums (PPS)
    else if(2 == this->_conflictList)
    {
      // Throw an error -- we aren't using this mode (yet)
    }

    // conflictList mode: ATOMIC
    else if(1 == this->_conflictList)
    {
      if(0 == this->_use_color_set)
      {
        // TODO: call functorFindConflicts_Atomic from here...
        functorFindConflicts_Atomic<adj_view_t> conf(this->nv,
                                                     xadj_,
                                                     adj_,
                                                     vertex_colors_,
                                                     current_vertexList_,
                                                     next_iteration_recolorList_,
                                                     next_iteration_recolorListLength_);
        Kokkos::parallel_reduce(my_exec_space(0, current_vertexListLength_), conf, output_numUncolored);
      }
    }
    else
    {
      // Throw an error becaue we should not be here...
    }

    std::cout << ">>> WCMCLEN num_uncolored: " << output_numUncolored << std::endl;

    return output_numUncolored;
  }



  // ------------------------------------------------------
  // Functors: Helpers
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



  // ------------------------------------------------------
  // Functors: Distance-2 Graph Coloring
  // ------------------------------------------------------

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
    nnz_lno_t                nv;                  // num vertices
    const_lno_row_view_t     _idx;                // vertex degree list
    const_lno_nnz_view_t     _adj;                // vertex adjacency list
    color_view_type          _colors;             // vertex colors
    nnz_lno_temp_work_view_t _vertexList;         // 
    nnz_lno_t                _vertexListLength;   // 
    nnz_lno_t                _chunkSize;          // 

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
//      std::cout << ">>> WCMCLEN functorGreedyColor() <<C'TOR>> (KokkosGraph_Distance2Color_impl.hpp)" << std::endl
//                << ">>> WCMCLEN - nv                = " << nv << std::endl
//                << ">>> WCMCLEN - _vertexListLength = " << _vertexListLength << std::endl
//                << ">>> WCMCLEN - _chunkSize        = " << _chunkSize << std::endl;
    }


    // Color vertex i with smallest available color.
    //
    // Each thread colors a chunk of vertices to prevent all vertices getting the same color.
    // 
    // This version uses a bool array of size FORBIDDEN_SIZE.
    //
    // param: ii = vertex id
    // 
    KOKKOS_INLINE_FUNCTION
    void operator()(const nnz_lno_t vid_) const
    {

//      std::cout << ">>> WCMCLEN functorGreedyColor::operator()(" << vid_ << ") (KokkosGraph_Distance2Color_impl.hpp)" << std::endl;

      nnz_lno_t vid = 0;
      for (nnz_lno_t ichunk=0; ichunk < _chunkSize; ichunk++)
      {
        if (vid_ * _chunkSize + ichunk < _vertexListLength)
          vid = _vertexList(vid_ * _chunkSize + ichunk);
        else
          continue;

//        std::cout << ">>> WCMCLEN vid_ = " << vid_ << std::endl
//                  << ">>> WCMCLEN vid  = " << vid  << std::endl;

        // Already colored this vertex.
        if(_colors(vid) > 0) { continue; }

        bool foundColor = false;    // Have we found a valid color?

        // Use forbidden array to find available color.
        // - should be small enough to fit into fast memory (use Kokkos memoryspace?)
        bool forbidden[VB_COLORING_FORBIDDEN_SIZE];     // Forbidden Colors

        // Do multiple passes if the array is too small.
        color_t degree = _idx(vid+1) - _idx(vid);
        color_t offset = 0;
        for( ; offset <= (degree + VB_COLORING_FORBIDDEN_SIZE) && (!foundColor); offset += VB_COLORING_FORBIDDEN_SIZE)
        {
          // initialize
          for(int j=0; j < VB_COLORING_FORBIDDEN_SIZE; j++)
          {
            forbidden[j] = false;
          }
          // by convention, start at 1
          if(offset == 0)
          {
            forbidden[0] = true;
          }

          // Check neighbors, fill forbidden array.
          // -- TODO: Edit this for neighbors of neighbors loop
          for(size_type vid_1adj=_idx(vid); vid_1adj < _idx(vid+1); vid_1adj++) 
          {
            size_type vid_1idx = _adj(vid_1adj);
//            std::cout << ">>> WCMCLEN vid_1idx = " << vid_1idx << std::endl;
//            std::cout << ">>> WCMCLEN     vid_2idx = ";
            for(size_type vid_2adj=_idx(vid_1idx); vid_2adj < _idx(vid_1idx+1); vid_2adj++)
            { 
              size_type vid_2idx = _adj(vid_2adj);
//              std::cout << vid_2idx;

              // Skip distance-2-self-loops
              if(vid_2idx == vid || vid_2idx >= nv) 
              { 
//                std::cout << "* ";
                continue;
              }
//              std::cout << " ";

              color_t c = _colors(vid_2idx);

              if((c >= offset) && (c - offset < VB_COLORING_FORBIDDEN_SIZE))
              {
                forbidden[c - offset] = true;
              }

            }
//            std::cout << std::endl;
          }

          // color vertex i with smallest available color (firstFit)
          for(int c=0; c < VB_COLORING_FORBIDDEN_SIZE; c++)
          {
            if(!forbidden[c]) 
            {
              _colors(vid) = offset + c;
              foundColor = true;
              break;
            }
          }   // for c...
        }   // for offset...
      }   // for ichunk...
    }   // operator() (end)
  };  // struct functorGreedyColor (end)



template <typename adj_view_t>
struct functorFindConflicts_Atomic
{
  nnz_lno_t                  nv;           // num verts
  const_lno_row_view_t       _idx;
  adj_view_t                 _adj;
  color_view_type            _colors;
  nnz_lno_temp_work_view_t   _vertexList;
  nnz_lno_temp_work_view_t   _recolorList;
  single_dim_index_view_type _recolorListLength;


  functorFindConflicts_Atomic(nnz_lno_t                  nv_,
                              const_lno_row_view_t       xadj_,
                              adj_view_t                 adj_,
                              color_view_type            colors,
                              nnz_lno_temp_work_view_t   vertexList,
                              nnz_lno_temp_work_view_t   recolorList,
                              single_dim_index_view_type recolorListLength)
           : nv (nv_),
             _idx(xadj_),
             _adj(adj_),
             _colors(colors),
             _vertexList(vertexList),
             _recolorList(recolorList),
             _recolorListLength(recolorListLength)
  { }

  KOKKOS_INLINE_FUNCTION
  void operator()(const nnz_lno_t vid_, nnz_lno_t& numConflicts) const
  {
//    std::cout << ">>> WCMCLEN functorFindConflicts_Atomic::operator()(" 
//              << vid_ << ", " << numConflicts
//              << ") (KokkosGraph_Distance2Color_impl.hpp)" 
//              << std::endl;

    nnz_lno_t vid      = _vertexList(vid_);
    color_t   my_color = _colors(vid);

    size_type vid_1adj     = _idx(vid);
    size_type vid_1adj_end = _idx(vid+1);

    for(; vid_1adj < vid_1adj_end; vid_1adj++)
    {
      nnz_lno_t vid_1idx = _adj(vid_1adj);

      for(size_type vid_2adj=_idx(vid_1idx); vid_2adj < _idx(vid_1idx+1); vid_2adj++)
      { 
        size_type vid_2idx = _adj(vid_2adj);

        if(vid == vid_2idx || vid_2idx >= nv) continue;

        if(_colors(vid_2idx) == my_color)
        {
          _colors(vid) = 0;   // uncolor vertex
          // Atomically add vertex to recolorList
          const nnz_lno_t k = Kokkos::atomic_fetch_add( &_recolorListLength(), 1);
          _recolorList(k) = vid;
          numConflicts += 1;
//          std::cout << ">>> WCMCLEN vertex " << vid << " marked as conflict" << std::endl;
          break;  // Can exit if vertex gets marked as a conflict.
        }
      }
    }
  }
}; // struct functorFindConflicts_Atomic (end)




};  // end class GraphColorD2


}  // end Impl namespace 
}  // end KokkosGraph namespace





#endif  // _KOKKOSCOLORINGD2IMP_HPP
