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
#include <iomanip>
#include <stdexcept>
#include <vector>

#include <KokkosSparse_spgemm.hpp>
#include <Kokkos_Atomic.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_Timer.hpp>

#include "KokkosGraph_GraphColorHandle.hpp"
#include "KokkosGraph_graph_color.hpp"
#include "KokkosKernels_Handle.hpp"

#ifndef _KOKKOSCOLORINGD2IMP_HPP
#define _KOKKOSCOLORINGD2IMP_HPP


namespace KokkosGraph {

namespace Impl {

#define VB_D2_COLORING_FORBIDDEN_SIZE 64
// #define VB_D2_COLORING_FORBIDDEN_SIZE 20000



/*!
 *  \brief Base class for graph coloring purposes.
 *  Each color represents the set of the vertices that are independent,
 *  e.g. no vertex having same color shares an edge.
 *  General aim is to find the minimum number of colors, minimum number of independent sets.
 */
template<typename HandleType, typename lno_row_view_t_, typename lno_nnz_view_t_, typename clno_row_view_t_, typename clno_nnz_view_t_>
class GraphColorD2
{
  public:
    typedef lno_row_view_t_ in_lno_row_view_t;
    typedef lno_nnz_view_t_ in_lno_nnz_view_t;

    typedef typename HandleType::GraphColoringHandleType::color_view_t color_view_type;
    typedef typename HandleType::GraphColoringHandleType::color_t color_t;

    typedef typename HandleType::GraphColoringHandleType::size_type size_type;
    typedef typename HandleType::GraphColoringHandleType::nnz_lno_t nnz_lno_t;

    typedef typename in_lno_row_view_t::HostMirror row_lno_host_view_t;                             // host view type
    typedef typename in_lno_nnz_view_t::HostMirror nnz_lno_host_view_t;                             // host view type
    typedef typename HandleType::GraphColoringHandleType::color_host_view_t color_host_view_t;      // host view type

    typedef typename HandleType::GraphColoringHandleType::HandleExecSpace MyExecSpace;
    typedef typename HandleType::GraphColoringHandleType::HandleTempMemorySpace MyTempMemorySpace;
    typedef typename HandleType::GraphColoringHandleType::const_size_type const_size_type;

    typedef typename lno_row_view_t_::device_type row_lno_view_device_t;
    typedef typename lno_row_view_t_::const_type const_lno_row_view_t;
    typedef typename lno_nnz_view_t_::const_type const_lno_nnz_view_t;
    typedef typename lno_nnz_view_t_::non_const_type non_const_lno_nnz_view_t;

    typedef typename clno_row_view_t_::const_type const_clno_row_view_t;
    typedef typename clno_nnz_view_t_::const_type const_clno_nnz_view_t;
    typedef typename clno_nnz_view_t_::non_const_type non_const_clno_nnz_view_t;

    typedef typename HandleType::GraphColoringHandleType::nnz_lno_temp_work_view_t nnz_lno_temp_work_view_t;
    typedef typename Kokkos::View<nnz_lno_t, row_lno_view_device_t> single_dim_index_view_type;

    typedef Kokkos::RangePolicy<MyExecSpace> my_exec_space;

    typedef Kokkos::TeamPolicy<MyExecSpace> team_policy_t ;
    typedef typename team_policy_t::member_type team_member_t ;


  protected:
    nnz_lno_t nr;                      // num_rows  (# verts)
    nnz_lno_t nc;                      // num cols
    size_type ne;                      // # edges
    const_lno_row_view_t xadj;         // rowmap, transpose of rowmap
    const_lno_nnz_view_t adj;          // entries, transpose of entries   (size = # edges)
    const_clno_row_view_t t_xadj;      // rowmap, transpose of rowmap
    const_clno_nnz_view_t t_adj;       // entries, transpose of entries
    nnz_lno_t nv;                      // num vertices

    typename HandleType::GraphColoringHandleType *gc_handle;      // pointer to the graph coloring handle

  private:
    int _chunkSize;      // the size of the minimum work unit assigned to threads.  Changes the convergence on GPUs
    int _max_num_iterations;
    char _conflictList;                  // 0: none, 1: atomic (default), 2: parallel prefix sums (0, 2 not implemented)
    bool _serialConflictResolution;      // true if using serial conflict resolution, false otherwise (default)
    char _use_color_set;                 // The VB Algorithm Type: 0: VB,  1: VBCS,  2: VBBIT  (1, 2 not implemented).
    bool _ticToc;                        // if true print info in each step

  public:
    /**
     * \brief GraphColor constructor.
     * \param nv_: number of vertices in the graph
     * \param ne_: number of edges in the graph
     * \param row_map: the xadj array of the graph. Its size is nv_ +1
     * \param entries: adjacency array of the graph. Its size is ne_
     * \param handle: GraphColoringHandle object that holds the specification about the graph coloring,
     *    including parameters.
     */
    GraphColorD2(nnz_lno_t nr_,
                 nnz_lno_t nc_,
                 size_type ne_,
                 const_lno_row_view_t row_map,
                 const_lno_nnz_view_t entries,
                 const_clno_row_view_t t_row_map,
                 const_clno_nnz_view_t t_entries,
                 HandleType *handle)
        : nr(nr_), nc(nc_), ne(ne_), xadj(row_map), adj(entries), t_xadj(t_row_map), t_adj(t_entries), nv(nr_),
          gc_handle(handle->get_graph_coloring_handle()), _chunkSize(handle->get_graph_coloring_handle()->get_vb_chunk_size()),
          _max_num_iterations(handle->get_graph_coloring_handle()->get_max_number_of_iterations()), _conflictList(1),
          _serialConflictResolution(false), _use_color_set(0), _ticToc(handle->get_verbose())
    {
        // std::cout << ">>> WCMCLEN GraphColorD2() (KokkosGraph_Distance2Color_impl.hpp)" << std::endl
        //           << ">>> WCMCLEN :    coloring_algo_type = " << handle->get_coloring_algo_type() << std::endl
        //           << ">>> WCMCLEN :    conflict_list_type = " << handle->get_conflict_list_type() << std::endl;
    }


    /**
     *  \brief GraphColor destructor.
     */
    virtual ~GraphColorD2() {}


    // -----------------------------------------------------------------
    //
    // GraphColorD2::execute()
    //
    // -----------------------------------------------------------------
    virtual void execute()
    {
        color_view_type colors_out("Graph Colors", this->nv);

        // Data:
        // gc_handle = graph coloring handle
        // nr        = num_rows  (scalar)
        // nc        = num_cols  (scalar)
        // xadj      = row_map   (view 1 dimension - [num_verts+1] - entries index into adj )
        // adj       = entries   (view 1 dimension - [num_edges]   - adjacency list )
        if(this->_ticToc)
        {
            std::cout << "\tcolor_graph_d2 params:" << std::endl
                      << "\t  algorithm                : " << (int)this->_use_color_set << std::endl
                      << "\t  useConflictList          : " << (int)this->_conflictList << std::endl
                      << "\t  ticToc                   : " << this->_ticToc << std::endl
                      << "\t  max_num_iterations       : " << this->_max_num_iterations << std::endl
                      << "\t  serialConflictResolution : " << (int)this->_serialConflictResolution << std::endl
                      << "\t  chunkSize                : " << this->_chunkSize << std::endl
                      << "\t  use_color_set            : " << (int)this->_use_color_set << std::endl
                      << "\tgraph information:" << std::endl
                      << "\t  nv                       : " << this->nv << std::endl
                      << "\t  ne                       : " << this->ne << std::endl;
        }

        // prettyPrint1DView(this->xadj, ">>> WCMCLEN xadj     ", 500);
        // prettyPrint1DView(this->adj,  ">>> WCMCLEN adj      ", 500);

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
            next_iteration_recolorList       = nnz_lno_temp_work_view_t(Kokkos::ViewAllocateWithoutInitializing("recolorList"), this->nv);
            next_iteration_recolorListLength = single_dim_index_view_type("recolorListLength");
        }

        nnz_lno_t numUncolored             = this->nv;
        nnz_lno_t current_vertexListLength = this->nv;

        double time;
        double total_time = 0.0;
        Kokkos::Impl::Timer timer;

        int iter = 0;
        for(; (iter < _max_num_iterations) && (numUncolored > 0); iter++)
        {
            // ------------------------------------------
            // Do greedy color
            // ------------------------------------------
            this->colorGreedy(this->xadj, this->adj, this->t_xadj, this->t_adj, colors_out, current_vertexList, current_vertexListLength);

            MyExecSpace::fence();

            if(this->_ticToc)
            {
                time = timer.seconds();
                total_time += time;
                std::cout << "\tTime speculative greedy phase " << std::setw(-2) << iter << " : " << time << std::endl;
                timer.reset();
                gc_handle->add_to_overall_coloring_time_phase1(time);
            }

            // prettyPrint1DView(colors_out, ">>> WCMCLEN colors_out", 100);

            // ------------------------------------------
            // Find conflicts
            // ------------------------------------------
            bool swap_work_arrays = true;      // NOTE: swap_work_arrays can go away in this example -- was only ever
                                               //       set false in the PPS code in the original D1 coloring...

            // NOTE: not using colorset algorithm in this so we don't include colorset data
            numUncolored = this->findConflicts(swap_work_arrays,
                                               this->xadj,
                                               this->adj,
                                               this->t_xadj,
                                               this->t_adj,
                                               colors_out,
                                               current_vertexList,
                                               current_vertexListLength,
                                               next_iteration_recolorList,
                                               next_iteration_recolorListLength);

            MyExecSpace::fence();

            if(_ticToc)
            {
                time = timer.seconds();
                total_time += time;
                std::cout << "\tTime conflict detection " << std::setw(-2) << iter << "       : " << time << std::endl;
                timer.reset();
                gc_handle->add_to_overall_coloring_time_phase2(time);
            }

            // If conflictList is used and we need to swap the work arrays
            if(this->_conflictList && swap_work_arrays)
            {
                // Swap Work Arrays
                if(iter + 1 < this->_max_num_iterations)
                {
                    nnz_lno_temp_work_view_t temp = current_vertexList;
                    current_vertexList            = next_iteration_recolorList;
                    next_iteration_recolorList    = temp;

                    current_vertexListLength         = numUncolored;
                    next_iteration_recolorListLength = single_dim_index_view_type("recolorListLength");
                }
            }
        }      // end for iter...

        // ------------------------------------------
        // clean up in serial (resolveConflicts)
        // ------------------------------------------
        if(numUncolored > 0)
        {
            this->resolveConflicts(
                    this->nv, this->xadj, this->adj, this->t_xadj, this->t_adj, colors_out, current_vertexList, current_vertexListLength);
        }

        MyExecSpace::fence();

        if(_ticToc)
        {
            time = timer.seconds();
            total_time += time;
            std::cout << "\tTime serial conflict resolution : " << time << std::endl;
            gc_handle->add_to_overall_coloring_time_phase3(time);
        }

        // Save out the number of phases and vertex colors
        this->gc_handle->set_vertex_colors(colors_out);
        this->gc_handle->set_num_phases((double)iter);

    }      // color_graph_d2 (end)



  private:
    // -----------------------------------------------------------------
    //
    // GraphColorD2::colorGreedy()
    //
    // -----------------------------------------------------------------
    void colorGreedy(const_lno_row_view_t xadj_,
                     const_lno_nnz_view_t adj_,
                     const_clno_row_view_t t_xadj_,
                     const_clno_nnz_view_t t_adj_,
                     color_view_type vertex_colors_,
                     nnz_lno_temp_work_view_t current_vertexList_,
                     nnz_lno_t current_vertexListLength_)
    {
        nnz_lno_t chunkSize_ = this->_chunkSize;

        if(current_vertexListLength_ < 100 * chunkSize_)
        {
            chunkSize_ = 1;
        }

        // Pick the right coloring algorithm to use based on which algorithm we're using
        switch(this->gc_handle->get_coloring_algo_type())
        {
            // Vertex Based without Team Policy
            case COLORING_D2:
            case COLORING_D2_VB:
                {
                functorGreedyColorVB gc(this->nv, xadj_, adj_, t_xadj_, t_adj_, vertex_colors_, current_vertexList_, current_vertexListLength_, chunkSize_);
                Kokkos::parallel_for(my_exec_space(0, current_vertexListLength_ / chunkSize_ + 1), gc);
                }
                break;

            // Vertex Based with Team Policy
            case COLORING_D2_VBTP:
                {
                functorGreedyColorVBTP gc(this->nv, xadj_, adj_, t_xadj_, t_adj_, vertex_colors_, current_vertexList_, current_vertexListLength_, chunkSize_);

                #if defined( KOKKOS_ENABLE_CUDA )
                const team_policy_t policy_inst(current_vertexListLength_ / chunkSize_ + 1, chunkSize_);
                #else
                const team_policy_t policy_inst(current_vertexListLength_ / chunkSize_ + 1, Kokkos::AUTO);
                #endif

                Kokkos::parallel_for(policy_inst, gc);
                }
                break;
            default:
                throw std::invalid_argument("Unknown Distance-2 Algorithm Type");
            }

    }      // colorGreedy (end)



    // -----------------------------------------------------------------
    //
    // GraphColorD2::findConflicts()
    //
    // -----------------------------------------------------------------
    // NOTE: not using colorset algorithm in this so we don't include colorset data
    template<typename adj_view_t>
    nnz_lno_t findConflicts(bool &swap_work_arrays,
                            const_lno_row_view_t xadj_,
                            adj_view_t adj_,
                            const_clno_row_view_t t_xadj_,
                            const_clno_nnz_view_t t_adj_,
                            color_view_type vertex_colors_,
                            nnz_lno_temp_work_view_t current_vertexList_,
                            nnz_lno_t current_vertexListLength_,
                            nnz_lno_temp_work_view_t next_iteration_recolorList_,
                            single_dim_index_view_type next_iteration_recolorListLength_)
    {
        swap_work_arrays              = true;
        nnz_lno_t output_numUncolored = 0;

        // conflictList mode:
        if(0 == this->_conflictList)
        {
            // Throw an error -- not implemented (yet)
            std::ostringstream os;
            os << "GraphColorD2::findConflicts() not implemented for conflictList == 0";
            Kokkos::Impl::throw_runtime_exception(os.str());
        }

        // conflictList mode: Parallel Prefix Sums (PPS)
        else if(2 == this->_conflictList)
        {
            // Throw an error -- not implemented (yet)
            std::ostringstream os;
            os << "GraphColorD2::findConflicts() not implemented for conflictList == 2";
            Kokkos::Impl::throw_runtime_exception(os.str());
        }

        // conflictList mode: ATOMIC
        else if(1 == this->_conflictList)
        {
            if(0 == this->_use_color_set)
            {
                functorFindConflicts_Atomic<adj_view_t> conf(this->nv,
                                                             xadj_,
                                                             adj_,
                                                             t_xadj_,
                                                             t_adj_,
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
            std::ostringstream os;
            os << "GraphColorD2::findConflicts() - unknown conflictList Flag value: " << this->_conflictList << " ";
            Kokkos::Impl::throw_runtime_exception(os.str());
        }
        return output_numUncolored;
    }      // findConflicts (end)



    // -----------------------------------------------------------------
    //
    // GraphColorD2::resolveConflicts()
    //
    // -----------------------------------------------------------------
    template<typename adj_view_t>
    void resolveConflicts(nnz_lno_t _nv,
                          const_lno_row_view_t xadj_,
                          adj_view_t adj_,
                          const_clno_row_view_t t_xadj_,
                          const_clno_nnz_view_t t_adj_,
                          color_view_type vertex_colors_,
                          nnz_lno_temp_work_view_t current_vertexList_,
                          size_type current_vertexListLength_)
    {
        color_t *forbidden = new color_t[_nv];
        nnz_lno_t vid      = 0;
        nnz_lno_t end      = _nv;

        typename nnz_lno_temp_work_view_t::HostMirror h_recolor_list;

        if(this->_conflictList)
        {
            end            = current_vertexListLength_;
            h_recolor_list = Kokkos::create_mirror_view(current_vertexList_);
            Kokkos::deep_copy(h_recolor_list, current_vertexList_);
        }

        color_host_view_t h_colors = Kokkos::create_mirror_view(vertex_colors_);

        typename const_lno_row_view_t::HostMirror h_idx = Kokkos::create_mirror_view(xadj_);
        typename adj_view_t::HostMirror h_adj           = Kokkos::create_mirror_view(adj_);

        typename const_clno_row_view_t::HostMirror h_t_idx = Kokkos::create_mirror_view(t_xadj_);
        typename const_clno_nnz_view_t::HostMirror h_t_adj = Kokkos::create_mirror_view(t_adj_);

        Kokkos::deep_copy(h_colors, vertex_colors_);

        Kokkos::deep_copy(h_idx, xadj_);
        Kokkos::deep_copy(h_adj, adj_);

        Kokkos::deep_copy(h_t_idx, t_xadj_);
        Kokkos::deep_copy(h_t_adj, t_adj_);

        for(nnz_lno_t k = 0; k < end; k++)
        {
            if(this->_conflictList)
            {
                vid = h_recolor_list(k);
            }
            else
            {
                vid = k;      // check for uncolored vertices
            }

            if(h_colors(vid) > 0)
                continue;

            // loop over distance-1 neighbors of vid
            for(size_type vid_1adj = h_idx(vid); vid_1adj < h_idx(vid + 1); vid_1adj++)
            {
                size_type vid_1idx = h_adj(vid_1adj);

                // loop over distance-1 neighbors of vid_1idx (distance-2 from vid)
                for(size_type vid_2adj = h_t_idx(vid_1idx); vid_2adj < h_t_idx(vid_1idx + 1); vid_2adj++)
                {
                    nnz_lno_t vid_2idx = h_t_adj(vid_2adj);

                    // skip over loops vid -- x -- vid
                    if(vid_2idx == vid)
                        continue;

                    forbidden[h_colors(vid_2idx)] = vid;
                }
            }

            // color vertex vid with smallest available color
            int c = 1;
            while(forbidden[c] == vid) c++;

            h_colors(vid) = c;
        }
        Kokkos::deep_copy(vertex_colors_, h_colors);
        delete[] forbidden;
    }      // resolveConflicts (end)


    // ------------------------------------------------------
    // Functions: Helpers
    // ------------------------------------------------------


    // pretty-print a 1D View with label
    template<typename kokkos_view_t>
    void prettyPrint1DView(kokkos_view_t &view, const char *label, const size_t max_entries = 500) const
    {
        int max_per_line = 20;
        int line_count   = 1;
        std::cout << label << " = [ \n\t";
        for(size_t i = 0; i < view.extent(0); i++)
        {
            std::cout << std::setw(5) << view(i) << " ";
            if(line_count >= max_per_line)
            {
                std::cout << std::endl << "\t";
                line_count = 0;
            }
            line_count++;
            if(i >= max_entries - 1)
            {
                std::cout << "<snip>";
                break;
            }
        }
        if(line_count > 1)
            std::cout << std::endl;
        std::cout << "\t ]" << std::endl;
    }      // prettyPrint1DView (end)



  public:



    // ------------------------------------------------------
    // Functors: Distance-2 Graph Coloring
    // ------------------------------------------------------

    /**
     * Functor to init a list sequentialy, that is list[i] = i
     */
    template<typename view_type>
    struct functorInitList
    {
        view_type _vertexList;
        functorInitList(view_type vertexList) : _vertexList(vertexList) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const nnz_lno_t i) const
        {
            // Natural order
            _vertexList(i) = i;
        }
    };      // struct functorInitList (end)



   /**
     * Functor for VB algorithm speculative coloring without edge filtering.
     */
    struct functorGreedyColorVB
    {
        nnz_lno_t nv;                              // num vertices
        const_lno_row_view_t _idx;                 // vertex degree list
        const_lno_nnz_view_t _adj;                 // vertex adjacency list
        const_clno_row_view_t _t_idx;              // transpose vertex degree list
        const_clno_nnz_view_t _t_adj;              // transpose vertex adjacency list
        color_view_type _colors;                   // vertex colors
        nnz_lno_temp_work_view_t _vertexList;      //
        nnz_lno_t _vertexListLength;               //
        nnz_lno_t _chunkSize;                      //

        functorGreedyColorVB(nnz_lno_t nv_,
                              const_lno_row_view_t xadj_,
                              const_lno_nnz_view_t adj_,
                              const_clno_row_view_t t_xadj_,
                              const_clno_nnz_view_t t_adj_,
                              color_view_type colors,
                              nnz_lno_temp_work_view_t vertexList,
                              nnz_lno_t vertexListLength,
                              nnz_lno_t chunkSize)
            : nv(nv_), _idx(xadj_), _adj(adj_), _t_idx(t_xadj_), _t_adj(t_adj_), _colors(colors), _vertexList(vertexList),
              _vertexListLength(vertexListLength), _chunkSize(chunkSize)
        {
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
            // std::cout << ">>> WCMCLEN functorGreedyColor::operator()(" << vid_ << ") (KokkosGraph_Distance2Color_impl.hpp)" << std::endl;
            nnz_lno_t vid = 0;
            for(nnz_lno_t ichunk = 0; ichunk < _chunkSize; ichunk++)
            {
                if(vid_ * _chunkSize + ichunk < _vertexListLength)
                    vid = _vertexList(vid_ * _chunkSize + ichunk);
                else
                    continue;

                // Already colored this vertex.
                if(_colors(vid) > 0)
                {
                    continue;
                }

                bool foundColor = false;      // Have we found a valid color?

                // Use forbidden array to find available color.
                // - should be small enough to fit into fast memory (use Kokkos memoryspace?)
                bool forbidden[VB_D2_COLORING_FORBIDDEN_SIZE];      // Forbidden Colors

                // Do multiple passes if the array is too small.
                // * The Distance-1 code used the knowledge of the degree of the vertex to cap the number of iterations
                //   but in distance-2 we'd need the total vertices at distance-2 which we don't easily have aprioi.
                //   This could be as big as all the vertices in the graph if diameter(G)=2...
                // * TODO: Determine a decent cap for this loop to prevent infinite loops (or prove infinite loop can't happen).
                color_t offset = 0;

                while(!foundColor)
                {
                    // initialize
                    for(int j = 0; j < VB_D2_COLORING_FORBIDDEN_SIZE; j++) { forbidden[j] = false; }
                    // by convention, start at 1
                    if(offset == 0)
                    {
                        forbidden[0] = true;
                    }

                    // Check neighbors, fill forbidden array.
                    for(size_type vid_1adj = _idx(vid); vid_1adj < _idx(vid + 1); vid_1adj++)
                    {
                        nnz_lno_t vid_1idx = _adj(vid_1adj);

                        for(size_type vid_2adj = _t_idx(vid_1idx); vid_2adj < _t_idx(vid_1idx + 1); vid_2adj++)
                        {
                            nnz_lno_t vid_2idx = _t_adj(vid_2adj);

                            // Skip distance-2-self-loops
                            if(vid_2idx == vid || vid_2idx >= nv)
                            {
                                continue;
                            }

                            color_t c = _colors(vid_2idx);

                            if((c >= offset) && (c - offset < VB_D2_COLORING_FORBIDDEN_SIZE))
                            {
                                forbidden[c - offset] = true;
                            }
                        }
                    }

                    // color vertex i with smallest available color (firstFit)
                    for(int c = 0; c < VB_D2_COLORING_FORBIDDEN_SIZE; c++)
                    {
                        if(!forbidden[c])
                        {
                            _colors(vid) = offset + c;
                            foundColor   = true;
                            break;
                        }
                    }      // for c...
                    offset += VB_D2_COLORING_FORBIDDEN_SIZE;
                }      // for offset...
            }          // for ichunk...
        }              // operator() (end)
    };                 // struct functorGreedyColorVB (end)



    /**
     * Functor for VB algorithm speculative coloring without edge filtering.
     * Team Policy Enabled
     */
    struct functorGreedyColorVBTP
    {
        nnz_lno_t nv;                              // num vertices
        const_lno_row_view_t _idx;                 // vertex degree list
        const_lno_nnz_view_t _adj;                 // vertex adjacency list
        const_clno_row_view_t _t_idx;              // transpose vertex degree list
        const_clno_nnz_view_t _t_adj;              // transpose vertex adjacency list
        color_view_type _colors;                   // vertex colors
        nnz_lno_temp_work_view_t _vertexList;      //
        nnz_lno_t _vertexListLength;               //
        nnz_lno_t _chunkSize;                      //

        functorGreedyColorVBTP(nnz_lno_t nv_,
                             const_lno_row_view_t xadj_,
                             const_lno_nnz_view_t adj_,
                             const_clno_row_view_t t_xadj_,
                             const_clno_nnz_view_t t_adj_,
                             color_view_type colors,
                             nnz_lno_temp_work_view_t vertexList,
                             nnz_lno_t vertexListLength,
                             nnz_lno_t chunkSize)
            : nv(nv_), _idx(xadj_), _adj(adj_), _t_idx(t_xadj_), _t_adj(t_adj_), _colors(colors), _vertexList(vertexList),
              _vertexListLength(vertexListLength), _chunkSize(chunkSize)
        {
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
        void operator()(const team_member_t &thread) const
        {
            nnz_lno_t chunk_id = thread.league_rank() * thread.team_size() + thread.team_rank();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, _chunkSize), [&](const nnz_lno_t ichunk)
            {
                if(chunk_id * _chunkSize + ichunk < _vertexListLength)
                {
                    const nnz_lno_t vid = _vertexList(chunk_id * _chunkSize + ichunk);

                    // Already colored this vertex.
                    if(_colors(vid) <= 0)
                    {
                        bool foundColor = false;      // Have we found a valid color?

                        // Use forbidden array to find available color.
                        // - should be small enough to fit into fast memory (use Kokkos memoryspace?)
                        // - If more levels of parallelism are addd in the loops over neighbors, then
                        //   atomics will be necessary for updating this.
                        bool forbidden[VB_D2_COLORING_FORBIDDEN_SIZE];      // Forbidden Colors

                        // Do multiple passes if the array is too small.
                        // * TODO: Determine a decent cap for this loop to prevent infinite loops (or prove infinite loop can't happen).
                        color_t offset = 0;

                        while(!foundColor && offset < nv)
                        {
                            // initialize
                            for(int j = 0; j < VB_D2_COLORING_FORBIDDEN_SIZE; j++) { forbidden[j] = false; }

                            // If the offset is 0 then we're looking at colors 0..63, but color 0 is reserved for
                            // UNCOLORED vertices so we should start coloring at 1.
                            if(0 == offset)
                            {
                                forbidden[0] = true;
                            }

                            // Loop over neighbors
                            for(size_type vid_d1_adj = _idx(vid); vid_d1_adj < _idx(vid + 1); vid_d1_adj++)
                            {
                                const nnz_lno_t vid_d1 = _adj(vid_d1_adj);

                                // Loop over distance-2 neighbors
                                for(size_type vid_d2_adj = _t_idx(vid_d1); vid_d2_adj < _t_idx(vid_d1 + 1); vid_d2_adj++)
                                {
                                    const nnz_lno_t vid_d2 = _t_adj(vid_d2_adj);

                                    // Skip distance-2 self loops
                                    if(vid_d2 != vid && vid_d2 < nv)
                                    {
                                        color_t c = _colors(vid_d2);

                                        // If color found is inside current 'range' then mark it as used.
                                        if((c >= offset) && (c - offset < VB_D2_COLORING_FORBIDDEN_SIZE))
                                        {
                                            Kokkos::atomic_fetch_or(&forbidden[c-offset], true);
                                            // forbidden[c - offset] = true;
                                        }
                                    }
                                }
                            }

                            // color vertex i with smallest available color (firstFit)
                            for(int c = 0; c < VB_D2_COLORING_FORBIDDEN_SIZE; c++)
                            {
                                if(!forbidden[c])
                                {
                                    _colors(vid) = offset + c;
                                    foundColor   = true;
                                    break;
                                }
                            }      // for c...
                            offset += VB_D2_COLORING_FORBIDDEN_SIZE;
                        }      // while(!foundColor)
                    }          // if _colors(vid) <= 0 ...
                }              // if chunk_id*...
            });                // for ichunk...
        }                      // operator() (end)

    };               // struct functorGreedyColorVBTP (end)



    template<typename adj_view_t>
    struct functorFindConflicts_Atomic
    {
        nnz_lno_t nv;      // num verts
        const_lno_row_view_t _idx;
        adj_view_t _adj;
        const_clno_row_view_t _t_idx;
        const_clno_nnz_view_t _t_adj;
        color_view_type _colors;
        nnz_lno_temp_work_view_t _vertexList;
        nnz_lno_temp_work_view_t _recolorList;
        single_dim_index_view_type _recolorListLength;


        functorFindConflicts_Atomic(nnz_lno_t nv_,
                                    const_lno_row_view_t xadj_,
                                    adj_view_t adj_,
                                    const_clno_row_view_t t_xadj_,
                                    const_clno_nnz_view_t t_adj_,
                                    color_view_type colors,
                                    nnz_lno_temp_work_view_t vertexList,
                                    nnz_lno_temp_work_view_t recolorList,
                                    single_dim_index_view_type recolorListLength)
            : nv(nv_), _idx(xadj_), _adj(adj_), _t_idx(t_xadj_), _t_adj(t_adj_), _colors(colors), _vertexList(vertexList), _recolorList(recolorList),
              _recolorListLength(recolorListLength)
        {
        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const nnz_lno_t vid_, nnz_lno_t &numConflicts) const
        {
            typedef typename std::remove_reference<decltype(_recolorListLength())>::type atomic_incr_type;
            const nnz_lno_t vid = _vertexList(vid_);
            color_t my_color = _colors(vid);

            size_type vid_1adj     = _idx(vid);
            const size_type vid_1adj_end = _idx(vid + 1);

            bool break_out = false;

            for(; !break_out && vid_1adj < vid_1adj_end; vid_1adj++)
            {
                nnz_lno_t vid_1idx = _adj(vid_1adj);

                for(size_type vid_2adj = _t_idx(vid_1idx); !break_out && vid_2adj < _t_idx(vid_1idx + 1); vid_2adj++)
                {
                    const nnz_lno_t vid_2idx = _t_adj(vid_2adj);

                    if(vid != vid_2idx && vid_2idx < nv)
                    {
                        if(_colors(vid_2idx) == my_color)
                        {
                            _colors(vid) = 0;      // uncolor vertex

                            // Atomically add vertex to recolorList
                            const nnz_lno_t k = Kokkos::atomic_fetch_add(&_recolorListLength(), atomic_incr_type(1));
                            _recolorList(k)   = vid;
                            numConflicts += 1;
                            break_out = true;
                            // break;      // Can exit if vertex gets marked as a conflict.
                        }
                    }
                }
            }
        }
    };      // struct functorFindConflicts_Atomic (end)


};      // end class GraphColorD2


}      // namespace Impl
}      // namespace KokkosGraph


#endif      // _KOKKOSCOLORINGD2IMP_HPP
