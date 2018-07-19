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
#include <Kokkos_UnorderedMap.hpp>

#include "KokkosGraph_GraphColor.hpp"
#include "KokkosGraph_GraphColorHandle.hpp"
#include "KokkosKernels_Handle.hpp"

#ifndef _KOKKOSCOLORINGD2IMP_HPP
#define _KOKKOSCOLORINGD2IMP_HPP


namespace KokkosGraph {

namespace Impl {

#define VB_D2_COLORING_FORBIDDEN_SIZE    64
#define VBBIT_D2_COLORING_FORBIDDEN_SIZE 64

#define WCMCLEN_EXPERIMENTAL  0

/*!
 * \brief Distance-2 Graph Coloring class
 *
 * This class supports direct methods for distance-2 graph coloring.
 *
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

    typedef Kokkos::TeamPolicy<MyExecSpace> team_policy_t;
    typedef typename team_policy_t::member_type team_member_t;

    typedef Kokkos::View<bool*> non_const_1d_bool_view_t;
    typedef Kokkos::View<size_type *> non_const_1d_size_type_view_t;

    typedef long long int bit_64_forbidden_t;

  protected:

    nnz_lno_t nv;                      // num vertices
    nnz_lno_t nr;                      // num_rows  (# verts)
    nnz_lno_t nc;                      // num cols
    size_type ne;                      // # edges
    const_lno_row_view_t xadj;         // rowmap, transpose of rowmap
    const_lno_nnz_view_t adj;          // entries, transpose of entries   (size = # edges)
    const_clno_row_view_t t_xadj;      // rowmap, transpose of rowmap
    const_clno_nnz_view_t t_adj;       // entries, transpose of entries

    typename HandleType::GraphColoringHandleType *gc_handle;      // pointer to the graph coloring handle

  private:

    int _chunkSize;      // the size of the minimum work unit assigned to threads.  Changes the convergence on GPUs
    int _max_num_iterations;
    char _conflictList;                  // 0: none, 1: atomic (default), 2: parallel prefix sums (0, 2 not implemented)
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
    GraphColorD2(nnz_lno_t nv_,
                 nnz_lno_t nc_,
                 size_type ne_,
                 const_lno_row_view_t row_map,
                 const_lno_nnz_view_t entries,
                 const_clno_row_view_t t_row_map,
                 const_clno_nnz_view_t t_entries,
                 HandleType *handle)
        : nv(nv_), nr(nv_), nc(nc_), ne(ne_), xadj(row_map), adj(entries), t_xadj(t_row_map), t_adj(t_entries),
          gc_handle(handle->get_graph_coloring_handle()), _chunkSize(handle->get_graph_coloring_handle()->get_vb_chunk_size()),
          _max_num_iterations(handle->get_graph_coloring_handle()->get_max_number_of_iterations()), _conflictList(1),
          _use_color_set(0), _ticToc(handle->get_verbose())
    {
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
        bool using_edge_filtering = false;

        color_view_type colors_out("Graph Colors", this->nv);

        // If the selected pass is using edge filtering we should set the flag.
        switch(this->gc_handle->get_coloring_algo_type())
        {
            case COLORING_D2_VB_BIT_EF:
                using_edge_filtering = true;
                break;
            default:
                break;
        }

        // EXPERIMENTAL Begin
        #if WCMCLEN_EXPERIMENTAL
        // Compute Distance-2 Degree of the vertices.
        non_const_1d_size_type_view_t degree_d2 = non_const_1d_size_type_view_t("degree d2", this->nv);
        if(using_edge_filtering)
        {
            this->calculate_d2_degree(degree_d2);
        }
        #endif
        // EXPERIMENTAL End

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
                      << "\t  chunkSize                : " << this->_chunkSize << std::endl
                      << "\t  use_color_set            : " << (int)this->_use_color_set << std::endl
                      << "\t  Edge Filtering Pass?     : " << (int)using_edge_filtering << std::endl
                      << "\tgraph information:" << std::endl
                      << "\t  nv                       : " << this->nv << std::endl
                      << "\t  ne                       : " << this->ne << std::endl;

            // prettyPrint1DView(this->xadj, ">>> xadj ", 500);
            // prettyPrint1DView(this->adj,  ">>>  adj ", 500);
        }

        // conflictlist - store conflicts that can happen when we're coloring in parallel.
        nnz_lno_temp_work_view_t current_vertexList = nnz_lno_temp_work_view_t(Kokkos::ViewAllocateWithoutInitializing("vertexList"), this->nv);

        // init conflictlist sequentially.
        Kokkos::parallel_for("InitList", my_exec_space(0, this->nv), functorInitList<nnz_lno_temp_work_view_t>(current_vertexList));

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

        nnz_lno_t numUncolored              = this->nv;
        nnz_lno_t current_vertexListLength  = this->nv;

        double time;
        double total_time = 0.0;
        Kokkos::Impl::Timer timer;

        int iter = 0;
        for(; (iter < _max_num_iterations) && (numUncolored > 0); iter++)
        {
            timer.reset();

            // ------------------------------------------
            // Do greedy color
            // ------------------------------------------
            if(using_edge_filtering)
            {
                // Temp copy of the adj array (mutable)
                // - This is required for edge-filtering passes to avoid
                //   side effects since edge filtering modifies the adj array.
                nnz_lno_temp_work_view_t adj_copy;
                adj_copy = nnz_lno_temp_work_view_t(Kokkos::ViewAllocateWithoutInitializing("adj copy"), this->ne);
                Kokkos::deep_copy(adj_copy, this->adj);

                non_const_clno_nnz_view_t t_adj_copy;
                t_adj_copy = non_const_clno_nnz_view_t(Kokkos::ViewAllocateWithoutInitializing("t_adj copy"), this->ne);
                Kokkos::deep_copy(t_adj_copy, this->t_adj);

                //prettyPrint1DView(t_adj_copy, "t_adj_copy", 100);
                //prettyPrint1DView(t_adj, "t_adj", 100);

                this->colorGreedyEF(this->xadj, adj_copy, this->t_xadj, t_adj_copy, colors_out, current_vertexList, current_vertexListLength);
            }
            else
            {
                this->colorGreedy(this->xadj, this->adj, this->t_xadj, this->t_adj, colors_out, current_vertexList, current_vertexListLength);
            }

            MyExecSpace::fence();

            if(this->_ticToc)
            {
                time = timer.seconds();
                total_time += time;
                std::cout << "\tIteration: " << iter << std::endl
                          << "\t  - Time speculative greedy phase : " << time << std::endl;
                          // << "\t      - Num Uncolored             : " << numUncolored << std::endl;

                gc_handle->add_to_overall_coloring_time_phase1(time);

                timer.reset();
            }

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
                std::cout << "\t  - Time conflict detection " << "      : " << time << std::endl;
                gc_handle->add_to_overall_coloring_time_phase2(time);
                timer.reset();
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
        // clean up in serial (resolveConflictsSerial)
        // ------------------------------------------
        if(numUncolored > 0)
        {
            this->resolveConflictsSerial(this->nv, this->xadj, this->adj, this->t_xadj, this->t_adj, colors_out, current_vertexList, current_vertexListLength);
        }

        MyExecSpace::fence();

        if(_ticToc)
        {
            time = timer.seconds();
            total_time += time;
            std::cout << "\tTime serial conflict resolution   : " << time << std::endl;
            gc_handle->add_to_overall_coloring_time_phase3(time);
        }

        // Save the number of phases and vertex colors to the graph coloring handle
        this->gc_handle->set_vertex_colors(colors_out);
        this->gc_handle->set_num_phases((double)iter);

        // ------------------------------------------
        // Print out histogram of colors
        // ------------------------------------------
        if(_ticToc)
        {
            //prettyPrint1DView(colors_out, ">>> colors_out ", 500);            // WCMCLEN
            //printDistance2ColorsHistogram();
        }
    }      // color_graph_d2 (end)



    /**
     *  Validate Distance 2 Graph Coloring
     *
     *  @param validation_flags is an array of 3 booleans.
     *         validation_flags[0] : True IF the distance-2 coloring is invalid.
     *         validation_flags[1] : True IF the coloring is bad because vertices are left uncolored.
     *         validation_flags[2] : True IF the coloring is bad because at least one pair of vertices
     *                               at distance=2 from each other has the same color.
     *         validation_flags[3] : True IF a vertex has a color that is greater than number of vertices in the graph.
     *                               This may not be an INVALID coloring, but can indicate poor quality in coloring.
     *
     *  @return boolean that is TRUE if the Distance-2 coloring is valid. False if otherwise.
     */
    bool verifyDistance2Coloring(const_lno_row_view_t xadj_,
                                 const_lno_nnz_view_t adj_,
                                 const_clno_row_view_t t_xadj_,
                                 const_clno_nnz_view_t t_adj_,
                                 color_view_type vertex_colors_,
                                 bool validation_flags[])
    {
        bool output = true;

        validation_flags[0] = false;      // True if an invalid coloring.
        validation_flags[1] = false;      // True if uncolored vertices exist.
        validation_flags[2] = false;      // True if an invalid color exists.

        nnz_lno_t chunkSize_ = this->_chunkSize;
        if(nv < 100 * chunkSize_)
        {
            chunkSize_ = 1;
        }
        const size_t num_chunks = this->nv / chunkSize_ + 1;

        non_const_1d_bool_view_t d_flags("flags", 4);

        // Create host mirror view.
        non_const_1d_bool_view_t::HostMirror h_flags = Kokkos::create_mirror_view(d_flags);

        // Initialize flags on host
        for(int i = 0; i < 4; i++) h_flags(i) = false;

        // Deep copy initialized flags to device
        Kokkos::deep_copy(d_flags, h_flags);

        functorVerifyDistance2Coloring vr(this->nv, xadj_, adj_, t_xadj_, t_adj_, vertex_colors_, d_flags, chunkSize_);
        Kokkos::parallel_for("ValidateD2Coloring", my_exec_space(0, num_chunks), vr);

        // Deep copy flags back to host memory
        Kokkos::deep_copy(h_flags, d_flags);

        validation_flags[0] = h_flags[0];
        validation_flags[1] = h_flags[1];
        validation_flags[2] = h_flags[2];
        validation_flags[3] = h_flags[3];

        output = !h_flags[0];

        return output;
    }      // verifyDistance2Coloring (end)



    /**
     * Print out the distance-2 coloring histogram.
     */
    void printDistance2ColorsHistogram(void)
    {
        nnz_lno_t num_colors = this->gc_handle->get_num_colors();
        std::cout << "num_colors: " << num_colors << std::endl;

        nnz_lno_temp_work_view_t histogram("histogram", num_colors + 1);
        MyExecSpace::fence();
        KokkosKernels::Impl::kk_get_histogram<typename HandleType::GraphColoringHandleType::color_view_t, nnz_lno_temp_work_view_t, MyExecSpace>(
                this->nv, this->gc_handle->get_vertex_colors(), histogram);

        std::cout << ">>> Histogram: " << std::endl;
        KokkosKernels::Impl::kk_print_1Dview(histogram);
        std::cout << std::endl;
    }




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

        const size_t num_chunks = current_vertexListLength_ / chunkSize_ + 1;

        // Pick the right coloring algorithm to use based on which algorithm we're using
        switch(this->gc_handle->get_coloring_algo_type())
        {
            // Single level parallelism on chunks
            // 1. [P] loop over chunks of vertices
            // 2. [S] loop over vertices in chunks
            // 3. [S] loop over vertex neighbors
            // 4. [S] loop over vertex neighbors of neighbors
            case COLORING_D2:
            case COLORING_D2_VB:
            {
                functorGreedyColorVB gc(this->nv, xadj_, adj_, t_xadj_, t_adj_, vertex_colors_, current_vertexList_, current_vertexListLength_, chunkSize_);
                Kokkos::parallel_for("LoopOverChunks", my_exec_space(0, num_chunks), gc);
            }
            break;

            // Two level parallelism:
            // 1. [P] loop over chunks of vertices
            // 2. [P] loop over vertices in chunks
            // 3. [S] loop over vertex neighbors
            // 4. [S] loop over vertex neighbors of neighbors
            case COLORING_D2_VBTP:
            {
                functorGreedyColorVBTP gc(this->nv, xadj_, adj_, t_xadj_, t_adj_, vertex_colors_, current_vertexList_, current_vertexListLength_, chunkSize_);

                #if defined(KOKKOS_ENABLE_CUDA)
                const team_policy_t policy_inst(num_chunks, chunkSize_);
                #else
                //const team_policy_t policy_inst(num_chunks, Kokkos::AUTO);
                const team_policy_t policy_inst(num_chunks, chunkSize_);
                #endif

                Kokkos::parallel_for("LoopOverChunks", policy_inst, gc);
            }
            break;

            // Two level parallelism:
            // 1. [P] loop over chunks of vertices
            // 2. [S] loop over vertices in chunks
            // 3. [P] loop over vertex neighbors
            // 4. [S] loop over vertex neighbors of neighbors
            case COLORING_D2_VBTP2:
            {
                functorGreedyColorVBTP2 gc(this->nv, xadj_, adj_, t_xadj_, t_adj_, vertex_colors_, current_vertexList_, current_vertexListLength_, chunkSize_);

                #if defined(KOKKOS_ENABLE_CUDA)
                const team_policy_t policy_inst(num_chunks, Kokkos::AUTO);
                #else
                const team_policy_t policy_inst(num_chunks, Kokkos::AUTO);
                #endif

                Kokkos::parallel_for("LoopOverChunks", policy_inst, gc);
            }
            break;

            // Two level parallelism:
            // 1. [P] loop over chunks of vertices
            // 2. [S] loop over vertices in chunks
            // 3. [S] loop over vertex neighbors
            // 4. [P] loop over vertex neighbors of neighbors
            case COLORING_D2_VBTP3:
            {
                functorGreedyColorVBTP3 gc(this->nv, xadj_, adj_, t_xadj_, t_adj_, vertex_colors_, current_vertexList_, current_vertexListLength_, chunkSize_);

                #if defined(KOKKOS_ENABLE_CUDA)
                const team_policy_t policy_inst(num_chunks, Kokkos::AUTO);
                #else
                const team_policy_t policy_inst(num_chunks, Kokkos::AUTO);
                #endif

                Kokkos::parallel_for("LoopOverChunks", policy_inst, gc);
            }
            break;

            // Three level parallelism:
            // 1. [P] loop over chunks of vertices
            // 2. [P] loop over vertices in chunks
            // 3. [P] loop over vertex neighbors
            // 4. [S] loop over vertex neighbors of neighbors
            case COLORING_D2_VBTPVR1:
            {
                functorGreedyColorVBTPVR1 gc(this->nv, xadj_, adj_, t_xadj_, t_adj_, vertex_colors_, current_vertexList_, current_vertexListLength_, chunkSize_);

                #if defined(KOKKOS_ENABLE_CUDA)
                const team_policy_t policy_inst(num_chunks, chunkSize_, 32);
                #else
                const team_policy_t policy_inst(num_chunks, Kokkos::AUTO, 1);
                #endif

                Kokkos::parallel_for("LoopOverChunks", policy_inst, gc);
            }
            break;

            // Three level parallelism:
            // 1. [P] loop over chunks of vertices
            // 2. [P] loop over vertices in chunks
            // 3. [S] loop over color offset blocks
            // 4. [S] loop over vertex neighbors
            // 5. [P] loop over vertex neighbors of neighbors
            case COLORING_D2_VBTPVR2:
            {
                functorGreedyColorVBTPVR2 gc(this->nv, xadj_, adj_, t_xadj_, t_adj_, vertex_colors_, current_vertexList_, current_vertexListLength_, chunkSize_);

                #if defined(KOKKOS_ENABLE_CUDA)
                const team_policy_t policy_inst(num_chunks, chunkSize_, 32);
                #else
                const team_policy_t policy_inst(num_chunks, Kokkos::AUTO, 1);
                #endif

                Kokkos::parallel_for("LoopOverChunks", policy_inst, gc);
            }
            break;

            // One level Perallelism, BIT Array for coloring
            // 1. [P] loop over chunks of vertices
            // 2. [S] loop over vertices in chunks
            // 3. [S] loop over color offset blocks
            // 4. [S] loop over vertex neighbors
            // 5. [S] loop over vertex neighbors of neighbors
            case COLORING_D2_VB_BIT:
            {
                functorGreedyColorVB_BIT gc(this->nv, xadj_, adj_, t_xadj_, t_adj_, vertex_colors_, current_vertexList_, current_vertexListLength_, chunkSize_);
                Kokkos::parallel_for("LoopOverChunks", my_exec_space(0, num_chunks), gc);
            }
            break;

            // Two level Perallelism, BIT Array for coloring
            // 1. [P] loop over chunks of vertices
            // 2. [P] loop over vertices in chunks
            // 3. [S] loop over color offset blocks
            // 4. [S] loop over vertex neighbors
            // 5. [S] loop over vertex neighbors of neighbors
            case COLORING_D2_VBTP_BIT:
            {
                functorGreedyColorVBTP_BIT gc(this->nv, xadj_, adj_, t_xadj_, t_adj_, vertex_colors_, current_vertexList_, current_vertexListLength_, chunkSize_);

                #if defined(KOKKOS_ENABLE_CUDA)
                const team_policy_t policy_inst(num_chunks, chunkSize_);
                #else
                const team_policy_t policy_inst(num_chunks, Kokkos::AUTO);
                #endif

                Kokkos::parallel_for("LoopOverChunks", policy_inst, gc);
            }
            break;

            default:
                throw std::invalid_argument("Unknown Distance-2 Algorithm Type or invalid for non Edge Filtering mode.");
        }

    }      // colorGreedy (end)



    // -----------------------------------------------------------------
    //
    // GraphColorD2::colorGreedyEF()
    //
    // -----------------------------------------------------------------
    void colorGreedyEF(const_lno_row_view_t xadj_,
                       nnz_lno_temp_work_view_t adj_copy_,
                       const_clno_row_view_t t_xadj_,
                       non_const_clno_nnz_view_t t_adj_copy_,
                       color_view_type vertex_colors_,
                       nnz_lno_temp_work_view_t current_vertexList_,
                       nnz_lno_t current_vertexListLength_)
    {
        nnz_lno_t chunkSize_ = this->_chunkSize;

        if(current_vertexListLength_ < 100 * chunkSize_)
        {
            chunkSize_ = 1;
        }

        const size_t num_chunks = current_vertexListLength_ / chunkSize_ + 1;

        // Pick the right coloring algorithm to use based on which algorithm we're using
        switch(this->gc_handle->get_coloring_algo_type())
        {
            // One level Perallelism, BIT Array for coloring + edge filtering
            // 1. [P] loop over chunks of vertices
            // 2. [S] loop over vertices in chunks
            // 3. [S] loop over color offset blocks
            // 4. [S] loop over vertex neighbors
            // 5. [S] loop over vertex neighbors of neighbors
            case COLORING_D2_VB_BIT_EF:
            {
                functorGreedyColorVB_BIT_EF gc(this->nv, xadj_, adj_copy_, t_xadj_, t_adj_copy_, vertex_colors_, current_vertexList_, current_vertexListLength_, chunkSize_);
                Kokkos::parallel_for("LoopOverChunks", my_exec_space(0, num_chunks), gc);
                //prettyPrint1DView(vertex_colors_, "COLORS_GC_VB_BIT",500);
            }
            break;

            default:
                throw std::invalid_argument("Unknown Distance-2 Algorithm Type or algorithm does not use Edge Filtering.");
        }
    }   // colorGreedyEF (end)



    // -----------------------------------------------------------------
    //
    // GraphColorD2::findConflicts()
    //
    // -----------------------------------------------------------------
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
                functorFindConflicts_Atomic<adj_view_t> conf(this->nv, xadj_, adj_, t_xadj_, t_adj_, vertex_colors_, current_vertexList_, next_iteration_recolorList_, next_iteration_recolorListLength_);
                Kokkos::parallel_reduce("FindConflicts", my_exec_space(0, current_vertexListLength_), conf, output_numUncolored);
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
    // GraphColorD2::resolveConflictsSerial()
    //
    // -----------------------------------------------------------------
    template<typename adj_view_t>
    void resolveConflictsSerial(nnz_lno_t _nv,
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
            for(size_type vid_d1_adj = h_idx(vid); vid_d1_adj < h_idx(vid + 1); vid_d1_adj++)
            {
                size_type vid_d1 = h_adj(vid_d1_adj);

                // loop over neighbors of vid_d1 (distance-2 from vid)
                for(size_type vid_d2_adj = h_t_idx(vid_d1); vid_d2_adj < h_t_idx(vid_d1 + 1); vid_d2_adj++)
                {
                    nnz_lno_t vid_d2 = h_t_adj(vid_d2_adj);

                    // skip over loops vid -- x -- vid
                    if(vid_d2 == vid)
                        continue;

                    forbidden[h_colors(vid_d2)] = vid;
                }
            }

            // color vertex vid with smallest available color
            int c = 1;
            while(forbidden[c] == vid) c++;

            h_colors(vid) = c;
        }
        Kokkos::deep_copy(vertex_colors_, h_colors);
        delete[] forbidden;
    }      // resolveConflictsSerial (end)


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


    // EXPERIMENTAL -- we might not need this for edge-filtering.
    void calculate_d2_degree(non_const_1d_size_type_view_t degree_d2)
    {
        nnz_lno_t chunk_size = this->_chunkSize;
        if(nv < 100 * chunk_size)
            {
                chunk_size = 1;
            }
            const size_t num_chunks = this->nv / chunk_size + 1;

            functorCalculateD2Degree calculateD2Degree(this->nv, this->xadj, this->adj, this->t_xadj, this->t_adj, chunk_size, degree_d2);
            Kokkos::parallel_for("Compute Degree D2", my_exec_space(0, num_chunks), calculateD2Degree);

            if(_ticToc)
            {
                prettyPrint1DView(degree_d2, "Degree D2", 150);
            }

            size_type degree_d2_max = 0;
            Kokkos::parallel_reduce("Max D2 Degree", this->nv, KOKKOS_LAMBDA(const int& i, size_type& lmax)
            {
                lmax = degree_d2(i) > lmax ? degree_d2(i) : lmax;
            }, Kokkos::Max<size_type>(degree_d2_max));
            std::cout << ">>> Max D2 Degree: " << degree_d2_max << std::endl;

            size_type degree_d2_sum = 0;
            Kokkos::parallel_reduce("Sum D2 Degree", this->nv, KOKKOS_LAMBDA(const int& i, size_type& lsum)
            {
                lsum += degree_d2(i);
            }, Kokkos::Sum<size_type>(degree_d2_sum));
            std::cout << ">>> D2 Degree Sum: " << degree_d2_sum << std::endl;
    }


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
     * Single level parallelism
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
            for(nnz_lno_t ichunk = 0; ichunk < _chunkSize; ichunk++)
            {
                if(vid_ * _chunkSize + ichunk < _vertexListLength)
                {
                    const nnz_lno_t vid = _vertexList(vid_ * _chunkSize + ichunk);

                    // If vertex is not already colored...
                    if(_colors(vid) <= 0)
                    {
                        bool foundColor = false;      // Have we found a valid color?

                        const size_type vid_adj_begin = _idx(vid);
                        const size_type vid_adj_end   = _idx(vid + 1);

                        // Use forbidden array to find available color.
                        // - should be small enough to fit into fast memory (use Kokkos memoryspace?)
                        bool forbidden[VB_D2_COLORING_FORBIDDEN_SIZE];      // Forbidden Colors

                        // Do multiple passes if the forbidden array is too small.
                        // * The Distance-1 code used the knowledge of the degree of the vertex to cap the number of iterations
                        //   but in distance-2 we'd need the total vertices at distance-2 which we don't easily have aprioi.
                        //   This could be as big as all the vertices in the graph if diameter(G)=2...
                        // * TODO: Determine if we can cap this at something lower than nv.
                        for(color_t offset = 0; !foundColor && offset < nv; offset += VB_D2_COLORING_FORBIDDEN_SIZE)
                        {
                            // initialize
                            for(int i = 0; i < VB_D2_COLORING_FORBIDDEN_SIZE; i++) { forbidden[i] = false; }

                            // Colors start at 1.  0 is special in that it means a vertex is uncolored.
                            // For the range 0..63 we mark forbidden[0] as true to take color 0 out of
                            // consideration.
                            if(0 == offset)
                            {
                                forbidden[0] = true;
                            }

                            // Check neighbors, fill forbidden array.
                            for(size_type vid_adj = vid_adj_begin; vid_adj < vid_adj_end; vid_adj++)
                            {
                                const nnz_lno_t vid_d1           = _adj(vid_adj);
                                const size_type vid_d1_adj_begin = _t_idx(vid_d1);
                                const size_type vid_d1_adj_end   = _t_idx(vid_d1 + 1);

                                for(size_type vid_d1_adj = vid_d1_adj_begin; vid_d1_adj < vid_d1_adj_end; vid_d1_adj++)
                                {
                                    const nnz_lno_t vid_d2 = _t_adj(vid_d1_adj);

                                    // Skip distance-2-self-loops
                                    if(vid_d2 != vid && vid_d2 < nv)
                                    {
                                        const color_t c = _colors(vid_d2);

                                        if((c >= offset) && (c - offset < VB_D2_COLORING_FORBIDDEN_SIZE))
                                        {
                                            forbidden[c - offset] = true;
                                        }
                                    }
                                }      // for vid_d1_adj...
                            }          // for vid_adj...

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
                        }          // for !foundColor...
                    }              // if _colors(vid) <= 0...
                }                  // if vid_ * _chunkSize...
            }                      // for ichunk...
        }                          // operator() (end)
    };                             // struct functorGreedyColorVB (end)



    /**
     * Functor for VB_BIT algorithm coloring without edge filtering.
     * Single level parallelism
     */
    struct functorGreedyColorVB_BIT
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

        functorGreedyColorVB_BIT(nnz_lno_t nv_,
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
            for(nnz_lno_t ichunk = 0; ichunk < _chunkSize; ichunk++)
            {
                if(vid_ * _chunkSize + ichunk < _vertexListLength)
                {
                    const nnz_lno_t vid = _vertexList(vid_ * _chunkSize + ichunk);

                    // If vertex is not colored yet...
                    if(_colors(vid) == 0)
                    {
                        const size_type vid_adj_begin = _idx(vid);
                        const size_type vid_adj_end   = _idx(vid + 1);

                        bool foundColor = false;
                        for(color_t offset = 0; !foundColor && offset <= (nv + VBBIT_D2_COLORING_FORBIDDEN_SIZE); offset += VBBIT_D2_COLORING_FORBIDDEN_SIZE)
                        {
                            // Forbidden colors
                            // - single long int for forbidden colors
                            bit_64_forbidden_t forbidden = 0;

                            // If all available colors for this range are unavailable we can break out of the nested loops
                            bool break_out = false;

                            // Loop over distance-1 neighbors of vid
                            for(size_type vid_adj = vid_adj_begin; !break_out && vid_adj < vid_adj_end; ++vid_adj)
                            {
                                const nnz_lno_t vid_d1           = _adj(vid_adj);
                                const size_type vid_d1_adj_begin = _t_idx(vid_d1);
                                const size_type vid_d1_adj_end   = _t_idx(vid_d1 + 1);

                                // Loop over distance-2 neighbors of vid
                                for(size_type vid_d1_adj = vid_d1_adj_begin; !break_out && vid_d1_adj < vid_d1_adj_end; ++vid_d1_adj)
                                {
                                    const nnz_lno_t vid_d2 = _t_adj(vid_d1_adj);

                                    // Ignore Distance-2 Self Loops
                                    if(vid_d2 != vid && vid_d2 < nv)
                                    {
                                        const color_t color        = _colors(vid_d2);
                                        const color_t color_offset = color - offset;

                                        // if color is within the current range, or if its color is in a previously traversed
                                        // range
                                        if(color && color_offset <= VBBIT_D2_COLORING_FORBIDDEN_SIZE)
                                        {
                                            // if it is in the current range, then add the color to the banned colors
                                            if(color > offset)
                                            {
                                                // convert color to bit representation
                                                bit_64_forbidden_t ban_color_bit = 1;

                                                ban_color_bit = ban_color_bit << (color_offset - 1);

                                                // add it to forbidden colors
                                                forbidden = forbidden | ban_color_bit;

                                                // if there are no available colors in this range then exit early,
                                                // no need to traverse the rest.
                                                if(0 == ~forbidden)
                                                {
                                                    break_out = true;
                                                }
                                            }      // if color > offset ...
                                        }          // if color && color_offset ...
                                    }              // if vid_d2 ...
                                }                  // for vid_d1_adj ...
                            }                      // for vid_adj ...
                            forbidden = ~(forbidden);

                            // check if an available color exists.
                            if(forbidden)
                            {
                                color_t val = 1;

                                // if there is an available color, choose the first color, using 2s complement.
                                bit_64_forbidden_t new_color = forbidden & (-forbidden);

                                // convert it back to decimal color.
                                while((new_color & 1) == 0)
                                {
                                    ++val;
                                    new_color = new_color >> 1;
                                }
                                _colors(vid) = val + offset;
                                foundColor   = true;
                                break;
                            }
                        }      // for !foundColor
                    }          // if _colors(vid)==0
                }              // if vid_ * _chunkSize ...
            }                  // for ichunk...
        }                      // operator() (end)
    };                         // struct functorGreedyColorVB_BIT (end)



    /**
     * Functor for VBTP_BIT algorithm coloring without edge filtering.
     * Two level parallelism
     * - [P] Loop over chunks
     * - [P] Loop over vertices in each chunk
     * - [S] Loop over offset range(s)
     * - [S] Loop over neighbors of vid: vid_d1
     * - [S] Loop over neighbors of vid_d1: vid_d2
     */
    struct functorGreedyColorVBTP_BIT
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

        functorGreedyColorVBTP_BIT(nnz_lno_t nv_,
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

                    // If vertex is not colored yet...
                    if(_colors(vid) == 0)
                    {
                        size_type vid_adj_begin = _idx(vid);
                        size_type vid_adj_end   = _idx(vid + 1);

                        bool foundColor = false;
                        for(color_t offset = 0; !foundColor && offset <= (nv + VBBIT_D2_COLORING_FORBIDDEN_SIZE); offset += VBBIT_D2_COLORING_FORBIDDEN_SIZE)
                        {
                            // Forbidden colors
                            // - single long int for forbidden colors
                            bit_64_forbidden_t forbidden = 0;

                            // If all available colors for this range are unavailable we can break out of the nested loops
                            bool break_out = false;

                            // Loop over distance-1 neighbors of vid
                            for(size_type vid_adj = vid_adj_begin; !break_out && vid_adj < vid_adj_end; ++vid_adj)
                            {
                                const nnz_lno_t vid_d1     = _adj(vid_adj);
                                size_type vid_d1_adj_begin = _t_idx(vid_d1);
                                size_type vid_d1_adj_end   = _t_idx(vid_d1 + 1);

                                // Loop over distance-2 neighbors of vid
                                for(size_type vid_d1_adj = vid_d1_adj_begin; !break_out && vid_d1_adj < vid_d1_adj_end; ++vid_d1_adj)
                                {
                                    const nnz_lno_t vid_d2 = _t_adj(vid_d1_adj);

                                    // Ignore Distance-2 Self Loops
                                    if(vid_d2 != vid && vid_d2 < nv)
                                    {
                                        color_t color        = _colors(vid_d2);
                                        color_t color_offset = color - offset;

                                        // if color is within the current range, or if its color is in a previously traversed
                                        // range
                                        if(color && color_offset <= VBBIT_D2_COLORING_FORBIDDEN_SIZE)
                                        {
                                            // if it is in the current range, then add the color to the banned colors
                                            if(color > offset)
                                            {
                                                // convert color to bit representation
                                                bit_64_forbidden_t ban_color_bit = 1;

                                                ban_color_bit = ban_color_bit << (color_offset - 1);

                                                // add it to forbidden colors
                                                forbidden = forbidden | ban_color_bit;

                                                // if there are no available colors in this range then exit early,
                                                // no need to traverse the rest.
                                                if(0 == ~forbidden)
                                                {
                                                    break_out = true;
                                                }
                                            }
                                        }
                                    }      // if vid_d2...
                                }          // for vid_d1_adj...
                            }              // for vid_adj...
                            forbidden = ~(forbidden);

                            // check if an available color exists.
                            if(forbidden)
                            {
                                // if there is an available color, choose the first color, using 2s complement.
                                bit_64_forbidden_t new_color = forbidden & (-forbidden);
                                color_t val                  = 1;

                                // convert it back to decimal color.
                                while((new_color & 1) == 0)
                                {
                                    ++val;
                                    new_color = new_color >> 1;
                                }
                                _colors(vid) = val + offset;
                                foundColor   = true;
                                break;
                            }
                        }      // for !foundColor
                    }          // if _colors(vid)==0
                }              // if vid_ * _chunkSize ...
            });                // parallel_for ichunk...
        }                      // operator() (end)
    };                         // struct functorGreedyColorVBTP_BIT (end)



   /**
     * Functor for VB_BIT_EF algorithm coloring without edge filtering.
     * Single level parallelism
     */
    struct functorGreedyColorVB_BIT_EF
    {
        nnz_lno_t _nv;                              // num vertices
        const_lno_row_view_t _idx;                 // vertex degree list
        nnz_lno_temp_work_view_t _adj;             // vertex adjacency list  (mutable)
        const_clno_row_view_t _t_idx;              // transpose vertex degree list
        non_const_clno_nnz_view_t _t_adj;          // transpose vertex adjacency list (mutable)
        color_view_type _colors;                   // vertex colors
        nnz_lno_temp_work_view_t _vertexList;      //
        nnz_lno_t _vertexListLength;               //
        nnz_lno_t _chunk_size;                     //

        functorGreedyColorVB_BIT_EF(nnz_lno_t nv,
                                    const_lno_row_view_t xadj,
                                    nnz_lno_temp_work_view_t adj,
                                    const_clno_row_view_t t_xadj,
                                    non_const_clno_nnz_view_t t_adj,
                                    color_view_type colors,
                                    nnz_lno_temp_work_view_t vertexList,
                                    nnz_lno_t vertexListLength,
                                    nnz_lno_t chunk_size)
            : _nv(nv), _idx(xadj), _adj(adj), _t_idx(t_xadj), _t_adj(t_adj), _colors(colors), _vertexList(vertexList),
              _vertexListLength(vertexListLength), _chunk_size(chunk_size)
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
        void operator()(const nnz_lno_t chunk_idx) const
        {
            for(nnz_lno_t ichunk = 0; ichunk < _chunk_size; ichunk++)
            {
                const nnz_lno_t vid_idx = chunk_idx * _chunk_size + ichunk;

                if(vid_idx < _vertexListLength)
                {
                    const nnz_lno_t vid = _vertexList(vid_idx);

                    // If vertex is not colored yet..
                    if(_colors(vid) == 0)
                    {
                        size_type vid_adj_begin = _idx(vid);
                        size_type vid_adj_end   = _idx(vid + 1);
                        //size_type degree_vid    = vid_adj_end - vid_adj_begin;                                                    // EXPERIMENTAL

                        bool foundColor = false;
                        for(color_t offset = 0; !foundColor && offset <= (_nv + VBBIT_D2_COLORING_FORBIDDEN_SIZE); offset += VBBIT_D2_COLORING_FORBIDDEN_SIZE)
                        {
                            // Forbidden colors
                            // - single long int for forbidden colors
                            bit_64_forbidden_t forbidden = 0;

                            // If all available colors for this range are unavailable we can break out of the nested loops
                            bool offset_colors_full = false;

                            // Loop over distance-1 neighbors of vid
                            for(size_type vid_adj = vid_adj_begin; !offset_colors_full && vid_adj < vid_adj_end; ++vid_adj)
                            {
                                const nnz_lno_t vid_d1 = _adj(vid_adj);

                                size_type vid_d1_adj_begin     = _t_idx(vid_d1);
                                const size_type vid_d1_adj_end = _t_idx(vid_d1 + 1);
                                const size_type degree_vid_d1  = vid_d1_adj_end - vid_d1_adj_begin;
                                size_type num_vid_d2_colored_in_range = 0;

                                // Store the maximum color value found in the vertices adjacent to vid_d1
                                color_t max_color_adj_to_d1 = 0;

                                // Loop over distance-2 neighbors of vid
                                for(size_type vid_d1_adj = vid_d1_adj_begin; !offset_colors_full && vid_d1_adj < vid_d1_adj_end; ++vid_d1_adj)
                                {
                                    const nnz_lno_t vid_d2 = _t_adj(vid_d1_adj);

                                    // Ignore Distance-2 Self Loops
                                    if(vid_d2 != vid && vid_d2 < _nv)
                                    {
                                        color_t color        = _colors(vid_d2);
                                        color_t color_offset = color - offset;          // color_offset < 0 means color is from a previous offset.

                                        // Update maximum color adjacent to vid_d1 found so far.
                                        max_color_adj_to_d1 = color > max_color_adj_to_d1 ? color : max_color_adj_to_d1;

                                        // if color is within the current range, or if its color is in a previously traversed
                                        // range
                                        if(color && color_offset <= VBBIT_D2_COLORING_FORBIDDEN_SIZE)
                                        {
                                            num_vid_d2_colored_in_range++;

                                            // if it is in the current range, then add the color to the banned colors
                                            if(color > offset)
                                            {
                                                // convert color to bit representation
                                                bit_64_forbidden_t ban_color_bit = 1;

                                                ban_color_bit = ban_color_bit << (color_offset - 1);

                                                // add it to forbidden colors
                                                forbidden = forbidden | ban_color_bit;

                                                // if there are no available colors in this range then exit early,
                                                // no need to traverse the rest.
                                                if(0 == ~forbidden)
                                                {
                                                    offset_colors_full = true;
                                                }
                                            }
                                        }      // if color && color_offset
                                    }          // if vid_d2 != vid ...
                                }              // for vid_d1_adj ...

                                    // Edge filtering on the neighbors of vid.  We can only do this if ALL neighbors of vid_d1
                                    // have been visited and if they're colored in current offset range or lower.
                                    if(degree_vid_d1 == num_vid_d2_colored_in_range)
                                    {
                                        if(vid_adj_begin > vid_adj)
                                        {
                                            _adj(vid_adj) = _adj(vid_adj_begin);
                                            _adj(vid_adj_begin) = vid_d1;
                                        }
                                        vid_adj_begin++;
                                    }

                            }      // for vid_adj
                            forbidden = ~(forbidden);

                            // check if an available color exists.
                            if(forbidden)
                            {
                                // if there is an available color, choose the first color, using 2s complement.
                                bit_64_forbidden_t new_color = forbidden & (-forbidden);
                                color_t val                  = 1;

                                // convert it back to decimal color.
                                while((new_color & 1) == 0)
                                {
                                    ++val;
                                    new_color = new_color >> 1;
                                }
                                _colors(vid) = val + offset;
                                foundColor   = true;
                                break;
                            }
                        }      // for offset=0...
                    }          // if _colors(vid)==0
                }              // if vid_ * _chunkSize ...
            }                  // for ichunk...
        }                      // operator() (end)
    };                         // struct functorGreedyColorVB_BIT_EF (end)



    /**
     * Functor for VB algorithm speculative coloring without edge filtering.
     * Team Policy Enabled on loop over chunks
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

            Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, _chunkSize), [&](const nnz_lno_t ichunk) {
                if(chunk_id * _chunkSize + ichunk < _vertexListLength)
                {
                    const nnz_lno_t vid = _vertexList(chunk_id * _chunkSize + ichunk);

                    // Already colored this vertex.
                    if(_colors(vid) <= 0)
                    {
                        bool foundColor = false;      // Have we found a valid color?

                        // Use forbidden array to find available color.
                        // - should be small enough to fit into fast memory (use Kokkos memoryspace?)
                        // - If more levels of parallelism are added in the loops over neighbors, then
                        //   atomics will be necessary for updating this.
                        bool forbidden[VB_D2_COLORING_FORBIDDEN_SIZE];      // Forbidden Colors

                        // Do multiple passes if we fill the current range of possible colors
                        // * TODO: Determine if we can cap this at something smaller than nv, or if it matters.
                        for(color_t offset=0; !foundColor && offset < nv; offset += VB_D2_COLORING_FORBIDDEN_SIZE)
                        {
                            // initialize the forbidden array to { false }
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
                                        const color_t c = _colors(vid_d2);

                                        // If color found is inside current 'range' then mark it as used.
                                        if((c >= offset) && (c - offset < VB_D2_COLORING_FORBIDDEN_SIZE))
                                        {
                                            forbidden[c - offset] = true;
                                        }
                                    }      // if vid_d2 != vid ...
                                }          // for vid_d2_adj ...
                            }              // for vid_d1_adj ...

                            // color vertex i with smallest available color (firstFit)
                            for(int c = 0; c < VB_D2_COLORING_FORBIDDEN_SIZE; c++)
                            {
                                if(!forbidden[c])
                                {
                                    _colors(vid) = offset + c;
                                    foundColor   = true;
                                    break;
                                }
                            }      // for c=0 ...
                        }      // for offset=0 ...
                    }          // if _colors(vid) <= 0 ...
                }              // if chunk_id* ...
            });                // for ichunk ...
        }                      // operator() (end)
    };                         // struct functorGreedyColorVBTP (end)



    /**
     * Functor for VB algorithm speculative coloring without edge filtering.
     * Team Policy Enabled on loop over neighbors
     * - Serialized the loop over chunks
     */
    struct functorGreedyColorVBTP2
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

        functorGreedyColorVBTP2(nnz_lno_t nv_,
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

            for(nnz_lno_t ichunk = 0; ichunk < _chunkSize; ichunk++)
            {
                const nnz_lno_t vertex_list_idx = chunk_id * _chunkSize + ichunk;
                if(vertex_list_idx < _vertexListLength)
                {
                    const nnz_lno_t vid = _vertexList(vertex_list_idx);

                    // Already colored this vertex.
                    if(_colors(vid) <= 0)
                    {
                        bool foundColor = false;      // Have we found a valid color?

                        // Use forbidden array to find available color.
                        // - should be small enough to fit into fast memory (use Kokkos memoryspace?)
                        // - If more levels of parallelism are added in the loops over neighbors, then
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
                            // for(size_type vid_d1_adj = _idx(vid); vid_d1_adj < _idx(vid + 1); vid_d1_adj++)
                            Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, _idx(vid + 1) - _idx(vid)), [&](const size_type &idx)
                            {
                                const size_type vid_d1_adj = _idx(vid) + idx;
                                const nnz_lno_t vid_d1     = _adj(vid_d1_adj);

                                // Loop over distance-2 neighbors
                                for(size_type vid_d2_adj = _t_idx(vid_d1); vid_d2_adj < _t_idx(vid_d1 + 1); vid_d2_adj++)
                                {
                                    const nnz_lno_t vid_d2 = _t_adj(vid_d2_adj);

                                    // Skip distance-2 self loops
                                    if(vid_d2 != vid && vid_d2 < nv)
                                    {
                                        const color_t c = _colors(vid_d2);

                                        // If color found is inside current 'range' then mark it as used.
                                        if((c >= offset) && (c - offset < VB_D2_COLORING_FORBIDDEN_SIZE))
                                        {
                                            Kokkos::atomic_fetch_or(&forbidden[c - offset], true);
                                        }
                                    }
                                }
                            });

                            thread.team_barrier();

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
                }              // if vid_ * ......
            }                  // for ichunk...
        }                      // operator() (end)
    };                         // struct functorGreedyColorVBTP2 (end)



    /**
     * Functor for VB algorithm speculative coloring without edge filtering.
     * Team Policy Enabled
     *  - trying out removing the loop-over-chunks stuff
     */
    struct functorGreedyColorVBTP3
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

        functorGreedyColorVBTP3(nnz_lno_t nv_,
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

            for(nnz_lno_t ichunk = 0; ichunk < _chunkSize; ichunk++)
            {
                const nnz_lno_t vertex_list_idx = chunk_id * _chunkSize + ichunk;
                if(vertex_list_idx < _vertexListLength)
                {
                    const nnz_lno_t vid = _vertexList(vertex_list_idx);

                    // Already colored this vertex.
                    if(_colors(vid) <= 0)
                    {
                        bool foundColor = false;      // Have we found a valid color?

                        // Use forbidden array to find available color.
                        // - should be small enough to fit into fast memory (use Kokkos memoryspace?)
                        // - If more levels of parallelism are added in the loops over neighbors, then
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
                                Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, _t_idx(vid_d1 + 1) - _t_idx(vid_d1)), [&](const size_type &idx)
                                {
                                    const size_type vid_d2_adj = _t_idx(vid_d1) + idx;
                                    const nnz_lno_t vid_d2     = _t_adj(vid_d2_adj);

                                    // Skip distance-2 self loops
                                    if(vid_d2 != vid && vid_d2 < nv)
                                    {
                                        const color_t c = _colors(vid_d2);

                                        // If color found is inside current 'range' then mark it as used.
                                        if((c >= offset) && (c - offset < VB_D2_COLORING_FORBIDDEN_SIZE))
                                        {
                                            Kokkos::atomic_fetch_or(&forbidden[c - offset], true);
                                        }
                                    }
                                });
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
                }              // if vertex_list_idx < _vertexListLength
            }                  // for ichunk...
        }                      // operator() (end)
    };                         // struct functorGreedyColorVBTP3 (end)



    /**
     * Functor for VB algorithm speculative coloring without edge filtering.
     * Team Policy Enabled on loop over chunks
     * Vector Range Enabled on loop over neighbors of neighbors
     */
    struct functorGreedyColorVBTPVR1
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

        functorGreedyColorVBTPVR1(nnz_lno_t nv_,
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
                        // - If more levels of parallelism are added in the loops over neighbors, then
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
                            Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, _idx(vid + 1) - _idx(vid)), [&](const size_type &idx)
                            {
                                //size_type vid_d1_adj   = idx + _idx(vid);
                                const size_type vid_d1_adj = _idx(vid) + idx;
                                const nnz_lno_t vid_d1 = _adj(vid_d1_adj);

                                // Loop over distance-2 neighbors
                                for(size_type vid_d2_adj = _t_idx(vid_d1); vid_d2_adj < _t_idx(vid_d1 + 1); vid_d2_adj++)
                                {
                                    const nnz_lno_t vid_d2 = _t_adj(vid_d2_adj);

                                    // Skip distance-2 self loops
                                    if(vid_d2 != vid && vid_d2 < nv)
                                    {
                                        const color_t c = _colors(vid_d2);

                                        // If color found is inside current 'range' then mark it as used.
                                        if((c >= offset) && (c - offset < VB_D2_COLORING_FORBIDDEN_SIZE))
                                        {
                                            Kokkos::atomic_fetch_or(&forbidden[c - offset], true);
                                        }
                                    }
                                }
                            });

                            thread.team_barrier();

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
    };                         // struct functorGreedyColorVBTPVR1 (end)



    /**
     * Functor for VB algorithm speculative coloring without edge filtering.
     * Team Policy Enabled on loop-over -chunks
     * Vector Range Enabled on loop-over-neighbors
     */
    struct functorGreedyColorVBTPVR2
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

        functorGreedyColorVBTPVR2(nnz_lno_t nv_,
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
                        // - If more levels of parallelism are added in the loops over neighbors, then
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

                            // Check neighbors, fill forbidden array.
                            for(size_type vid_d1_adj = _idx(vid); vid_d1_adj < _idx(vid + 1); vid_d1_adj++)
                            {
                                const nnz_lno_t vid_d1 = _adj(vid_d1_adj);

                                // Loop over distance-2 neighbors
                                Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, _t_idx(vid_d1 + 1) - _t_idx(vid_d1)), [&](const size_type &idx)
                                {
                                    //const size_type vid_d2_adj = idx + _t_idx(vid);
                                    const size_type vid_d2_adj = _t_idx(vid_d1) + idx;
                                    const nnz_lno_t vid_d2     = _t_adj(vid_d2_adj);

                                    // Skip distance-2 self loops
                                    if(vid_d2 != vid && vid_d2 < nv)
                                    {
                                        const color_t c = _colors(vid_d2);

                                        // If color found is inside current 'range' then mark it as used.
                                        if((c >= offset) && (c - offset < VB_D2_COLORING_FORBIDDEN_SIZE))
                                        {
                                            Kokkos::atomic_fetch_or(&forbidden[c - offset], true);
                                        }
                                    }
                                });      // for vid_d2_adj...
                                thread.team_barrier();
                            }            // for vid_d1_adj...

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
    };                         // struct functorGreedyColorVBTPVR2 (end)



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
            const nnz_lno_t vid    = _vertexList(vid_);
            const color_t my_color = _colors(vid);

            size_type vid_d1_adj           = _idx(vid);
            const size_type vid_d1_adj_end = _idx(vid + 1);

            bool break_out = false;

            for(; !break_out && vid_d1_adj < vid_d1_adj_end; vid_d1_adj++)
            {
                nnz_lno_t vid_d1 = _adj(vid_d1_adj);

                for(size_type vid_d2_adj = _t_idx(vid_d1); !break_out && vid_d2_adj < _t_idx(vid_d1 + 1); vid_d2_adj++)
                {
                    const nnz_lno_t vid_d2 = _t_adj(vid_d2_adj);

                    if(vid != vid_d2 && vid_d2 < nv)
                    {
                        if(_colors(vid_d2) == my_color)
                        {
                            _colors(vid) = 0;      // uncolor vertex

                            // Atomically add vertex to recolorList
                            const nnz_lno_t k = Kokkos::atomic_fetch_add(&_recolorListLength(), atomic_incr_type(1));
                            _recolorList(k)   = vid;
                            numConflicts += 1;
                            break_out = true;
                            // break;      // Can exit if vertex gets marked as a conflict.
                        }
                    }      // if vid != vid_d2 ...
                }          // for vid_d2_adj ...
            }              // for vid_d1_adj ...
        }                  // operator() (end)
    };                     // struct functorFindConflicts_Atomic (end)



    /**
     * functorVerifyDistance2Coloring
     *  - Validate correctness of the distance-2 coloring
     */
    struct functorVerifyDistance2Coloring
    {
        nnz_lno_t nv;                          // num vertices
        const_lno_row_view_t _idx;             // vertex degree list
        const_lno_nnz_view_t _adj;             // vertex adjacency list
        const_clno_row_view_t _t_idx;          // transpose vertex degree list
        const_clno_nnz_view_t _t_adj;          // transpose vertex adjacency list
        color_view_type _colors;               // vertex colors
        non_const_1d_bool_view_t _flags;       // [0]: valid or not?  [1]=true means something is uncolored. [2]=true means something has distance-2 neighbor of same color
        nnz_lno_t _chunkSize;                  //

        functorVerifyDistance2Coloring(nnz_lno_t nv_,
                                       const_lno_row_view_t xadj_,
                                       const_lno_nnz_view_t adj_,
                                       const_clno_row_view_t t_xadj_,
                                       const_clno_nnz_view_t t_adj_,
                                       color_view_type colors,
                                       non_const_1d_bool_view_t flags,
                                       nnz_lno_t chunkSize)
            : nv(nv_)
              , _idx(xadj_)
              , _adj(adj_)
              , _t_idx(t_xadj_)
              , _t_adj(t_adj_)
              , _colors(colors)
              , _flags(flags)
              , _chunkSize(chunkSize)
        {
        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const nnz_lno_t chunk_id) const
        {
            bool has_uncolored_vertex            = false;
            bool has_invalid_color               = false;
            bool has_color_bigger_than_num_verts = false;

            // Loop over vertices in chunks
            for(nnz_lno_t chunk_idx = 0; chunk_idx < _chunkSize; chunk_idx++)
            {
                bool break_out = false;
                nnz_lno_t vid  = chunk_id * _chunkSize + chunk_idx;

                if(vid < nv)
                {
                    const color_t color_vid          = _colors(vid);
                    const size_type vid_d1_adj_begin = _idx(vid);
                    const size_type vid_d1_adj_end   = _idx(vid + 1);

                    if(color_vid == 0)
                    {
                        has_uncolored_vertex = true;
                    }

                    if(color_vid > nv)
                    {
                        has_color_bigger_than_num_verts = true;
                    }

                    // Loop over neighbors of vid (distance-1 from vid)
                    for(size_type vid_d1_adj = vid_d1_adj_begin; !break_out && vid_d1_adj < vid_d1_adj_end; ++vid_d1_adj)
                    {
                        const nnz_lno_t vid_d1           = _adj(vid_d1_adj);
                        const size_type vid_d2_adj_begin = _t_idx(vid_d1);
                        const size_type vid_d2_adj_end   = _t_idx(vid_d1 + 1);

                        // Loop over neighbors of vid_d1 (distance-2 from vid)
                        for(size_type vid_d2_adj = vid_d2_adj_begin; !break_out && vid_d2_adj < vid_d2_adj_end; ++vid_d2_adj)
                        {
                            const nnz_lno_t vid_d2 = _t_adj(vid_d2_adj);

                            // Ignore Distance-2 self loops
                            if(vid != vid_d2)
                            {
                                const color_t color_d2 = _colors(vid_d2);
                                // If distance-2 neighbor of vid has the same color as vid, then the coloring is invalid
                                if(color_vid == color_d2)
                                {
                                    has_invalid_color = true;
                                    break_out         = true;
                                    std::cout << ">>> Invalid color match: " << vid << ", " << vid_d2 << " both have color " << color_vid << std::endl;
                                }
                            }
                        }
                    }
                }
            }
            if(has_uncolored_vertex || has_invalid_color)
                Kokkos::atomic_fetch_or(&_flags[0], has_uncolored_vertex || has_invalid_color);
            if(has_uncolored_vertex)
                Kokkos::atomic_fetch_or(&_flags[1], has_uncolored_vertex);
            if(has_invalid_color)
                Kokkos::atomic_fetch_or(&_flags[2], has_invalid_color);
            if(has_color_bigger_than_num_verts)
                Kokkos::atomic_fetch_or(&_flags[3], has_color_bigger_than_num_verts);
        }      // operator()
    };         // struct functorGreedyColorVB (end)


    // EXPERIMENTAL Begin
    /**
     * functorVerifyDistance2Coloring
     *  - Validate correctness of the distance-2 coloring
     */
    struct functorCalculateD2Degree
    {
        nnz_lno_t num_verts;                           // num vertices
        const_lno_row_view_t _idx;                     // vertex degree list
        const_lno_nnz_view_t _adj;                     // vertex adjacency list
        const_clno_row_view_t _t_idx;                  // transpose vertex degree list
        const_clno_nnz_view_t _t_adj;                  // transpose vertex adjacency list
        nnz_lno_t _chunk_size;                         // Chunksize (threads work on |chunk_size| number of verts in batch.
        non_const_1d_size_type_view_t _degree_d2;      // Distance-2 Degree (assumes all are initialized to 0)

        functorCalculateD2Degree(nnz_lno_t num_verts,
                                 const_lno_row_view_t xadj,
                                 const_lno_nnz_view_t adj,
                                 const_clno_row_view_t t_xadj,
                                 const_clno_nnz_view_t t_adj,
                                 nnz_lno_t chunk_size,
                                 non_const_1d_size_type_view_t degree_d2)
            : num_verts(num_verts), _idx(xadj), _adj(adj), _t_idx(t_xadj), _t_adj(t_adj), _chunk_size(chunk_size), _degree_d2(degree_d2)
        {
        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const nnz_lno_t chunk_id) const
        {
            // Loop over vertices in chunks
            for(nnz_lno_t chunk_idx = 0; chunk_idx < _chunk_size; chunk_idx++)
            {
                nnz_lno_t vid  = chunk_id * _chunk_size + chunk_idx;
                if(vid < num_verts)
                {
                    const size_type vid_d1_adj_begin = _idx(vid);
                    const size_type vid_d1_adj_end   = _idx(vid + 1);

                    // Loop over neighbors of vid (distance-1 from vid)
                    for(size_type vid_d1_adj = vid_d1_adj_begin; vid_d1_adj < vid_d1_adj_end; ++vid_d1_adj)
                    {
                        const nnz_lno_t vid_d1           = _adj(vid_d1_adj);
                        const size_type vid_d2_adj_begin = _t_idx(vid_d1);
                        const size_type vid_d2_adj_end   = _t_idx(vid_d1 + 1);
                        const size_type degree_vid_d1    = vid_d2_adj_end - vid_d2_adj_begin;
                        _degree_d2(vid) += degree_vid_d1;
                    }      // for vid_d1_adj ...
                }          // if vid < nv ...
            }              // for chunk_idx ...
        }                  // operator() (end)
    };                     // struct functorCalculateD2Degree (end)
    // EXPERIMENTAL End




};      // end class GraphColorD2



}      // namespace Impl
}      // namespace KokkosGraph


#endif      // _KOKKOSCOLORINGD2IMP_HPP
