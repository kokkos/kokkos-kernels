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
#ifndef _KOKKOS_GRAPH_COLORD2_HPP
#define _KOKKOS_GRAPH_COLORD2_HPP

#include "KokkosGraph_GraphColor_impl.hpp"
#include "KokkosGraph_Distance2Color_impl.hpp"
#include "KokkosGraph_Distance2Color_MatrixSquared_impl.hpp"
#include "KokkosGraph_GraphColorHandle.hpp"
#include "KokkosKernels_Utils.hpp"

namespace KokkosGraph{

namespace Experimental{



/**
 *  Distance 2 Graph Coloring
 */
template<class KernelHandle, typename lno_row_view_t_, typename lno_nnz_view_t_, typename lno_col_view_t_, typename lno_colnnz_view_t_>
void graph_color_d2(KernelHandle *handle,
                    typename KernelHandle::nnz_lno_t num_rows,
                    typename KernelHandle::nnz_lno_t num_cols,
                    lno_row_view_t_ row_map,
                    lno_nnz_view_t_ row_entries,
                    // If graph is symmetric, simply give same for col_map and row_map, and row_entries and col_entries.
                    lno_col_view_t_ col_map,
                    lno_colnnz_view_t_ col_entries)
{
    Kokkos::Impl::Timer timer;

    // Set our handle pointer to a GraphColoringHandleType.
    typename KernelHandle::GraphColoringHandleType *gch = handle->get_graph_coloring_handle();

    // Get the algorithm we're running from the graph coloring handle.
    ColoringAlgorithm algorithm = gch->get_coloring_algo_type();

    // Create a view to save the colors to.
    // - Note: color_view_t is a Kokkos::View<color_t *, HandlePersistentMemorySpace> color_view_t    (KokkosGraph_GraphColorHandle.hpp)
    //         a 1D array of color_t
    typedef typename KernelHandle::GraphColoringHandleType::color_view_t color_view_type;
    color_view_type colors_out("Graph Colors", num_rows);

    gch->set_tictoc(handle->get_verbose());

    switch(algorithm)
    {
        case COLORING_SPGEMM:      // WCMCLEN: Remove SPGEMM coloring references for D2 Graph Coloring?
        case COLORING_D2_MATRIX_SQUARED:
        {
            Impl::GraphColorD2_MatrixSquared<KernelHandle, lno_row_view_t_, lno_nnz_view_t_, lno_col_view_t_, lno_colnnz_view_t_>
            gc(num_rows, num_cols, row_entries.extent(0), row_map, row_entries, col_map, col_entries, handle);
            gc.execute();
            break;
        }

        case COLORING_D2_SERIAL:
        {
            #if defined KOKKOS_ENABLE_SERIAL
            int num_phases = 0;
            std::cout << "KokkosGraph_Distance2Color.hpp::COLORING_D2_SERIAL" << std::endl;
            Impl::GraphColor<typename KernelHandle::GraphColoringHandleType, lno_row_view_t_, lno_nnz_view_t_>
            gc(num_rows, row_entries.extent(0), row_map, row_entries, gch);
            gc.d2_color_graph(colors_out, num_phases, num_cols, col_map, col_entries);

            // Save out the number of phases and vertex colors
            gch->set_vertex_colors(colors_out);
            gch->set_num_phases((double)num_phases);
            #else
            throw std::runtime_error("Kokkos-Kernels must be built with Serial enabled to use COLORING_D2_SERIAL");
            #endif
            break;
        }

        case COLORING_D2:
        case COLORING_D2_VB:
        case COLORING_D2_VB_BIT:
        case COLORING_D2_VB_BIT_EF:
        case COLORING_D2_VBTP:
        case COLORING_D2_VBTP_BIT:
        case COLORING_D2_VBTP2:
        case COLORING_D2_VBTP3:
        case COLORING_D2_VBTPVR1:
        case COLORING_D2_VBTPVR2:
        {
            Impl::GraphColorD2<KernelHandle, lno_row_view_t_, lno_nnz_view_t_, lno_col_view_t_, lno_colnnz_view_t_>
            gc(num_rows, num_cols, row_entries.extent(0), row_map, row_entries, col_map, col_entries, handle);
            gc.execute();
            break;
        }

        default:
        {
            break;
        }
    }

    double coloring_time = timer.seconds();
    gch->add_to_overall_coloring_time(coloring_time);
    gch->set_coloring_time(coloring_time);
}



/**
 *  Validate Distance 2 Graph Coloring
 *
 *  @param validation_flags is an array of 4 booleans.
 *         validation_flags[0] : True IF the distance-2 coloring is invalid.
 *         validation_flags[1] : True IF the coloring is bad because vertices are left uncolored.
 *         validation_flags[2] : True IF the coloring is bad because at least one pair of vertices
 *                               at distance=2 from each other has the same color.
 *         validation_flags[3] : True IF a vertex has a color that is greater than number of vertices in the graph.
 *                               May not be an INVALID coloring, but can indicate poor quality in coloring.
 *
 *  @return boolean that is TRUE if the Distance-2 coloring is valid. False if otherwise.
 */
template<class KernelHandle, typename lno_row_view_t_, typename lno_nnz_view_t_, typename lno_col_view_t_, typename lno_colnnz_view_t_>
bool verifyDistance2Coloring(KernelHandle *handle,
                             typename KernelHandle::nnz_lno_t num_rows,
                             typename KernelHandle::nnz_lno_t num_cols,
                             lno_row_view_t_ row_map,
                             lno_nnz_view_t_ row_entries,
                             // If graph is symmetric, simply give same for col_map and row_map, and row_entries and col_entries.
                             lno_col_view_t_ col_map,
                             lno_colnnz_view_t_ col_entries,
                             bool validation_flags[])
{
    bool output = true;

    typename KernelHandle::GraphColoringHandleType *gch = handle->get_graph_coloring_handle();

    Impl::GraphColorD2<KernelHandle, lno_row_view_t_, lno_nnz_view_t_, lno_col_view_t_, lno_colnnz_view_t_> gc(
            num_rows, num_cols, row_entries.extent(0), row_map, row_entries, col_map, col_entries, handle);

    output = gc.verifyDistance2Coloring(row_map, row_entries, col_map, col_entries, gch->get_vertex_colors(), validation_flags);

    return output;
}



/**
 * Print out a histogram of graph colors for Distance-2 Graph Coloring
 */
template<class KernelHandle, typename lno_row_view_t_, typename lno_nnz_view_t_, typename lno_col_view_t_, typename lno_colnnz_view_t_>
void printDistance2ColorsHistogram(KernelHandle *handle,
                                   typename KernelHandle::nnz_lno_t num_rows,
                                   typename KernelHandle::nnz_lno_t num_cols,
                                   lno_row_view_t_ row_map,
                                   lno_nnz_view_t_ row_entries,
                                   // If graph is symmetric, simply give same for col_map and row_map, and row_entries and col_entries.
                                   lno_col_view_t_ col_map,
                                   lno_colnnz_view_t_ col_entries)
{
    Impl::GraphColorD2<KernelHandle, lno_row_view_t_, lno_nnz_view_t_, lno_col_view_t_, lno_colnnz_view_t_> gc(
            num_rows, num_cols, row_entries.extent(0), row_map, row_entries, col_map, col_entries, handle);

    gc.printDistance2ColorsHistogram();
}




}      // end namespace Experimental
}      // end namespace KokkosGraph

#endif      //_KOKKOS_GRAPH_COLOR_HPP
