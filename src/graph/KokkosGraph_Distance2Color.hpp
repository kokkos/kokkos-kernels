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
// Questions? Contact William McLendon (wcmclen@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#ifndef _KOKKOS_GRAPH_COLORDISTANCE2_HPP
#define _KOKKOS_GRAPH_COLORDISTANCE2_HPP

#include "KokkosGraph_Distance1ColorHandle.hpp"
#include "KokkosGraph_Distance2ColorHandle.hpp"
#include "KokkosGraph_Distance1Color_impl.hpp" 
#include "KokkosGraph_Distance2Color_impl.hpp"
#include "KokkosGraph_Distance2Color_MatrixSquared_impl.hpp"

#include "KokkosKernels_Utils.hpp"


namespace KokkosGraph {

namespace Experimental {



/**
 * Compute the distance-2 coloring of the matrix/graph.
 *
 * If the graph is symmetric, give the same value for col_map and row_map,
 * and for row_entries and col_entries.
 *
 * @param[in]  handle         The Kernel Handle
 * @param[in]  num_rows       Number of rows in the matrix (number of vertices)
 * @param[in]  num_cols       Number of columns in the matrix
 * @param[in]  row_map        Row map
 * @param[in]  row_entries    Row entries
 * @param[in]  col_map        Column map
 * @param[in]  col_entries    Column entries
 *
 * @return Nothing
 */
template<class KernelHandle, typename lno_row_view_t_, typename lno_nnz_view_t_, typename lno_col_view_t_, typename lno_colnnz_view_t_>
void graph_compute_distance2_color(KernelHandle *handle,
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
    typename KernelHandle::GraphColorDistance2HandleType *gch_d2 = handle->get_distance2_graph_coloring_handle();
    // Get the algorithm we're running from the graph coloring handle.
    GraphColoringAlgorithmDistance2 algorithm = gch_d2->get_coloring_algo_type();
    // Create a view to save the colors to.
    using color_view_type = typename KernelHandle::GraphColorDistance2HandleType::color_view_type;
    color_view_type colors_out("Graph Colors", num_rows);

    if(algorithm == COLORING_D2_MATRIX_SQUARED ||
        algorithm == COLORING_D2_SPGEMM)
    {
      Impl::GraphColorDistance2MatrixSquared<KernelHandle, lno_row_view_t_, lno_nnz_view_t_, lno_col_view_t_, lno_colnnz_view_t_>
        gc(num_rows, num_cols, row_entries.extent(0), row_map, row_entries, col_map, col_entries, handle);
      gc.compute_distance2_color();
    }
    else
    {
      Impl::GraphColorDistance2<typename KernelHandle::GraphColorDistance2HandleType, lno_row_view_t_, lno_nnz_view_t_, lno_col_view_t_, lno_colnnz_view_t_>
        gc(num_rows, num_cols, row_entries.extent(0), row_map, row_entries, col_map, col_entries, gch_d2);
      if(algorithm == COLORING_D2_SERIAL)
      {
        gc.compute_distance2_color_serial();
      }
      else if(algorithm == COLORING_D2_NB_BIT)
      {
        gc.compute_d2_coloring_dynamic();
      }
      else
      {
        gc.compute_distance2_color();
      }
    }
    gch_d2->add_to_overall_coloring_time(timer.seconds());
    gch_d2->set_coloring_time(timer.seconds());
}


}      // end namespace Experimental
}      // end namespace KokkosGraph

#endif //_KOKKOS_GRAPH_COLORDISTANCE2_HPP
