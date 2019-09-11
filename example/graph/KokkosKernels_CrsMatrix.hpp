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
#ifndef __KOKKOSKERNELS_CRSMATRIX_HPP__
#define __KOKKOSKERNELS_CRSMATRIX_HPP__


#include "KokkosKernels_Utils.hpp"
#include "KokkosKernels_StaticCrsGraph.hpp"



namespace KokkosKernelsGraphExample {



template<typename ScalarType, typename OrdinalType, typename Device, typename MemoryTraits, typename SizeType>
class CrsMatrix
{
  public:
    typedef typename Kokkos::ViewTraits<ScalarType*, Device, void, void>::host_mirror_space host_mirror_space;

    typedef typename Device::execution_space              execution_space;
    typedef typename Device::memory_space                 memory_space;
    typedef Kokkos::Device<execution_space, memory_space> device_type;
    typedef ScalarType                                    value_type;
    typedef OrdinalType                                   ordinal_type;
    typedef MemoryTraits                                  memory_traits;
    typedef SizeType                                      size_type;

    typedef StaticCrsGraph<OrdinalType, Device, MemoryTraits, SizeType>                   StaticCrsGraphType;
    typedef typename StaticCrsGraphType::entries_type                                     index_type;
    typedef typename index_type::non_const_value_type                                     const_ordinal_type;
    typedef typename index_type::non_const_value_type                                     non_const_ordinal_type;
    typedef typename StaticCrsGraphType::row_map_type                                     row_map_type;
    typedef Kokkos::View<value_type*, Kokkos::LayoutRight, device_type, MemoryTraits>     values_type;
    typedef CrsMatrix<ScalarType, OrdinalType, host_mirror_space, MemoryTraits, SizeType> HostMirror;

    StaticCrsGraphType graph;
    values_type        values;
    CrsMatrix() : numCols_(0) {}
    CrsMatrix(const std::string& label, const OrdinalType& ncols, const values_type& vals, const StaticCrsGraphType& graph_)
        : graph(graph_), values(vals), numCols_(ncols)
    {
    }

    //! The number of rows in the sparse matrix.
    KOKKOS_INLINE_FUNCTION ordinal_type numRows() const { return graph.numRows(); }

    //! The number of columns in the sparse matrix.
    KOKKOS_INLINE_FUNCTION ordinal_type numCols() const { return numCols_; }

    //! The number of stored entries in the sparse matrix.
    KOKKOS_INLINE_FUNCTION size_type nnz() const { return graph.entries.extent(0); }
    ordinal_type                     numCols_;
};



template<typename myExecSpace, typename crs_matrix_type>
crs_matrix_type
get_crsmat(typename crs_matrix_type::row_map_type::non_const_type::value_type* xadj,
           typename crs_matrix_type::index_type::non_const_type::value_type*   adj,
           typename crs_matrix_type::values_type::non_const_type::value_type*  ew,
           typename crs_matrix_type::row_map_type::non_const_type::value_type  ne,
           typename crs_matrix_type::index_type::non_const_type::value_type    nv,
           int                                                                 is_one_based)
{

    typedef typename crs_matrix_type::StaticCrsGraphType           graph_type;
    typedef typename crs_matrix_type::row_map_type::non_const_type row_map_view_type;
    typedef typename crs_matrix_type::index_type::non_const_type   cols_view_type;
    typedef typename crs_matrix_type::values_type::non_const_type  values_view_type;

    typedef typename row_map_view_type::value_type size_type;
    typedef typename cols_view_type::value_type    lno_type;
    typedef typename values_view_type::value_type  scalar_type;



    row_map_view_type rowmap_view("rowmap_view", nv + 1);
    cols_view_type    columns_view("colsmap_view", ne);
    values_view_type  values_view("values_view", ne);

    KokkosKernels::Impl::copy_vector<scalar_type*, values_view_type, myExecSpace>(ne, ew, values_view);
    KokkosKernels::Impl::copy_vector<lno_type*, cols_view_type, myExecSpace>(ne, adj, columns_view);
    KokkosKernels::Impl::copy_vector<size_type*, row_map_view_type, myExecSpace>(nv + 1, xadj, rowmap_view);

    size_type ncols = 0;
    KokkosKernels::Impl::view_reduce_max<cols_view_type, myExecSpace>(ne, columns_view, ncols);
    ncols += 1;

    if(is_one_based)
    {
        // if algorithm is mkl_csrmultcsr convert to 1 base so that we dont dublicate the memory at the experiments/
        KokkosKernels::Impl::kk_a_times_x_plus_b<row_map_view_type, row_map_view_type, int, int, myExecSpace>(
          nv + 1, rowmap_view, rowmap_view, 1, 1);
        KokkosKernels::Impl::kk_a_times_x_plus_b<cols_view_type, cols_view_type, int, int, myExecSpace>(
          ne, columns_view, columns_view, 1, 1);
    }

    graph_type      static_graph(columns_view, rowmap_view);
    crs_matrix_type crsmat("CrsMatrix", ncols, values_view, static_graph);
    return crsmat;
}


}      // namespace KokkosKernelsGraphExample


#endif //  __KOKKOSKERNELS_CRSMATRIX_HPP__
