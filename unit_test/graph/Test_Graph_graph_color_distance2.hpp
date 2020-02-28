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

#include <gtest/gtest.h>
#include <random>
#include <Kokkos_Core.hpp>

#include "KokkosGraph_Distance2Color.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_IOUtils.hpp"
#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosKernels_Handle.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"

using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

using namespace KokkosGraph;
using namespace KokkosGraph::Experimental;

namespace Test {

template<typename crsMat_type, typename device>
void
run_graphcolor_d2(crsMat_type input_mat, crsMat_type input_mat_T, const std::vector<GraphColoringAlgorithmDistance2>& algos)
{

    using graph_type        = typename crsMat_type::StaticCrsGraphType;
    using lno_view_type     = typename graph_type::row_map_type;
    using lno_nnz_view_type = typename graph_type::entries_type;
    using scalar_view_type  = typename crsMat_type::values_type::non_const_type;

    using size_type   = typename lno_view_type::value_type;
    using lno_type    = typename lno_nnz_view_type::value_type;
    using scalar_type = typename scalar_view_type::value_type;

    using KernelHandle = KokkosKernelsHandle<size_type,
                                             lno_type,
                                             scalar_type,
                                             typename device::execution_space,
                                             typename device::memory_space,
                                             typename device::memory_space>;

    for(auto algo : algos)
    {
      KernelHandle kh;
      kh.set_team_work_size(16);
      kh.set_dynamic_scheduling(true);

      kh.create_distance2_graph_coloring_handle(algo);

      EXPECT_EQ(input_mat.numRows(), input_mat_T.numCols());
      EXPECT_EQ(input_mat.numCols(), input_mat_T.numRows());

      const lno_type nv = input_mat.numRows();
      const lno_type nc = input_mat.numCols();

      // Compute the Distance-2 graph coloring.
      graph_compute_distance2_color<KernelHandle, lno_view_type, lno_nnz_view_type>
          (&kh, nv, nc, input_mat.graph.row_map, input_mat.graph.entries, input_mat_T.graph.row_map, input_mat_T.graph.entries);

      kh.get_distance2_graph_coloring_handle()->get_num_colors();
      kh.get_distance2_graph_coloring_handle()->get_vertex_colors();

      // Verify the Distance-2 graph coloring is valid.
      bool d2_coloring_is_valid = false;
      bool d2_coloring_validation_flags[4] = { false };

      d2_coloring_is_valid = KokkosGraph::Impl::graph_verify_distance2_color(
          &kh, nv, nc,
          input_mat.graph.row_map, input_mat.graph.entries,
          input_mat_T.graph.row_map, input_mat_T.graph.entries,
          d2_coloring_validation_flags);

      // Print out messages based on coloring validation check.
      if(!d2_coloring_is_valid)
      {
          std::cout << std::endl
                    << "Distance-2 Graph Coloring is NOT VALID" << std::endl
                    << "  - Vert(s) left uncolored : " << d2_coloring_validation_flags[1] << std::endl
                    << "  - Invalid D2 Coloring    : " << d2_coloring_validation_flags[2] << std::endl
                    << std::endl;
      }
      if(d2_coloring_validation_flags[3])
      {
          std::cout << "Distance-2 Graph Coloring may have poor quality." << std::endl
                    << "  - Vert(s) have high color value : " << d2_coloring_validation_flags[3] << std::endl
                    << std::endl;
      }

      kh.destroy_distance2_graph_coloring_handle();

      EXPECT_TRUE(d2_coloring_is_valid);
    }
}

}      // namespace Test


//Symmetric test: generates a random matrix, and then symmetrizes it (unions 
template<typename scalar_type, typename lno_type, typename size_type, typename device>
void
test_d2_symmetric(lno_type numRows, size_type nnz, lno_type bandwidth, lno_type row_size_variance)
{
    using namespace Test;

    using crsMat_type       = KokkosSparse::CrsMatrix<scalar_type, lno_type, device, void, size_type>;
    using graph_type        = typename crsMat_type::StaticCrsGraphType;
    using lno_view_type     = typename graph_type::row_map_type;
    using lno_nnz_view_type = typename graph_type::entries_type;
    using lno_view_nc       = typename graph_type::row_map_type::non_const_type;
    using lno_nnz_view_nc   = typename graph_type::entries_type::non_const_type;
    using scalar_view_type  = typename crsMat_type::values_type::non_const_type;

    lno_type    numCols = numRows;
    crsMat_type input_mat = KokkosKernels::Impl::kk_generate_sparse_matrix<crsMat_type>(numRows, numCols, nnz, row_size_variance, bandwidth);

    typename lno_view_type::non_const_type     sym_xadj;
    typename lno_nnz_view_type::non_const_type sym_adj;

    KokkosKernels::Impl::symmetrize_graph_symbolic_hashmap
      <lno_view_type, lno_nnz_view_type, lno_view_nc, lno_nnz_view_nc, device>
      (numRows, input_mat.graph.row_map, input_mat.graph.entries, sym_xadj, sym_adj);

    size_type        numentries = sym_adj.extent(0);
    scalar_view_type newValues("vals", numentries);

    graph_type static_graph(sym_adj, sym_xadj);
    input_mat = crsMat_type("CrsMatrix", numCols, newValues, static_graph);
    run_graphcolor_d2<crsMat_type, device>(input_mat, input_mat,
        {COLORING_D2_MATRIX_SQUARED, COLORING_D2_SERIAL, COLORING_D2, COLORING_D2_VB, COLORING_D2_VB_BIT, COLORING_D2_VB_BIT_EF, COLORING_D2_VB_DYNAMIC});

}

//Asymmetric test: generates a random matrix (with nv != nc), and uses the explicit transpose as input_mat_T
template<typename scalar_type, typename lno_type, typename size_type, typename device>
void
test_d2_asymmetric(lno_type numRows, lno_type numCols, size_type nnz, lno_type bandwidth, lno_type row_size_variance)
{
    using namespace Test;

    using crsMat_type       = KokkosSparse::CrsMatrix<scalar_type, lno_type, device, void, size_type>;
    using graph_type        = typename crsMat_type::StaticCrsGraphType;
    using lno_view_type     = typename graph_type::row_map_type;
    using lno_nnz_view_type = typename graph_type::entries_type;
    using lno_view_nc       = typename graph_type::row_map_type::non_const_type;
    using lno_nnz_view_nc   = typename graph_type::entries_type::non_const_type;

    crsMat_type input_mat = KokkosKernels::Impl::kk_generate_sparse_matrix<crsMat_type>(numRows, numCols, nnz, row_size_variance, bandwidth);

    lno_view_nc transRowmap("input^T rowmap", numCols + 1);
    lno_nnz_view_nc transColinds("input^T colinds", input_mat.nnz());

    KokkosKernels::Impl::kk_transpose_graph<lno_view_type, lno_nnz_view_type, lno_view_nc, lno_nnz_view_nc, lno_view_nc, typename device::execution_space>
      (numRows, numCols, input_mat.graph.row_map, input_mat.graph.entries, transRowmap, transColinds);
    crsMat_type input_mat_T("input^T", numCols, numRows, input_mat.nnz(), input_mat.values, transRowmap, transColinds);
    run_graphcolor_d2<crsMat_type, device>(input_mat, input_mat_T,
        {COLORING_D2_MATRIX_SQUARED, COLORING_D2_SERIAL, COLORING_D2, COLORING_D2_VB, COLORING_D2_VB_BIT, COLORING_D2_VB_BIT_EF, COLORING_D2_VB_DYNAMIC});
}

//Filtering test: start with a symmetric matrix, but intentionally add some entries that are have out-of-bounds columns.
//These entries should be ignored by D2. This test simulates the MueLu use case.
template<typename scalar_type, typename lno_type, typename size_type, typename device>
void
test_d2_filtering(lno_type numRows, lno_type numCols, size_type nnz, lno_type bandwidth, lno_type row_size_variance)
{
    using namespace Test;

    using crsMat_type       = KokkosSparse::CrsMatrix<scalar_type, lno_type, device, void, size_type>;
    using graph_type        = typename crsMat_type::StaticCrsGraphType;
    using lno_view_type     = typename graph_type::row_map_type;
    using lno_nnz_view_type = typename graph_type::entries_type;
    using lno_view_nc       = typename graph_type::row_map_type::non_const_type;
    using lno_nnz_view_nc   = typename graph_type::entries_type::non_const_type;
    using scalar_view_type  = typename crsMat_type::values_type::non_const_type;

    crsMat_type input_mat = KokkosKernels::Impl::kk_generate_sparse_matrix<crsMat_type>(numRows, numRows, nnz, row_size_variance, bandwidth);

    lno_view_nc sym_xadj;
    lno_nnz_view_nc sym_adj;

    KokkosKernels::Impl::symmetrize_graph_symbolic_hashmap
      <lno_view_type, lno_nnz_view_type, lno_view_nc, lno_nnz_view_nc, device>
      (numRows, input_mat.graph.row_map, input_mat.graph.entries, sym_xadj, sym_adj);

    auto xadj_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sym_xadj);
    auto adj_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sym_adj);
    
    Kokkos::View<size_type*, Kokkos::HostSpace> newRowmap("Unfiltered rowmap", numRows + 1);

    //Precompute the new rowmap
    size_type accum = 0;
    for(lno_type i = 0; i <= numRows; i++)
    {
      newRowmap(i) = accum;
      if(i == numRows)
        break;
      size_type oldRowSize = xadj_host(i + 1) - xadj_host(i);
      accum += oldRowSize * 1.5;
    }
    size_type newNNZ = newRowmap(numRows);
    Kokkos::View<lno_type*, Kokkos::HostSpace> newEntries("Unfiltered entries", newNNZ);
    //Randomly shuffle each row so that the entries to filter are spread throughout
    std::random_device rd;
    std::mt19937 g(rd());
    for(lno_type i = 0; i < numRows; i++)
    {
      size_type oldRowBegin = xadj_host(i);
      size_type oldRowEnd = xadj_host(i + 1);
      size_type newRowSize = newRowmap(i + 1) - newRowmap(i);
      std::vector<lno_type> ents;
      ents.reserve(newRowSize);
      for(size_type j = oldRowBegin; j < oldRowEnd; j++)
        ents.push_back(adj_host(j));
      for(size_type j = oldRowEnd - oldRowBegin; j < newRowSize; j++)
        ents.push_back(numRows + rand() % (numCols - numRows));
      std::shuffle(ents.begin(), ents.end(), g);
      for(size_t j = 0; j < ents.size(); j++)
        newEntries(newRowmap(i) + j) = ents[j];
    }
    lno_view_nc finalRowmap("Rowmap", numRows + 1);
    lno_nnz_view_nc finalEntries("Entries", newNNZ);
    Kokkos::deep_copy(finalRowmap, newRowmap);
    Kokkos::deep_copy(finalEntries, newEntries);
    scalar_view_type newValues("Vals", newNNZ);

    input_mat = crsMat_type("input", numRows, numRows, newNNZ, newValues, finalRowmap, finalEntries);

    //Note that MATRIX_SQUARED cannot be used here, because SPGEMM doesn't accept out-of-bounds indices
    run_graphcolor_d2<crsMat_type, device>(input_mat, input_mat,
        {COLORING_D2_SERIAL, COLORING_D2, COLORING_D2_VB, COLORING_D2_VB_BIT, COLORING_D2_VB_BIT_EF, COLORING_D2_VB_DYNAMIC});
}

#define EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE)                                           \
    TEST_F(TestCategory, graph##_##graph_color_d2_symmetric##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) \
    {                                                                                           \
      test_d2_symmetric<SCALAR, ORDINAL, OFFSET, DEVICE>(5000, 5000 * 30, 200, 10);          \
      test_d2_symmetric<SCALAR, ORDINAL, OFFSET, DEVICE>(5000, 5000 * 5, 100, 10);          \
    } \
    TEST_F(TestCategory, graph##_##graph_color_d2_asymmetric##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) \
    {                                                                                           \
      test_d2_asymmetric<SCALAR, ORDINAL, OFFSET, DEVICE>(5000, 6000, 5000 * 30, 200, 10);          \
      test_d2_asymmetric<SCALAR, ORDINAL, OFFSET, DEVICE>(5000, 6000, 5000 * 5, 100, 10);          \
    } \
    TEST_F(TestCategory, graph##_##graph_color_d2_filtering##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) \
    {                                                                                           \
      test_d2_filtering<SCALAR, ORDINAL, OFFSET, DEVICE>(5000, 7000, 5000 * 10, 200, 10); \
    }

#if defined(KOKKOSKERNELS_INST_DOUBLE)
#if(defined(KOKKOSKERNELS_INST_ORDINAL_INT) && defined(KOKKOSKERNELS_INST_OFFSET_INT)) \
  || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_TEST(double, int, int, TestExecSpace)
#endif

#if(defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && defined(KOKKOSKERNELS_INST_OFFSET_INT)) \
  || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_TEST(double, int64_t, int, TestExecSpace)
#endif

#if(defined(KOKKOSKERNELS_INST_ORDINAL_INT) && defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) \
  || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_TEST(double, int, size_t, TestExecSpace)
#endif

#if(defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) \
  || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_TEST(double, int64_t, size_t, TestExecSpace)
#endif
#endif
