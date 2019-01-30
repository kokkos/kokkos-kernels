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
int
run_graphcolor_d2(crsMat_type                                                             input_mat,
                  GraphColoringAlgorithmDistance2                                         coloring_algorithm,
                  size_t&                                                                 num_colors,
                  typename crsMat_type::StaticCrsGraphType::entries_type::non_const_type& vertex_colors)
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

    KernelHandle kh;
    kh.set_team_work_size(16);
    kh.set_dynamic_scheduling(true);

    kh.create_distance2_graph_coloring_handle(coloring_algorithm);

    const size_t num_rows_1 = input_mat.numRows();
    const size_t num_cols_1 = input_mat.numCols();

    // Compute the Distance-2 graph coloring.
    graph_compute_distance2_color<KernelHandle, lno_view_type, lno_nnz_view_type>
        (&kh, num_rows_1, num_cols_1, input_mat.graph.row_map, input_mat.graph.entries, input_mat.graph.row_map, input_mat.graph.entries);

    num_colors    = kh.get_distance2_graph_coloring_handle()->get_num_colors();
    vertex_colors = kh.get_distance2_graph_coloring_handle()->get_vertex_colors();

    // Verify the Distance-2 graph coloring is valid.
    bool d2_coloring_is_valid = false;
    bool d2_coloring_validation_flags[4] = { false };

    d2_coloring_is_valid = graph_verify_distance2_color(&kh, num_rows_1, num_cols_1,
                                                        input_mat.graph.row_map,input_mat.graph.entries,
                                                        input_mat.graph.row_map, input_mat.graph.entries,
                                                        d2_coloring_validation_flags);



    // Print out messages based on coloring validation check.
    if(d2_coloring_is_valid)
    {
        std::cout << std::endl << "Distance-2 Graph Coloring is VALID" << std::endl << std::endl;
    }
    else
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

    // return 0 if the coloring is valid, 1 otherwise.
    return d2_coloring_is_valid ? 0 : 1;
}

}      // namespace Test



template<typename scalar_type, typename lno_type, typename size_type, typename device>
void
test_coloring_d2(lno_type numRows, size_type nnz, lno_type bandwidth, lno_type row_size_variance)
{
    using namespace Test;

    using crsMat_type       = KokkosSparse::CrsMatrix<scalar_type, lno_type, device, void, size_type>;
    using graph_type        = typename crsMat_type::StaticCrsGraphType;
    using lno_view_type     = typename graph_type::row_map_type;
    using lno_nnz_view_type = typename graph_type::entries_type;
    using scalar_view_type  = typename crsMat_type::values_type::non_const_type;

    using color_view_type = typename graph_type::entries_type::non_const_type;

    lno_type    numCols = numRows;
    crsMat_type input_mat = KokkosKernels::Impl::kk_generate_sparse_matrix<crsMat_type>(numRows, numCols, nnz, row_size_variance, bandwidth);

    typename lno_view_type::non_const_type     sym_xadj;
    typename lno_nnz_view_type::non_const_type sym_adj;

    KokkosKernels::Impl::symmetrize_graph_symbolic_hashmap<lno_view_type,
                                                           lno_nnz_view_type,
                                                           typename lno_view_type::non_const_type,
                                                           typename lno_nnz_view_type::non_const_type,
                                                           device>(
      numRows, input_mat.graph.row_map, input_mat.graph.entries, sym_xadj, sym_adj);

    size_type        numentries = sym_adj.extent(0);
    scalar_view_type newValues("vals", numentries);

    graph_type static_graph(sym_adj, sym_xadj);
    input_mat = crsMat_type("CrsMatrix", numCols, newValues, static_graph);

    typedef KokkosKernelsHandle<size_type,
                                lno_type,
                                scalar_type,
                                typename device::execution_space,
                                typename device::memory_space,
                                typename device::memory_space>
      KernelHandle;

    KernelHandle cp;

    std::string algName = "SPGEMM_KK_MEMSPEED";
    cp.create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(algName));
    typename graph_type::row_map_type::non_const_type cRowptrs("cRowptrs", numRows + 1);

    // Call symbolic multiplication of graph with itself (no transposes, and A and B are the same)
    KokkosSparse::Experimental::spgemm_symbolic(&cp,
                                                numRows,
                                                numRows,
                                                numRows,
                                                input_mat.graph.row_map,
                                                input_mat.graph.entries,
                                                false,
                                                input_mat.graph.row_map,
                                                input_mat.graph.entries,
                                                false,
                                                cRowptrs);
    // Get num nz in C
    auto Cnnz = cp.get_spgemm_handle()->get_c_nnz();

    // Must create placeholder value views for A and C (values are meaningless)
    // Said above that the scalar view type is the same as the colinds view type
    scalar_view_type aFakeValues("A/B placeholder values (meaningless)", input_mat.graph.entries.size());

    // Allocate C entries array, and placeholder values
    typename graph_type::entries_type::non_const_type cColinds("C colinds", Cnnz);
    scalar_view_type                                  cFakeValues("C placeholder values (meaningless)", Cnnz);

    // Run the numeric kernel
    KokkosSparse::Experimental::spgemm_numeric(&cp,
                                               numRows,
                                               numRows,
                                               numRows,
                                               input_mat.graph.row_map,
                                               input_mat.graph.entries,
                                               aFakeValues,
                                               false,
                                               input_mat.graph.row_map,
                                               input_mat.graph.entries,
                                               aFakeValues,
                                               false,
                                               cRowptrs,
                                               cColinds,
                                               cFakeValues);
    // done with spgemm
    cp.destroy_spgemm_handle();

    GraphColoringAlgorithmDistance2 coloring_algorithms[] = {
      COLORING_D2_MATRIX_SQUARED, COLORING_D2_SERIAL, COLORING_D2, COLORING_D2_VB, COLORING_D2_VB_BIT, COLORING_D2_VB_BIT_EF};

    int num_algorithms = 6;

    for(int ii = 0; ii < num_algorithms; ++ii)
    {

        GraphColoringAlgorithmDistance2 coloring_algorithm = coloring_algorithms[ ii ];
        color_view_type                 vector_colors;
        size_t                          num_colors;

        Kokkos::Impl::Timer timer1;
        crsMat_type         output_mat;
        int res = run_graphcolor_d2<crsMat_type, device>(input_mat, coloring_algorithm, num_colors, vector_colors);

        // double coloring_time = timer1.seconds();
        EXPECT_TRUE((res == 0));

        #if 0  // no need to check distance-1 coloring for validity.
        const lno_type num_rows_1 = input_mat.numRows();
        const lno_type num_cols_1 = input_mat.numCols();

        lno_type num_conflict = KokkosKernels::Impl::
          kk_is_d1_coloring_valid<lno_view_type, lno_nnz_view_type, color_view_type, typename device::execution_space>(
            num_rows_1, num_cols_1, cRowptrs, cColinds, vector_colors);

        lno_type conf = 0;
        {
            // also check the correctness of the validation code :)
            typename lno_view_type::HostMirror     hrm      = Kokkos::create_mirror_view(cRowptrs);
            typename lno_nnz_view_type::HostMirror hentries = Kokkos::create_mirror_view(cColinds);
            typename color_view_type::HostMirror   hcolor   = Kokkos::create_mirror_view(vector_colors);
            Kokkos::deep_copy(hrm, cRowptrs);
            Kokkos::deep_copy(hentries, cColinds);
            Kokkos::deep_copy(hcolor, vector_colors);

            for(lno_type i = 0; i < num_rows_1; ++i)
            {
                const size_type b = hrm(i);
                const size_type e = hrm(i + 1);
                for(size_type j = b; j < e; ++j)
                {
                    lno_type d = hentries(j);
                    if(i != d)
                    {
                        if(hcolor(d) == hcolor(i))
                        {
                            conf++;
                        }
                    }
                }
            }
        }

        EXPECT_TRUE((num_conflict == conf));

        EXPECT_TRUE((num_conflict == 0));
        #endif

        /*
            ::testing::internal::CaptureStdout();
            std::cout << "num_colors:" << num_colors << " num_conflict:" << num_conflict << " conf:" << conf << std::endl;
            std::string capturedStdout = ::testing::internal::GetCapturedStdout().c_str();
            EXPECT_STREQ("something", capturedStdout.c_str());
        */
    }

    // device::execution_space::finalize();
}

#define EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE)                                           \
    TEST_F(TestCategory, graph##_##graph_color_d2##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) \
    {                                                                                           \
        test_coloring_d2<SCALAR, ORDINAL, OFFSET, DEVICE>(50000, 50000 * 30, 200, 10);          \
        test_coloring_d2<SCALAR, ORDINAL, OFFSET, DEVICE>(50000, 50000 * 30, 100, 10);          \
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
