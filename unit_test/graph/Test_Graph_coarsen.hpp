/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Brian Kelley (bmkelle@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

//#include <gtest/gtest.h>
#include <random>
#include <set>
#include <Kokkos_Core.hpp>

#include "KokkosGraph_CoarsenConstruct.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_IOUtils.hpp"
#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosKernels_Handle.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"

using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

using namespace KokkosGraph;
using namespace KokkosGraph::Experimental;

//namespace Test {

template<class coarsener_t>
bool verify_coarsening(typename coarsener_t::coarse_level_triple fine_l, typename coarsener_t::coarse_level_triple coarse_l){
    using crsMat = typename coarsener_t::matrix_t;
    using graph_type  = typename crsMat::StaticCrsGraphType;
    using c_rowmap_t  = typename graph_type::row_map_type;
    using c_entries_t = typename graph_type::entries_type;
    using rowmap_t    = typename c_rowmap_t::non_const_type;
    using entries_t   = typename c_entries_t::non_const_type;
    using svt = typename coarsener_t::wgt_view_t;
    using ordinal_t = typename entries_t::value_type;
    using edge_t = typename rowmap_t::value_type;

    
    bool correct = true;
    crsMat A = fine_l.coarse_mtx;
    crsMat coarse_A = coarse_l.coarse_mtx;
    typename c_rowmap_t::HostMirror f_rowmap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.row_map);
    typename c_rowmap_t::HostMirror c_rowmap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coarse_A.graph.row_map);
    typename c_entries_t::HostMirror f_entries = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.entries);
    typename c_entries_t::HostMirror vcmap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coarse_l.interp_mtx.graph.entries);
    typename svt::HostMirror few = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.values);
    typename svt::HostMirror cew = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coarse_A.values);
    typename entries_t::HostMirror fvw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fine_l.coarse_vtx_wgts);
    typename entries_t::HostMirror cvw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coarse_l.coarse_vtx_wgts);
    ordinal_t f_size = 0;
    ordinal_t c_size = 0;
    for(ordinal_t i = 0; i < fvw.extent(0); i++){
        f_size += fvw(i);
    }
    for(ordinal_t i = 0; i < cvw.extent(0); i++){
        c_size += cvw(i);
    }
    //number of columns in interpolation matrix should give number of rows in coarse matrix
    if(coarse_l.interp_mtx.numCols() != coarse_l.coarse_mtx.numRows()){
        correct = false;
    }
    //sum of vertex weights in each graph should be equal
    if(f_size != c_size){
        correct = false;
    }
    typename svt::value_type f_edges = 0, c_edges = 0;
    for(ordinal_t i = 0; i < A.numRows(); i++){
        for(edge_t j = f_rowmap(i); j < f_rowmap(i + 1); j++){
            ordinal_t v = f_entries(j);
            if(vcmap(i) != vcmap(v)){
                f_edges += few(j);
            }
        }
    }
    for(ordinal_t i = 0; i < coarse_A.numRows(); i++){
        for(edge_t j = c_rowmap(i); j < c_rowmap(i + 1); j++){
            c_edges += cew(j);
        }
    }
    //sum of inter-aggregate edges in fine graph should be sum of all edges in coarse graph
    if(f_edges != c_edges){
        correct = false;
    }
    return correct;
}

template<class crsMat>
bool verify_is_graph(crsMat A){
    using graph_type  = typename crsMat::StaticCrsGraphType;
    using c_rowmap_t  = typename graph_type::row_map_type;
    using c_entries_t = typename graph_type::entries_type;
    using rowmap_t    = typename c_rowmap_t::non_const_type;
    using entries_t   = typename c_entries_t::non_const_type;
    using ordinal_t = typename entries_t::value_type;
    using edge_t = typename rowmap_t::value_type;
    typename c_rowmap_t::HostMirror rowmap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.row_map);
    typename c_entries_t::HostMirror entries = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.entries);
    
    bool correct = true;
    for(ordinal_t i = 0; i < A.numRows(); i++){
        std::set<ordinal_t> adjset;
        for(edge_t j = rowmap(i); j < rowmap(i + 1); j++){
            ordinal_t v = entries(j);
            //A should not contain out-of-bounds columns
            if(v >= A.numRows()){
                correct = false;
            }
            //Each row should not contain duplicate columns
            if(adjset.find(v) != adjset.end()){
                correct = false;
            }
            adjset.insert(v);
        }
    }
    return correct;
}

template<class crsMat>
bool verify_aggregator(crsMat A, crsMat agg){
    using graph_type  = typename crsMat::StaticCrsGraphType;
    using c_rowmap_t  = typename graph_type::row_map_type;
    using c_entries_t = typename graph_type::entries_type;
    using rowmap_t    = typename c_rowmap_t::non_const_type;
    using entries_t   = typename c_entries_t::non_const_type;
    using ordinal_t = typename entries_t::value_type;
    using edge_t = typename rowmap_t::value_type;

    bool correct = true;
    //aggregator should have as many rows as A
    if(A.numRows() != agg.numRows()){
        correct = false;
    }
    //aggregator should have as many entries as A has rows
    if(A.numRows() != agg.nnz()){
        correct = false;
    }
    //aggregator should have fewer columns than A has rows
    if(A.numRows() <= agg.numCols()){
        correct = false;
    }
    typename c_entries_t::HostMirror entries = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), agg.graph.entries);

    std::vector<int> aggregateSizes(agg.numCols(), 0);
    for(ordinal_t i = 0; i < agg.nnz(); i++){
        ordinal_t v = entries(i);
        //aggregator should not have out-of-bounds columns
        if(v >= agg.numCols()){
            correct = false;
        }
        aggregateSizes[v]++;
    }
    for(ordinal_t i = 0; i < agg.numCols(); i++){
        //each aggregate label should contain at least one fine vertex
        if(aggregateSizes[i] == 0){
            correct = false;
        }
    }
    return correct;
}

template <typename scalar, typename lno_t, typename size_type,
          typename device>
void test_coarsen(lno_t numVerts, size_type nnz, lno_t bandwidth,
               lno_t row_size_variance) {
  using execution_space = typename device::execution_space;
  using crsMat =
      KokkosSparse::CrsMatrix<scalar, lno_t, device, void, size_type>;
  using graph_type  = typename crsMat::StaticCrsGraphType;
  using c_rowmap_t  = typename graph_type::row_map_type;
  using c_entries_t = typename graph_type::entries_type;
  using rowmap_t    = typename c_rowmap_t::non_const_type;
  using entries_t   = typename c_entries_t::non_const_type;
  using svt = Kokkos::View<scalar*>;
  // Generate graph, and add some out-of-bounds columns
  crsMat A = KokkosKernels::Impl::kk_generate_sparse_matrix<crsMat>(
      numVerts, numVerts, nnz, row_size_variance, bandwidth);
  auto G = A.graph;
  // Symmetrize the graph
  rowmap_t symRowmap;
  entries_t symEntries;
  KokkosKernels::Impl::symmetrize_graph_symbolic_hashmap<
      c_rowmap_t, c_entries_t, rowmap_t, entries_t, execution_space>(
      numVerts, G.row_map, G.entries, symRowmap, symEntries);
  graph_type GS(symEntries, symRowmap);
  svt symValues("sym values", symEntries.extent(0));
  entries_t vWgts("vertex weights", numVerts);
  Kokkos::deep_copy(symValues, static_cast<scalar>(2.5));
  Kokkos::deep_copy(vWgts, static_cast<typename entries_t::value_type>(1));
  crsMat AS("A symmetric", numVerts, symValues, GS);
  using coarsener_t = coarse_builder<typename entries_t::value_type, typename rowmap_t::value_type, scalar, device>;
  coarsener_t coarsener;
  using clt = typename coarsener_t::coarse_level_triple;
  clt fine_A;
  fine_A.coarse_mtx = AS;
  fine_A.coarse_vtx_wgts = vWgts;
  fine_A.level = 0;
  fine_A.uniform_weights = true;
  std::vector<typename coarsener_t::Heuristic> heuristics = { coarsener_t::HECv1, /*coarsener_t::HECv2,*/ 
      coarsener_t::HECv3, coarsener_t::Match, 
      coarsener_t::MtMetis, coarsener_t::MIS2, 
      coarsener_t::GOSHv1, coarsener_t::GOSHv2 };
  for(auto h : heuristics){
    coarsener.set_heuristic(h);
    printf("testing heuristic: %i\n", static_cast<int>(h));
    crsMat aggregator = coarsener.generate_coarse_mapping(AS, true);
    bool correct_aggregator = verify_aggregator(A, aggregator);
    clt coarse_A = coarsener.build_coarse_graph(fine_A, aggregator);
    bool correct_graph = verify_is_graph<crsMat>(coarse_A.coarse_mtx);
    bool correct_coarsening = verify_coarsening<coarsener_t>(fine_A, coarse_A);
    if(!correct_aggregator){
        printf("Aggregator is invalid\n");
    }
    if(!correct_graph){
        printf("Coarse graph is invalid!\n");
    }
    if(!correct_coarsening){
        printf("Coarsening is incorrect\n");
    }
  }
  //bool success = Test::verifyD2MIS<lno_t, size_type, decltype(rowmapHost),
  //                                 decltype(entriesHost), decltype(misHost)>(
  //    numVerts, rowmapHost, entriesHost, misHost);
  //EXPECT_TRUE(success) << "Dist-2 MIS (algo " << (int)algo
  //                     << ") produced invalid set.";
}

//#define EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE)                                 \
//  TEST_F(TestCategory,                                                                \
//         graph##_##graph_mis2##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) {         \
//    test_mis2<SCALAR, ORDINAL, OFFSET, DEVICE>(5000, 5000 * 20, 1000, 10);            \
//    test_mis2<SCALAR, ORDINAL, OFFSET, DEVICE>(50, 50 * 10, 40, 10);                  \
//    test_mis2<SCALAR, ORDINAL, OFFSET, DEVICE>(5, 5 * 3, 5, 0);                       \
//  }                                                                                   \
//  TEST_F(                                                                             \
//      TestCategory,                                                                   \
//      graph##_##graph_mis2_coarsening##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) { \
//    test_mis2_coarsening<SCALAR, ORDINAL, OFFSET, DEVICE>(5000, 5000 * 200,           \
//                                                          2000, 10);                  \
//    test_mis2_coarsening<SCALAR, ORDINAL, OFFSET, DEVICE>(5000, 5000 * 20,            \
//                                                          1000, 10);                  \
//    test_mis2_coarsening<SCALAR, ORDINAL, OFFSET, DEVICE>(50, 50 * 10, 40,            \
//                                                          10);                        \
//    test_mis2_coarsening<SCALAR, ORDINAL, OFFSET, DEVICE>(5, 5 * 3, 5, 0);            \
//    test_mis2_coarsening_zero_rows<SCALAR, ORDINAL, OFFSET, DEVICE>();                \
//  }
//
//// FIXME_SYCL
//#ifndef KOKKOS_ENABLE_SYCL
//#if defined(KOKKOSKERNELS_INST_DOUBLE)
//#if (defined(KOKKOSKERNELS_INST_ORDINAL_INT) && \
//     defined(KOKKOSKERNELS_INST_OFFSET_INT)) || \
//    (!defined(KOKKOSKERNELS_ETI_ONLY) &&        \
//     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
//EXECUTE_TEST(double, int, int, TestExecSpace)
//#endif
//#endif
//
//#if (defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && \
//     defined(KOKKOSKERNELS_INST_OFFSET_INT)) ||     \
//    (!defined(KOKKOSKERNELS_ETI_ONLY) &&            \
//     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
//EXECUTE_TEST(double, int64_t, int, TestExecSpace)
//#endif
//
//#if (defined(KOKKOSKERNELS_INST_ORDINAL_INT) &&    \
//     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) || \
//    (!defined(KOKKOSKERNELS_ETI_ONLY) &&           \
//     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
//EXECUTE_TEST(double, int, size_t, TestExecSpace)
//#endif
//
//#if (defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && \
//     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) ||  \
//    (!defined(KOKKOSKERNELS_ETI_ONLY) &&            \
//     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
//EXECUTE_TEST(double, int64_t, size_t, TestExecSpace)
//#endif
//#endif

//#undef EXECUTE_TEST
