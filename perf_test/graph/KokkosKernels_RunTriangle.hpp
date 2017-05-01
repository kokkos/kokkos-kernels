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


#include "KokkosKernels_Triangle.hpp"
#include "KokkosKernels_TestParameters.hpp"

#define TRANPOSEFIRST false
#define TRANPOSESECOND false

namespace KokkosKernels{

namespace Experiment{
template <typename crsGraph_t, typename device>
bool is_same_graph(crsGraph_t output_mat1, crsGraph_t output_mat2){

  //typedef typename crsGraph_t::StaticCrsGraphType crsGraph_t;
  typedef typename crsGraph_t::row_map_type::non_const_type lno_view_t;
  typedef typename crsGraph_t::entries_type::non_const_type   lno_nnz_view_t;
  //typedef typename crsGraph_t::values_type::non_const_type scalar_view_t;

  size_t nrows1 = output_mat1.row_map.dimension_0();
  size_t nentries1 = output_mat1.entries.dimension_0() ;

  size_t nrows2 = output_mat2.row_map.dimension_0();
  size_t nentries2 = output_mat2.entries.dimension_0() ;
  //size_t nvals2 = output_mat2.values.dimension_0();


  lno_nnz_view_t h_ent1 (Kokkos::ViewAllocateWithoutInitializing("e1"), nentries1);
  lno_nnz_view_t h_vals1 (Kokkos::ViewAllocateWithoutInitializing("v1"), nentries1);


  KokkosKernels::Experimental::Util::kk_sort_graph<typename crsGraph_t::row_map_type,
    typename crsGraph_t::entries_type,
    lno_nnz_view_t,
    lno_nnz_view_t,
    lno_nnz_view_t,
    typename device::execution_space
    >(
    output_mat1.row_map, output_mat1.entries,h_vals1,
    h_ent1, h_vals1
  );

  lno_nnz_view_t h_ent2 (Kokkos::ViewAllocateWithoutInitializing("e1"), nentries2);
  lno_nnz_view_t h_vals2 (Kokkos::ViewAllocateWithoutInitializing("v1"), nentries2);

  if (nrows1 != nrows2) return false;
  if (nentries1 != nentries2) return false;

  KokkosKernels::Experimental::Util::kk_sort_graph
      <typename crsGraph_t::row_map_type,
      typename crsGraph_t::entries_type,
      lno_nnz_view_t,
      lno_nnz_view_t,
      lno_nnz_view_t,
      typename device::execution_space
      >(
      output_mat2.row_map, output_mat2.entries, h_vals2,
      h_ent2, h_vals2
    );

  bool is_identical = true;
  is_identical = KokkosKernels::Experimental::Util::kk_is_identical_view
      <typename crsGraph_t::row_map_type, typename crsGraph_t::row_map_type, typename lno_view_t::value_type,
      typename device::execution_space>(output_mat1.row_map, output_mat2.row_map, 0);
  if (!is_identical) return false;

  is_identical = KokkosKernels::Experimental::Util::kk_is_identical_view
      <lno_nnz_view_t, lno_nnz_view_t, typename lno_nnz_view_t::value_type,
      typename device::execution_space>(h_ent1, h_ent2, 0 );
  if (!is_identical) return false;

  if (!is_identical) {
    std::cout << "Incorret values" << std::endl;
  }
  return true;
}


template <typename ExecSpace, typename crsGraph_t, typename crsGraph_t2 , typename crsGraph_t3 , typename TempMemSpace , typename PersistentMemSpace >
crsGraph_t3 run_experiment(
    crsGraph_t crsGraph, crsGraph_t2 crsGraph2, Parameters params){
    //int algorithm, int repeat, int chunk_size ,int multi_color_scale, int shmemsize, int team_size, int use_dynamic_scheduling, int verbose){
  int algorithm = params.algorithm;
  int repeat = params.repeat;
  int chunk_size = params.chunk_size;

  int shmemsize = params.shmemsize;
  int team_size = params.team_size;
  int use_dynamic_scheduling = params.use_dynamic_scheduling;
  int verbose = params.verbose;

  char *coloring_input_file = params.coloring_input_file;
  char *coloring_output_file = params.coloring_output_file;
  //char spgemm_step = params.spgemm_step;
  int vector_size = params.vector_size;
  int check_output = params.check_output;

  //spgemm_step++;

  typedef typename crsGraph_t3::row_map_type::non_const_type lno_view_t;
  typedef typename crsGraph_t3::entries_type::non_const_type lno_nnz_view_t;

  lno_view_t row_mapC;
  lno_nnz_view_t entriesC;
  lno_nnz_view_t valuesC;

  typedef KokkosKernels::Experimental::KokkosKernelsHandle
      <lno_view_t,lno_nnz_view_t, lno_nnz_view_t,
      ExecSpace, TempMemSpace,PersistentMemSpace > KernelHandle;

  typedef typename lno_nnz_view_t::value_type lno_t;
  typedef typename lno_view_t::value_type size_type;

  KernelHandle kh;
  kh.set_team_work_size(chunk_size);
  kh.set_shmem_size(shmemsize);
  kh.set_suggested_team_size(team_size);
  kh.set_suggested_vector_size(vector_size);

  if (use_dynamic_scheduling){
    kh.set_dynamic_scheduling(true);
  }
  if (verbose){
    kh.set_verbose(true);
  }

  const lno_t m = crsGraph.numRows();
  const lno_t n = crsGraph2.numRows();
  const lno_t k = crsGraph2.numCols();

  std::cout << "m:" << m << " n:" << n << " k:" << k << std::endl;
  if (n < crsGraph.numCols()){
    std::cout << "left.numCols():" << crsGraph.numCols() << " right.numRows():" << crsGraph2.numRows() << std::endl;
    exit(1);
  }

  lno_view_t row_mapC_ref;
  lno_nnz_view_t entriesC_ref;
  lno_nnz_view_t valuesC_ref;
  crsGraph_t3 Ccrsgraph_ref;
  if (check_output)
  {
    std::cout << "Running a reference algorithm" << std::endl;
    row_mapC_ref = lno_view_t ("non_const_lnow_row", m + 1);
    entriesC_ref = lno_nnz_view_t ("");

    KernelHandle kh;
    kh.set_team_work_size(chunk_size);
    kh.set_shmem_size(shmemsize);
    kh.set_suggested_team_size(team_size);
    kh.create_spgemm_handle(KokkosKernels::Experimental::Graph::SPGEMM_KK_TRIANGLE_DEFAULT);

    if (use_dynamic_scheduling){
      kh.set_dynamic_scheduling(true);
    }

    KokkosKernels::Experimental::Graph::triangle_count (
        &kh,
        m,
        n,
        k,
        crsGraph.row_map,
        crsGraph.entries,
        TRANPOSEFIRST,
        crsGraph2.row_map,
        crsGraph2.entries,
        TRANPOSESECOND,
        row_mapC_ref
    );

    ExecSpace::fence();


    size_type c_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
    if (c_nnz_size){
      entriesC_ref = lno_nnz_view_t (Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size);
    }

    KokkosKernels::Experimental::Graph::triangle_enumerate(
        &kh,
        m,
        n,
        k,
        crsGraph.row_map,
        crsGraph.entries,
        TRANPOSEFIRST,

        crsGraph2.row_map,
        crsGraph2.entries,
        TRANPOSESECOND,
        row_mapC_ref,
        entriesC_ref,
        valuesC_ref
    );
    ExecSpace::fence();

    crsGraph_t3 static_graph (entriesC_ref, row_mapC_ref, k);
    Ccrsgraph_ref = static_graph;
  }


  switch (algorithm){
  case 16:
    kh.create_spgemm_handle(KokkosKernels::Experimental::Graph::SPGEMM_KK_TRIANGLE_DEFAULT);
    break;
  case 17:
    kh.create_spgemm_handle(KokkosKernels::Experimental::Graph::SPGEMM_KK_TRIANGLE_MEM);
    break;
  case 18:
    kh.create_spgemm_handle(KokkosKernels::Experimental::Graph::SPGEMM_KK_TRIANGLE_DENSE);
    break;
  case 19:
    kh.create_spgemm_handle(KokkosKernels::Experimental::Graph::SPGEMM_KK_TRIANGLE_IA_DEFAULT);
    break;
  case 20:
    kh.create_spgemm_handle(KokkosKernels::Experimental::Graph::SPGEMM_KK_TRIANGLE_IA_MEM);
    break;
  case 21:
    kh.create_spgemm_handle(KokkosKernels::Experimental::Graph::SPGEMM_KK_TRIANGLE_IA_DENSE);
    break;

  default:
    kh.create_spgemm_handle(KokkosKernels::Experimental::Graph::SPGEMM_KK_TRIANGLE_DEFAULT);
    break;
  }

  /*
  kh.get_spgemm_handle()->set_multi_color_scale(multi_color_scale);
  kh.get_spgemm_handle()->mkl_keep_output = mkl_keep_output;
  kh.get_spgemm_handle()->mkl_convert_to_1base = false;
  kh.get_spgemm_handle()->set_read_write_cost_calc (calculate_read_write_cost);
  */
  if (coloring_input_file){
    kh.get_spgemm_handle()->coloring_input_file =  std::string(coloring_input_file);
  }
  if (coloring_output_file){
    kh.get_spgemm_handle()->coloring_output_file = std::string(coloring_output_file);
  }


  kh.get_spgemm_handle()->set_compression_steps(!params.compression2step);



  for (int i = 0; i < repeat; ++i){

    KokkosKernels::Experimental::Util::print_1Dview(crsGraph.row_map);
    KokkosKernels::Experimental::Util::print_1Dview(crsGraph.entries);


    KokkosKernels::Experimental::Util::print_1Dview(crsGraph2.row_map);
    KokkosKernels::Experimental::Util::print_1Dview(crsGraph2.entries);

    row_mapC = lno_view_t
              ("non_const_lnow_row",
                  m + 1);
    entriesC = lno_nnz_view_t ("");
    valuesC  = lno_nnz_view_t ("");

    double symbolic_time = 0, numeric_time = 0;
    if (params.triangle_options == 0 || params.triangle_options == 1){
      Kokkos::Impl::Timer timer1;
      KokkosKernels::Experimental::Graph::triangle_count (
          &kh,
          m,
          n,
          k,
          crsGraph.row_map,
          crsGraph.entries,
          TRANPOSEFIRST,
          crsGraph2.row_map,
          crsGraph2.entries,
          TRANPOSESECOND,
          row_mapC
      );

      ExecSpace::fence();
      symbolic_time = timer1.seconds();
    }
    if (params.triangle_options == 1){
      Kokkos::Impl::Timer timer3;
      size_type c_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
      if (c_nnz_size){
        entriesC = lno_nnz_view_t (Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size);
        //valuesC = scalar_view_t (Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);
      }
      KokkosKernels::Experimental::Graph::triangle_enumerate(
          &kh,
          m,
          n,
          k,
          crsGraph.row_map,
          crsGraph.entries,
          TRANPOSEFIRST,

          crsGraph2.row_map,
          crsGraph2.entries,
          TRANPOSESECOND,
          row_mapC,
          entriesC,
          valuesC
      );

      ExecSpace::fence();
      numeric_time = timer3.seconds();

    }

    if (params.triangle_options == 2 ){
      Kokkos::Impl::Timer timer1;
      KokkosKernels::Experimental::Graph::triangle_generic (
          &kh,
          m,
          n,
          k,
          crsGraph.row_map,
          crsGraph.entries,
          TRANPOSEFIRST,
          crsGraph2.row_map,
          crsGraph2.entries,
          TRANPOSESECOND,

          KOKKOS_LAMBDA(const lno_t& row, const lno_t &col ) {
            row_mapC(row) += 1;
          }
      );

      size_type num_triangles = 0;
      KokkosKernels::Experimental::Util::kk_reduce_view<lno_view_t, ExecSpace>(m, row_mapC, num_triangles);
      ExecSpace::fence();

      symbolic_time = timer1.seconds();
      std::cout << "num_triangles:" << num_triangles << std::endl;
    }

    if (params.triangle_options == 3 ){
      //Kokkos::View<lno_t *, Kokkos::MemoryTraits<Kokkos::Atomic> > num_triangle_per_vertex("triangle_vertex", k);
      lno_view_t num_triangle_per_vertex("triangle_vertex", k);

      Kokkos::Impl::Timer timer1;
      KokkosKernels::Experimental::Graph::triangle_generic (
          &kh,
          m,
          n,
          k,
          crsGraph.row_map,
          crsGraph.entries,
          TRANPOSEFIRST,
          crsGraph2.row_map,
          crsGraph2.entries,
          TRANPOSESECOND,

          KOKKOS_LAMBDA(const lno_t& row, const lno_t &col ) {
            row_mapC(row) += 1;
	    //Kokkos::atomic_fetch_add(&(num_triangle_per_vertex(col)),1);
            //Kokkos::atomic_fetch_add(&(num_triangle_per_vertex(crsGraph.entries(row * 2 ))),1);
            //Kokkos::atomic_fetch_add(&(num_triangle_per_vertex(crsGraph.entries(row * 2 + 1) )),1);

            num_triangle_per_vertex(col) +=1;
            //below assumes that crsGraph is the incidence matrix.
            //row corresponds to edge index,
            //col corresponds to vertex index in the triangle.
            //num_triangle_per_vertex(crsGraph.entries(row * 2 )) += 1;
            //num_triangle_per_vertex(crsGraph.entries(row * 2 + 1)) += 1;
          }
      );

      size_type num_triangles = 0;
      KokkosKernels::Experimental::Util::kk_reduce_view<lno_view_t, ExecSpace>(m, row_mapC, num_triangles);
      ExecSpace::fence();

      symbolic_time = timer1.seconds();
      std::cout << "num_triangles:" << num_triangles << std::endl;

      num_triangles = 0;
      KokkosKernels::Experimental::Util::kk_reduce_view<lno_view_t, ExecSpace>(k, num_triangle_per_vertex, num_triangles);
      //KokkosKernels::Experimental::Util::kk_reduce_view<Kokkos::View<lno_t* , Kokkos::MemoryTraits<Kokkos::Atomic> >, ExecSpace>(k, num_triangle_per_vertex, num_triangles);
      
      ExecSpace::fence();

      std::cout << "num_triangles:" << num_triangles << std::endl;

      KokkosKernels::Experimental::Util::print_1Dview(num_triangle_per_vertex);
    }

    std::cout
    << "mm_time:" << symbolic_time + numeric_time
    << " symbolic_time:" << symbolic_time
    << " numeric_time:" << numeric_time << std::endl;


  }

  std::cout << "row_mapC:" << row_mapC.dimension_0() << std::endl;
  std::cout << "entriesC:" << entriesC.dimension_0() << std::endl;
  std::cout << "valuesC:" << valuesC.dimension_0() << std::endl;

  KokkosKernels::Experimental::Util::print_1Dview(entriesC);


  crsGraph_t3 static_graph (entriesC, row_mapC, k);

  if (check_output){
    bool is_identical = is_same_graph<crsGraph_t3, typename crsGraph_t3::device_type>(Ccrsgraph_ref, static_graph);
    if (!is_identical){
      std::cout << "Result is wrong." << std::endl;
      exit(1);
    }
  }


  return static_graph;

}


};
};
