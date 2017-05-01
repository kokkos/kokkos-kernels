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

#include "KokkosKernels_RunTriangle.hpp"
#include "KokkosKernels_MyCRSMatrix.hpp"
namespace KokkosKernels{

namespace Experiment{

  template <typename size_type, typename lno_t,
            typename exec_space, typename hbm_mem_space, typename sbm_mem_space>
  void run_multi_mem_triangle(Parameters params){

    typedef exec_space myExecSpace;
    typedef Kokkos::Device<exec_space, hbm_mem_space> myFastDevice;
    typedef Kokkos::Device<exec_space, sbm_mem_space> mySlowExecSpace;

    typedef typename MyKokkosSparse::CrsMatrix<double, lno_t, myFastDevice, void, size_type > fast_crstmat_t;
    typedef typename fast_crstmat_t::StaticCrsGraphType fast_graph_t;
    typedef typename fast_graph_t::row_map_type::non_const_type fast_row_map_view_t;
    typedef typename fast_graph_t::entries_type::non_const_type   fast_cols_view_t;

    typedef typename fast_graph_t::row_map_type::const_type const_fast_row_map_view_t;
    typedef typename fast_graph_t::entries_type::const_type   const_fast_cols_view_t;

    typedef typename MyKokkosSparse::CrsMatrix<double, lno_t, mySlowExecSpace, void, size_type > slow_crstmat_t;
    typedef typename slow_crstmat_t::StaticCrsGraphType slow_graph_t;

    typedef typename slow_graph_t::row_map_type::non_const_type slow_row_map_view_t;
    typedef typename slow_graph_t::entries_type::non_const_type   slow_cols_view_t;
    typedef typename slow_graph_t::row_map_type::const_type const_slow_row_map_view_t;
    typedef typename slow_graph_t::entries_type::const_type   const_slow_cols_view_t;

    char *a_mat_file = params.a_mtx_bin_file;
    char *b_mat_file = params.b_mtx_bin_file;
    //char *c_mat_file = params.c_mtx_bin_file;

    slow_graph_t a_slow_crsgraph, b_slow_crsgraph, c_slow_crsgraph;
    fast_graph_t a_fast_crsgraph, b_fast_crsgraph, c_fast_crsgraph;


    double preprocess_time = 0;
    //read a and b matrices and store them on slow or fast memory.
    if (params.a_mem_space == 1){


      fast_crstmat_t a_fast_crsmat;

      a_fast_crsmat = KokkosKernels::Experimental::Util::read_kokkos_crst_matrix<fast_crstmat_t>(a_mat_file);
      fast_cols_view_t new_indices;
      Kokkos::Impl::Timer timer1;
      if (params.left_sort){
        new_indices = fast_cols_view_t(Kokkos::ViewAllocateWithoutInitializing("new indices"), a_fast_crsmat.numRows());
        KokkosKernels::Experimental::Util::kk_sort_by_row_size <size_type, lno_t,exec_space>(
            a_fast_crsmat.numRows(), a_fast_crsmat.graph.row_map.data(), new_indices.data()
            //&(new_indices[0])
            );
      }
      if (params.left_lower_triangle){
        a_fast_crsmat = KokkosKernels::Experimental::Util::
            kk_get_lower_crs_matrix(a_fast_crsmat,new_indices.data()/* , params.use_dynamic_scheduling */);
      }
      a_fast_crsgraph = a_fast_crsmat.graph;
      a_fast_crsgraph.num_cols = a_fast_crsmat.numCols();
      preprocess_time = timer1.seconds();
    }
    else {
      slow_crstmat_t a_slow_crsmat;
      a_slow_crsmat = KokkosKernels::Experimental::Util::read_kokkos_crst_matrix<slow_crstmat_t>(a_mat_file);
      fast_cols_view_t new_indices;
      Kokkos::Impl::Timer timer1;
      if (params.left_sort){
        new_indices = fast_cols_view_t(Kokkos::ViewAllocateWithoutInitializing("new indices"), a_slow_crsmat.numRows());

        KokkosKernels::Experimental::Util::kk_sort_by_row_size<size_type, lno_t,exec_space>(
            a_slow_crsmat.numRows(), a_slow_crsmat.graph.row_map.data(),
            new_indices.data());
      }
      if (params.left_lower_triangle){
        a_slow_crsmat = KokkosKernels::Experimental::Util::
                    kk_get_lower_crs_matrix(a_slow_crsmat,new_indices.data()/* , params.use_dynamic_scheduling */);
      }

      a_slow_crsgraph = a_slow_crsmat.graph;
      a_slow_crsgraph.num_cols = a_slow_crsmat.numCols();
      preprocess_time = timer1.seconds();
    }


    if ((b_mat_file == NULL || strcmp(b_mat_file, a_mat_file) == 0) && params.b_mem_space == params.a_mem_space){
      std::cout << "Using A matrix for B as well" << std::endl;
      b_fast_crsgraph = a_fast_crsgraph;
      b_slow_crsgraph = a_slow_crsgraph;
    }
    else if (params.b_mem_space == 1){


      fast_crstmat_t b_fast_crsmat;

      if (b_mat_file == NULL) b_mat_file = a_mat_file;
      b_fast_crsmat = KokkosKernels::Experimental::Util::read_kokkos_crst_matrix<fast_crstmat_t>(b_mat_file);
      fast_cols_view_t new_indices;
      Kokkos::Impl::Timer timer1;
      if (params.right_sort){
        new_indices = fast_cols_view_t(Kokkos::ViewAllocateWithoutInitializing("new indices"), b_fast_crsmat.numRows());

        KokkosKernels::Experimental::Util::kk_sort_by_row_size<size_type, lno_t,exec_space>(
            b_fast_crsmat.numRows(), b_fast_crsmat.graph.row_map.data(),
            new_indices.data());
      }
      if (params.right_lower_triangle){
        b_fast_crsmat = KokkosKernels::Experimental::Util::
                            kk_get_lower_crs_matrix(b_fast_crsmat,new_indices.data()/* , params.use_dynamic_scheduling */);
      }
      b_fast_crsgraph = b_fast_crsmat.graph;
      b_fast_crsgraph.num_cols = b_fast_crsmat.numCols();

      preprocess_time = timer1.seconds();
    }
    else {
      slow_crstmat_t b_slow_crsmat;
      if (b_mat_file == NULL) b_mat_file = a_mat_file;
      b_slow_crsmat = KokkosKernels::Experimental::Util::read_kokkos_crst_matrix<slow_crstmat_t>(b_mat_file);
      fast_cols_view_t new_indices;
      Kokkos::Impl::Timer timer1;
      if (params.right_sort){
        new_indices = fast_cols_view_t(Kokkos::ViewAllocateWithoutInitializing("new indices"), b_slow_crsmat.numRows());

        KokkosKernels::Experimental::Util::kk_sort_by_row_size<size_type, lno_t,exec_space>(
            b_slow_crsmat.numRows(), b_slow_crsmat.graph.row_map.data(),
            new_indices.data());

      }
      if (params.right_lower_triangle){
        b_slow_crsmat = KokkosKernels::Experimental::Util::
                            kk_get_lower_crs_matrix(b_slow_crsmat,new_indices.data()/* , params.use_dynamic_scheduling */);
      }
      b_slow_crsgraph = b_slow_crsmat.graph;
      b_slow_crsgraph.num_cols = b_slow_crsmat.numCols();

      preprocess_time = timer1.seconds();
    }


    std::cout << "preprocess_time:" << preprocess_time << std::endl;


    if (params.a_mem_space == 1){
      if (params.b_mem_space == 1){
        if (params.c_mem_space == 1){
          if (params.work_mem_space == 1){
            c_fast_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_graph_t,fast_graph_t,fast_graph_t, hbm_mem_space, hbm_mem_space>
                  (a_fast_crsgraph, b_fast_crsgraph, params);
          }
          else {
            c_fast_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_graph_t,fast_graph_t,fast_graph_t, sbm_mem_space, sbm_mem_space>
                  (a_fast_crsgraph, b_fast_crsgraph, params);
          }

        }
        else {
          //C is in slow memory.
          if (params.work_mem_space == 1){
            c_slow_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_graph_t,fast_graph_t,slow_graph_t, hbm_mem_space, hbm_mem_space>
                  (a_fast_crsgraph, b_fast_crsgraph, params);
          }
          else {
            c_slow_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_graph_t,fast_graph_t,slow_graph_t, sbm_mem_space, sbm_mem_space>
                  (a_fast_crsgraph, b_fast_crsgraph, params);
          }
        }
      }
      else {
        //B is in slow memory
        if (params.c_mem_space == 1){
          if (params.work_mem_space == 1){
            c_fast_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_graph_t,slow_graph_t,fast_graph_t, hbm_mem_space, hbm_mem_space>
                  (a_fast_crsgraph, b_slow_crsgraph, params);
          }
          else {
            c_fast_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_graph_t,slow_graph_t,fast_graph_t, sbm_mem_space, sbm_mem_space>
                  (a_fast_crsgraph, b_slow_crsgraph, params);
          }

        }
        else {
          //C is in slow memory.
          if (params.work_mem_space == 1){
            c_slow_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_graph_t,slow_graph_t,slow_graph_t, hbm_mem_space, hbm_mem_space>
                  (a_fast_crsgraph, b_slow_crsgraph, params);
          }
          else {
            c_slow_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_graph_t,slow_graph_t,slow_graph_t, sbm_mem_space, sbm_mem_space>
                  (a_fast_crsgraph, b_slow_crsgraph, params);
          }
        }

      }
    }
    else {
      //A is in slow memory
      if (params.b_mem_space == 1){
        if (params.c_mem_space == 1){
          if (params.work_mem_space == 1){
            c_fast_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_graph_t,fast_graph_t,fast_graph_t, hbm_mem_space, hbm_mem_space>
                  (a_slow_crsgraph, b_fast_crsgraph, params);
          }
          else {
            c_fast_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_graph_t,fast_graph_t,fast_graph_t, sbm_mem_space, sbm_mem_space>
                  (a_slow_crsgraph, b_fast_crsgraph, params);
          }

        }
        else {
          //C is in slow memory.
          if (params.work_mem_space == 1){
            c_slow_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_graph_t,fast_graph_t,slow_graph_t, hbm_mem_space, hbm_mem_space>
                  (a_slow_crsgraph, b_fast_crsgraph, params);
          }
          else {
            c_slow_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_graph_t,fast_graph_t,slow_graph_t, sbm_mem_space, sbm_mem_space>
                  (a_slow_crsgraph, b_fast_crsgraph, params);
          }
        }
      }
      else {
        //B is in slow memory
        if (params.c_mem_space == 1){
          if (params.work_mem_space == 1){
            c_fast_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_graph_t,slow_graph_t,fast_graph_t, hbm_mem_space, hbm_mem_space>
                  (a_slow_crsgraph, b_slow_crsgraph, params);
          }
          else {
            c_fast_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_graph_t,slow_graph_t,fast_graph_t, sbm_mem_space, sbm_mem_space>
                  (a_slow_crsgraph, b_slow_crsgraph, params);
          }

        }
        else {
          //C is in slow memory.
          if (params.work_mem_space == 1){
            c_slow_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_graph_t,slow_graph_t,slow_graph_t, hbm_mem_space, hbm_mem_space>
                  (a_slow_crsgraph, b_slow_crsgraph, params);
          }
          else {
            c_slow_crsgraph =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_graph_t,slow_graph_t,slow_graph_t, sbm_mem_space, sbm_mem_space>
                  (a_slow_crsgraph, b_slow_crsgraph, params);
          }
        }

      }

    }
  }


}
}
