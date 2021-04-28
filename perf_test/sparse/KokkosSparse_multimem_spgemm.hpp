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
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_run_spgemm.hpp"
#include <PerfTestUtilities.hpp>
namespace KokkosKernels {

namespace Experiment {

template <typename size_type, typename lno_t, typename scalar_t,
          typename exec_space, typename hbm_mem_space, typename sbm_mem_space>
test_list run_multi_mem_spgemm(rajaperf::RunParams rajaperf_params, Parameters params) {
  test_list list;
  typedef exec_space myExecSpace;
  typedef Kokkos::Device<exec_space, hbm_mem_space> myFastDevice;
  typedef Kokkos::Device<exec_space, sbm_mem_space> mySlowExecSpace;

  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, myFastDevice, void,
                                           size_type>
      fast_crstmat_t;
  data_retriever<fast_crstmat_t, fast_crstmat_t, fast_crstmat_t, fast_crstmat_t,
                 fast_crstmat_t, fast_crstmat_t>
      retriever("sparse/spmv", "sample.mtx", "sample.mtx", "sample.mtx",
                "sample.mtx", "sample.mtx", "sample.mtx");
  for(auto test_case : retriever.test_cases) {
    //test_case.filename;
    //test_case.test_data;
    // typedef typename fast_crstmat_t::StaticCrsGraphType fast_graph_t;
    // typedef typename fast_crstmat_t::row_map_type::non_const_type
    // fast_row_map_view_t;
    typedef
        typename fast_crstmat_t::index_type::non_const_type fast_cols_view_t;
    typedef
        typename fast_crstmat_t::values_type::non_const_type fast_values_view_t;
    typedef typename fast_crstmat_t::row_map_type::const_type
        const_fast_row_map_view_t;
    typedef
        typename fast_crstmat_t::index_type::const_type const_fast_cols_view_t;
    typedef typename fast_crstmat_t::values_type::const_type
        const_fast_values_view_t;

    typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, mySlowExecSpace,
                                             void, size_type>
        slow_crstmat_t;
    // typedef typename slow_crstmat_t::StaticCrsGraphType slow_graph_t;
    // typedef typename slow_crstmat_t::row_map_type::non_const_type
    // slow_row_map_view_t;
    typedef
        typename slow_crstmat_t::index_type::non_const_type slow_cols_view_t;
    typedef
        typename slow_crstmat_t::values_type::non_const_type slow_values_view_t;
    typedef typename slow_crstmat_t::row_map_type::const_type
        const_slow_row_map_view_t;
    typedef
        typename slow_crstmat_t::index_type::const_type const_slow_cols_view_t;
    typedef typename slow_crstmat_t::values_type::const_type
        const_slow_values_view_t;
    list.push_back(rajaperf::make_kernel_base("dogs"+test_case.filename,rajaperf_params, [=](const int, const int){
      slow_crstmat_t a_slow_crsmat, b_slow_crsmat, c_slow_crsmat;
      fast_crstmat_t a_fast_crsmat, b_fast_crsmat, c_fast_crsmat;

      a_slow_crsmat = std::get<0>(test_case.test_data);
      b_slow_crsmat = std::get<1>(test_case.test_data);
      a_fast_crsmat = std::get<2>(test_case.test_data);
      b_fast_crsmat = std::get<3>(test_case.test_data);
      using experiment_functions = decltype(KokkosKernels::Experiment::build_experiment<
          myExecSpace, fast_crstmat_t, fast_crstmat_t, fast_crstmat_t,
          hbm_mem_space, hbm_mem_space>(a_fast_crsmat, b_fast_crsmat,
                                        params));
      using setup_function_type = typename experiment_functions::first_type;
      using test_function_type = typename experiment_functions::second_type;
      setup_function_type setup_function;
      test_function_type test_function;
      // This is where we start running
      if (params.a_mem_space == 1) {
        if (params.b_mem_space == 1) {
          if (params.c_mem_space == 1) {
            if (params.work_mem_space == 1) {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, fast_crstmat_t, fast_crstmat_t, fast_crstmat_t,
                  hbm_mem_space, hbm_mem_space>(a_fast_crsmat, b_fast_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            } else {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, fast_crstmat_t, fast_crstmat_t, fast_crstmat_t,
                  sbm_mem_space, sbm_mem_space>(a_fast_crsmat, b_fast_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            }

          } else {
            // C is in slow memory.
            if (params.work_mem_space == 1) {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, fast_crstmat_t, fast_crstmat_t, slow_crstmat_t,
                  hbm_mem_space, hbm_mem_space>(a_fast_crsmat, b_fast_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            } else {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, fast_crstmat_t, fast_crstmat_t, slow_crstmat_t,
                  sbm_mem_space, sbm_mem_space>(a_fast_crsmat, b_fast_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            }
          }
        } else {
          // B is in slow memory
          if (params.c_mem_space == 1) {
            if (params.work_mem_space == 1) {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, fast_crstmat_t, slow_crstmat_t, fast_crstmat_t,
                  hbm_mem_space, hbm_mem_space>(a_fast_crsmat, b_slow_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            } else {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, fast_crstmat_t, slow_crstmat_t, fast_crstmat_t,
                  sbm_mem_space, sbm_mem_space>(a_fast_crsmat, b_slow_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            }

          } else {
            // C is in slow memory.
            if (params.work_mem_space == 1) {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, fast_crstmat_t, slow_crstmat_t, slow_crstmat_t,
                  hbm_mem_space, hbm_mem_space>(a_fast_crsmat, b_slow_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            } else {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, fast_crstmat_t, slow_crstmat_t, slow_crstmat_t,
                  sbm_mem_space, sbm_mem_space>(a_fast_crsmat, b_slow_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            }
          }
        }
      } else {
        // A is in slow memory
        if (params.b_mem_space == 1) {
          if (params.c_mem_space == 1) {
            if (params.work_mem_space == 1) {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, slow_crstmat_t, fast_crstmat_t, fast_crstmat_t,
                  hbm_mem_space, hbm_mem_space>(a_slow_crsmat, b_fast_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            } else {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, slow_crstmat_t, fast_crstmat_t, fast_crstmat_t,
                  sbm_mem_space, sbm_mem_space>(a_slow_crsmat, b_fast_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            }

          } else {
            // C is in slow memory.
            if (params.work_mem_space == 1) {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, slow_crstmat_t, fast_crstmat_t, slow_crstmat_t,
                  hbm_mem_space, hbm_mem_space>(a_slow_crsmat, b_fast_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            } else {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, slow_crstmat_t, fast_crstmat_t, slow_crstmat_t,
                  sbm_mem_space, sbm_mem_space>(a_slow_crsmat, b_fast_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            }
          }
        } else {
          // B is in slow memory
          if (params.c_mem_space == 1) {
            if (params.work_mem_space == 1) {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, slow_crstmat_t, slow_crstmat_t, fast_crstmat_t,
                  hbm_mem_space, hbm_mem_space>(a_slow_crsmat, b_slow_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            } else {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, slow_crstmat_t, slow_crstmat_t, fast_crstmat_t,
                  sbm_mem_space, sbm_mem_space>(a_slow_crsmat, b_slow_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            }

          } else {
            // C is in slow memory.
            if (params.work_mem_space == 1) {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, slow_crstmat_t, slow_crstmat_t, slow_crstmat_t,
                  hbm_mem_space, hbm_mem_space>(a_slow_crsmat, b_slow_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            } else {
              auto functions = KokkosKernels::Experiment::build_experiment<
                  myExecSpace, slow_crstmat_t, slow_crstmat_t, slow_crstmat_t,
                  sbm_mem_space, sbm_mem_space>(a_slow_crsmat, b_slow_crsmat,
                                                params);
              setup_function = functions.first;
              test_function = functions.second;
            }
          }
        }
      }
    },[&](const int, const int, auto exec) {
}));
    //if (c_mat_file != NULL) {
    //  if (params.c_mem_space == 1) {
    //    fast_cols_view_t sorted_adj("sorted adj",
    //                                c_fast_crsmat.graph.entries.extent(0));
    //    fast_values_view_t sorted_vals("sorted vals",
    //                                   c_fast_crsmat.graph.entries.extent(0));

    //    KokkosKernels::Impl::kk_sort_graph<
    //        const_fast_row_map_view_t, const_fast_cols_view_t,
    //        const_fast_values_view_t, fast_cols_view_t, fast_values_view_t,
    //        myExecSpace>(c_fast_crsmat.graph.row_map,
    //                     c_fast_crsmat.graph.entries, c_fast_crsmat.values,
    //                     sorted_adj, sorted_vals);

    //    KokkosKernels::Impl::write_graph_bin(
    //        (lno_t)(c_fast_crsmat.numRows()),
    //        (size_type)(c_fast_crsmat.graph.entries.extent(0)),
    //        c_fast_crsmat.graph.row_map.data(), sorted_adj.data(),
    //        sorted_vals.data(), c_mat_file);
    //  } else {
    //    slow_cols_view_t sorted_adj("sorted adj",
    //                                c_fast_crsmat.graph.entries.extent(0));
    //    slow_values_view_t sorted_vals("sorted vals",
    //                                   c_fast_crsmat.graph.entries.extent(0));

    //    KokkosKernels::Impl::kk_sort_graph<
    //        const_slow_row_map_view_t, const_slow_cols_view_t,
    //        const_slow_values_view_t, slow_cols_view_t, slow_values_view_t,
    //        myExecSpace>(c_slow_crsmat.graph.row_map,
    //                     c_slow_crsmat.graph.entries, c_slow_crsmat.values,
    //                     sorted_adj, sorted_vals);

    //    KokkosKernels::Impl::write_graph_bin(
    //        (lno_t)c_slow_crsmat.numRows(),
    //        (size_type)c_slow_crsmat.graph.entries.extent(0),
    //        c_slow_crsmat.graph.row_map.data(), sorted_adj.data(),
    //        sorted_vals.data(), c_mat_file);
    //  }
    //}
  }
  return list;
}

}  // namespace Experiment
}  // namespace KokkosKernels
