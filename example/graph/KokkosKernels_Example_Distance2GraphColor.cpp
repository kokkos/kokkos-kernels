/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
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
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <algorithm>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#include <iostream>
#include <iomanip>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <string>
#include <sys/time.h>

#include <Kokkos_Core.hpp>
#include <KokkosKernels_IOUtils.hpp>
#include <KokkosGraph_Distance2Color.hpp>

#include "KokkosKernels_CrsMatrix.hpp"
#include "KokkosKernels_Parameters.hpp"
#include "KokkosKernels_StaticCrsGraph.hpp"

using namespace KokkosGraph;

#ifdef KOKKOSKERNELS_INST_DOUBLE
using kk_scalar_type = double;
#else
#ifdef KOKKOSKERNELS_INST_FLOAT
using kk_scalar_type = float;
#endif
#endif

#ifdef KOKKOSKERNELS_INST_OFFSET_INT
using kk_size_type = int;
#else
#ifdef KOKKOSKERNELS_INST_OFFSET_SIZE_T
using kk_size_type   = size_t;
#endif
#endif

#ifdef KOKKOSKERNELS_INST_ORDINAL_INT
using kk_lno_type = int;
#else
#ifdef KOKKOSKERNELS_INST_ORDINAL_INT64_T
using kk_lno_type    = int64_t;
#endif
#endif



using namespace KokkosGraph;



void
print_options(std::ostream& os, const char* app_name, unsigned int indent = 0)
{
    std::string spaces(indent, ' ');
    os << "Usage:" << std::endl
       << spaces << "  " << app_name << " [parameters]" << std::endl
       << std::endl
       << spaces << "Parameters:" << std::endl
       << spaces << "  Parallelism (select one of the following):" << std::endl
       << spaces << "      --serial <N>        Execute serially." << std::endl
       << spaces << "      --threads <N>       Use N posix threads." << std::endl
       << spaces << "      --openmp <N>        Use OpenMP with N threads." << std::endl
       << spaces << "      --cuda              Use CUDA" << std::endl
       << std::endl
       << spaces << "  Required Parameters:" << std::endl
       << spaces << "      --amtx <filename>   Input file in Matrix Market format (.mtx)." << std::endl
       << std::endl
       << spaces << "      --algorithm <algorithm_name>   Set the algorithm to use.  Allowable values are:" << std::endl
       << spaces << "                 COLORING_D2_MATRIX_SQUARED  - Matrix-squared + Distance-1 method." << std::endl
       << spaces << "                 COLORING_D2_SERIAL          - Serial algorithm (must use with 'serial' mode)" << std::endl
       << spaces << "                 COLORING_D2_VB              - Vertex Based method using boolean forbidden array (Default)." << std::endl
       << spaces << "                 COLORING_D2_VB_BIT          - VB with Bitvector Forbidden Array" << std::endl
       << spaces << "                 COLORING_D2_VB_BIT_EF       - VB_BIT with Edge Filtering" << std::endl
       << std::endl
       << spaces << "  Optional Parameters:" << std::endl
       << spaces << "      --output-histogram              Print out a histogram of the colors." << std::endl
       << spaces << "      --output-graphviz               Write the output to a graphviz file (G.dot)." << std::endl
       << spaces << "                                      Note: Vertices with color 0 will be filled in and colored" << std::endl
       << spaces << "      --output-graphviz-vert-max <N>  Upper limit of vertices in G to allow graphviz output. Default=1500." << std::endl
       << spaces << "                                      Requires --output-graphviz to also be enabled." << std::endl
       << spaces << "      --validate                      Check that the coloring is a valid distance-2 graph coloring" << std::endl
       << spaces << "      --verbose-level <N>             Set verbosity level [0..5] where N > 0 means print verbose messags." << std::endl
       << spaces << "                                      Default: 0" << std::endl
       << spaces << "      --help                          Print out command line help." << std::endl
       << spaces << " " << std::endl;
}


int
parse_inputs(KokkosKernels::Example::Parameters& params, int argc, char** argv)
{
    bool got_required_param_amtx      = false;
    bool got_required_param_algorithm = false;

    for(int i = 1; i < argc; ++i)
    {
        if(0 == strcasecmp(argv[ i ], "--threads"))
        {
            params.use_threads = atoi(argv[ ++i ]);
        }
        else if(0 == strcasecmp(argv[ i ], "--serial"))
        {
            params.use_serial = atoi(argv[ ++i ]);
        }
        else if(0 == strcasecmp(argv[ i ], "--openmp"))
        {
            params.use_openmp = atoi(argv[ ++i ]);
            std::cout << "use_openmp = " << params.use_openmp << std::endl;
        }
        else if(0 == strcasecmp(argv[ i ], "--cuda"))
        {
            params.use_cuda = 1;
        }
        else if(0 == strcasecmp(argv[ i ], "--amtx"))
        {
            got_required_param_amtx = true;
            params.mtx_bin_file     = argv[ ++i ];
        }
        else if(0 == strcasecmp(argv[ i ], "--validate"))
        {
            params.validate = 1;
        }
        else if(0 == strcasecmp(argv[ i ], "--verbose-level"))
        {
            params.verbose_level = atoi( argv[++i] );
            params.verbose_level = std::min(5, params.verbose_level);
            params.verbose_level = std::max(0, params.verbose_level);
        }
        else if(0 == strcasecmp(argv[ i ], "--output-histogram"))
        {
            params.output_histogram = 1;
        }
        else if(0 == strcasecmp(argv[ i ], "--output-graphviz"))
        {
            params.output_graphviz = 1;
        }
        else if(0 == strcasecmp(argv[ i ], "--output-graphviz-vert-max"))
        {
            params.output_graphviz_vert_max = atoi( argv[++i] );
        }
        else if(0 == strcasecmp(argv[ i ], "--algorithm"))
        {
            ++i;
            if(0 == strcasecmp(argv[ i ], "COLORING_D2_MATRIX_SQUARED"))
            {
                params.algorithm             = 1;
                got_required_param_algorithm = true;
            }
            else if(0 == strcasecmp(argv[ i ], "COLORING_D2_SERIAL"))
            {
                params.algorithm             = 2;
                got_required_param_algorithm = true;
            }
            else if(0 == strcasecmp(argv[ i ], "COLORING_D2_VB") || 0 == strcasecmp(argv[ i ], "COLORING_D2"))
            {
                params.algorithm             = 3;
                got_required_param_algorithm = true;
            }
            else if(0 == strcasecmp(argv[ i ], "COLORING_D2_VB_BIT"))
            {
                params.algorithm             = 4;
                got_required_param_algorithm = true;
            }
            else if(0 == strcasecmp(argv[ i ], "COLORING_D2_VB_BIT_EF"))
            {
                params.algorithm             = 5;
                got_required_param_algorithm = true;
            }
            else
            {
                std::cerr << "2-Unrecognized command line argument #" << i << ": " << argv[ i ] << std::endl;
                print_options(std::cout, argv[ 0 ]);
                return 1;
            }
        }
        else if(0 == strcasecmp(argv[ i ], "--help") || 0 == strcasecmp(argv[ i ], "-h"))
        {
            print_options(std::cout, argv[ 0 ]);
            return 1;
        }
        else
        {
            std::cerr << "3-Unrecognized command line argument #" << i << ": " << argv[ i ] << std::endl;
            print_options(std::cout, argv[ 0 ]);
            return 1;
        }
    }

    if(!got_required_param_amtx)
    {
        std::cout << "Missing required parameter amtx" << std::endl << std::endl;
        print_options(std::cout, argv[ 0 ]);
        return 1;
    }
    if(!got_required_param_algorithm)
    {
        std::cout << "Missing required parameter algorithm" << std::endl << std::endl;
        print_options(std::cout, argv[ 0 ]);
        return 1;
    }
    if(!params.use_serial && !params.use_threads && !params.use_openmp && !params.use_cuda)
    {
        print_options(std::cout, argv[ 0 ]);
        return 1;
    }
    return 0;
}


namespace KokkosKernels {
namespace Example {


std::string
getCurrentDateTimeStr()
{
    // Note: This could be replaced with `std::put_time(&tm, "%FT%T%z")` but std::put_time isn't
    //       supported on the intel C++ compilers as of v. 17.0.x
    time_t now = time(0);
    char   output[ 100 ];
    std::strftime(output, sizeof(output), "%FT%T%Z", std::localtime(&now));
    return output;
}


template<typename ExecSpace, 
         typename CrsGraph_type1, 
         typename CrsGraph_type2, 
         typename CrsGraph_type3, 
         typename TempMemSpace, 
         typename PersistentMemSpace>
void
run_experiment(CrsGraph_type1 crsGraph, Parameters params)
{
    using namespace KokkosGraph;
    using namespace KokkosGraph::Experimental;

    int algorithm = params.algorithm;
    int shmemsize = params.shmemsize;
//    int verbose   = params.verbose_level;

    using lno_view_type     = typename CrsGraph_type3::row_map_type::non_const_type;
    using lno_nnz_view_type = typename CrsGraph_type3::entries_type::non_const_type;
    using size_type         = typename lno_view_type::non_const_value_type;
    using lno_type          = typename lno_nnz_view_type::non_const_value_type;
    using KernelHandle_type = KokkosKernels::Experimental::KokkosKernelsHandle<size_type, lno_type, kk_scalar_type, ExecSpace, TempMemSpace, PersistentMemSpace>;

    // Get Date/Time stamps of start to use later when printing out summary data.

    // Note: crsGraph.numRows() == number of vertices in the 'graph'
    //       crsGraph.entries.extent(0) == number of edges in the 'graph'
    std::cout << "Num verts: " << crsGraph.numRows() << std::endl << "Num edges: " << crsGraph.entries.extent(0) << std::endl;

    // Create a kernel handle
    KernelHandle_type kh;
    kh.set_shmem_size(shmemsize);

    if(params.verbose_level > 0)
    {
        kh.set_verbose(true);
    }

    // accumulators for average stats
    size_t total_colors = 0;
    size_t total_phases = 0;

    std::string label_algorithm;
    switch(algorithm)
    {
        case 1:
            kh.create_distance2_graph_coloring_handle(COLORING_D2_MATRIX_SQUARED);
            label_algorithm = "COLORING_D2_MATRIX_SQUARED";
            break;
        case 2:
            kh.create_distance2_graph_coloring_handle(COLORING_D2_SERIAL);
            label_algorithm = "COLORING_D2_SERIAL";
            break;
        case 3:
            kh.create_distance2_graph_coloring_handle(COLORING_D2_VB);
            label_algorithm = "COLORING_D2_VB";
            break;
        case 4:
            kh.create_distance2_graph_coloring_handle(COLORING_D2_VB_BIT);
            label_algorithm = "COLORING_D2_VB_BIT";
            break;
        case 5:
            kh.create_distance2_graph_coloring_handle(COLORING_D2_VB_BIT_EF);
            label_algorithm = "COLORING_D2_VB_BIT_EF";
            break;
        default:
            kh.create_distance2_graph_coloring_handle(COLORING_D2_VB);
            label_algorithm = "COLORING_D2_VB";
            break;
    }

    if(params.verbose_level > 0)
    {
        std::cout << std::endl << "Run Graph Color D2 (" << label_algorithm << ")" << std::endl;
    }


    // Call the distance-2 graph coloring routine
    graph_compute_distance2_color(&kh, crsGraph.numRows(), crsGraph.numCols(), crsGraph.row_map, crsGraph.entries, crsGraph.row_map, crsGraph.entries);


    total_colors += kh.get_distance2_graph_coloring_handle()->get_num_colors();
    total_phases += kh.get_distance2_graph_coloring_handle()->get_num_phases();


    if(params.verbose_level > 0)
    {
        std::cout << "Total Time: " << kh.get_distance2_graph_coloring_handle()->get_overall_coloring_time() << std::endl
                  << "Num colors: " << kh.get_distance2_graph_coloring_handle()->get_num_colors() << std::endl
                  << "Num Phases: " << kh.get_distance2_graph_coloring_handle()->get_num_phases() << std::endl
                  << "Colors:\n\t";
        KokkosKernels::Impl::print_1Dview(kh.get_distance2_graph_coloring_handle()->get_vertex_colors());
        std::cout << std::endl;
    }

    // Write out the results to a GraphViz file if enabled
    if(params.output_graphviz && crsGraph.numRows() <= params.output_graphviz_vert_max)
    {
        auto colors = kh.get_distance2_graph_coloring_handle()->get_vertex_colors();

        std::ofstream os("G.dot", std::ofstream::out);

        kh.get_distance2_graph_coloring_handle()->dump_graphviz(os, crsGraph.numRows(), crsGraph.row_map, crsGraph.entries, colors);
    }

    // ------------------------------------------
    // Verify correctness
    // ------------------------------------------
    std::string str_color_is_valid = "UNKNOWN";
    if(0 != params.validate)
    {
        str_color_is_valid = "VALID";
    
        bool d2_coloring_is_valid              = false;
        bool d2_coloring_validation_flags[ 4 ] = {false};

        d2_coloring_is_valid = KokkosGraph::Impl::graph_verify_distance2_color(&kh,
                                                                               crsGraph.numRows(),
                                                                               crsGraph.numCols(),
                                                                               crsGraph.row_map,
                                                                               crsGraph.entries,
                                                                               crsGraph.row_map,
                                                                               crsGraph.entries,
                                                                               d2_coloring_validation_flags);
    
        // Print out messages based on coloring validation check.
        if(d2_coloring_is_valid)
        {
            std::cout << std::endl << "Distance-2 Graph Coloring is VALID" << std::endl << std::endl;
        }
        else
        {
            str_color_is_valid = "INVALID";
            std::cout << std::endl
                      << "Distance-2 Graph Coloring is NOT VALID" << std::endl
                      << "  - Vert(s) left uncolored : " << d2_coloring_validation_flags[ 1 ] << std::endl
                      << "  - Invalid D2 Coloring    : " << d2_coloring_validation_flags[ 2 ] << std::endl
                      << std::endl;
        }
        if(d2_coloring_validation_flags[ 3 ])
        {
            std::cout << "Distance-2 Graph Coloring may have poor quality." << std::endl
                      << "  - Vert(s) have high color value : " << d2_coloring_validation_flags[ 3 ] << std::endl
                      << std::endl;
        }
    }

    // ------------------------------------------
    // Print out the colors histogram
    // ------------------------------------------
    if(0 != params.output_histogram)
    {
        KokkosGraph::Impl::graph_print_distance2_color_histogram(&kh, 
                                                                 crsGraph.numRows(), 
                                                                 crsGraph.numCols(), 
                                                                 crsGraph.row_map, 
                                                                 crsGraph.entries, 
                                                                 crsGraph.row_map, 
                                                                 crsGraph.entries, 
                                                                 false);
    }


    Kokkos::Impl::Timer timer;

    double total_time                   = kh.get_distance2_graph_coloring_handle()->get_overall_coloring_time();
    double total_time_color_greedy      = kh.get_distance2_graph_coloring_handle()->get_overall_coloring_time_phase1();
    double total_time_find_conflicts    = kh.get_distance2_graph_coloring_handle()->get_overall_coloring_time_phase2();
    double total_time_resolve_conflicts = kh.get_distance2_graph_coloring_handle()->get_overall_coloring_time_phase3();
    double total_time_matrix_squared    = kh.get_distance2_graph_coloring_handle()->get_overall_coloring_time_phase4();
    double total_time_matrix_squared_d1 = kh.get_distance2_graph_coloring_handle()->get_overall_coloring_time_phase5();

    std::string mtx_bin_file = params.mtx_bin_file;
    mtx_bin_file             = mtx_bin_file.substr(mtx_bin_file.find_last_of("/\\") + 1);

    int  result;
    char hostname[ 100 ];
    char username[ 100 ];

    result = gethostname(hostname, 100);
    if(result)
    {
        perror("gethostname");
    }

    result = getlogin_r(username, 100);
    if(result)
    {
        perror("getlogin_r");
    }

    std::string currentDateTimeStr = getCurrentDateTimeStr();

    std::cout << std::endl
              << "Summary" << std::endl
              << "-------" << std::endl
              << "    Date/Time      : " << currentDateTimeStr << std::endl
              << "    KExecSName     : " << Kokkos::DefaultExecutionSpace::name() << std::endl
              << "    Filename       : " << mtx_bin_file << std::endl
              << "    Num Verts      : " << crsGraph.numRows() << std::endl
              << "    Num Edges      : " << crsGraph.entries.extent(0) << std::endl
              << "    Concurrency    : " << Kokkos::DefaultExecutionSpace::concurrency() << std::endl
              << "    Algorithm      : " << label_algorithm << std::endl
              << "Overall Time/Stats" << std::endl
              << "    Total Time     : " << total_time << std::endl
              << "    Avg Time       : " << total_time << std::endl
              << "VB Distance[1|2] Stats " << std::endl
              << "    Avg Time CG    : " << total_time_color_greedy << std::endl
              << "    Avg Time FC    : " << total_time_find_conflicts << std::endl
              << "    Avg Time RC    : " << total_time_resolve_conflicts << std::endl
              << "Matrix-Squared + D1 Stats" << std::endl
              << "    Avg Time to M^2: " << total_time_matrix_squared << std::endl
              << "    Avg Time to D1 : " << total_time_matrix_squared_d1 << std::endl
              << "Coloring Stats" << std::endl
              << "    Avg colors     : " << total_colors << std::endl
              << "    Avg Phases     : " << total_phases << std::endl
              << "    Validation     : " << str_color_is_valid << std::endl
              << std::endl
              << std::endl;

}   // run_experiment()


template<typename size_type, typename lno_type, typename exec_space, typename hbm_mem_space>
void
driver(Parameters params)
{
    using myExecSpace       = exec_space;
    using myFastDevice      = Kokkos::Device<exec_space, hbm_mem_space>;
    using fast_crstmat_type = typename KokkosKernelsGraphExample::CrsMatrix<double, lno_type, myFastDevice, void, size_type>;
    using fast_graph_type   = typename fast_crstmat_type::StaticCrsGraphType;

    char* mat_file = params.mtx_bin_file;

    fast_graph_type fast_crsgraph;

    fast_crstmat_type fast_crsmat;
    fast_crsmat            = KokkosKernels::Impl::read_kokkos_crst_matrix<fast_crstmat_type>(mat_file);
    fast_crsgraph          = fast_crsmat.graph;
    fast_crsgraph.num_cols = fast_crsmat.numCols();

    KokkosKernels::Example::run_experiment<myExecSpace, fast_graph_type, fast_graph_type, fast_graph_type, hbm_mem_space, hbm_mem_space>(fast_crsgraph, params);

} // driver()


}      // namespace Example
}      // namespace KokkosKernels



int
main(int argc, char* argv[])
{
    KokkosKernels::Example::Parameters params;

    if(parse_inputs(params, argc, argv))
    {
        return 1;
    }

    if(params.mtx_bin_file == NULL)
    {
        std::cerr << "Provide a matrix file" << std::endl;
        return 0;
    }

    std::cout << "Sizeof(kk_lno_type) : " << sizeof(kk_lno_type) << std::endl << "Sizeof(size_type): " << sizeof(kk_size_type) << std::endl;

    const int num_threads = params.use_openmp;      // Assumption is that use_openmp variable is provided as number of threads
    const int device_id   = 0;
    Kokkos::initialize(Kokkos::InitArguments(num_threads, -1, device_id));

    // Print out information about the configuration of the run if verbose_level >= 5
    if(params.verbose_level >= 5)
    {
        Kokkos::print_configuration(std::cout);
    }

#if defined(KOKKOS_ENABLE_OPENMP)
    if(params.use_openmp)
    {
        KokkosKernels::Example::driver<kk_size_type, kk_lno_type, Kokkos::OpenMP, Kokkos::OpenMP::memory_space>(params);
    }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
    if(params.use_cuda)
    {
        KokkosKernels::Example::driver<kk_size_type, kk_lno_type, Kokkos::Cuda, Kokkos::Cuda::memory_space>(params);
    }
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
    if(params.use_serial)
    {
        KokkosKernels::Example::driver<kk_size_type, kk_lno_type, Kokkos::Serial, Kokkos::Serial::memory_space>(params);
    }
#endif

    Kokkos::finalize();

    return 0;
}
