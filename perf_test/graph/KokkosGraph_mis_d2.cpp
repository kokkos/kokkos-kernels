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

#include "KokkosKernels_Utils.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spadd.hpp"
#include "KokkosGraph_MIS2.hpp"
#include "KokkosKernels_default_types.hpp"

using namespace KokkosGraph;

struct MIS2Parameters
{
  int repeat = 1;
  bool verbose = false;
  int use_threads = 0;
  int use_openmp = 0;
  int use_cuda = 0;
  int use_serial = 0;
  const char* mtx_file = NULL;
  MIS2_Algorithm algo = MIS2_FAST;
};

void print_options(std::ostream &os, const char *app_name, unsigned int indent = 0)
{
    std::string spaces(indent, ' ');
    os << "Usage:" << std::endl
       << spaces << "  " << app_name << " [parameters]" << std::endl
       << std::endl
       << spaces << "Parameters:" << std::endl
       << spaces << "  Required Parameters:" << std::endl
       << spaces << "      --amtx <filename>   Input file in Matrix Market format (.mtx)." << std::endl
       << std::endl
       << spaces << "      Device type (the following are enabled in this build):" << std::endl
#ifdef KOKKOS_ENABLE_SERIAL
       << spaces << "          --serial            Execute serially." << std::endl
#endif
#ifdef KOKKOS_ENABLE_THREADS
       << spaces << "          --threads           Use posix threads.\n"
#endif
#ifdef KOKKOS_ENABLE_OPENMP
       << spaces << "          --openmp            Use OpenMP.\n"
#endif
#ifdef KOKKOS_ENABLE_CUDA
       << spaces << "          --cuda              Use CUDA.\n"
#endif
       << std::endl
       << spaces << "  Optional Parameters:" << std::endl
       << spaces << "      --algo alg          alg: fast, quality" << std::endl
       << spaces << "      --repeat <N>        Set number of test repetitions (Default: 1) " << std::endl
       << spaces << "      --verbose           Enable verbose mode (record and print timing + extra information)" << std::endl
       << spaces << "      --help              Print out command line help." << std::endl
       << spaces << " " << std::endl;
}

static char* getNextArg(int& i, int argc, char** argv)
{
  i++;
  if(i >= argc)
  {
    std::cerr << "Error: expected additional command-line argument!\n";
    exit(1);
  }
  return argv[i];
}

int parse_inputs(MIS2Parameters &params, int argc, char **argv)
{
    bool got_required_param_amtx      = false;
    for(int i = 1; i < argc; ++i)
    {
        if(0 == strcasecmp(argv[i], "--threads"))
        {
            params.use_threads = 1;
        }
        else if(0 == strcasecmp(argv[i], "--serial"))
        {
            params.use_serial = 1;
        }
        else if(0 == strcasecmp(argv[i], "--openmp"))
        {
            params.use_openmp = 1;
        }
        else if(0 == strcasecmp(argv[i], "--cuda"))
        {
            params.use_cuda = 1;
        }
        else if(0 == strcasecmp(argv[i], "--repeat"))
        {
            params.repeat = atoi(getNextArg(i, argc, argv));
            if(params.repeat <= 0)
            {
              std::cout << "*** Repeat count must be positive, defaulting to 1.\n";
              params.repeat = 1;
            }
        }
        else if(0 == strcasecmp(argv[i], "--amtx"))
        {
            got_required_param_amtx = true;
            params.mtx_file  = getNextArg(i, argc, argv);
        }
        else if(0 == strcasecmp(argv[i], "--algo"))
        {
            const char* algName = getNextArg(i, argc, argv);
            if(!strcasecmp(algName, "fast"))
              params.algo = MIS2_FAST;
            else if(!strcasecmp(algName, "quality"))
              params.algo = MIS2_QUALITY;
            else
              throw std::invalid_argument("Algorithm not valid: must be 'fast' or 'quality'");
        }
        else if(0 == strcasecmp(argv[i], "--verbose"))
        {
            params.verbose = true;
        }
        else if(0 == strcasecmp(argv[i], "--help") || 0 == strcasecmp(argv[i], "-h"))
        {
            print_options(std::cout, argv[0]);
            return 1;
        }
        else
        {
            std::cerr << "Unrecognized command line argument #" << i << ": " << argv[i] << std::endl;
            print_options(std::cout, argv[0]);
            return 1;
        }
    }

    if(!got_required_param_amtx)
    {
        std::cout << "Missing required parameter amtx" << std::endl << std::endl;
        print_options(std::cout, argv[0]);
        return 1;
    }
    if(!params.use_serial && !params.use_threads && !params.use_openmp && !params.use_cuda)
    {
        print_options(std::cout, argv[0]);
        return 1;
    }
    return 0;
}

template<typename device_t>
void run_mis2(const MIS2Parameters& params)
{
    using size_type = default_size_type;
    using lno_t = default_lno_t;
    using exec_space = typename device_t::execution_space;
    using mem_space = typename device_t::memory_space;
    using crsMat_t = typename KokkosSparse::CrsMatrix<default_scalar, default_lno_t, device_t, void, default_size_type>;
    using lno_view_t = typename crsMat_t::index_type::non_const_type;
    using KKH = KokkosKernels::Experimental::KokkosKernelsHandle<size_type, lno_t, double, exec_space, mem_space, mem_space>;
 
    Kokkos::Timer t;
    crsMat_t A_in = KokkosKernels::Impl::read_kokkos_crst_matrix<crsMat_t>(params.mtx_file);
    std::cout << "I/O time: " << t.seconds() << " s\n";
    t.reset();
    //Symmetrize the matrix just in case
    crsMat_t At_in = KokkosKernels::Impl::transpose_matrix(A_in);
    crsMat_t A;
    KKH kkh;
    kkh.create_spadd_handle(false);
    KokkosSparse::spadd_symbolic(&kkh, A_in, At_in, A);
    KokkosSparse::spadd_numeric(&kkh, 1.0, A_in, 1.0, At_in, A);
    kkh.destroy_spadd_handle();
    std::cout << "Time to symmetrize: " << t.seconds() << " s\n";
    auto rowmap = A.graph.row_map;
    auto entries = A.graph.entries;
    lno_t numVerts = A.numRows();

    std::cout << "Num verts: " << numVerts << '\n'
              << "Num edges: " << A.nnz() << '\n';

    lno_view_t mis;

    t.reset();
    for(int rep = 0; rep < params.repeat; rep++)
    {
      mis = KokkosGraph::Experimental::graph_d2_mis<device_t, decltype(rowmap), decltype(entries)>(rowmap, entries, params.algo);
      exec_space().fence();
    }
    double totalTime = t.seconds();
    std::cout << "MIS-2 average time: " << totalTime / params.repeat << '\n';
    std::cout << "MIS size: " << mis.extent(0) << '\n';

    if(params.verbose)
    {
      std::cout << "Vertices in independent set:\n";
      KokkosKernels::Impl::print_1Dview(mis);
    }
}

int main(int argc, char *argv[])
{
    MIS2Parameters params;

    if(parse_inputs(params, argc, argv))
    {
        return 1;
    }

    if(params.mtx_file == NULL)
    {
        std::cerr << "Provide a matrix file" << std::endl;
        return 0;
    }

    Kokkos::initialize();

    bool run = false;

    #if defined(KOKKOS_ENABLE_OPENMP)
    if(params.use_openmp)
    {
      run_mis2<Kokkos::OpenMP>(params);
      run = true;
    }
    #endif

    #if defined(KOKKOS_ENABLE_THREADS)
    if(params.use_threads)
    {
      run_mis2<Kokkos::Threads>(params);
      run = true;
    }
    #endif

    #if defined(KOKKOS_ENABLE_CUDA)
    if(params.use_cuda)
    {
      run_mis2<Kokkos::Cuda>(params);
      run = true;
    }
    #endif

    #if defined(KOKKOS_ENABLE_SERIAL)
    if(params.use_serial)
    {
      run_mis2<Kokkos::Serial>(params);
      run = true;
    }
    #endif

    if(!run)
    {
      std::cerr << "*** ERROR: did not run, none of the supported device types were selected.\n";
    }

    Kokkos::finalize();

    return 0;
}
