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
#include <stdlib.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <sys/time.h>

// Kokkos Includes
#include <Kokkos_Core.hpp>
#include <Kokkos_UniqueToken.hpp>

// Kokkos Kernels Includes
#include <KokkosKernels_HashmapAccumulator.hpp>
#include <KokkosKernels_TestParameters.hpp>
#include <KokkosKernels_Uniform_Initialized_MemoryPool.hpp>



namespace KokkosKernels {
namespace Experiment {

    template<typename execution_space, typename uniform_memory_pool_t>
    struct functorTestHashmapAccumulator
    {
        //typedef ExecutionSpace execution_space;
        //typedef typename MemoryPoolType uniform_memory_pool_t;
        typedef typename Kokkos::View<size_t*> data_view_t;

        const size_t _num_entries;
        const data_view_t _data;
        uniform_memory_pool_t _memory_pool;
        const size_t _hash_size;
        const size_t _max_hash_entries;

        functorTestHashmapAccumulator( const size_t num_entries,
                                       const data_view_t data,
                                       uniform_memory_pool_t memory_pool,
                                       const size_t hash_size,
                                       const size_t max_hash_entries)
            : _num_entries(num_entries)
            , _data(data)
            , _memory_pool(memory_pool)
            , _hash_size(hash_size)
            , _max_hash_entries(max_hash_entries)
        {
        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const size_t idx) const
        {
            // TODO: Do something with HashmapAccumulator here.
        }

    };  // functorTestHashmapAccumulator


    template<typename execution_space>
    void experiment(size_t num_entries)
    {
        //typedef ExecutionSpace execution_space;
        typedef typename KokkosKernels::Impl::UniformMemoryPool<execution_space, size_t> uniform_memory_pool_t;
        typedef typename Kokkos::View<size_t*> data_view_t;
        typedef typename data_view_t::HostMirror data_view_hostmirror_t;

        // Get the concurrecny
        size_t concurrency = execution_space::concurrency();

        // Set up random number generator
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_int_distribution<size_t> distr(0, num_entries);

        // Create a view of random values
        data_view_t d_data("data", num_entries);
        data_view_hostmirror_t h_data = Kokkos::create_mirror_view(d_data);

        for(size_t i=0; i<num_entries; i++)
        {
            h_data(i) = distr(eng);
            std::cout << h_data(i) << " ";
        }
        std::cout << std::endl;

        // Deep copy initialized values to device memory.
        Kokkos::deep_copy(d_data, h_data);

        // Set Hash Table Parameters
        size_t max_hash_entries = num_entries;  // Max number of entries that can be inserted.
        size_t hash_size_hint   = 12;           // How many hash keys are allowed?

        // Set the hash_size as the next power of 2 bigger than hash_size_hint.
        size_t hash_size = 1;
        while(hash_size < hash_size_hint) { hash_size *= 2; }

        // Create Uniform Initialized Memory Pool
        KokkosKernels::Impl::PoolType pool_type = KokkosKernels::Impl::OneThread2OneChunk;

        // Determine memory chunk size for UniformMemoryPool
        size_t mem_chunk_size = hash_size;      // for hash indices
        mem_chunk_size += hash_size;            // for hash begins
        mem_chunk_size += max_hash_entries;     // for hash nexts
        mem_chunk_size += max_hash_entries;     // for hash keys
        //mem_chunk_size += max_entries;          // for hash values

        // Set a cap on # of chunks to 32.  In application something else should be done
        // here differently if we're OpenMP vs. GPU but for this example we can just cap
        // our number of chunks at 32.
        size_t num_chunks  = KOKKOSKERNELS_MACRO_MIN(32, concurrency);

        //KokkosKernels::Impl::UniformMemoryPool<Kokkos::DefaultExecutionSpace, size_t> m_space(num_chunks, mem_chunk_size, -1, pool_type);
        uniform_memory_pool_t memory_pool(num_chunks, mem_chunk_size, -1, pool_type);

        functorTestHashmapAccumulator<execution_space, uniform_memory_pool_t>
        testHashmapAccumulator(num_entries, h_data, memory_pool, hash_size, max_hash_entries);

        Kokkos::parallel_for("testHashmapAccumulator", execution_space(0, num_entries), testHashmapAccumulator);

    }

}   // namespace Experiment
}   // namespace KokkosKernels



void print_options(std::ostream &os, const char *app_name, unsigned int indent = 0)
{
    std::string spaces(indent, ' ');
    os << "Usage:" << std::endl
       << spaces << "  " << app_name << " [parameters]" << std::endl
       << std::endl
       << spaces << "Parameters:" << std::endl
       << spaces << "  Parallelism (select one of the following):" << std::endl
       << spaces << "      serial <N>        Execute serially." << std::endl
       << spaces << "      threads <N>       Use N posix threads." << std::endl
       << spaces << "      openmp <N>        Use OpenMP with N threads." << std::endl
       << spaces << "      cuda              Use CUDA" << std::endl
       << spaces << "      help              Print out command line help." << std::endl
       << spaces << " " << std::endl;
}   // print_options



int parse_inputs(KokkosKernels::Experiment::Parameters &params, int argc, char **argv)
{
    if(argc==1)
    {
        print_options(std::cout, argv[0]);
        return 1;
    }

    for(int i = 1; i < argc; ++i)
    {
        if(0 == strcasecmp(argv[i], "threads"))
        {
            params.use_threads = atoi(argv[++i]);
        }
        else if(0 == strcasecmp(argv[i], "serial"))
        {
            params.use_serial = atoi(argv[++i]);
        }
        else if(0 == strcasecmp(argv[i], "openmp"))
        {
            params.use_openmp = atoi(argv[++i]);
        }
        else if(0 == strcasecmp(argv[i], "cuda"))
        {
            params.use_cuda = 1;
        }
        else if(0 == strcasecmp(argv[i], "help") || 0 == strcasecmp(argv[i], "-h"))
        {
            print_options(std::cout, argv[0]);
            return 1;
        }
        else
        {
            std::cerr << "3-Unrecognized command line argument #" << i << ": " << argv[i] << std::endl;
            print_options(std::cout, argv[0]);
            return 1;
        }
    }
    return 0;
}   // parse_inputs



int main(int argc, char *argv[])
{
    KokkosKernels::Experiment::Parameters params;

    // Override default repeats (default is 6)
    params.repeat = 1;

    if(parse_inputs(params, argc, argv))
    {
        return 1;
    }

    const int num_threads = params.use_openmp;      // Assumption is that use_openmp variable is provided as number of threads
    const int device_id   = 0;
    Kokkos::initialize(Kokkos::InitArguments(num_threads, -1, device_id));

    // Work goes here.
    KokkosKernels::Experiment::experiment<Kokkos::DefaultExecutionSpace>(20);

    Kokkos::finalize();
    std::cout << "Done" << std::endl;
    return 0;
}
