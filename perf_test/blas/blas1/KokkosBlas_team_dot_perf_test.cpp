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


///////////////////////////////////////////////////////////////////////////////////////////////////
// The Level 1 BLAS perform scalar, vector and vector-vector operations;

// https://github.com/kokkos/kokkos-kernels/wiki/BLAS-1%3A%3Ateam-dot
//
// Usage: result = KokkosBlas::Experimental::dot(team,x,y);
// 
// Multiplies each value of x(i) with y(i), and computes the total within 
// a parallel kernel using a TeamPolicy execution policy

// Interface Single Vector only
//
// Parameters:
/*
    TeamType: A Kokkos::TeamPolicy<...>::member_type
    VectorX: A rank-1 Kokkos::View
    VectorY: A rank-1 Kokkos::View
*/
// REQUIREMENTS:
// Y.rank == 1 or X.rank == 1
// Y.extent(0) == X.extent(0)

// Dot Test WITH TEAM POLICY design:
// 1) create 1D View containing 1D matrix, aka a vector; this will be your X
// input matrix; 2) create 1D View containing 1D matrix, aka a vector; this will
// be your Y input matrix; 3) perform the dot operation on the two inputs, and
// capture result in "result"

// Here, m represents the desired length for each 1D matrix / vector;
// "m" is used here, because code from another test was adapted for this test.
//  
//  Top Level Execution Policies:
//  https://github.com/kokkos/kokkos/wiki/Execution-Policies
//
//  TeamPolicy
//  https://github.com/kokkos/kokkos/wiki/Kokkos%3A%3ATeamPolicy
//
/*
Policy 	        Description
RangePolicy 	Each iterate is an integer in a contiguous range
MDRangePolicy 	Each iterate for each rank is an integer in a contiguous range
TeamPolicy      Assigns to each iterate in a contiguous range a team of threads
*/

/// Nested Execution Policies are used to dispatch parallel work inside of an already executing parallel region,
//  either dispatched with a TeamPolicy or a task policy.
/*
 * Policy 	            Description
TeamThreadRange 	Used inside of a TeamPolicy kernel to perform nested parallel loops split over threads of a team.
TeamVectorRange 	Used inside of a TeamPolicy kernel to perform nested parallel loops split over threads of a team and their vector lanes.
ThreadVectorRange 	Used inside of a TeamPolicy kernel to perform nested parallel loops with vector lanes of a thread.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_team_dot.hpp>
#include <Kokkos_Random.hpp>

struct Params {
  int use_cuda    = 0;
  int use_openmp  = 0;
  int use_threads = 0;
  // m is vector length, or number of rows
  int m           = 100000;
  int repeat      = 1;
};

void print_options() {
  std::cerr << "Options:\n" << std::endl;

  std::cerr << "\tBACKEND: '--threads[numThreads]' | '--openmp [numThreads]' | "
               "'--cuda [cudaDeviceIndex]'"
            << std::endl;
  std::cerr << "\tIf no BACKEND selected, serial is the default." << std::endl;
  std::cerr << "\t[Optional] --repeat :: how many times to repeat overall "
               "dot (symbolic + repeated numeric)"
            << std::endl;
}

int parse_inputs(Params& params, int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (0 == strcasecmp(argv[i], "--help") || 0 == strcasecmp(argv[i], "-h")) {
      print_options();
      exit(0);  // note: this is before Kokkos::initialize
    } else if (0 == strcasecmp(argv[i], "--threads")) {
      params.use_threads = atoi(argv[++i]);
    } else if (0 == strcasecmp(argv[i], "--openmp")) {
      params.use_openmp = atoi(argv[++i]);
    } else if (0 == strcasecmp(argv[i], "--cuda")) {
      params.use_cuda = atoi(argv[++i]) + 1;
    } else if (0 == strcasecmp(argv[i], "--m")) {
      params.m = atoi(argv[++i]);
    } else if (0 == strcasecmp(argv[i], "--repeat")) {
      // if provided, C will be written to given file.
      // has to have ".bin", or ".crs" extension.
      params.repeat = atoi(argv[++i]);
    } else {
      std::cerr << "Unrecognized command line argument #" << i << ": "
                << argv[i] << std::endl;
      print_options();
      return 1;
    }
  }
  return 0;
}


template <class ExecSpace>
void run(int m, int repeat) {
  // Declare type aliases
    using Scalar   = double;
    using MemSpace = typename ExecSpace::memory_space;
    using Device   = Kokkos::Device<ExecSpace, MemSpace>;

     // For the Team implementation of dot; ExecSpace is implicit;
     using policy = Kokkos::TeamPolicy<ExecSpace>;
     using member_type = typename policy::member_type;

  // Create 1D view w/ Device as the ExecSpace; this is an input vector
  Kokkos::View<Scalar*, MemSpace> x("X", m);

  // Create 1D view w/ Device as the ExecSpace; this is the output vector
  Kokkos::View<Scalar*, MemSpace> y("Y", m);

  //
  // Here, deep_copy is filling / copying values into Host memory from Views X
  // and Y
  Kokkos::deep_copy(x, 3.0);
  Kokkos::deep_copy(y, 2.0);

  std::cout << "Running BLAS Level 1 Kokkos Teams-based implementation DOT performance experiment ("
            << ExecSpace::name() << ")\n";

  std::cout << "Each test input vector has a length of " << m << std::endl;

  // Warm up run of dot:

  Kokkos::parallel_for("TeamDotDemoUsage",
                  policy(1, Kokkos::AUTO),
                  KOKKOS_LAMBDA(const member_type& team){
                       });

  // To guarantee a kernel has finished, a developer should call the fence of the execution space on which the kernel is being executed. 
  // Otherwise, it depends on the execution space where the loop executes, and whether this execution space implements a barrier.

  Kokkos::fence();
  Kokkos::Timer timer;

  // Live test of dot:
  Kokkos::parallel_for("TeamDotDemoUsage",
                       policy(1, Kokkos::AUTO),
                       KOKKOS_LAMBDA (const member_type& team)
                       {
                       double result = KokkosBlas::Experimental::dot(team, x, y);
                       }
                       );

  ExecSpace().fence();

  // Kokkos Timer set up and data capture
  double total = timer.seconds();
  double avg   = total / repeat;
  // Flops calculation for a 1D matrix dot product per test run;
  size_t flopsPerRun = (size_t)2 * m;
  printf("Avg DOT time: %f s.\n", avg);
  printf("Avg DOT FLOP/s: %.3e\n", flopsPerRun / avg);
}

int main(int argc, char** argv) {
  Params params;

  if (parse_inputs(params, argc, argv)) {
    return 1;
  }

  const int device_id = params.use_cuda - 1;

  const int num_threads = std::max(params.use_openmp, params.use_threads);

  Kokkos::initialize(Kokkos::InitArguments(num_threads, -1, device_id));

  bool useThreads = params.use_threads != 0;
  bool useOMP     = params.use_openmp != 0;
  bool useCUDA    = params.use_cuda != 0;

  bool useSerial = !useOMP && !useCUDA;

  if (useThreads)
  {
#if defined(KOKKOS_ENABLE_THREADS)
    if (params.use_threads)
      run<Kokkos::Threads, Kokkos::LayoutLeft>(params.m, params.repeat);
    else
      run<Kokkos::Threads, Kokkos::LayoutRight>(params.m, params.repeat);
#else
    std::cout << "ERROR:  PThreads requested, but not available.\n";
  return 1;
#endif
}

    if (useOMP) {
#if defined(KOKKOS_ENABLE_OPENMP)
        run<Kokkos::OpenMP>(params.m, params.repeat);
#else
  std::cout << "ERROR: OpenMP requested, but not available.\n";
  return 1;
#endif
    }

    if (useCUDA) {
#if defined(KOKKOS_ENABLE_CUDA)
        run<Kokkos::Cuda>(params.m, params.repeat);
#else
  std::cout << "ERROR: CUDA requested, but not available.\n";
  return 1;
#endif
    }
    if (useSerial) {
#if defined(KOKKOS_ENABLE_SERIAL)
        run<Kokkos::Serial>(params.m, params.repeat);
#else
  std::cout << "ERROR: Serial device requested, but not available; here, implementation of dot is explicitly parallel.\n";
  return 1;
#endif
    }
    Kokkos::finalize();
    return 0;
  }
