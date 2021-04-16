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

#include "KokkosBlas2_gemv.hpp"
#include <Kokkos_Random.hpp>

struct Params
{
  int use_cuda = 0;
  int use_openmp = 0;
  int use_threads = 0;
  int m = 5000;
  int n = 5000;
  int repeat = 1;
  bool layoutLeft = true;
};

void print_options(){
  std::cerr << "Options\n" << std::endl;

  std::cerr << "\tBACKEND: '--threads[numThreads]' | '--openmp [numThreads]' | '--cuda [cudaDeviceIndex]'" << std::endl;
  std::cerr << "\tIf none selected, serial is used." << std::endl;
  std::cerr << "\t[Optional] --repeat :: how many times to repeat overall spadd (symbolic + repeated numeric)" << std::endl;
  std::cerr << "\t[Optional] --layout :: matrix layout ('left' or 'right', default 'left')" << std::endl;
  std::cerr << "\t[Optional] --m      :: number of rows to generate" << std::endl;
  std::cerr << "\t[Optional] --n      :: number of cols to generate" << std::endl;
}

int parse_inputs (Params& params, int argc, char **argv){
  for ( int i = 1 ; i < argc ; ++i ) {
    if ( 0 == strcasecmp( argv[i] , "--help") || 0 == strcasecmp( argv[i] , "-h" )) {
      print_options();
      exit(0);  //note: this is before Kokkos::initialize
    }
    else if ( 0 == strcasecmp( argv[i] , "--threads" ) ) {
      params.use_threads = atoi( argv[++i] );
    }
    else if ( 0 == strcasecmp( argv[i] , "--openmp" ) ) {
      params.use_openmp = atoi( argv[++i] );
    }
    else if ( 0 == strcasecmp( argv[i] , "--cuda" ) ) {
      params.use_cuda = atoi( argv[++i] ) + 1;
    }
    else if ( 0 == strcasecmp( argv[i] , "--layout" ) ) {
      i++;
      if(0 == strcasecmp( argv[i] , "left"))
        params.layoutLeft = true;
      else if(0 == strcasecmp( argv[i] , "right"))
        params.layoutLeft = false;
      else
      {
        std::cerr << "Invalid layout: must be 'left' or 'right'.\n";
        exit(1);
      }
    }
    else if( 0 == strcasecmp( argv[i], "--m" ))
    {
      params.m = atoi(argv[++i]);
    }
    else if( 0 == strcasecmp( argv[i], "--n" ))
    {
      params.n = atoi(argv[++i]);
    }
    else if ( 0 == strcasecmp( argv[i] , "--repeat" ) ) {
      //if provided, C will be written to given file.
      //has to have ".bin", or ".crs" extension.
      params.repeat = atoi( argv[++i] );
    }
    else {
      std::cerr << "Unrecognized command line argument #" << i << ": " << argv[i] << std::endl ;
      print_options();
      return 1;
    }
  }
  return 0;
}

template<typename ExecSpace, typename Layout>
void run(int m, int n, int repeat)
{
  using Scalar = double;
  using MemSpace = typename ExecSpace::memory_space;
  using Device = Kokkos::Device<ExecSpace, MemSpace>;
  std::cout << "Running GEMV experiment (" << ExecSpace::name() << ")\n";
  Kokkos::View<Scalar**, Layout, Device> A(Kokkos::ViewAllocateWithoutInitializing("A"), m, n);
  Kokkos::View<Scalar*, Device> x(Kokkos::ViewAllocateWithoutInitializing("x"), n);
  Kokkos::View<Scalar*, Device> y(Kokkos::ViewAllocateWithoutInitializing("y"), m);
  Kokkos::Random_XorShift64_Pool<ExecSpace> pool(123);
  Kokkos::fill_random(A, pool, 10.0);
  Kokkos::fill_random(x, pool, 10.0);
  //Do a warm-up run
  KokkosBlas::gemv("N", 1.0, A, x, 0.0, y);
  //Now, start timing
  Kokkos::fence();
  Kokkos::Timer timer;
  for(int i = 0; i < repeat; i++)
  {
    KokkosBlas::gemv("N", 1.0, A, x, 0.0, y);
    ExecSpace().fence();
  }
  double total = timer.seconds();
  double avg = total / repeat;
  size_t flopsPerRun = (size_t) m * n;
  printf("Avg GEMV time: %f s.\n", avg);
  printf("Avg GEMV FLOP/s: %.3e\n", flopsPerRun / avg);
}

int main (int argc, char ** argv){
  Params params;

  if (parse_inputs (params, argc, argv) ){
    return 1;
  }
  const int num_threads = params.use_openmp; // Assumption is that use_openmp variable is provided as number of threads
  const int device_id = params.use_cuda - 1;

  Kokkos::initialize( Kokkos::InitArguments( num_threads, -1, device_id ) );

  bool useOMP = params.use_openmp != 0;
  bool useCUDA = params.use_cuda != 0;

  bool useSerial = !useOMP && !useCUDA;

  if(useOMP)
  {
#if defined( KOKKOS_ENABLE_OPENMP )
    if(params.layoutLeft)
      run<Kokkos::OpenMP, Kokkos::LayoutLeft>(params.m, params.n, params.repeat);
    else
      run<Kokkos::OpenMP, Kokkos::LayoutRight>(params.m, params.n, params.repeat);
#else
    std::cout << "ERROR: OpenMP requested, but not available.\n";
    return 1;
#endif
  }
  if(useCUDA)
  {
#if defined( KOKKOS_ENABLE_CUDA )
    if(params.layoutLeft)
      run<Kokkos::Cuda, Kokkos::LayoutLeft>(params.m, params.n, params.repeat);
    else
      run<Kokkos::Cuda, Kokkos::LayoutRight>(params.m, params.n, params.repeat);
#else
    std::cout << "ERROR: CUDA requested, but not available.\n";
    return 1;
#endif
  }
  if(useSerial)
  {
#if defined( KOKKOS_ENABLE_SERIAL )
    if(params.layoutLeft)
      run<Kokkos::Serial, Kokkos::LayoutLeft>(params.m, params.n, params.repeat);
    else
      run<Kokkos::Serial, Kokkos::LayoutRight>(params.m, params.n, params.repeat);
#else
    std::cout << "ERROR: Serial device requested, but not available.\n";
    return 1;
#endif
  }
  Kokkos::finalize(); 
  return 0;
}

