

// Created by David Poliakoff and Amy Powell on 6/15/2021

#ifndef KOKKOSKERNELS_KOKKOSBLAS_DOT_TEST_RPS_HPP
#define KOKKOSKERNELS_KOKKOSBLAS_DOT_TEST_RPS_HPP

#include <Kokkos_Core.hpp>
#include "blas/KokkosBlas1_dot.hpp"
#include <Kokkos_Random.hpp>

// These headers are required for RPS perf test implementation
//#include <common/KernelBase.hpp>
//#include <common/RunParams.hpp>
//#include <common/QuickKernelBase.hpp>
//
#ifdef KOKKOSKERNELS_ENABLE_TESTS_AND_PERFSUITE
#include <PerfTestUtilities.hpp>

test_list make_dot_kernel_base(const rajaperf::RunParams& params);
test_list construct_dot_kernel_base(const rajaperf::RunParams& run_params);

#endif //KOKKOSKERNELS_ENABLE_TESTS_AND_PERFSUITE

template <class ExecSpace, class Layout>
struct testData {
  
  // type aliases
  using Scalar   = double;
  using MemSpace = typename ExecSpace::memory_space;
  using Device   = Kokkos::Device<ExecSpace, MemSpace>;

// Run Time Info for KK implementation
//  int use_cuda    = 0;                             
//  int use_openmp  = 0;                             
//  int use_threads = 0;                             
  
  // m is vector length                            
  int m           = 100000;         
  int repeat      = 1;              
  bool layoutLeft = true;

  // Test Matrices x and y, View declaration
  
    // Create 1D view w/ Device as the ExecSpace; this is an input vector
  Kokkos::View<Scalar*, Device> x;

  // Create 1D view w/ Device as the ExecSpace; this is the output vector
  Kokkos::View<Scalar*, Device> y;

  // A function with no return type whose name is the name of the class is a
  // constructor or a destructor;
  // Constructor -- create function:
  testData(int m) {
          x = Kokkos::View<Scalar*, Device>(Kokkos::ViewAllocateWithoutInitializing("x"), m);
          y = Kokkos::View<Scalar*, Device>(Kokkos::ViewAllocateWithoutInitializing("y"), m);
          
          Kokkos::Random_XorShift64_Pool<ExecSpace> pool(123);

          Kokkos::fill_random(x, pool, 10.0);
          Kokkos::fill_random(y, pool, 10.0);
  }
};
////////////////////////////////////////////////////////////////////////////////////////////////

/* Taking in by reference avoids making a copy of the data in memory, whereas
 * taking in by value would make a copy in memory.  Copying operations do not
 * enhance performance.
 *  
 *  A function takes data as a pointer when you're dealing with a collection of
 *  things, such as 8 test datasets
 *
 */
// Creating the machinery needed to run as an RPS test



// Templated function 
template<typename ExecSpace, typename Layout>
testData<ExecSpace, Layout> setup_test(int m,
                                       int repeat,
                                       bool layoutLeft
                                       );

                                

// Must have the full function body, as templated functions are recipes for
// functions
//
/*
template <class ExecSpace, class Layout>
void run(int m, int repeat)
// COMMENT OUT TO CLEAN UP

{

  std::cout << "Running BLAS Level 1 DOT performance experiment ("
            << ExecSpace::name() << ")\n";

  std::cout << "Each test input vector has a length of " << m << std::endl;

  // Declaring variable pool w/ a seeded random number;
  // a parallel random number generator, so you
  // won't get the same number with a given seed each time

  // We're constructing an instance of testData, which takes m as a param
  testData<ExecSpace,Layout> testMatrices(m);

  // do a warm up run of dot:
  KokkosBlas::dot(testMatrices.x, testMatrices.y);

  // The live test of dot:

  Kokkos::fence();
  Kokkos::Timer timer;

    for (int i = 0; i < testMatrices.repeat; i++) {
    double result = KokkosBlas::dot(testMatrices.x, testMatrices.y);
    ExecSpace().fence();
  }

  // Kokkos Timer set up
  double total = timer.seconds();
  double avg   = total / testMatrices.repeat;
  // Flops calculation for a 1D matrix dot product per test run;
  size_t flopsPerRun = (size_t)2 * m;
  printf("Avg DOT time: %f s.\n", avg);
  printf("Avg DOT FLOP/s: %.3e\n", flopsPerRun / avg);
}
*/

#endif //KOKKOSKERNELS_KOKKOSBLAS_DOT_TEST_HPP
