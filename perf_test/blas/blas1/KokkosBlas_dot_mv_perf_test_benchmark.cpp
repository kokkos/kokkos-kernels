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

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "KokkosBlas_dot_perf_test.hpp"
#include "Benchmark_Context.hpp"
#include <benchmark/benchmark.h>

using KokkosKernelsBenchmark::Params;

struct Blas1_Params {
  Blas1_Params(benchmark::State& state)
      : use_cuda(Params::get_param_or_default("--use_cuda", 0)),
        use_hip(Params::get_param_or_default("--use_hip", 0)),
        use_sycl(Params::get_param_or_default("--use_sycl", 0)),
        use_openmp(Params::get_param_or_default("--use_openmp", 0)),
        use_threads(Params::get_param_or_default("--use_threads", 0)),
        m(Params::get_param_or_default("--m", 100000)),
        n(Params::get_param_or_default("--n", 5)),
        repeat(Params::get_param_or_default("--repeat", 20)) {
    report(state);
  };

  void report(benchmark::State& state) {
    state.counters["Params::use_cuda"]    = use_cuda;
    state.counters["Params::use_hip"]     = use_hip;
    state.counters["Params::use_sycl"]    = use_sycl;
    state.counters["Params::use_openmp"]  = use_openmp;
    state.counters["Params::use_threads"] = use_threads;
    state.counters["Params::m"]           = m;
    state.counters["Params::n"]           = n;
    state.counters["Params::repeat"]      = repeat;
  }

  const int use_cuda;
  const int use_hip;
  const int use_sycl;
  const int use_openmp;
  const int use_threads;
  const int m;  // vector length
  const int n;  // number of columns
  const int repeat;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// The Level 1 BLAS perform scalar, vector and vector-vector operations;
//
// https://github.com/kokkos/kokkos-kernels/wiki/BLAS-1%3A%3Adot
//
// Usage: result = KokkosBlas::dot(x,y); KokkosBlas::dot(r,x,y);
// Multiplies each value of x(i) [x(i,j)] with y(i) or [y(i,j)] and computes the
// sum. (If x and y have scalar type Kokkos::complex, the complex conjugate of
// x(i) or x(i,j) will be used.) VectorX: A rank-1 Kokkos::View VectorY: A
// rank-1 Kokkos::View ReturnVector: A rank-0 or rank-1 Kokkos::View
//
// REQUIREMENTS:
// Y.rank == 1 or X.rank == 1
// Y.extent(0) == X.extent(0)

// Dot Test design:
// 1) create 1D View containing 1D matrix, aka a vector; this will be your X
// input matrix; 2) create 1D View containing 1D matrix, aka a vector; this will
// be your Y input matrix; 3) perform the dot operation on the two inputs, and
// capture result in "result"

// Here, m represents the desired length for each 1D matrix;
// "m" is used here, because code from another test was adapted for this test.
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Scalar, class ExecSpace>
static void run(benchmark::State& state, const Blas1_Params& params) {
  state.counters[ExecSpace::name()] = 1;

  // Declare type aliases
  using MemSpace = typename ExecSpace::memory_space;
  using Device   = Kokkos::Device<ExecSpace, MemSpace>;

  Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device> x(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "x"), params.m, params.n);

  Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device> y(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "y"), params.m, params.n);

  Kokkos::View<Scalar*, Device> result(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "x dot y"), params.n);

  // Declaring variable pool w/ a seeded random number;
  // a parallel random number generator, so you
  // won't get the same number with a given seed each time
  Kokkos::Random_XorShift64_Pool<ExecSpace> pool(123);

  Kokkos::fill_random(x, pool, 10.0);
  Kokkos::fill_random(y, pool, 10.0);

  for (auto _ : state) {
    // do a warm up run of dot:
    KokkosBlas::dot(result, x, y);

    // The live test of dot:

    Kokkos::fence();
    Kokkos::Timer timer;

    for (int i = 0; i < params.repeat; i++) {
      KokkosBlas::dot(result, x, y);
      ExecSpace().fence();
    }

    // Kokkos Timer set up
    double total = timer.seconds();
    double avg   = total / params.repeat;
    // Flops calculation for a 1D matrix dot product per test run;
    size_t flopsPerRun = (size_t)2 * params.m * params.n;
    state.SetIterationTime(total);

    state.counters["Avg DOT time (s):"] = avg;
    state.counters["Avg DOT FLOP/s:"]   = flopsPerRun / avg;
  }
}

template <class Scalar>
static void Blas1_dot_mv(benchmark::State& state) {
  Blas1_Params params(state);

  if (params.use_threads != 0) {
#if defined(KOKKOS_ENABLE_THREADS)
    run<Kokkos::Threads>(state, params);
    return;
#else
    state.SkipWithError(" PThreads requested, but not available.");
#endif
  }

  if (params.use_openmp != 0) {
#if defined(KOKKOS_ENABLE_OPENMP)
    run<double, Kokkos::OpenMP>(state, params);
    return;
#else
    state.SkipWithError("OpenMP requested, but not available.");
#endif
  }

  if (params.use_cuda != 0) {
#if defined(KOKKOS_ENABLE_CUDA)
    run<double, Kokkos::Cuda>(state, params);
    return;
#else
    state.SkipWithError("CUDA requested, but not available.");
#endif
  }
  if (params.use_hip != 0) {
#if defined(KOKKOS_ENABLE_HIP)
    run<double, Kokkos::Experimental::HIP>(state, params);
    return;
#else
    state.SkipWithError("HIP requested, but not available.");
#endif
  }
  if (params.use_sycl != 0) {
#if defined(KOKKOS_ENABLE_SYCL)
    run<double, Kokkos::Experimental::SYCL>(state, params);
    return;
#else
    state.SkipWithError("SYCL requested, but not available.");
#endif
  }
  if (!params.use_threads && !params.use_openmp && !params.use_cuda &&
      !params.use_hip && !params.use_sycl) {
#if defined(KOKKOS_ENABLE_SERIAL)
    run<double, Kokkos::Serial>(state, params);
    return;
#else
    state.SkipWithError("Serial device requested, but not available.");
#endif
  }
}

BENCHMARK(Blas1_dot_mv<double>)->UseManualTime();
