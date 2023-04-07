//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "KokkosBlas2_gemv.hpp"

#include "KokkosKernels_TestUtils.hpp"
#include "KokkosKernels_perf_test_utilities.hpp"

#include <Benchmark_Context.hpp>
#include <benchmark/benchmark.h>

struct blas2_gemv_params : public perf_test::CommonInputParams {
  int m = 5000;
  int n = 5000;
  // bool layoutLeft = true;
};

void print_options() {
  std::cerr << "Options\n" << std::endl;
  std::cerr << perf_test::list_common_options();

  std::cerr << "\t[Optional] --m      :: number of rows to generate"
            << std::endl;
  std::cerr << "\t[Optional] --n      :: number of cols to generate"
            << std::endl;
}

blas2_gemv_params parse_blas2_gemv_options(int& argc, char** argv) {
  blas2_gemv_params params;
  perf_test::parse_common_options(argc, argv, params);

  for (int i = 1; i < argc; ++i) {
    if (perf_test::check_arg_int(i, argc, argv, "--m", params.m)) {
      ++i;
    } else if (perf_test::check_arg_int(i, argc, argv, "--n", params.n)) {
      ++i;
    } else {
      std::cerr << "Unrecognized command line argument #" << i << ": "
                << argv[i] << std::endl;
      print_options();
      return params;
    }
  }
  return params;
}

template <typename Scalar, typename Layout, typename ExecSpace>
static void KokkosBlas2_gemv(benchmark::State& state) {
  const auto m = state.range(0);
  const auto n = state.range(1);

  // Declare type aliases
  using MemSpace  = typename ExecSpace::memory_space;
  using Device    = Kokkos::Device<ExecSpace, MemSpace>;

  // Create a View containing a 2D matrix; allocate KokkosView with template
  // args of Scalar**, a layout, and
  Kokkos::View<Scalar**, Layout, Device> A(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "A"), m, n);
  // Create Views containing 1D matrix; allocate (without) matrix "x" of size n
  Kokkos::View<Scalar*, Device> x(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "x"), n);
  // Create Views containing 1D matrix; allocate (without) matrix "y" of size m
  Kokkos::View<Scalar*, Device> y(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "y"), m);

  // Declaring variable pool w/ a number seed;
  // a parallel random number generator, so you
  // won't get the same number with a given seed each time
  Kokkos::Random_XorShift64_Pool<ExecSpace> pool(123);

  // Fill 2D Matrix "A" and 1D matrix (i.e., a vector) "x" with random values;
  // Here, 10 is the max value of the random generator between 1 and 10
  // (uniform )
  Kokkos::fill_random(A, pool, 10.0);
  Kokkos::fill_random(x, pool, 10.0);

  // Do a warm-up run
  KokkosBlas::gemv("N", 1.0, A, x, 0.0, y);
  Kokkos::fence();
  double total_time = 0.0;

  for (auto _ : state) {
    // Start timing
    Kokkos::Timer timer;
    KokkosBlas::gemv("N", 1.0, A, x, 0.0, y);
    ExecSpace().fence();

    double time = timer.seconds();
    total_time += time;
    state.SetIterationTime(time);
  }

  state.counters[ExecSpace::name()] = 1;
  state.counters["Avg GEMV time (s):"] =
      benchmark::Counter(total_time, benchmark::Counter::kAvgIterations);
  size_t flopsPerRun                 = (size_t)2 * m * n;
  state.counters["Avg GEMV FLOP/s:"] = benchmark::Counter(
      flopsPerRun, benchmark::Counter::kIsIterationInvariantRate);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  benchmark::Initialize(&argc, argv);
  benchmark::SetDefaultTimeUnit(benchmark::kSecond);
  KokkosKernelsBenchmark::add_benchmark_context(true);

  const auto params    = parse_blas2_gemv_options(argc, argv);
  const auto arg_names = std::vector<std::string>{"m", "n"};
  const auto args      = std::vector<int64_t>{params.m, params.n};

  if (params.use_openmp) {
#if defined(KOKKOS_ENABLE_OPENMP)
    benchmark::RegisterBenchmark(
        "KokkosBlas2_gemv",
        KokkosBlas2_gemv<double, Kokkos::LayoutRight, Kokkos::OpenMP>)
        ->ArgNames(arg_names)
        ->Args(args)
        ->UseManualTime();
#else
    std::cout << "ERROR: OpenMP requested, but not available.\n";
    return 1;
#endif
  }

  if (params.use_cuda) {
#if defined(KOKKOS_ENABLE_CUDA)
    benchmark::RegisterBenchmark(
        "KokkosBlas2_gemv",
        KokkosBlas2_gemv<double, Kokkos::LayoutRight, Kokkos::Cuda>)
        ->ArgNames(arg_names)
        ->Args(args)
        ->UseManualTime();
#else
    std::cout << "ERROR: CUDA requested, but not available.\n";
    return 1;
#endif
  }

  if (true) {  // serial
#if defined(KOKKOS_ENABLE_SERIAL)
    benchmark::RegisterBenchmark(
        "KokkosBlas2_gemv",
        KokkosBlas2_gemv<double, Kokkos::LayoutRight, Kokkos::Serial>)
        ->ArgNames({"m", "n"})
        ->Args({params.m, params.n})
        ->UseManualTime();
#else
    std::cout << "ERROR: Serial device requested, but not available.\n";
    return 1;
#endif
  }

  benchmark::RunSpecifiedBenchmarks();

  benchmark::Shutdown();
  Kokkos::finalize();
  return 0;
}
