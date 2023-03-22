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
#include <benchmark/benchmark.h>

template <typename Scalar, typename Layout>
static void KokkosBlas2_gemv(benchmark::State& state) {
  const auto m = state.range(0);
  const auto n = state.range(1);

  // Declare type aliases
  using ExecSpace = Kokkos::DefaultExecutionSpace;
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

  for (auto _ : state) {
    // Do a warm-up run
    KokkosBlas::gemv("N", 1.0, A, x, 0.0, y);

    // Start timing
    Kokkos::fence();
    Kokkos::Timer timer;
    KokkosBlas::gemv("N", 1.0, A, x, 0.0, y);
    ExecSpace().fence();

    // Kokkos Timer set up
    double time = timer.seconds();
    // Flops calculation
    size_t flopsPerRun = (size_t)2 * m * n;
    state.SetIterationTime(time);

    state.counters["Avg GEMV time (s):"] =
        benchmark::Counter(time, benchmark::Counter::kDefaults);
    state.counters["Avg GEMV FLOP/s:"] = benchmark::Counter(
        flopsPerRun, benchmark::Counter::kIsIterationInvariantRate);
  }
}

BENCHMARK(KokkosBlas2_gemv<double, Kokkos::LayoutLeft>)
    ->ArgNames({"m", "n", Kokkos::DefaultExecutionSpace::name()})
    ->Args({5000, 5000, 1})
    ->UseManualTime();

BENCHMARK(KokkosBlas2_gemv<double, Kokkos::LayoutRight>)
    ->ArgNames({"m", "n", Kokkos::DefaultExecutionSpace::name()})
    ->Args({5000, 5000, 1})
    ->UseManualTime();
