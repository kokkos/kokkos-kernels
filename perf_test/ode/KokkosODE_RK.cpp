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

#include "KokkosODE_RungeKuttaTables_impl.hpp"
#include "KokkosODE_RungeKutta_impl.hpp"
#include "KokkosKernels_TestUtils.hpp"

#ifdef KOKKOSKERNELS_ENABLE_BENCHMARK
#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"
#else
#include "KokkosKernels_perf_test_utilities.hpp"
#endif

namespace {
// R1 = 1e-6*1.85e10 * exp(-15618 / T) * (reac) ( 1 – (1- 10^-9) reac)
// d(reac)/dt = -R1
// d(prod)/dt = R1
struct chem_model_1 {

  constexpr static int neqs = 2;
  // constexpr static double alpha = 1e-6*1.85e10;
  constexpr static double alpha = 1.85e10;
  constexpr static double beta  = 15618;
  constexpr static double gamma = 1 - 10^-9;

  const double tstart, tend, T0, T1;

  chem_model_1(const double tstart_ = 0, const double tend_ = 300,
	       const double T0_ = 300, const double T1_ = 800) : tstart(tstart_), tend(tend_), T0(T0_), T1(T1_) {};

  template <class vec_type1, class vec_type2>
  KOKKOS_FUNCTION
  void evaluate_function(const double t, const double /*dt*/, const vec_type1& y, const vec_type2& f) const {
    // First compute the temperature
    // using linear ramp from T0 to T1
    // between tstart and tend.
    double T = (T1 - T0) * (t - tstart) / (tend - tstart) + T0;

    // Evaluate the chemical reaction rate
    f(0) = -alpha * Kokkos::exp(-beta / T) * y(0) * (1 - gamma * y(0));
    f(1) = -f(0);
  }

};

// More complex chemical reaction involving two reacting
// species foam A and foam B, that become 5 products.
// The temperature is capped at 1000K once t reaches 1500s
struct chem_model_2 {

  constexpr static int neqs = 7;
  constexpr static double alpha1 = 1e-6*3334169440721739.0*1500;
  constexpr static double beta1  = 207850000.0 / 8314.0;
  constexpr static double alpha2 = 1e-6*49997793980831.89*1500;
  constexpr static double beta2  = 207850000.0 / 8314.0;

  const double tstart, tend, T0, T1;

  chem_model_2(const double tstart_ = 0, const double tend_ = 2000,
	       const double T0_ = 300, const double T1_ = 1000) : tstart(tstart_), tend(tend_), T0(T0_), T1(T1_) {};

  template <class vec_type1, class vec_type2>
  KOKKOS_FUNCTION
  void evaluate_function(const double t, const double /*dt*/, const vec_type1& y, const vec_type2& f) const {
    // First compute the temperature
    // using linear ramp from T0 to T1
    // between tstart and tend.
    double T = ((T1 - T0) * (t - tstart) / (1500 - tstart) + T0 < 1000) ? (T1 - T0) * (t - tstart) / (1500 - tstart) + T0 : 1000;

    // Evaluate the chemical reaction rates
    double R1 = y(0) * alpha1 * Kokkos::exp(-beta1 / T);
    double R2 = y(1) * alpha2 * Kokkos::exp(-beta2 / T);

    // Evaluate the chemical reaction rate
    f(0) = -R1;
    f(1) = -R2;
    f(2) = R1 + 0.08 * R2;
    f(3) = 0.147 * R2;
    f(4) = 0.453 * R2;
    f(5) = 0.187 * R2;
    f(6) = 0.133 * R2;
  }

};

template <class ode_type, class table_type, class vec_type, class mv_type, class scalar_type>
struct RKSolve_wrapper {

  ode_type my_ode;
  table_type table;
  scalar_type tstart, tend, dt;
  int max_steps;
  vec_type y_old, y_new, tmp;
  mv_type kstack;

  RKSolve_wrapper(const ode_type& my_ode_, const table_type& table_,
		  const scalar_type tstart_, const scalar_type tend_, const scalar_type dt_,
		  const int max_steps_, const vec_type& y_old_, const vec_type& y_new_,
		  const vec_type& tmp_, const mv_type& kstack_) :
    my_ode(my_ode_), table(table_), tstart(tstart_), tend(tend_), dt(dt_), max_steps(max_steps_),
    y_old(y_old_), y_new(y_new_), tmp(tmp_), kstack(kstack_) {}

  KOKKOS_FUNCTION
  void operator() (const int idx) const {

    // Take subviews to create the local problem
    auto local_y_old  = Kokkos::subview( y_old, Kokkos::pair(2*idx, 2*idx + 1));
    auto local_y_new  = Kokkos::subview( y_new, Kokkos::pair(2*idx, 2*idx + 1));
    auto local_tmp    = Kokkos::subview(   tmp, Kokkos::pair(2*idx, 2*idx + 1));
    auto local_kstack = Kokkos::subview(kstack, Kokkos::pair(2*idx, 2*idx + 1), Kokkos::ALL());

    // Run Runge-Kutta time integrator
    KokkosODE::Impl::RKSolve<ode_type, table_type, vec_type, mv_type, double>(my_ode, table, tstart, tend, dt, max_steps,
									       local_y_old, local_y_new, local_tmp, local_kstack);
  }
};

struct rk_input_parameters{

  int num_odes;
  int model;
  
  rk_input_parameters(const int num_odes_, const int model_) : num_odes(num_odes_), model(model_) {};

};

} // namespace (anonymous)


#ifdef KOKKOSKERNELS_ENABLE_BENCHMARK
void run_ode_chem(benchmark::State& state, const rk_input_parameters& params) {
#else
void run_ode_chem(const std::vector<int>& state, const rk_input_parameters& params) {
#endif
  using execution_space = Kokkos::DefaultExecutionSpace;
  using vec_type   = Kokkos::View<double*,  execution_space>;
  using mv_type    = Kokkos::View<double**, execution_space>;
  using table_type = KokkosODE::Impl::ButcherTableau<4, 5, 1>;

  const int num_odes = params.num_odes;
  const int model    = params.model;

  switch (model) {
  case 1: {
    chem_model_1 chem_model;
    const int neqs = chem_model.neqs;
    const int max_steps = 15000;
    const double dt = 0.1;

    table_type table;
    vec_type tmp("tmp vector", neqs*num_odes);
    mv_type kstack("k stack", neqs*num_odes, table.nstages);

    // Set initial conditions
    vec_type y_new("solution", neqs*num_odes);
    vec_type y_old("initial conditions", neqs*num_odes);
    auto y_old_h = Kokkos::create_mirror(y_old);
    y_old_h(0) = 1; y_old_h(1) = 0;
    Kokkos::deep_copy(y_old, y_old_h);
    Kokkos::deep_copy(y_new, y_old_h);

    Kokkos::RangePolicy<execution_space> my_policy(0, num_odes);
    RKSolve_wrapper solve_wrapper(chem_model, table, chem_model.tstart, chem_model.tend,
				  dt, max_steps, y_old, y_new, tmp, kstack);

    Kokkos::Timer time;
    time.reset();
    for(auto _ : state) {
      (void) _;
      Kokkos::parallel_for(my_policy, solve_wrapper);
      Kokkos::fence();
    }
    double run_time = time.seconds();

    auto y_new_h = Kokkos::create_mirror(y_new);
    Kokkos::deep_copy(y_new_h, y_new);
    std::cout << "\nChem model 1" << std::endl;
    std::cout << "  t0=" << chem_model.tstart << ", tn=" << chem_model.tend << std::endl;
    std::cout << "  T0=" << chem_model.T0 << ", Tn=" << chem_model.T1 << std::endl;
    std::cout << "  dt=" << dt << std::endl;
    std::cout << "  y(t0)={" << y_old_h(0) << ", " << y_old_h(1) << "}" << std::endl;
    std::cout << "  y(tn)={" << y_new_h(0) << ", " << y_new_h(1) << "}" << std::endl;
    std::cout << "  num odes: " << num_odes << std::endl;
    std::cout << "  time elapsed: " << run_time << std::endl;
    break;
  }
  case 2: {
    chem_model_2 chem_model;
    const int neqs = chem_model.neqs;
    const int max_steps = 15000;
    const double dt = 0.1;

    table_type table;
    vec_type tmp("tmp vector", neqs*num_odes);
    mv_type kstack("k stack", neqs*num_odes, table.nstages);

    // Set initial conditions
    vec_type y_new("solution", neqs*num_odes);
    vec_type y_old("initial conditions", neqs*num_odes);
    auto y_old_h = Kokkos::create_mirror(y_old);
    y_old_h(0) = 0.25; y_old_h(1) = 0.25; y_old(2) = 0;
    y_old(3) = 0; y_old(4) = 0; y_old(5) = 0; y_old(6) = 0;
    Kokkos::deep_copy(y_old, y_old_h);
    Kokkos::deep_copy(y_new, y_old_h);

    Kokkos::RangePolicy<execution_space> my_policy(0, num_odes);
    RKSolve_wrapper solve_wrapper(chem_model, table, chem_model.tstart, chem_model.tend,
				  dt, max_steps, y_old, y_new, tmp, kstack);

    Kokkos::Timer time;
    time.reset();
    for(auto _ : state) {
      (void) _;
      Kokkos::parallel_for(my_policy, solve_wrapper);
      Kokkos::fence();
    }
    double run_time = time.seconds();

    auto y_new_h = Kokkos::create_mirror(y_new);
    Kokkos::deep_copy(y_new_h, y_new);
    std::cout << "\nChem model 2" << std::endl;
    std::cout << "  t0=" << chem_model.tstart << ", tn=" << chem_model.tend << std::endl;
    std::cout << "  T0=" << chem_model.T0 << ", Tn=" << chem_model.T1 << std::endl;
    std::cout << "  dt=" << dt << std::endl;
    std::cout << "  y(t0)={" << y_old_h(0) << ", " << y_old_h(1) << "}" << std::endl;
    std::cout << "  y(tn)={" << y_new_h(0) << ", " << y_new_h(1) << "}" << std::endl;
    std::cout << "  num odes: " << num_odes << std::endl;
    std::cout << "  time elapsed: " << run_time << std::endl;
    break;
  }
  }
}

#ifdef KOKKOSKERNELS_ENABLE_BENCHMARK
void run_benchmark_wrapper(benchmark::State& state) {
  rk_input_parameters params(state.range(0), state.range(1));

  run_ode_chem(state, params);
}

#else
template <class execution_space>
  void run_perftest_wrapper(int argc, char** argv, perf_test::CommonInputParams) {
  rk_input_parameters params(1000, 1);

  // Loop over arguments, parse them and fill params struct
  for(int argIdx = 1; argIdx < argc - 1; ++argIdx) {
    if((0 == Test::string_compare_no_case(argv[argIdx], "-N"))) {
      params.num_odes = atoi(argv[argIdx + 1]);
    }
    if((0 == Test::string_compare_no_case(argv[argIdx], "--model"))) {
      params.model = atoi(argv[argIdx + 1]);
    }
  }

  // Basically we call the run routine
  // with std::vector state = {1} which
  // means we will do the perf test once.
  // we could change that with command
  // line argument logic in the future.
  // std::vector<int> tmp({1});
  run_ode_chem({1}, params);
}
#endif


#ifdef KOKKOSKERNELS_ENABLE_BENCHMARK
// Benchmark style call for performance
// monitoring over time in a highly controlled
// environment
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);

  benchmark::Initialize(&argc, argv);
  benchmark::SetDefaultTimeUnit(benchmark::kMillisecond);
  KokkosKernelsBenchmark::add_benchmark_context(true);


  std::string bench_name = "KokkosODE_chem_models";
  benchmark::RegisterBenchmark(bench_name.c_str(), run_benchmark_wrapper)
    ->UseRealTime()
    ->ArgNames({"n", "model"})
    ->Args({1000, 1});


  benchmark::RunSpecifiedBenchmarks();

  benchmark::Shutdown();
  Kokkos::finalize();

  return 0;
}

#else

// perf test style call for testing and
// tunning of algorithms in a convenient way
#define KOKKOSKERNELS_PERF_TEST_NAME run_perftest_wrapper
#include "KokkosKernels_perf_test_instantiation.hpp"
int main(int argc, char** argv) {
  return main_instantiation(argc, argv);
}  // main

#endif // KOKKOSKERNELS_ENABLE_BENCHMARK
