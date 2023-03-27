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

#include <gtest/gtest.h>
#include "KokkosKernels_TestUtils.hpp"

#include "KokkosODE_RungeKutta_impl.hpp"
#include "KokkosODE_RungeKuttaTables_impl.hpp"

namespace Test {

// damped harmonic undriven oscillator
// m y'' + c y' + k y = 0
// solution: y=A * exp(-xi * omega_0 * t) * sin(sqrt(1-xi^2) * omega_0 * t + phi)
// omega_0 = sqrt(k/m); xi = c / sqrt(4*m*k)
// A and phi depend on y(0) and y'(0);
// Change of variables: x(t) = y(t)*exp(-c/(2m)*t) = y(t)*exp(-xi * omega_0 * t)
// Change of variables: X = [x ]
//                          [x']
// Leads to X' = A*X  with A = [ 0  1]
//                             [-d  0]
// with d = k/m - (c/(2m)^2) = (1 - xi^2)*omega_0^2
struct duho {

  constexpr static int neqs = 2;
  const double m, c, k, d;
  const double a11 = 0, a12 = 1, a21, a22;

  duho(const double m_, const double c_, const double k_) : m(m_), c(c_), k(k_), d(k_ / m_ - (c_*c_) / (4*m_*m_)), a21(-k / m), a22(-c / m) {};

  template <class vec_type1, class vec_type2>
  KOKKOS_FUNCTION
  void evaluate_function(const double /*t*/, const double /*dt*/, const vec_type1& y, const vec_type2& f) const {
    f(0) = a11*y(0) + a12*y(1);
    f(1) = a21*y(0) + a22*y(1);
  }

  template <class vec_type>
  KOKKOS_FUNCTION
  void solution(const double t, const vec_type& y0, const vec_type& y) const {
    using KAT = Kokkos::ArithTraits<double>;

    const double gamma       = c / (2 * m);
    const double omega       = KAT::sqrt(k / m - gamma * gamma);
    const double phi         = KAT::atan((y0(1) + gamma * y0(0)) / (y0(0) * omega));
    const double A           = y0(0) / KAT::cos(phi);

    y(0) = A * KAT::cos(omega * t - phi) * KAT::exp(-t * gamma);
    y(1) = -y(0) * gamma - omega * A * KAT::sin(omega * t - phi) * KAT::exp(-t * gamma);
  }

}; // duho

template <class ode_type, class vec_type, class scalar_type>
struct solution_wrapper{

  ode_type ode;
  scalar_type t;
  vec_type y_old, y_ref;

  solution_wrapper(const ode_type& ode_, const scalar_type t_, const vec_type& y_old_, const vec_type& y_ref_)
    : ode(ode_), t(t_), y_old(y_old_), y_ref(y_ref_) {};

  KOKKOS_FUNCTION
  void operator() (const int /*idx*/) const {
    ode.solution(t, y_old, y_ref);
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
  void operator() (const int /*idx*/) const {
    KokkosODE::Impl::RKSolve<ode_type, table_type, vec_type, mv_type, double>(my_ode, table, tstart, tend, dt, max_steps, y_old, y_new, tmp, kstack);
  }
};

template <class ode_type, class table_type, class vec_type, class mv_type, class scalar_type>
void test_method(const std::string label, ode_type& my_ode,
		 const scalar_type& tstart, const scalar_type& tend, scalar_type& dt,
		 const int max_steps, vec_type& y_old, vec_type& y_new,
		 const Kokkos::View<double**, Kokkos::HostSpace>& ks,
		 const Kokkos::View<double*, Kokkos::HostSpace>& sol,
		 typename vec_type::HostMirror y_ref_h) {
  using execution_space = typename vec_type::execution_space;

  table_type table;
  vec_type tmp("tmp vector", my_ode.neqs);
  mv_type kstack("k stack", my_ode.neqs, table.nstages);

  Kokkos::RangePolicy<execution_space> my_policy(0, 1);
  RKSolve_wrapper solve_wrapper(my_ode, table, tstart, tend, dt, max_steps, y_old, y_new, tmp, kstack);
  Kokkos::parallel_for(my_policy, solve_wrapper);

  auto y_new_h = Kokkos::create_mirror_view(y_new);
  Kokkos::deep_copy(y_new_h, y_new);
  auto kstack_h = Kokkos::create_mirror_view(kstack);
  Kokkos::deep_copy(kstack_h, kstack);

#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  std::cout << "\n" << label << std::endl;
#endif
  for(int stageIdx = 0; stageIdx < table.nstages; ++stageIdx) {
    EXPECT_NEAR_KK(ks(0, stageIdx), kstack_h(0, stageIdx), 1e-8);
    EXPECT_NEAR_KK(ks(1, stageIdx), kstack_h(1, stageIdx), 1e-8);
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
    std::cout << "  k" << stageIdx << "={" << kstack_h(0, stageIdx) << ", " << kstack_h(1, stageIdx) << "}" << std::endl;
#endif
  }
  EXPECT_NEAR_KK(sol(0), y_new_h(0), 1e-8);
  EXPECT_NEAR_KK(sol(1), y_new_h(1), 1e-8);
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  std::cout << "  y={" << y_new_h(0) << ", " << y_new_h(1) << "}" << std::endl;
  std::cout << "  error={" << Kokkos::abs(y_new_h(0) - y_ref_h(0)) / Kokkos::abs(y_ref_h(0))
	    << ", " << Kokkos::abs(y_new_h(1) - y_ref_h(1)) / Kokkos::abs(y_ref_h(1)) << "}" << std::endl;
#endif

} // test_method

template <class execution_space>
void test_RK() {
  using vec_type   = Kokkos::View<double*,  execution_space>;
  using mv_type    = Kokkos::View<double**, execution_space>;

  duho my_oscillator(1, 1, 4);
  const int neqs    = my_oscillator.neqs;
  
  vec_type y("solution", neqs), f("function", neqs);
  auto y_h = Kokkos::create_mirror(y);
  y_h(0) = 1; y_h(1) = 0;
  Kokkos::deep_copy(y, y_h);

  constexpr double tstart = 0, tend = 10;
  constexpr int max_steps = 1000;
  double dt = (tend - tstart) / max_steps;
  vec_type y_new("y new", neqs), y_old("y old", neqs);

  // Since y_old_h will be reused to set initial conditions
  // for each method tested we do not want to use
  // create_mirror_view which would not do a copy
  // when y_old is in HostSpace.
  typename vec_type::HostMirror y_old_h = Kokkos::create_mirror(y_old);
  y_old_h(0) = 1; y_old_h(1) = 0;

  // First compute analytical solution as reference
  // and to evaluate the error from each RK method.
  vec_type y_ref("reference value", neqs);
  auto y_ref_h = Kokkos::create_mirror(y_ref);
  {
    Kokkos::deep_copy(y_old, y_old_h);
    Kokkos::RangePolicy<execution_space> my_policy(0, 1);
    solution_wrapper wrapper(my_oscillator, tstart + dt, y_old, y_ref);
    Kokkos::parallel_for(my_policy, wrapper);

    Kokkos::deep_copy(y_ref_h, y_ref);
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
    std::cout << "\nAnalytical solution" << std::endl;
    std::cout << "  y={" << y_ref_h(0) << ", " << y_ref_h(1) << "}" << std::endl;
#endif
  }

  // We perform a single step using a RK method
  // and check the values for ki and y_new against
  // expected values.
  {
    Kokkos::deep_copy(y_old, y_old_h);
    double ks_raw[2] = {0, -4};
    Kokkos::View<double**, Kokkos::HostSpace> ks(ks_raw, 2, 1);
    double sol_raw[2] = {1, -0.04};
    Kokkos::View<double*, Kokkos::HostSpace> sol(sol_raw, 2);
    test_method<duho, KokkosODE::Impl::ButcherTableau<0, 0>, vec_type, mv_type, double>("Euler-Forward", my_oscillator, tstart, tend, dt, 1, y_old, y_new, ks, sol, y_ref_h);
  }

  {
    Kokkos::deep_copy(y_old, y_old_h);
    double ks_raw[4] = {0, -0.04,
			-4, -3.96};
    Kokkos::View<double**, Kokkos::HostSpace> ks(ks_raw, 2, 2);
    double sol_raw[2] = {0.9998, -0.0398};
    Kokkos::View<double*, Kokkos::HostSpace> sol(sol_raw, 2);
    test_method<duho, KokkosODE::Impl::ButcherTableau<1, 1>, vec_type, mv_type, double>("Euler-Heun", my_oscillator, tstart, tend, dt, 1, y_old, y_new, ks, sol, y_ref_h);
  }

  {
    Kokkos::deep_copy(y_old, y_old_h);
    double ks_raw[6]  = {0, -0.02, -0.03980078,
			 -4, -3.98, -3.95940234};
    Kokkos::View<double**, Kokkos::HostSpace> ks(ks_raw, 2, 3);
    double sol_raw[2] = {0.9998, -0.03979999};
    Kokkos::View<double*, Kokkos::HostSpace> sol(sol_raw, 2);
    test_method<duho, KokkosODE::Impl::ButcherTableau<1, 2>, vec_type, mv_type, double>("RKF-12", my_oscillator, tstart, tend, dt, 1, y_old, y_new, ks, sol, y_ref_h);
  }

  {
    Kokkos::deep_copy(y_old, y_old_h);
    double ks_raw[8]  = {0, -0.02, -0.02985, -0.039798,
			 -4, -3.98, -3.96955, -3.95940467};
    Kokkos::View<double**, Kokkos::HostSpace> ks(ks_raw, 2, 4);
    double sol_raw[2] = {0.99980067, -0.039798};
    Kokkos::View<double*, Kokkos::HostSpace> sol(sol_raw, 2);
    test_method<duho, KokkosODE::Impl::ButcherTableau<2, 3>, vec_type, mv_type, double>("RKBS", my_oscillator, tstart, tend, dt, 1, y_old, y_new, ks, sol, y_ref_h);
  }

  {
    Kokkos::deep_copy(y_old, y_old_h);
    double ks_raw[12] = {0, -0.01, -0.01497188, -0.03674986, -0.03979499, -0.0199505,
			 -4, -3.99, -3.98491562, -3.96257222, -3.95941166, -3.97984883};
    Kokkos::View<double**, Kokkos::HostSpace> ks(ks_raw, 2, 6);
    double sol_raw[2] = { 0.99980067, -0.03979801};
    Kokkos::View<double*, Kokkos::HostSpace> sol(sol_raw, 2);
    test_method<duho, KokkosODE::Impl::ButcherTableau<4, 5>, vec_type, mv_type, double>("RKF-45", my_oscillator, tstart, tend, dt, 1, y_old, y_new, ks, sol, y_ref_h);
  }

  {
    Kokkos::deep_copy(y_old, y_old_h);
    double ks_raw[12] = {0, -0.008, -0.011982, -0.02392735, -0.03979862, -0.03484563,
			 -4, -3.992, -3.987946, -3.97578551, -3.95940328, -3.96454357};
    Kokkos::View<double**, Kokkos::HostSpace> ks(ks_raw, 2, 6);
    double sol_raw[2] = { 0.99980067, -0.03979801};
    Kokkos::View<double*, Kokkos::HostSpace> sol(sol_raw, 2);
    test_method<duho, KokkosODE::Impl::ButcherTableau<4, 5, 1>, vec_type, mv_type, double>("Cash-Karp", my_oscillator, tstart, tend, dt, 1, y_old, y_new, ks, sol, y_ref_h);
  }

} // test_RK

template <class ode_type, class table_type, class vec_type, class mv_type, class scalar_type>
void test_rate(ode_type& my_ode, const scalar_type& tstart, const scalar_type& tend,
	       Kokkos::View<scalar_type*, Kokkos::HostSpace> dt, const int max_steps,
	       typename vec_type::HostMirror& y_old_h, typename vec_type::HostMirror& y_ref_h,
	       typename vec_type::HostMirror& error) {
  using execution_space = typename vec_type::execution_space;

  table_type table;
  vec_type tmp("tmp vector", my_ode.neqs);
  mv_type kstack("k stack", my_ode.neqs, table.nstages);

  vec_type y_new("solution", my_ode.neqs);
  vec_type y_old("intial conditions", my_ode.neqs);
  auto y_new_h = Kokkos::create_mirror(y_new);

  Kokkos::RangePolicy<execution_space> my_policy(0, 1);
  for(int idx = 0; idx < dt.extent_int(0); ++idx) {
    Kokkos::deep_copy(y_old, y_old_h);
    Kokkos::deep_copy(y_new, y_old_h);
    RKSolve_wrapper solve_wrapper(my_ode, table, tstart, tend, dt(idx), max_steps, y_old, y_new, tmp, kstack);
    Kokkos::parallel_for(my_policy, solve_wrapper);

    Kokkos::deep_copy(y_new_h, y_new);
    error(idx) = Kokkos::abs(y_new_h(0) - y_ref_h(0)) / Kokkos::abs(y_ref_h(0));

#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
    std::cout << "dt=" << dt(idx) << ", error=" << error(idx)
	      << ", solution: {" << y_new_h(0) << ", " << y_new_h(1) << "}" << std::endl;
#endif
  }

} // test_method

template<class execution_space>
void test_convergence_rate() {
  using vec_type   = Kokkos::View<double*,  execution_space>;
  using mv_type    = Kokkos::View<double**, execution_space>;

  duho my_oscillator(1, 1, 4);
  const int neqs    = my_oscillator.neqs;
  
  vec_type y("solution", neqs), f("function", neqs);
  auto y_h = Kokkos::create_mirror(y);
  y_h(0) = 1; y_h(1) = 0;
  Kokkos::deep_copy(y, y_h);

  constexpr double tstart = 0, tend = 1.024;
  constexpr int max_steps = 1024;
  Kokkos::View<double*, Kokkos::HostSpace> dt("Time Steps", 8);
  dt(0) = 0.002; dt(1) = 0.004; dt(2) = 0.008; dt(3) = 0.016;
  dt(4) = 0.032; dt(5) = 0.064; dt(6) = 0.128; dt(7) = 0.256;
  vec_type y_new("y new", neqs), y_old("y old", neqs);

  // Since y_old_h will be reused to set initial conditions
  // for each method tested we do not want to use
  // create_mirror_view which would not do a copy
  // when y_old is in HostSpace.
  typename vec_type::HostMirror y_old_h = Kokkos::create_mirror(y_old);
  y_old_h(0) = 1; y_old_h(1) = 0;

  // First compute analytical solution as reference
  // and to evaluate the error from each RK method.
  vec_type y_ref("reference value", neqs);
  auto y_ref_h = Kokkos::create_mirror(y_ref);
  {
    Kokkos::deep_copy(y_old, y_old_h);
    Kokkos::RangePolicy<execution_space> my_policy(0, 1);
    solution_wrapper wrapper(my_oscillator, tend, y_old, y_ref);
    Kokkos::parallel_for(my_policy, wrapper);

    Kokkos::deep_copy(y_ref_h, y_ref);
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
    std::cout << "\nAnalytical solution" << std::endl;
    std::cout << "  y={" << y_ref_h(0) << ", " << y_ref_h(1) << "}" << std::endl;
#endif
  }

  typename vec_type::HostMirror error("error", dt.extent(0));
  test_rate<duho, KokkosODE::Impl::ButcherTableau<1, 1>, vec_type, mv_type, double>(my_oscillator, tstart, tend, dt, max_steps, y_old_h, y_ref_h, error);

  for(int idx = 1; idx < dt.extent_int(0) - 2; ++idx) {
    double expected_ratio = Kokkos::pow(dt(idx + 1) / dt(idx), KokkosODE::Impl::ButcherTableau<1, 1>::order);
    double actual_ratio = error(idx+1) / error(idx);
    EXPECT_NEAR_KK_REL(actual_ratio, expected_ratio, 0.15);

#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
    double rel_ratio_diff = Kokkos::abs(actual_ratio - expected_ratio) / Kokkos::abs(expected_ratio);
    std::cout << "error ratio: " << actual_ratio << ", expected ratio: " << expected_ratio
	      << ", rel diff: " << rel_ratio_diff << std::endl;
#endif
  }

  Kokkos::deep_copy(error, 0);
  test_rate<duho, KokkosODE::Impl::ButcherTableau<2, 3>, vec_type, mv_type, double>(my_oscillator, tstart, tend, dt, max_steps, y_old_h, y_ref_h, error);

  for(int idx = 1; idx < dt.extent_int(0) - 2; ++idx) {
    double expected_ratio = Kokkos::pow(dt(idx + 1) / dt(idx), KokkosODE::Impl::ButcherTableau<2, 3>::order);
    double actual_ratio = error(idx+1) / error(idx);
    EXPECT_NEAR_KK_REL(actual_ratio, expected_ratio, 0.05);

#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
    double rel_ratio_diff = Kokkos::abs(actual_ratio - expected_ratio) / Kokkos::abs(expected_ratio);
    std::cout << "error ratio: " << actual_ratio << ", expected ratio: " << expected_ratio
	      << ", rel diff: " << rel_ratio_diff << std::endl;
#endif
  }

  Kokkos::deep_copy(error, 0);
  test_rate<duho, KokkosODE::Impl::ButcherTableau<4, 5>, vec_type, mv_type, double>(my_oscillator, tstart, tend, dt, max_steps, y_old_h, y_ref_h, error);

  for(int idx = 1; idx < dt.extent_int(0) - 2; ++idx) {
    double expected_ratio = Kokkos::pow(dt(idx + 1) / dt(idx), KokkosODE::Impl::ButcherTableau<4, 5>::order);
    double actual_ratio = error(idx+1) / error(idx);
    EXPECT_NEAR_KK_REL(actual_ratio, expected_ratio, 0.05);

#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
    double rel_ratio_diff = Kokkos::abs(actual_ratio - expected_ratio) / Kokkos::abs(expected_ratio);
    std::cout << "error ratio: " << actual_ratio << ", expected ratio: " << expected_ratio
	      << ", rel diff: " << rel_ratio_diff << std::endl;
#endif
  }
} // test_convergence_rate
} // namespace Test

int test_RK() {
  Test::test_RK<TestExecSpace>();

  return 1;
}

int test_RK_conv_rate() {
  Test::test_convergence_rate<TestExecSpace>();
  return 1;
}

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, RKSolve_serial) { test_RK(); }
TEST_F(TestCategory, RK_conv_rate) { test_RK_conv_rate(); }
#endif
