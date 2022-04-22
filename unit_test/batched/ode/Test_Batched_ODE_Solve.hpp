#ifdef KOKKOSKERNELS_INST_DOUBLE

#include <gtest/gtest.h>

#include <KokkosBatched_ODE_RKSolve.hpp>
#include <KokkosBatched_ODE_TestProblems.hpp>
#include <KokkosBatched_ODE_Args.hpp>
#include <KokkosBatched_ODE_AllocationState.hpp>

namespace KokkosBatched {
namespace Experimental {
namespace ODE {

template <typename SolverState, typename MemorySpace, typename ODEType,
          typename TableType>
void kernel(int nelems, const ODEType &ode, ODEArgs &args, double tstart,
            double tend, const int ndofs,
            const RkDynamicAllocation<MemorySpace> &pool) {
  int glob_status = 0;

  Kokkos::parallel_reduce(
      "ODESolverKernel",
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, nelems),
      KOKKOS_LAMBDA(const int elem, int &status) {
        typename SolverState::StackType stack{};
        SolverState s;
        s.set_views(stack, pool, elem);

        for (int dof = 0; dof < ndofs; ++dof) {
          s.y[dof] = ode.expected_val(ode.tstart(), dof);
        }

        auto thread_status = static_cast<int>(SerialRKSolve<TableType>::invoke(
            ode, args, s.y, s.y0, s.dydt, s.ytemp, s.k, tstart, tend));
        status             = thread_status > status ? thread_status : status;

        for (int dof = 0; dof < ndofs; ++dof) {
          pool.y(elem, dof) = s.y[dof];
        }
      },
      Kokkos::Max<int>(glob_status));

  ASSERT_EQ(ODESolverStatus(glob_status), ODESolverStatus::SUCCESS);
}

template <typename SolverState, typename MemorySpace, typename ODEType,
          typename TableType>
void compute_errors(const RkDynamicAllocation<MemorySpace> &pool,
                    Kokkos::View<double ***, Kokkos::HostSpace> errs,
                    const ODEType &ode, ODEArgs &args, const int nelems,
                    const int ndofs, const int level) {
  kernel<SolverState, MemorySpace, ODEType, TableType>(
      nelems, ode, args, ode.tstart(), ode.tend(), ndofs, pool);

  auto y_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pool.y);

  for (int elem = 0; elem < nelems; ++elem) {
    for (int n = 0; n < ndofs; ++n) {
      const double err =
          Kokkos::fabs(ode.expected_val(ode.tend(), n) - y_host(elem, n));
      errs(elem, n, level) = err;
    }
  }
}

template <typename ODEType>
using ErrorCheck = std::function<void(const int, const double, const ODEType &,
                                      const double, const double)>;

template <typename ODEType>
void empty_check(const int /*dof*/, const double /*err*/,
                 const ODEType /*&ode*/, const double /*rel_tol*/,
                 const double /*abs_tol*/) {}

template <typename ODEType>
void error_check(const int dof, const double err, const ODEType &ode,
                 const double rel_tol, const double abs_tol) {
  const double val     = Kokkos::fabs(ode.expected_val(ode.tend(), dof));
  const bool condition = (err < rel_tol * val) || (err < abs_tol);
  EXPECT_TRUE(condition);
}

template <typename MemorySpace, typename TableType, typename ODEType, int ndofs>
struct RKTest {
  using SolverStateStack = RkSolverState<RkStack<ndofs, TableType::nstages>>;
  using SolverStateDyn   = RkSolverState<RkDynamicAllocation<MemorySpace>>;

  RKTest(const bool use_stack_, const int nelems_, ODEArgs args_, ODEType ode_,
         ErrorCheck<ODEType> check_, const bool do_convergence_ = false,
         const double rel_tol_ = 1e-12, const double abs_tol_ = 1e-16)
      : use_stack(use_stack_),
        nelems(nelems_),
        args(args_),
        ode(ode_),
        pool(nelems, ndofs, TableType::nstages),
        errs(Kokkos::ViewAllocateWithoutInitializing("errs"), nelems, ndofs,
             nlevels),
        check(std::move(check_)),
        do_convergence(do_convergence_),
        rel_tol(rel_tol_),
        abs_tol(abs_tol_) {}
  const int nlevels = 5;

  const bool use_stack;
  const int nelems;
  ODEArgs args;
  const ODEType ode;

  RkDynamicAllocation<MemorySpace> pool;
  Kokkos::View<double ***, Kokkos::HostSpace> errs;

  ErrorCheck<ODEType> check;
  const bool do_convergence;
  const double rel_tol;
  const double abs_tol;

  void run() { do_convergence ? test_convergence() : test_tolerance(); }

  void test_convergence() {
    const int num_substeps = args.num_substeps;
    ODEArgs args_local     = args;

    for (int m = 0; m < nlevels; ++m) {
      const int reduction     = 1 << m;
      args_local.num_substeps = num_substeps * reduction;

      if (use_stack) {
        compute_errors<SolverStateStack, MemorySpace, ODEType, TableType>(
            pool, errs, ode, args_local, nelems, ndofs, m);
      } else {
        compute_errors<SolverStateDyn, MemorySpace, ODEType, TableType>(
            pool, errs, ode, args_local, nelems, ndofs, m);
      }

      for (int e = 0; e < nelems; ++e) {
        for (int n = 0; n < ndofs; ++n) {
          double err_ratio = 0.0;
          double slope     = 0.0;

          if (m >= 1) {
            err_ratio = errs(e, n, m - 1) / errs(e, n, m);
            slope     = Kokkos::log2(err_ratio);
          }

          // check the slope of last few refinements
          if (m >= (nlevels - 2)) {
            EXPECT_TRUE(slope > 0.98 * TableType::order);
          }
        }
      }
    }
  }

  void test_tolerance() {
    if (use_stack) {
      compute_errors<SolverStateStack, MemorySpace, ODEType, TableType>(
          pool, errs, ode, args, nelems, ndofs, 0);
    } else {
      compute_errors<SolverStateDyn, MemorySpace, ODEType, TableType>(
          pool, errs, ode, args, nelems, ndofs, 0);
    }

    for (int e = 0; e < nelems; ++e) {
      for (int n = 0; n < ndofs; ++n) {
        check(n, errs(e, n, 0), ode, rel_tol, abs_tol);
      }
    }
  }
};

template <typename MemorySpace, typename ODEType, int NDOFS>
void run_all_rks_verify(const bool use_stack, const int nelems, ODEArgs &args,
                        const ODEType &ode,
                        const std::vector<char> &do_conv = {true, true, true,
                                                            true, true, true},
                        ErrorCheck<ODEType> check = error_check<ODEType>) {
  RKTest<MemorySpace, RKEH, ODEType, NDOFS> t1(use_stack, nelems, args, ode,
                                               check, do_conv[0]);
  RKTest<MemorySpace, RK12, ODEType, NDOFS> t2(use_stack, nelems, args, ode,
                                               check, do_conv[1]);
  RKTest<MemorySpace, BS, ODEType, NDOFS> t3(use_stack, nelems, args, ode,
                                             check, do_conv[2]);
  RKTest<MemorySpace, RKF45, ODEType, NDOFS> t4(use_stack, nelems, args, ode,
                                                check, do_conv[3]);
  RKTest<MemorySpace, CashKarp, ODEType, NDOFS> t5(use_stack, nelems, args, ode,
                                                   check, do_conv[4]);
  RKTest<MemorySpace, DormandPrince, ODEType, NDOFS> t6(use_stack, nelems, args,
                                                        ode, check, do_conv[5]);

  t1.run();
  t2.run();
  t3.run();
  t4.run();
  t5.run();
  t6.run();
}

template <typename MemorySpace, int NDOFS>
void run_rk_verification_tests(const bool use_stack, const int nelems) {
  const int ndofs = NDOFS;
  ODEArgs args;
  args.is_adaptive = false;

  // For the stiffer ODES, params are chosen to be suitable for a convergence
  // study i.e. time step for stability is sufficiently large so we avoid
  // reaching machine precision too quickly

  // True indicates whether to do a convergence study,
  // otherwise we test for exact reproduction
  const std::vector<char> do_conv = {false, false, false, false, false, false};
  run_all_rks_verify<MemorySpace, DegreeOnePoly, NDOFS>(
      use_stack, nelems, args, DegreeOnePoly(ndofs), do_conv);
  run_all_rks_verify<MemorySpace, DegreeTwoPoly, NDOFS>(
      use_stack, nelems, args, DegreeTwoPoly(ndofs), do_conv);
  run_all_rks_verify<MemorySpace, DegreeThreePoly, NDOFS>(
      use_stack, nelems, args, DegreeThreePoly(ndofs),
      {true, true, false, false, false, false});
  run_all_rks_verify<MemorySpace, DegreeFivePoly, NDOFS>(
      use_stack, nelems, args, DegreeFivePoly(ndofs),
      {true, true, true, false, false, false});
  run_all_rks_verify<MemorySpace, SpringMassDamper, 2>(
      use_stack, nelems, args, SpringMassDamper(2, 21., 100.));
  run_all_rks_verify<MemorySpace, CosExp, NDOFS>(use_stack, nelems, args,
                                                 CosExp(ndofs, -1., 1., 2.));

  // coarsen step
  args.num_substeps = 4;
  run_all_rks_verify<MemorySpace, StiffChemicalDecayProcess, 3>(
      use_stack, nelems, args, StiffChemicalDecayProcess(3, 10., 1.));
  run_all_rks_verify<MemorySpace, Exponential, NDOFS>(use_stack, nelems, args,
                                                      Exponential(ndofs, -2.));
}

template <typename MemorySpace, typename ODEType, int NDOFS>
void run_all_rks_adapt(const bool use_stack, const int nelems, ODEArgs &args,
                       const ODEType &ode, const double rel_tol = 0.0,
                       const double abs_tol      = 0.0,
                       ErrorCheck<ODEType> check = empty_check<ODEType>) {
  RKTest<MemorySpace, RKEH, ODEType, NDOFS> t1(use_stack, nelems, args, ode,
                                               check, false, rel_tol, abs_tol);
  RKTest<MemorySpace, RK12, ODEType, NDOFS> t2(use_stack, nelems, args, ode,
                                               check, false, rel_tol, abs_tol);
  RKTest<MemorySpace, BS, ODEType, NDOFS> t3(use_stack, nelems, args, ode,
                                             check, false, rel_tol, abs_tol);
  RKTest<MemorySpace, RKF45, ODEType, NDOFS> t4(use_stack, nelems, args, ode,
                                                check, false, rel_tol, abs_tol);
  RKTest<MemorySpace, CashKarp, ODEType, NDOFS> t5(
      use_stack, nelems, args, ode, check, false, rel_tol, abs_tol);
  RKTest<MemorySpace, DormandPrince, ODEType, NDOFS> t6(
      use_stack, nelems, args, ode, check, false, rel_tol, abs_tol);

  t1.run();
  t2.run();
  t3.run();
  t4.run();
  t5.run();
  t6.run();
}

template <typename MemorySpace, int NDOFS>
void run_rk_adaptive_tests(const bool use_stack, const int nelems) {
  const int ndofs = NDOFS;
  ODEArgs args;
  args.maxSubSteps = 1e6;

  // Linear problems - both real and complex eigenvalues, checking for rel
  // errors

  // No. fcn evals
  // CosExp =                    {56340, 5367, 6200, 3492, 2670, 3752}
  // SpringMassDamper =          {7824, 3384, 3432, 3570, 3438, 4557}
  // StiffChemicalDecayProcess = {32124, 8178, 8084, 7722, 6870, 9737}
  // B5 =                        {249k, 24765, 24800, 6990, 4800, 7357}

  // the relative tolerance is set based on the dof w/ largest rel. err across
  // all solvers for the higher order solvers the rel tol reached is much better
  // than shown e.g. EnrightB5 limited by RK12 and BS, otherwise < 5e-5 rel. tol

  run_all_rks_adapt<MemorySpace, CosExp, NDOFS>(
      use_stack, nelems, args, CosExp(ndofs, -10., 2., 1.), 5.3e-5, 0.,
      error_check<CosExp>);
  run_all_rks_adapt<MemorySpace, SpringMassDamper, 2>(
      use_stack, nelems, args, SpringMassDamper(2, 1001., 1000.), 1e-4, 0.,
      error_check<SpringMassDamper>);
  run_all_rks_adapt<MemorySpace, StiffChemicalDecayProcess, 3>(
      use_stack, nelems, args, StiffChemicalDecayProcess(3, 1e4, 1.), 4e-9,
      1.8e-10, error_check<StiffChemicalDecayProcess>);
  run_all_rks_adapt<MemorySpace, EnrightB5, 6>(use_stack, nelems, args,
                                               EnrightB5(6), 1.3e-2, 0.0,
                                               error_check<EnrightB5>);

  // Nonlinear problems -- Enright problems are checking only for solver success

  // No. fcn evals
  // Tracer = {111k, 10425, 16176, 2724, 1914, 2996}
  // C1 =     {48.6k, 4635, 4448, 2238, 1572, 2415}
  // C5 =     {15114, 6393, 8876, 9078, 8334, 11165}
  // D2 =     {170k, 318k, 316k, 373k, 367k, 484k} // Tend = 40
  // D4 =     {265k, 529k, 529k, 623k, 614k, 809k} // Tend = 50
  run_all_rks_adapt<MemorySpace, Tracer, 2>(
      use_stack, nelems, args, Tracer(2, 10.0), 0.0, 1e-3, error_check<Tracer>);
  run_all_rks_adapt<MemorySpace, EnrightC1, 4>(use_stack, nelems, args,
                                               EnrightC1(4));
  run_all_rks_adapt<MemorySpace, EnrightC5, 4>(use_stack, nelems, args,
                                               EnrightC5(4));
  run_all_rks_adapt<MemorySpace, EnrightD2, 3>(use_stack, nelems, args,
                                               EnrightD2(3));
  run_all_rks_adapt<MemorySpace, EnrightD4, 3>(use_stack, nelems, args,
                                               EnrightD4(3));
}

TEST_F(TestCategory, ODE_RKVerificationTests) {
  run_rk_verification_tests<TestExecSpace, 1>(true, 2);
  run_rk_verification_tests<TestExecSpace, 1>(false, 2);
}

TEST_F(TestCategory, ODE_RKAdaptiveTests) {
  run_rk_adaptive_tests<TestExecSpace, 1>(true, 2);
  run_rk_adaptive_tests<TestExecSpace, 1>(false, 2);
}

TEST_F(TestCategory, ODE_RKSolverStatus) {
  constexpr int ndofs = 1;
  using TableType     = RKEH;
  using Stack         = RkStack<ndofs, TableType::nstages>;
  Exponential ode(ndofs, -10);
  double tstart = 0.0;
  double tend   = 1.0;

  Stack stack{};
  RkSolverState<Stack> state;
  state.set_views(stack);

  {
    ODEArgs args;
    state.y[0] = std::numeric_limits<double>::quiet_NaN();

    auto status = SerialRKSolve<TableType>::invoke(ode, args, state.y, state.y0,
                                                   state.dydt, state.ytemp,
                                                   state.k, tstart, tend);
    EXPECT_TRUE(status == ODESolverStatus::NONFINITE_STATE);
  }

  state.y[0] = ode.expected_val(tstart, 0);

  {
    ODEArgs args;
    args.maxSubSteps = 3;
    auto status = SerialRKSolve<TableType>::invoke(ode, args, state.y, state.y0,
                                                   state.dydt, state.ytemp,
                                                   state.k, tstart, tend);
    EXPECT_TRUE(status == ODESolverStatus::FAILED_TO_CONVERGE);
  }

  {
    ODEArgs args;
    args.minStepSize = 1.0;
    auto status = SerialRKSolve<TableType>::invoke(ode, args, state.y, state.y0,
                                                   state.dydt, state.ytemp,
                                                   state.k, tstart, tend);
    EXPECT_TRUE(status == ODESolverStatus::MINIMUM_TIMESTEP_REACHED);
  }
}

template <typename TableType, typename StateType, typename Arr>
void check_single_step(const double dt, const TableType &table,
                       const StateType &s, const Arr &ke) {
  using Kokkos::fabs;

  const double tol = 1e-15;
  for (unsigned dof = 0; dof < s.y.extent(0); ++dof) {
    double b_dot_k = 0.0;
    for (int j = 0; j < table.nstages; ++j) {
      const double err = fabs(ke[j][dof] - s.k(j, dof));
      EXPECT_TRUE(err <= fabs(tol * ke[j][dof]));
      b_dot_k += table.b[j] * ke[j][dof];
    }

    EXPECT_NEAR(s.y[dof], s.y0[dof] + dt * b_dot_k, tol);
  }
}

// Manually verify the j RK stages are computed properly
// RK solvers go through the same step(...)..use the RK45 table which is
// sufficiently complex. Note the ODE must have a dependence on y to fully test
// the stages
TEST_F(TestCategory, ODE_RKSingleStep) {
  constexpr int ndofs = 2;
  using Arr           = Kokkos::Array<double, ndofs>;
  using Stack         = RkStack<ndofs, RKF45::nstages>;
  SpringMassDamper ode(ndofs, 1001, 1000.);
  ODEArgs args = ODEArgs();
  RKF45 table;

  const double t0 = 0.1;
  const double dt = 1e-3;

  double est_err = 0.0;

  Stack stack{};
  RkSolverState<Stack> state;
  state.set_views(stack);
  auto y0 = state.y0;
  y0(0)   = ode.expected_val(t0, 0);
  y0(1)   = ode.expected_val(t0, 1);
  SerialRKSolveInternal::step(ode, table, t0, dt, state.y, state.y0,
                              state.ytemp, state.k, args, est_err);

  Kokkos::Array<Arr, RKF45::nstages> ke{};

  Arr tmp;
  ode.derivatives(t0, y0, ke[0]);

  // clang-format off
  tmp[0] = y0[0] + dt * 0.25 * ke[0][0];
  tmp[1] = y0[1] + dt * 0.25 * ke[0][1];
  ode.derivatives(t0 + dt * 0.25, tmp, ke[1]);

  tmp[0] = y0[0] + dt * (ke[0][0] * 3. / 32 + ke[1][0] * 9. / 32);
  tmp[1] = y0[1] + dt * (ke[0][1] * 3. / 32 + ke[1][1] * 9. / 32);
  ode.derivatives(t0 + dt * (3. / 8), tmp, ke[2]);

  tmp[0] = y0[0] + dt * (ke[0][0] * 1932. / 2197 - ke[1][0] * 7200. / 2197 + ke[2][0] * 7296. / 2197);
  tmp[1] = y0[1] + dt * (ke[0][1] * 1932. / 2197 - ke[1][1] * 7200. / 2197 + ke[2][1] * 7296. / 2197);
  ode.derivatives(t0 + dt * (12. / 13), tmp, ke[3]);

  tmp[0] = y0[0] + dt * (ke[0][0] * 439. / 216 - ke[1][0] * 8. + ke[2][0] * 3680. / 513 - ke[3][0] * 845. / 4104);
  tmp[1] = y0[1] + dt * (ke[0][1] * 439. / 216 - ke[1][1] * 8. + ke[2][1] * 3680. / 513 - ke[3][1] * 845. / 4104);
  ode.derivatives(t0 + dt, tmp, ke[4]);

  tmp[0] = y0[0] + dt * (-ke[0][0] * 8. / 27 + ke[1][0] * 2. - ke[2][0] * 3544. / 2565 + ke[3][0] * 1859. / 4104 - ke[4][0] * 11. / 40);
  tmp[1] = y0[1] + dt * (-ke[0][1] * 8. / 27 + ke[1][1] * 2. - ke[2][1] * 3544. / 2565 + ke[3][1] * 1859. / 4104 - ke[4][1] * 11. / 40);
  // clang-format on

  ode.derivatives(t0 + 0.5 * dt, tmp, ke[5]);

  check_single_step(dt, table, state, ke);
}

}  // namespace ODE
}  // namespace Experimental
}  // namespace KokkosBatched

#endif
