#ifdef KOKKOSKERNELS_INST_DOUBLE

#include <gtest/gtest.h>

#include <KokkosBatched_ODE_RKSolvers.h>
#include <KokkosBatched_ODE_TestProblems.h>
#include <KokkosBatched_ODE_Args.h>

namespace KokkosBatched {
namespace ode {

template <typename MemorySpace, typename ODEType, typename SolverType>
void scratch_kernel(const ODEType &ode, const SolverType &solver,
                    const int nelems) {
  using ScratchSpace =
      Kokkos::ScratchMemorySpace<typename MemorySpace::execution_space>;
  using ScratchView =
      Kokkos::View<double ***, ScratchSpace, Kokkos::MemoryUnmanaged>;
  using SolverState = RkSolverState<RkSharedAllocation<MemorySpace>>;
  const int team_size =
      std::is_same<MemorySpace, Kokkos::HostSpace>::value ? 1 : 32;
  const int scratch_size =
      ScratchView::shmem_size(team_size, (4 + SolverType::nstages), ode.neqs);

  using member_type = typename Kokkos::TeamPolicy<
      typename MemorySpace::execution_space>::member_type;
  auto team_policy = Kokkos::TeamPolicy<typename MemorySpace::execution_space>(
      nelems / team_size, team_size);
  team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size),
                               Kokkos::PerThread(0));

  Kokkos::parallel_for(
      "ODESolverKernel", team_policy, KOKKOS_LAMBDA(const member_type &team) {
        SolverState s;
        s.set_views(team.team_scratch(0), ode.neqs, SolverType::nstages);

        for (int dof = 0; dof < ode.neqs; ++dof) {
          s.y[dof] = ode.expected_val(ode.tstart(), dof);
        }

        solver.solve(ode, ode.tstart(), ode.tend(), s);
      });
  Kokkos::fence();
}

template <typename SolverState, typename MemorySpace, typename ODEType,
          typename SolverType>
void kernel(const RkDynamicAllocation<MemorySpace> &pool, const ODEType &ode,
            const SolverType &solver, int nelems) {
  Kokkos::parallel_for(
      "ODESolverKernel",
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, nelems),
      KOKKOS_LAMBDA(const int elem) {
        typename SolverState::StackType stack{};
        SolverState s;
        s.set_views(stack, pool, elem);

        const int ndofs = static_cast<int>(s.y.extent(0));

        for (int dof = 0; dof < ndofs; ++dof) {
          s.y[dof] = ode.expected_val(ode.tstart(), dof);
        }

        solver.solve(ode, ode.tstart(), ode.tend(), s);
      });
  Kokkos::fence();
}

template <typename MemorySpace, typename ODEType, typename SolverType>
void scratch_perf_run(const ODEType &ode, const SolverType &solver,
                      const int nelems) {
  scratch_kernel<MemorySpace>(ode, solver, nelems);
}

template <typename MemorySpace, typename SolverState, typename ODEType,
          typename SolverType>
void perf_run(RkDynamicAllocation<MemorySpace> &pool, const ODEType &ode,
              const SolverType &solver, const int nelems) {
  kernel<SolverState>(pool, ode, solver, nelems);
}

template <typename MemorySpace, typename TableType, typename ODEType, int ndofs>
struct RKPerfTest {
  using SolverType       = RungeKuttaSolver<TableType>;
  using SolverStateStack = RkSolverState<RkStack<ndofs, TableType::n>>;
  using SolverStateDyn   = RkSolverState<RkDynamicAllocation<MemorySpace>>;

  RKPerfTest(const int nelems, const ODEArgs &args, const ODEType &ode_,
             const bool use_stack)
      : pool(nelems, ode_.neqs, TableType::n), solver(args), ode(ode_) {
    if (use_stack) {
      perf_run<MemorySpace, SolverStateStack>(pool, ode, solver, nelems);
    } else {
      perf_run<MemorySpace, SolverStateDyn>(pool, ode, solver, nelems);
    }
  }
  RkDynamicAllocation<MemorySpace> pool;
  SolverType solver;
  ODEType ode;
};

template <typename MemorySpace, typename TableType, typename ODEType>
struct RKScratchPerfTest {
  using SolverType = RungeKuttaSolver<TableType>;
  RKScratchPerfTest(const int nelems, const ODEArgs &args, const ODEType &ode_)
      : solver(args) {
    scratch_perf_run<MemorySpace>(ode_, solver, nelems);
  }
  SolverType solver;
};

template <typename MemorySpace, typename TableType>
void run_enright(const bool use_stack, const int nelems) {
  ODEArgs args;
  RKPerfTest<MemorySpace, TableType, EnrightB5, 6> t1(nelems, args,
                                                      EnrightB5(6), use_stack);
  RKPerfTest<MemorySpace, TableType, EnrightC1, 4> t2(nelems, args,
                                                      EnrightC1(4), use_stack);
  RKPerfTest<MemorySpace, TableType, EnrightC5, 4> t3(nelems, args,
                                                      EnrightC5(4), use_stack);
}

template <typename MemorySpace, typename TableType>
void run_enright_scratch(const int nelems) {
  ODEArgs args;
  RKScratchPerfTest<MemorySpace, TableType, EnrightB5> t1(nelems, args,
                                                          EnrightB5(6));
  RKScratchPerfTest<MemorySpace, TableType, EnrightC1> t2(nelems, args,
                                                          EnrightC1(4));
  RKScratchPerfTest<MemorySpace, TableType, EnrightC5> t3(nelems, args,
                                                          EnrightC5(4));
}

template <typename MemorySpace>
void run_all_tables_scratch(const int nelems) {
  run_enright_scratch<MemorySpace, RKEH>(nelems);
  run_enright_scratch<MemorySpace, RK12>(nelems);
  run_enright_scratch<MemorySpace, BS>(nelems);
  run_enright_scratch<MemorySpace, RKF45>(nelems);
  run_enright_scratch<MemorySpace, CashKarp>(nelems);
  run_enright_scratch<MemorySpace, DormandPrince>(nelems);
}

template <typename MemorySpace>
void run_all_tables(const bool use_stack, const int nelems) {
  run_enright<MemorySpace, RKEH>(use_stack, nelems);
  run_enright<MemorySpace, RK12>(use_stack, nelems);
  run_enright<MemorySpace, BS>(use_stack, nelems);
  run_enright<MemorySpace, RKF45>(use_stack, nelems);
  run_enright<MemorySpace, CashKarp>(use_stack, nelems);
  run_enright<MemorySpace, DormandPrince>(use_stack, nelems);
}

TEST_F(TestCategory, ODE_RKPerformance) {
  const int nelems_host   = 64;
  const int nelems_device = nelems_host * 512;

  double dt_host_scratch = 0.0;
  double dt_host_dynamic = 0.0;
  double dt_host_stack   = 0.0;

  double dt_device_scratch = 0.0;
  double dt_device_dynamic = 0.0;
  double dt_device_stack   = 0.0;

  {
    Kokkos::Timer timer;
    run_all_tables_scratch<Kokkos::HostSpace>(nelems_host);
    dt_host_scratch = timer.seconds();
    std::cout << "RK Performance - Host (Scratch) time = " << dt_host_scratch
              << "\n";
  }

  {
    Kokkos::Timer timer;
    run_all_tables<Kokkos::HostSpace>(false, nelems_host);
    dt_host_dynamic = timer.seconds();
    std::cout << "RK Performance - Host (Dynamic) time = " << dt_host_dynamic
              << "\n";
  }

  {
    Kokkos::Timer timer;
    run_all_tables<Kokkos::HostSpace>(true, nelems_host);
    dt_host_stack = timer.seconds();
    std::cout << "RK Performance - Host (Stack) time = " << dt_host_stack
              << "\n";
  }
  if (1
#ifdef KOKKOS_ENABLE_OPENMP
      && !std::is_same<TestExecSpace, Kokkos::OpenMP>::value
#endif
#ifdef KOKKOS_ENABLE_SERIAL
      && !std::is_same<TestExecSpace, Kokkos::Serial>::value
#endif
#ifdef KOKKOS_ENABLE_THREADS
      && std::is_same<ExecutionSpace, Kokkos::Threads>::value
#endif
      )
  // See if this is a better way to fix it??
  // https://github.com/google/googletest/blob/main/docs/advanced.md#skipping-test-execution
  // See word doc with notes on this.
  // But the original test with Kokkos::Host was giving valid info.... Hmm. What
  // was Host at the time?
  {
    {
      Kokkos::Timer timer;
      run_all_tables_scratch<TestExecSpace>(nelems_device);
      dt_device_scratch = timer.seconds();
      std::cout << "RK Performance - Device (Scratch) time = "
                << dt_device_scratch << "\n";
    }

    {
      Kokkos::Timer timer;
      run_all_tables<TestExecSpace>(false, nelems_device);
      dt_device_dynamic = timer.seconds();
      std::cout << "RK Performance - Device (Dynamic) time = "
                << dt_device_dynamic << "\n";
    }
    {
      Kokkos::Timer timer;
      run_all_tables<TestExecSpace>(true, nelems_device);
      dt_device_stack = timer.seconds();
      std::cout << "RK Performance - Device (Stack) time = " << dt_device_stack
                << "\n";
    }

    std::cout << "RK Performance - Device (Dynamic / Stack) time = "
              << dt_device_dynamic / dt_device_stack << "\n";
    std::cout << "RK Performance - Device (Shared / Stack) time = "
              << dt_device_scratch / dt_device_stack << "\n";
    std::cout << "RK Performance - Host / Device (Stack) time = "
              << dt_host_stack / dt_device_stack << "\n";
  }

  std::cout << "RK Performance - Host (Dynamic / Stack) time = "
            << dt_host_dynamic / dt_host_stack << "\n";
  std::cout << "RK Performance - Host (Scratch / Stack) time = "
            << dt_host_scratch / dt_host_stack << "\n";
}

}  // namespace ode
}  // namespace KokkosBatched

#endif
