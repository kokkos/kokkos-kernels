#include "Kokkos_Core.hpp"
#include "KokkosBatched_SVD_Decl.hpp"

template <typename ExecSpace>
void call_svd_in_parallel_for() {
  Kokkos::TeamPolicy<ExecSpace> team_pol(1, Kokkos::AUTO);
  using ScratchMatrix = Kokkos::View<double[3][3], typename ExecSpace::scratch_memory_space>;
  using ScratchVector = Kokkos::View<double[3], typename ExecSpace::scratch_memory_space>;
  team_pol.set_scratch_size(0, Kokkos::PerThread(3 * ScratchMatrix::shmem_size() + 3 * ScratchVector::shmem_size()));
  Kokkos::parallel_for(
      team_pol, KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<ExecSpace>::member_type &team) {
        ScratchMatrix A(team.thread_scratch(0));
        ScratchMatrix U(team.thread_scratch(0));
        ScratchMatrix V(team.thread_scratch(0));
        ScratchVector S(team.thread_scratch(0));
        ScratchVector work(team.thread_scratch(0));

        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          printf("A = %p\n", A.data());
          printf("U = %p\n", U.data());
          printf("V = %p\n", V.data());
          printf("S = %p\n", S.data());
          printf("work = %p\n", work.data());

          A(0, 0) = 0.000000;
          A(1, 0) = 3.58442287931538747e-02;
          A(2, 0) = 0.000000;
          A(0, 1) = 0.000000;
          A(1, 1) = 3.81743062695684907e-02;
          A(2, 1) = 0.000000;
          A(0, 2) = 0.000000;
          A(1, 2) = 0.000000;
          A(2, 2) = -5.55555555555555733e-02;

          KokkosBatched::SerialSVD::invoke(KokkosBatched::SVD_USV_Tag{}, A, U, S, V, work);

          printf("S = {%.16f %.16f %.16f}\n", S(0), S(1), S(2));
          printf("A(0) = {%.16f %.16f %.16f}\n", A(0, 0), A(0, 1), A(0, 2));
          printf("A(1) = {%.16f %.16f %.16f}\n", A(1, 0), A(1, 1), A(1, 2));
          printf("A(2) = {%.16f %.16f %.16f}\n", A(2, 0), A(2, 1), A(2, 2));
          printf("U(0) = {%.16f %.16f %.16f}\n", U(0, 0), U(0, 1), U(0, 2));
          printf("U(1) = {%.16f %.16f %.16f}\n", U(1, 0), U(1, 1), U(1, 2));
          printf("U(2) = {%.16f %.16f %.16f}\n", U(2, 0), U(2, 1), U(2, 2));
          printf("V(0) = {%.16f %.16f %.16f}\n", V(0, 0), V(0, 1), V(0, 2));
          printf("V(1) = {%.16f %.16f %.16f}\n", V(1, 0), V(1, 1), V(1, 2));
          printf("V(2) = {%.16f %.16f %.16f}\n", V(2, 0), V(2, 1), V(2, 2));
        });
      });
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);

  {
    printf("Running on host\n");
    call_svd_in_parallel_for<Kokkos::DefaultHostExecutionSpace>();
    Kokkos::fence();
    printf("Done\n");

    printf("Running on device\n");
    call_svd_in_parallel_for<Kokkos::DefaultExecutionSpace>();
    Kokkos::fence();
    printf("Done\n");
  }

  Kokkos::finalize();
  return 0;
}
