#include<gtest/gtest.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas_geqrf.hpp>
#include<KokkosBlas_unmqr.hpp>
#include<KokkosKernels_TestUtils.hpp>

namespace Test {

  template<class ViewTypeC, class ExecutionSpace>
  struct DiffGEMM_QR {
    int N;
    ViewTypeC C,C2;

    typedef typename ViewTypeC::value_type ScalarC;
    typedef Kokkos::Details::ArithTraits<ScalarC> APT;
    typedef typename APT::mag_type mag_type;

    KOKKOS_INLINE_FUNCTION
    void operator() (const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type& team, mag_type& diff) const {
      const int i = team.league_rank();
      mag_type diff_row = 0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,N), [&] (const int& j,mag_type& diff_ij) {
        //printf("A (%i %i) (%i %i) (%i %i)\n",C.extent(0),C.extent(1),C2.extent(0),C2.extent(1),i,j);
        diff_ij += APT::abs(C(i,j)-C2(i,j));
        //printf("B (%i %i) (%i %i) (%i %i)\n",C.extent(0),C.extent(1),C2.extent(0),C2.extent(1),i,j);
      },diff_row);
      Kokkos::single(Kokkos::PerTeam(team), [&] () {
        diff += diff_row;
      });
    }
  };

  template<class ViewTypeC, class ExecutionSpace>
  struct Identity_QR {
    int N;
    ViewTypeC C;

    typedef typename ViewTypeC::value_type ScalarC;
    typedef Kokkos::Details::ArithTraits<ScalarC> APT;
    typedef typename APT::mag_type mag_type;

    KOKKOS_INLINE_FUNCTION
    void operator() (const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type& team) const {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,N), [&] (const int& j) {
        //printf("A (%i %i) (%i %i) (%i %i)\n",C.extent(0),C.extent(1),C2.extent(0),C2.extent(1),i,j);
        const int i = team.league_rank();
        const ScalarC one = 1.0;
        const ScalarC zero = 0.0;
        if(i == j){
          C(i,j) = one;
        }
        else{
          C(i, j) = zero;
        }
      });
    }
  };

  template<class ViewTypeC, class ExecutionSpace>
  struct CopyUpper_QR {
    int N;
    ViewTypeC C;

    typedef typename ViewTypeC::value_type ScalarC;
    typedef Kokkos::Details::ArithTraits<ScalarC> APT;
    typedef typename APT::mag_type mag_type;

    KOKKOS_INLINE_FUNCTION
    void operator() (const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type& team) const {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,N), [&] (const int& j) {
        //printf("A (%i %i) (%i %i) (%i %i)\n",C.extent(0),C.extent(1),C2.extent(0),C2.extent(1),i,j);
        const int i = team.league_rank();
        const ScalarC zero = 0.0;
        if(j < i){
          C(i,j) = zero;
        }
      });
    }
  };

  template<class ViewTypeA, class ViewTypeT, class Device>
  void impl_test_qr(int M, int N) {

    typedef typename ViewTypeA::device_type::execution_space execution_space;
    typedef typename ViewTypeA::value_type ScalarA;
    typedef Kokkos::Details::ArithTraits<ScalarA> APT;
    typedef typename APT::mag_type mag_type;

    double machine_eps = APT::epsilon();
    double eps = 10*machine_eps;

    ViewTypeA A("A",M,N);

    int minmn = M < N? M : N;

    ViewTypeA Aref("Aref", M, N);
    ViewTypeT T("Tau", minmn); 
    ViewTypeA Q("Q", M, M);
    ViewTypeA R("R", M, N);
    ViewTypeA Iref("Iref", M, M);

    typename ViewTypeA::HostMirror host_A        = Kokkos::create_mirror_view(Aref);
    typename ViewTypeA::HostMirror host_Q = Kokkos::create_mirror_view(Q);
    typename ViewTypeA::HostMirror host_Iref = Kokkos::create_mirror_view(Iref);

    uint64_t seed = Kokkos::Impl::clock_tic();
    Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed);

    Kokkos::fill_random(A,rand_pool, Kokkos::rand<typename Kokkos::Random_XorShift64_Pool<execution_space>::generator_type,ScalarA>::max());
    
    //Make Copy of A
    Kokkos::deep_copy(Aref, A);    

    //Take QR of A
    KokkosBlas::geqrf(A, T);

    //Extract upper portion of R
    Kokkos::deep_copy(R, A);
    struct CopyUpper_QR<ViewTypeA, execution_space> copy_upper;
    copy_upper.C = R;
    copy_upper.N = N;
    Kokkos::parallel_for("KokkosBlas::Test::CopyUpper", Kokkos::TeamPolicy<execution_space>(M,Kokkos::AUTO,16), copy_upper);

    //Fill Iref with Identity
    struct Identity_QR<ViewTypeA, execution_space> make_id;
    make_id.C = Iref;
    make_id.N = M;
    Kokkos::parallel_for("KokkosBlas::Test::Identity", Kokkos::TeamPolicy<execution_space>(M,Kokkos::AUTO,16), make_id);
    Kokkos::deep_copy(host_Iref, Iref);

    //Fill Q with Identity
    Kokkos::deep_copy(Q, Iref);

    //Compute Q @ R
    KokkosBlas::unmqr("L", "N", minmn, A, T, R);

    //Compare Aref with R
    mag_type diff = 0;
    struct DiffGEMM_QR<ViewTypeA,execution_space> diffgemm;
    diffgemm.N = N;
    diffgemm.C = Aref;
    diffgemm.C2 = R;
    Kokkos::parallel_reduce("KokkosBlas::Test::DiffGEMM", Kokkos::TeamPolicy<execution_space>(M,Kokkos::AUTO,16), diffgemm, diff);

    //Check Aref vs QR
    if( N!=0 && M!=0) {
      double diff_average = diff/(N*M);
      // Expected Result: Random Walk in the least significant bit (i.e. ~ sqrt(K)*eps
      // eps scales with the total sum and has a factor in it for the accuracy of the operations ->
      // eps = K * 75 * machine_eps * 7
      double diff_expected = 5*machine_eps;

      if ( (diff_average >= diff_expected ) ) {
        printf("Result: %e %e\n",diff_average,diff_expected);
      }

      EXPECT_TRUE( (diff_average < diff_expected ) );
    }

    //Compute QI = Q
    KokkosBlas::unmqr("L", "N", minmn, A, T, Q);

    //Compute Q^TQ = I
    KokkosBlas::unmqr("L", "T", minmn, A, T, Q);

    //Check Identity
    Kokkos::deep_copy(host_Q, Q);
    bool test_flag = true;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < M; j++) {
        if (APT::abs(host_Iref(i, j) - host_Q(i, j)) > eps) {
          test_flag = false;
          break;
        }
      }
      if (!test_flag) break;
    }
    ASSERT_EQ(test_flag, true);

    //Reset
    Kokkos::deep_copy(Q, Iref);
    //Compute IQ = Q
    KokkosBlas::unmqr("R", "N", minmn, A, T, Q);

    //Compute QQ^T = I
    KokkosBlas::unmqr("R", "T", minmn, A, T, Q);

    //Check Identity
    Kokkos::deep_copy(host_Q, Q);

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < M; j++) {
        if (APT::abs(host_Iref(i, j) - host_Q(i, j)) > eps) {
          test_flag = false;
          break;
        }
      }
      if (!test_flag) break;
    }
    ASSERT_EQ(test_flag, true);
  }

} //namespace Test


template<class ScalarA, class Device>
int test_qr() {

#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutLeft, Device> view_type_a_ll;
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutLeft, Device> view_type_b_ll;
  Test::impl_test_qr<view_type_a_ll, view_type_b_ll, Device>(0,0);
  Test::impl_test_qr<view_type_a_ll, view_type_b_ll, Device>(13,15);
  Test::impl_test_qr<view_type_a_ll, view_type_b_ll, Device>(179,15);
  Test::impl_test_qr<view_type_a_ll, view_type_b_ll, Device>(12,323);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutRight, Device> view_type_a_lr;
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutRight, Device> view_type_b_lr;
  Test::impl_test_qr<view_type_a_ll, view_type_b_ll, Device>(0,0);
  Test::impl_test_qr<view_type_a_ll, view_type_b_ll, Device>(13,15);
  Test::impl_test_qr<view_type_a_ll, view_type_b_ll, Device>(179,15);
  Test::impl_test_qr<view_type_a_ll, view_type_b_ll, Device>(12,323);
#endif

  return 1;
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, qr_float ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::qr_float");
    test_qr<float,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, qr_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::qr_double");
    test_qr<double,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, qr_complex_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::qr_complex_double");
    test_qr<Kokkos::complex<double>,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, qr_complex_float ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::qr_complex_float");
    test_qr<Kokkos::complex<float>,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
#endif

