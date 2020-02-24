#include<gtest/gtest.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlasLapack_trtri.hpp>
#include<KokkosKernels_TestUtils.hpp>

namespace Test {

  template<class ViewTypeA, class ExecutionSpace>
  struct UnitDiagTRTRI {
    ViewTypeA A_;
    using ScalarA = typename ViewTypeA::value_type;

    UnitDiagTRTRI (const ViewTypeA& A) : A_(A) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const int& i) const {
      A_(i,i) = ScalarA(1);
    }
  };
  template<class ViewTypeA, class ExecutionSpace>
  struct NonUnitDiagTRTRI {
    ViewTypeA A_;
    using ScalarA = typename ViewTypeA::value_type;

    NonUnitDiagTRTRI (const ViewTypeA& A) : A_(A) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const int& i) const {
      A_(i,i) = A_(i,i)+10;
    }
  };
  template<class ViewTypeA, class ViewTypeB, class ViewTypeC, class ExecutionSpace>
  struct VanillaGEMM {
    bool A_t, B_t, A_c, B_c;
    int N,K;
    ViewTypeA A;
    ViewTypeB B;
    ViewTypeC C;

    typedef typename ViewTypeA::value_type ScalarA;
    typedef typename ViewTypeB::value_type ScalarB;
    typedef typename ViewTypeC::value_type ScalarC;
    typedef Kokkos::Details::ArithTraits<ScalarC> APT;
    typedef typename APT::mag_type mag_type;
    ScalarA alpha;
    ScalarC beta;

    KOKKOS_INLINE_FUNCTION
    void operator() (const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type& team) const {
// GNU COMPILER BUG WORKAROUND
#if defined(KOKKOS_COMPILER_GNU) && !defined(__CUDA_ARCH__)
      int i = team.league_rank();
#else
      const int i = team.league_rank();
#endif
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,N), [&] (const int& j) {
        ScalarC C_ij = 0.0;

        // GNU 5.3, 5.4 and 6.1 (and maybe more) crash with another nested lambda here

#if defined(KOKKOS_COMPILER_GNU) && !defined(KOKKOS_COMPILER_NVCC)
        for(int k=0; k<K; k++) {
          ScalarA A_ik = A_t?(A_c?APT::conj(A(k,i)):A(k,i)):A(i,k);
          ScalarB B_kj = B_t?(B_c?APT::conj(B(j,k)):B(j,k)):B(k,j);
          C_ij += A_ik*B_kj;
        }
#else
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,K), [&] (const int& k, ScalarC& lsum) {
           ScalarA A_ik = A_t?(A_c?APT::conj(A(k,i)):A(k,i)):A(i,k);
           ScalarB B_kj = B_t?(B_c?APT::conj(B(j,k)):B(j,k)):B(k,j);
           lsum += A_ik*B_kj;
        },C_ij);
#endif

        C(i,j) = beta*C(i,j) + alpha*C_ij;
      });
    }
  };

  template<class ViewTypeA, class Device>
  int impl_test_trtri(const char* uplo, const char* diag, 
                      const int M, const int N) {

    using execution_space = typename ViewTypeA::device_type::execution_space;
    using ScalarA         = typename ViewTypeA::value_type;
    using APT             = Kokkos::Details::ArithTraits<ScalarA>;
    using mag_type        = typename APT::mag_type;

    double machine_eps = APT::epsilon();
    const mag_type eps = 1.0e8 * machine_eps; //~1e-10 for double
    bool A_l = (uplo[0]=='L') || (uplo[0]=='l');
    int ret;
    ViewTypeA A ("A", M,N);
    ViewTypeA A_original ("A_original", M,N);
    ViewTypeA A_I ("A_I", M,N); // is A_I taken...?
    uint64_t seed = Kokkos::Impl::clock_tic();
    ScalarA beta       = ScalarA(0);
    ScalarA cur_check_val; // Either 1 or 0, to check A_I

    printf("KokkosBlas::trtri test for %c %c, M %d, N %d, eps %g, ViewType: %s START\n", uplo[0],diag[0],M,N,eps,typeid(ViewTypeA).name());

    if (M != N)
      return KokkosBlas::trtri(uplo, diag, A);

    typename ViewTypeA::HostMirror host_A  = Kokkos::create_mirror_view(A);
    typename ViewTypeA::HostMirror host_I  = Kokkos::create_mirror_view(A);

    Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed);

    // Initialize A with deterministic random numbers
    Kokkos::fill_random(A, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<execution_space>, ScalarA>::max()+1);
    if((diag[0]=='U')||(diag[0]=='u')) {
      using functor_type = UnitDiagTRTRI<ViewTypeA,execution_space>;
      functor_type udtrtri(A);
      // Initialize As diag with 1s
      Kokkos::parallel_for("KokkosBlas::Test::UnitDiagTRTRI", Kokkos::RangePolicy<execution_space>(0,M), udtrtri);
    } else {//(diag[0]=='N')||(diag[0]=='n')
      using functor_type = NonUnitDiagTRTRI<ViewTypeA,execution_space>;
      functor_type nudtrtri(A);
      // Initialize As diag with A(i,i)+10
      Kokkos::parallel_for("KokkosBlas::Test::NonUnitDiagTRTRI", Kokkos::RangePolicy<execution_space>(0,M), nudtrtri);
    }
    Kokkos::fence();
    Kokkos::deep_copy(host_A,  A);
    // Make host_A a lower triangle
    if (A_l) {
      for (int i = 0; i < M-1; i++)
        for (int j = i+1; j < N; j++)
          host_A(i,j) = ScalarA(0);
    }
    else {
      // Make host_A a upper triangle
      for (int i = 1; i < M; i++)
        for (int j = 0; j < i; j++)
          host_A(i,j) = ScalarA(0); 
    }
    Kokkos::deep_copy(A, host_A);
    Kokkos::deep_copy(A_original, A);

    // A = A^-1
    ret = KokkosBlas::trtri(uplo, diag, A);
    Kokkos::fence();

    if (ret) {
      printf("KokkosBlas::trtri(%c, %c, %s) returned %d\n", uplo[0],diag[0],typeid(ViewTypeA).name(), ret);
      return ret;
    }

    // A_I = A_original * A
    struct VanillaGEMM<ViewTypeA,ViewTypeA,ViewTypeA,execution_space> vgemm;
    vgemm.A_t = false; vgemm.B_t = false;
    vgemm.A_c = false; vgemm.B_c = false;
    vgemm.N = N;    vgemm.K = M;
    vgemm.A = A;    vgemm.B = A_original;
    vgemm.C = A_I; // out
    vgemm.alpha = ScalarA(1);
    vgemm.beta = beta;
    Kokkos::parallel_for("KokkosBlas::Test::VanillaGEMM", Kokkos::TeamPolicy<execution_space>(M,Kokkos::AUTO,16), vgemm);
    Kokkos::fence();
    Kokkos::deep_copy(host_I, A_I);

    bool test_flag = true;
    for (int i=0; i<M; i++) {
      for (int j=0; j<N; j++) {
        // Set check value
        cur_check_val = (i==j) ? ScalarA(1) : ScalarA(0);//APT::abs(host_A(i,j));

        // Check how close |A_I - cur_check_val| is to 0.
        if (APT::abs(APT::abs(host_I(i,j)) - cur_check_val) > eps) {
            test_flag = false;
            printf("   Error: eps ( %g ), host_I ( %.15lf ) != cur_check_val ( %.15lf ) (abs result-cur_check_val %g) at (i %d, j %d)\n", 
                  eps, APT::abs(host_I(i,j)), cur_check_val, APT::abs(host_I(i,j) - cur_check_val), i, j);
            break;
        }
      }
      if (!test_flag) break;
    }
    EXPECT_EQ( test_flag, true );
    return ret;
  }
}

template<class ScalarA, class Device>
int test_trtri(const char* mode) {
  int ret;
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  using view_type_a = Kokkos::View<ScalarA**, Kokkos::LayoutLeft, Device>;
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  using view_type_a = Kokkos::View<ScalarA**, Kokkos::LayoutRight, Device>;
#endif

  ret = Test::impl_test_trtri<view_type_a, Device>(&mode[0],&mode[1],0,0);
  EXPECT_EQ(ret, 0);

  ret = Test::impl_test_trtri<view_type_a, Device>(&mode[0],&mode[1],1,1);
  EXPECT_EQ(ret, 0);

  ret = Test::impl_test_trtri<view_type_a, Device>(&mode[0],&mode[1],473,473);
  EXPECT_EQ(ret, 0);

  ret = Test::impl_test_trtri<view_type_a, Device>(&mode[0],&mode[1],1002,1002);
  EXPECT_EQ(ret, 0);

  // One time check, disabled due to runtime throw instead of return here
  //ret = Test::impl_test_trtri<view_type_a, Device>(&mode[0],&mode[1],1031,731);
  //EXPECT_NE(ret, 0);

  return 1;
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, trtri_float ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::trtri_float");
    test_trtri<float,TestExecSpace> ("UN");
    test_trtri<float,TestExecSpace> ("UU");
    test_trtri<float,TestExecSpace> ("LN");
    test_trtri<float,TestExecSpace> ("LU");
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, trtri_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::trtri_double");
    test_trtri<double,TestExecSpace> ("UN");
    test_trtri<double,TestExecSpace> ("UU");
    test_trtri<double,TestExecSpace> ("LN");
    test_trtri<double,TestExecSpace> ("LU");
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, trtri_complex_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::trtri_complex_double");
    test_trtri<Kokkos::complex<double>,TestExecSpace> ("UN");
    test_trtri<Kokkos::complex<double>,TestExecSpace> ("UU");
    test_trtri<Kokkos::complex<double>,TestExecSpace> ("LN");
    test_trtri<Kokkos::complex<double>,TestExecSpace> ("LU");
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, trtri_complex_float ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::trtri_complex_float");
    test_trtri<Kokkos::complex<float>,TestExecSpace> ("UN");
    test_trtri<Kokkos::complex<float>,TestExecSpace> ("UU");
    test_trtri<Kokkos::complex<float>,TestExecSpace> ("LN");
    test_trtri<Kokkos::complex<float>,TestExecSpace> ("LU");
  Kokkos::Profiling::popRegion();
}
#endif
