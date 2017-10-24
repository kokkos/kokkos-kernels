#include<gtest/gtest.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas3_gemm.hpp>
#include<KokkosKernels_TestUtils.hpp>

namespace Test {
  template<class ViewTypeA, class ViewTypeB, class ViewTypeC, class Device>
  void impl_test_gemm(const char* TA, const char* TB, int M, int N, int K) {


    bool A_t = (TA[0]!='N') && (TA[0]!='n');
    bool B_t = (TB[0]!='N') && (TB[0]!='n');
    bool A_c = (TA[0]=='C') || (TA[0]=='c');
    bool B_c = (TB[0]=='C') || (TB[0]=='c');
    typedef typename ViewTypeA::device_type::execution_space execution_space;
    typedef typename ViewTypeA::value_type ScalarA;
    typedef typename ViewTypeB::value_type ScalarB;
    typedef typename ViewTypeC::value_type ScalarC;
    typedef Kokkos::Details::ArithTraits<ScalarC> APT;
    typedef typename APT::mag_type mag_type;

    ScalarA alpha = 3;
    ScalarC beta = 5;
    double machine_eps = APT::epsilon();

    ViewTypeA A("A",A_t?K:M,A_t?M:K);
    ViewTypeB B("B",B_t?N:K,B_t?K:N);
    ViewTypeC C("C",M,N);
    ViewTypeC C2("C",M,N);

    uint64_t seed = Kokkos::Impl::clock_tic();
    Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed);

    Kokkos::fill_random(A,rand_pool,ScalarA(10));
    Kokkos::fill_random(B,rand_pool,ScalarB(10));
    Kokkos::fill_random(C,rand_pool,ScalarC(10));
    
    Kokkos::deep_copy(C2,C);

    Kokkos::fence();
 

    Kokkos::parallel_for(Kokkos::TeamPolicy<execution_space>(M,Kokkos::AUTO,16),
      KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<execution_space>::member_type& team) {
      const int i = team.league_rank();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,N), [=] (const int& j) {
        ScalarC C_ij = 0.0;

        // GNU 5.3, 5.4 and 6.1 (and maybe more) crash with another nested lambda here
#ifdef KOKKOS_COMPILER_GNU
        for(int k=0; k<K; k++) {
          ScalarA A_ik = A_t?(A_c?APT::conj(A(k,i)):A(k,i)):A(i,k);
          ScalarB B_kj = B_t?(B_c?APT::conj(B(j,k)):B(j,k)):B(k,j);
          C_ij += A_ik*B_kj;
        }
#else
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,K), [=] (const int& k, ScalarC& lsum) {
           ScalarA A_ik = A_t?(A_c?APT::conj(A(k,i)):A(k,i)):A(i,k);
           ScalarB B_kj = B_t?(B_c?APT::conj(B(j,k)):B(j,k)):B(k,j);
           lsum += A_ik*B_kj;
        },C_ij);
#endif

        C2(i,j) = beta*C2(i,j) + alpha*C_ij;
      });
    });
    int strides[8];
    A.stride(strides);
    const int LDA = strides[1];
    B.stride(strides);
    const int LDB = strides[1];
    C.stride(strides);
    const int LDC = strides[1];

    KokkosBlas::gemm(TA,TB,alpha,A,B,beta,C);

    Kokkos::fence();

    mag_type diff_C = 0;
    Kokkos::parallel_reduce(Kokkos::TeamPolicy<execution_space>(M,Kokkos::AUTO),
      KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<execution_space>::member_type& team, mag_type& diff) {
      const int i = team.league_rank();
      mag_type diff_row = 0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,N), [&] (const int& j,mag_type& diff_ij) {
        diff_ij += APT::abs(C(i,j)-C2(i,j));//sqrt((C(i,j) - C2(i,j)) * (C(i,j) - C2(i,j)));
      },diff_row);
      Kokkos::single(Kokkos::PerTeam(team), [&] () {
        diff += diff_row;
      });
    },diff_C);
    
    mag_type abs_C = 0;
    Kokkos::parallel_reduce(Kokkos::TeamPolicy<execution_space>(M,Kokkos::AUTO),
      KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<execution_space>::member_type& team, mag_type& abs_col) {
      const int i = team.league_rank();
      mag_type abs_row = 0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,N), [&] (const int& j,mag_type& abs_ij) {
        abs_ij += APT::abs(C(i,j));
      },abs_row);
      Kokkos::single(Kokkos::PerTeam(team), [&] () {
        abs_col += abs_row;
      });
    },abs_C);


    if( N!=0 && M!=0 && K!=0 ) {
      double diff_C_average = diff_C/(N*M);
      // Expected Result: Random Walk in the least significant bit (i.e. ~ sqrt(K)*eps
      // eps scales with the total sum and has a factor in it for the accuracy of the operations ->
      // eps = K * 75 * machine_eps * 7
      double diff_C_expected = 1.0*sqrt(K)*K*75*machine_eps*7;

      //printf("Result: %e %e\n",diff_C_average,diff_C_expected);
      EXPECT_TRUE( (diff_C_average < 1.05*diff_C_expected ) );
    }
  }
}



template<class ScalarA, class ScalarB, class ScalarC, class Device>
int test_gemm(const char* mode) {

#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutLeft, Device> view_type_a_ll;
  typedef Kokkos::View<ScalarB**, Kokkos::LayoutLeft, Device> view_type_b_ll;
  typedef Kokkos::View<ScalarC**, Kokkos::LayoutLeft, Device> view_type_c_ll;
  Test::impl_test_gemm<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(&mode[0],&mode[1],0,0,0);
  Test::impl_test_gemm<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(&mode[0],&mode[1],13,15,17);
  Test::impl_test_gemm<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(&mode[0],&mode[1],179,15,211);
  Test::impl_test_gemm<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(&mode[0],&mode[1],12,3071,517);
  Test::impl_test_gemm<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(&mode[0],&mode[1],1024,1024,2048);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutRight, Device> view_type_a_lr;
  typedef Kokkos::View<ScalarB**, Kokkos::LayoutRight, Device> view_type_b_lr;
  typedef Kokkos::View<ScalarC**, Kokkos::LayoutRight, Device> view_type_c_lr;
  Test::impl_test_gemm<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(&mode[0],&mode[1],0,0,0);
  Test::impl_test_gemm<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(&mode[0],&mode[1],13,15,17);
  Test::impl_test_gemm<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(&mode[0],&mode[1],179,15,211);
  Test::impl_test_gemm<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(&mode[0],&mode[1],12,3071,517);
  Test::impl_test_gemm<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(&mode[0],&mode[1],1024,1024,2048);
#endif
/*
#if defined(KOKKOSKERNELS_INST_LAYOUTSTRIDE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutStride, Device> view_type_a_ls;
  typedef Kokkos::View<ScalarX*, Kokkos::LayoutStride, Device> view_type_b_ls;
  typedef Kokkos::View<ScalarY*, Kokkos::LayoutStride, Device> view_type_c_ls;
  Test::impl_test_gemv<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(mode,0,1024);
  Test::impl_test_gemv<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(mode,13,1024);
  Test::impl_test_gemv<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(mode,1024,1024);
  Test::impl_test_gemv<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(mode,132231,1024);
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
  Test::impl_test_gemv<view_type_a_ls, view_type_b_ll, view_type_c_lr, Device>(mode,1024,1024);
  Test::impl_test_gemv<view_type_a_ll, view_type_b_ls, view_type_c_lr, Device>(mode,1024,1024);
#endif
*/
  return 1;
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, gemm_float ) {
    test_gemm<float,float,float,TestExecSpace> ("NN");
    test_gemm<float,float,float,TestExecSpace> ("TN");
    test_gemm<float,float,float,TestExecSpace> ("NT");
    test_gemm<float,float,float,TestExecSpace> ("TT");
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, gemm_double ) {
    test_gemm<double,double,double,TestExecSpace> ("NN");
    test_gemm<double,double,double,TestExecSpace> ("TN");
    test_gemm<double,double,double,TestExecSpace> ("NT");
    test_gemm<double,double,double,TestExecSpace> ("TT");
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, gemm_complex_double ) {
    test_gemm<Kokkos::complex<double>,Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("NN");
    test_gemm<Kokkos::complex<double>,Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("CN");
    test_gemm<Kokkos::complex<double>,Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("NC");
    test_gemm<Kokkos::complex<double>,Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("CC");
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, gemm_complex_float ) {
    test_gemm<Kokkos::complex<float>,Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("NN");
    test_gemm<Kokkos::complex<float>,Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("CN");
    test_gemm<Kokkos::complex<float>,Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("NC");
    test_gemm<Kokkos::complex<float>,Kokkos::complex<float>,Kokkos::complex<float>,TestExecSpace> ("CC");
}
#endif

/*
#if defined(KOKKOSKERNELS_INST_INT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, gemm_int ) {
    test_gemm<int,int,int,TestExecSpace> ("N");
}
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
TEST_F( TestCategory, gemm_double_int ) {
    test_gemm<double,int,float,TestExecSpace> ("N");
}
#endif
*/
