#include<gtest/gtest.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas3_gemm.hpp>
#include<KokkosKernels_TestUtils.hpp>

namespace Test {
  template<class ViewTypeA, class ViewTypeB, class ViewTypeC, class Device>
  void impl_test_gemm(const char* TA, const char* TB, int N, int M, int K) {

    bool A_t = TA[0]!='N';
    bool B_t = TB[0]!='N';
    typedef typename ViewTypeA::device_type::execution_space execution_space;
    typedef typename ViewTypeA::value_type ScalarA;
    typedef typename ViewTypeB::value_type ScalarB;
    typedef typename ViewTypeC::value_type ScalarC;

    ScalarA alpha = 3;
    ScalarC beta = 5;
    double eps = std::is_same<ScalarC,float>::value?1e-7:1e-15;

    ViewTypeA A("A",A_t?K:N,A_t?N:K);
    ViewTypeB B("B",B_t?M:K,B_t?K:M);
    ViewTypeC C("C",N,M);
    ViewTypeC C2("C",N,M);

    Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);

    Kokkos::fill_random(A,rand_pool,ScalarA(10));
    Kokkos::fill_random(B,rand_pool,ScalarB(10));
    Kokkos::fill_random(C,rand_pool,ScalarC(10));
    
    Kokkos::deep_copy(C2,C);

    Kokkos::fence();
 

    Kokkos::parallel_for(Kokkos::TeamPolicy<execution_space>(N,Kokkos::AUTO,16), 
      KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<execution_space>::member_type& team) {
      const int i = team.league_rank();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,M), [&] (const int& j) {
        ScalarC C_ij = 0.0;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,K), [&] (const int& k, ScalarC& lsum) {
           ScalarA A_ik = A_t?A(k,i):A(i,k);
           ScalarB B_kj = B_t?B(j,k):B(k,j);
           lsum += A_ik*B_kj;
        },C_ij);
        C2(i,j) = beta*C2(i,j) + alpha*C_ij;
      });
    });
    
    KokkosBlas::gemm(TA,TB,alpha,A,B,beta,C);

    Kokkos::fence();

    ScalarC diff_C = 0;
    Kokkos::parallel_reduce(Kokkos::TeamPolicy<execution_space>(N,Kokkos::AUTO), 
      KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<execution_space>::member_type& team, ScalarC& diff) {
      const int i = team.league_rank();
      ScalarC diff_row = 0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,M), [&] (const int& j,ScalarC& diff_ij) {
        diff_ij += (C(i,j) - C2(i,j)) * (C(i,j) - C2(i,j));
      },diff_row);
      diff += diff_row;
    },diff_C);
    
    printf("%i %i %i %e %e: %e\n",N,M,K,diff_C,eps,sqrt(diff_C)/eps);
    //EXPECT_NEAR_KK( const_const_result, expected_result, eps*expected_result);
  }
}



template<class ScalarA, class ScalarB, class ScalarC, class Device>
int test_gemm(const char* mode) {

#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutLeft, Device> view_type_a_ll;
  typedef Kokkos::View<ScalarB**, Kokkos::LayoutLeft, Device> view_type_b_ll;
  typedef Kokkos::View<ScalarC**, Kokkos::LayoutLeft, Device> view_type_c_ll;
  Test::impl_test_gemm<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(&mode[0],&mode[1],0,0,0);
  //Test::impl_test_gemm<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(&mode[0],&mode[1],13,15,17);
  //Test::impl_test_gemm<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(&mode[0],&mode[1],179,15,211);
  //Test::impl_test_gemm<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(&mode[0],&mode[1],1031,3071,1024);
  //Test::impl_test_gemm<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(&mode[0],&mode[1],1031,1031,1031);
  //Test::impl_test_gemm<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(&mode[0],&mode[1],2048,2048,2048);
#endif
/*
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutRight, Device> view_type_a_lr;
  typedef Kokkos::View<ScalarX*, Kokkos::LayoutRight, Device> view_type_b_lr;
  typedef Kokkos::View<ScalarY*, Kokkos::LayoutRight, Device> view_type_c_lr;
  Test::impl_test_gemv<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(mode,0,1024);
  Test::impl_test_gemv<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(mode,13,1024);
  Test::impl_test_gemv<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(mode,1024,1024);
  Test::impl_test_gemv<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(mode,132231,1024);
#endif

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
/*
#if defined(KOKKOSKERNELS_INST_FLOAT) || defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, gemm_float ) {
    test_gemm<float,float,float,TestExecSpace> ("NN");
}
#endif
*/
#if defined(KOKKOSKERNELS_INST_DOUBLE) || defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, gemm_double ) {
    test_gemm<double,double,double,TestExecSpace> ("NN");
}
#endif
/*
#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, gemv_complex_double ) {
    test_gemm<Kokkos::complex<double>,Kokkos::complex<double>,Kokkos::complex<double>,TestExecSpace> ("N");
}
#endif

#if defined(KOKKOSKERNELS_INST_INT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, gemv_int ) {
    test_gemm<int,int,int,TestExecSpace> ("N");
}
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
TEST_F( TestCategory, gemv_double_int ) {
    test_gemm<double,int,float,TestExecSpace> ("N");
}
#endif
*/
