#include<gtest/gtest.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<Kokkos_Blas1_MV.hpp>

namespace Test {
  template<class ViewTypeA, class ViewTypeB, class Device>
  void impl_test_dot(int N) {

    typedef typename ViewTypeA::value_type ScalarA;
    typedef typename ViewTypeB::value_type ScalarB;

    typedef Kokkos::View<ScalarA*[2],
       typename std::conditional<
                std::is_same<typename ViewTypeA::array_layout,Kokkos::LayoutStride>::value,
                Kokkos::LayoutRight, Kokkos::LayoutLeft>::type,Device> BaseTypeA;
    typedef Kokkos::View<ScalarB*[2],
       typename std::conditional<
                std::is_same<typename ViewTypeB::array_layout,Kokkos::LayoutStride>::value,
                Kokkos::LayoutRight, Kokkos::LayoutLeft>::type,Device> BaseTypeB;


    BaseTypeA b_a("A",N);
    BaseTypeB b_b("B",N);

    ViewTypeA a = Kokkos::subview(b_a,Kokkos::ALL(),0);
    ViewTypeB b = Kokkos::subview(b_b,Kokkos::ALL(),0);

    typename BaseTypeA::HostMirror h_b_a = Kokkos::create_mirror_view(b_a);
    typename BaseTypeB::HostMirror h_b_b = Kokkos::create_mirror_view(b_b);

    typename ViewTypeA::HostMirror h_a = Kokkos::subview(h_b_a,Kokkos::ALL(),0);
    typename ViewTypeB::HostMirror h_b = Kokkos::subview(h_b_b,Kokkos::ALL(),0);

    Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);

    Kokkos::fill_random(b_a,rand_pool,ScalarA(10));
    Kokkos::fill_random(b_b,rand_pool,ScalarB(10));

    Kokkos::deep_copy(h_b_a,b_a);
    Kokkos::deep_copy(h_b_b,b_b);

    ScalarA expected_result = 0;
    for(int i=0;i<N;i++)
      expected_result += h_a(i)*h_b(i);

    ScalarA nonconst_nonconst_result = KokkosBlas::dot(a,b);
    double eps = std::is_same<ScalarA,float>::value?2*1e-5:1e-7;
    EXPECT_NEAR( nonconst_nonconst_result, expected_result, eps*expected_result);
    typename ViewTypeA::const_type c_a = a;
    typename ViewTypeB::const_type c_b = b;

    ScalarA const_const_result = KokkosBlas::dot(c_a,c_b);
    EXPECT_NEAR( const_const_result, expected_result, eps*expected_result);

    ScalarA nonconst_const_result = KokkosBlas::dot(a,c_b);
    EXPECT_NEAR( nonconst_const_result, expected_result, eps*expected_result);

    ScalarA const_nonconst_result = KokkosBlas::dot(c_a,b);
    EXPECT_NEAR( const_nonconst_result, expected_result, eps*expected_result);
  }
}

template<class ScalarA, class ScalarB, class Device>
int test_dot() {
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutLeft, Device> view_type_a_ll;
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutRight, Device> view_type_a_lr;
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutStride, Device> view_type_a_ls;
  typedef Kokkos::View<ScalarB*, Kokkos::LayoutLeft, Device> view_type_b_ll;
  typedef Kokkos::View<ScalarB*, Kokkos::LayoutRight, Device> view_type_b_lr;
  typedef Kokkos::View<ScalarB*, Kokkos::LayoutStride, Device> view_type_b_ls;

  Test::impl_test_dot<view_type_a_ll, view_type_b_ll, Device>(0);
  Test::impl_test_dot<view_type_a_lr, view_type_b_lr, Device>(0);
  Test::impl_test_dot<view_type_a_ls, view_type_b_ls, Device>(0);

  Test::impl_test_dot<view_type_a_ll, view_type_b_ll, Device>(13);
  Test::impl_test_dot<view_type_a_lr, view_type_b_lr, Device>(13);
  Test::impl_test_dot<view_type_a_ls, view_type_b_ls, Device>(13);

  Test::impl_test_dot<view_type_a_ll, view_type_b_ll, Device>(1024);
  Test::impl_test_dot<view_type_a_lr, view_type_b_lr, Device>(1024);
  Test::impl_test_dot<view_type_a_ls, view_type_b_ls, Device>(1024);

  Test::impl_test_dot<view_type_a_ll, view_type_b_ll, Device>(132231);
  Test::impl_test_dot<view_type_a_lr, view_type_b_lr, Device>(132231);
  Test::impl_test_dot<view_type_a_ls, view_type_b_ls, Device>(132231);

  #if !defined(KOKKOSKERNELS_ETI_ONLY)
  Test::impl_test_dot<view_type_a_ls, view_type_b_ll, Device>(1024);
  Test::impl_test_dot<view_type_a_ll, view_type_b_ls, Device>(1024);
  #endif

  return 1;
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || !defined(KOKKOSKERNELS_ETI_ONLY)
TEST_F( TestCategory, dot_float ) {
    test_dot<float,float,TestExecSpace> ();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || !defined(KOKKOSKERNELS_ETI_ONLY)
TEST_F( TestCategory, dot_double ) {
    test_dot<double,double,TestExecSpace> ();
}
#endif

#if defined(KOKKOSKERNELS_INST_INT) || !defined(KOKKOSKERNELS_ETI_ONLY)
TEST_F( TestCategory, dot_int ) {
    test_dot<int,int,TestExecSpace> ();
}
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY)
TEST_F( TestCategory, dot_double_int ) {
    test_dot<double,int,TestExecSpace> ();
}
#endif
