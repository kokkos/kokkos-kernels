#include<gtest/gtest.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<Kokkos_Blas1_MV.hpp>

namespace Test {
  template<class ViewType, class Device>
  void impl_test_dot(int N) {
    typedef typename ViewType::value_type Scalar;
    typedef Kokkos::View<Scalar*[2],
       typename std::conditional<
                std::is_same<typename ViewType::array_layout,Kokkos::LayoutStride>::value,
                Kokkos::LayoutRight, Kokkos::LayoutLeft>::type,Device> BaseType;
    BaseType b_a("A",N);
    BaseType b_b("B",N);

    ViewType a = Kokkos::subview(b_a,Kokkos::ALL(),0);
    ViewType b = Kokkos::subview(b_b,Kokkos::ALL(),0);

    typename BaseType::HostMirror h_b_a = Kokkos::create_mirror_view(b_a);
    typename BaseType::HostMirror h_b_b = Kokkos::create_mirror_view(b_b);

    typename ViewType::HostMirror h_a = Kokkos::subview(h_b_a,Kokkos::ALL(),0);
    typename ViewType::HostMirror h_b = Kokkos::subview(h_b_b,Kokkos::ALL(),0);

    Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);

    Kokkos::fill_random(b_a,rand_pool,Scalar(10));
    Kokkos::fill_random(b_b,rand_pool,Scalar(10));

    Kokkos::deep_copy(h_b_a,b_a);
    Kokkos::deep_copy(h_b_b,b_b);

    Scalar expected_result = 0;
    for(int i=0;i<N;i++)
      expected_result += h_a(i)*h_b(i);

    Scalar nonconst_nonconst_result = KokkosBlas::dot(a,b);
    double eps = std::is_same<Scalar,float>::value?2*1e-5:1e-7;
    EXPECT_NEAR( nonconst_nonconst_result, expected_result, eps*expected_result);
    typename ViewType::const_type c_a = a;
    typename ViewType::const_type c_b = b;

    Scalar const_const_result = KokkosBlas::dot(c_a,c_b);
    EXPECT_NEAR( const_const_result, expected_result, eps*expected_result);

    Scalar nonconst_const_result = KokkosBlas::dot(a,c_b);
    EXPECT_NEAR( nonconst_const_result, expected_result, eps*expected_result);

    Scalar const_nonconst_result = KokkosBlas::dot(c_a,b);
    EXPECT_NEAR( const_nonconst_result, expected_result, eps*expected_result);
  }
}

template<class Scalar, class Device>
int test_dot() {
  Test::impl_test_dot<Kokkos::View<Scalar*, Kokkos::LayoutLeft, Device>, Device>(13);
  Test::impl_test_dot<Kokkos::View<Scalar*, Kokkos::LayoutLeft, Device>, Device>(13);
  Test::impl_test_dot<Kokkos::View<Scalar*, Kokkos::LayoutLeft, Device>, Device>(13);

  Test::impl_test_dot<Kokkos::View<Scalar*, Kokkos::LayoutRight, Device>, Device>(1024);
  Test::impl_test_dot<Kokkos::View<Scalar*, Kokkos::LayoutRight, Device>, Device>(1024);
  Test::impl_test_dot<Kokkos::View<Scalar*, Kokkos::LayoutRight, Device>, Device>(1024);

  Test::impl_test_dot<Kokkos::View<Scalar*, Kokkos::LayoutStride, Device>, Device>(133131);
  Test::impl_test_dot<Kokkos::View<Scalar*, Kokkos::LayoutStride, Device>, Device>(133131);
  Test::impl_test_dot<Kokkos::View<Scalar*, Kokkos::LayoutStride, Device>, Device>(133131);
  return 1;
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || !defined(KOKKOSKERNELS_ETI_ONLY)
TEST_F( TestCategory, dot_float ) {
    test_dot<float,TestExecSpace> ();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || !defined(KOKKOSKERNELS_ETI_ONLY)
TEST_F( TestCategory, dot_double ) {
    test_dot<double,TestExecSpace> ();
}
#endif

#if defined(KOKKOSKERNELS_INST_INT) || !defined(KOKKOSKERNELS_ETI_ONLY)
TEST_F( TestCategory, dot_int ) {
    test_dot<int,TestExecSpace> ();
}
#endif
