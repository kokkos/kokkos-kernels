// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas1_mult.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {
template <class ViewTypeA, class ViewTypeB, class ViewTypeC, class Device>
void impl_test_mult(int N) {
  typedef typename ViewTypeA::value_type ScalarA;
  typedef typename ViewTypeB::value_type ScalarB;
  typedef typename ViewTypeC::value_type ScalarC;

  ScalarA a  = 3;
  ScalarB b  = 5;
  double eps = TestUtils::selectEpsilonForScalarType<ScalarC>(1e-7, 1e-4, 0.1);

  TestUtils::view_stride_adapter<ViewTypeA> x("X", N);
  TestUtils::view_stride_adapter<ViewTypeB> y("Y", N);
  TestUtils::view_stride_adapter<ViewTypeC> z("Z", N);
  TestUtils::view_stride_adapter<ViewTypeC> org_z("Org_Z", N);

  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);

  {
    ScalarA randStart, randEnd;
    double randMag = std::is_same<ScalarA, Kokkos::Experimental::half_t>::value ? 1.0 : 10.0;
    TestUtils::getRandomBounds(randMag, randStart, randEnd);
    Kokkos::fill_random(x.d_view, rand_pool, randStart, randEnd);
  }
  {
    ScalarB randStart, randEnd;
    double randMag = std::is_same<ScalarB, Kokkos::Experimental::half_t>::value ? 1.0 : 10.0;
    TestUtils::getRandomBounds(randMag, randStart, randEnd);
    Kokkos::fill_random(y.d_view, rand_pool, randStart, randEnd);
  }
  {
    ScalarC randStart, randEnd;
    double randMag = std::is_same<ScalarC, Kokkos::Experimental::half_t>::value ? 1.0 : 10.0;
    TestUtils::getRandomBounds(randMag, randStart, randEnd);
    Kokkos::fill_random(z.d_view, rand_pool, randStart, randEnd);
  }

  Kokkos::deep_copy(org_z.h_base, z.d_base);

  Kokkos::deep_copy(x.h_base, x.d_base);
  Kokkos::deep_copy(y.h_base, y.d_base);

  KokkosBlas::mult(b, z.d_view, a, x.d_view, y.d_view);
  Kokkos::deep_copy(z.h_base, z.d_base);
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR_KK(static_cast<ScalarC>(a * x.h_view(i) * y.h_view(i) + b * org_z.h_view(i)), z.h_view(i), eps);
  }

  Kokkos::deep_copy(z.d_base, org_z.h_base);
  KokkosBlas::mult(b, z.d_view, a, x.d_view, y.d_view_const);
  Kokkos::deep_copy(z.h_base, z.d_base);
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR_KK(static_cast<ScalarC>(a * x.h_view(i) * y.h_view(i) + b * org_z.h_view(i)), z.h_view(i), eps);
  }

  Kokkos::deep_copy(z.d_base, org_z.h_base);
  KokkosBlas::mult(b, z.d_view, a, x.d_view_const, y.d_view_const);
  Kokkos::deep_copy(z.h_base, z.d_base);
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR_KK(static_cast<ScalarC>(a * x.h_view(i) * y.h_view(i) + b * org_z.h_view(i)), z.h_view(i), eps);
  }
}

template <class ViewTypeA, class ViewTypeB, class ViewTypeC, class Device>
void impl_test_mult_mv(int N, int K) {
  typedef typename ViewTypeA::value_type ScalarA;
  typedef typename ViewTypeB::value_type ScalarB;
  typedef typename ViewTypeC::value_type ScalarC;

  // x is rank-1, all others are rank-2
  TestUtils::view_stride_adapter<ViewTypeA> x("X", N);
  TestUtils::view_stride_adapter<ViewTypeB> y("Y", N, K);
  TestUtils::view_stride_adapter<ViewTypeC> z("Z", N, K);
  TestUtils::view_stride_adapter<ViewTypeC> org_z("Org_Z", N, K);

  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);

  {
    ScalarA randStart, randEnd;
    double randMag = std::is_same<ScalarA, Kokkos::Experimental::half_t>::value ? 1.0 : 10.0;
    TestUtils::getRandomBounds(randMag, randStart, randEnd);
    Kokkos::fill_random(x.d_view, rand_pool, randStart, randEnd);
  }
  {
    ScalarB randStart, randEnd;
    double randMag = std::is_same<ScalarB, Kokkos::Experimental::half_t>::value ? 1.0 : 10.0;
    TestUtils::getRandomBounds(randMag, randStart, randEnd);
    Kokkos::fill_random(y.d_view, rand_pool, randStart, randEnd);
  }
  {
    ScalarC randStart, randEnd;
    double randMag = std::is_same<ScalarC, Kokkos::Experimental::half_t>::value ? 1.0 : 10.0;
    TestUtils::getRandomBounds(randMag, randStart, randEnd);
    Kokkos::fill_random(z.d_view, rand_pool, randStart, randEnd);
  }

  Kokkos::deep_copy(org_z.h_base, z.d_base);
  Kokkos::deep_copy(x.h_base, x.d_base);
  Kokkos::deep_copy(y.h_base, y.d_base);

  ScalarA a = 3;
  ScalarB b = 5;

  double eps = TestUtils::selectEpsilonForScalarType<ScalarA>(1e-7, 1e-4, 0.1);

  KokkosBlas::mult(b, z.d_view, a, x.d_view, y.d_view);
  Kokkos::deep_copy(z.h_base, z.d_base);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      EXPECT_NEAR_KK(static_cast<ScalarC>(a * x.h_view(i) * y.h_view(i, j) + b * org_z.h_view(i, j)), z.h_view(i, j),
                     eps);
    }
  }

  Kokkos::deep_copy(z.d_base, org_z.h_base);
  KokkosBlas::mult(b, z.d_view, a, x.d_view, y.d_view_const);
  Kokkos::deep_copy(z.h_base, z.d_base);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      EXPECT_NEAR_KK(static_cast<ScalarC>(a * x.h_view(i) * y.h_view(i, j) + b * org_z.h_view(i, j)), z.h_view(i, j),
                     eps);
    }
  }
}
}  // namespace Test

template <class ScalarA, class ScalarB, class ScalarC, class Device>
int test_mult() {
  const int Nbig = std::is_same<ScalarA, Kokkos::Experimental::half_t>::value ? 128 : 1024;
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutLeft, Device> view_type_a_ll;
  typedef Kokkos::View<ScalarB*, Kokkos::LayoutLeft, Device> view_type_b_ll;
  typedef Kokkos::View<ScalarC*, Kokkos::LayoutLeft, Device> view_type_c_ll;
  Test::impl_test_mult<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(0);
  Test::impl_test_mult<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(13);
  Test::impl_test_mult<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(Nbig);
  // Test::impl_test_mult<view_type_a_ll, view_type_b_ll, view_type_c_ll,
  // Device>(132231);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutRight, Device> view_type_a_lr;
  typedef Kokkos::View<ScalarB*, Kokkos::LayoutRight, Device> view_type_b_lr;
  typedef Kokkos::View<ScalarC*, Kokkos::LayoutRight, Device> view_type_c_lr;
  Test::impl_test_mult<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(0);
  Test::impl_test_mult<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(13);
  Test::impl_test_mult<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(Nbig);
  // Test::impl_test_mult<view_type_a_lr, view_type_b_lr, view_type_c_lr,
  // Device>(132231);
#endif

#if (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutStride, Device> view_type_a_ls;
  typedef Kokkos::View<ScalarB*, Kokkos::LayoutStride, Device> view_type_b_ls;
  typedef Kokkos::View<ScalarC*, Kokkos::LayoutStride, Device> view_type_c_ls;
  Test::impl_test_mult<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(0);
  Test::impl_test_mult<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(13);
  Test::impl_test_mult<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(Nbig);
  // Test::impl_test_mult<view_type_a_ls, view_type_b_ls, view_type_c_ls,
  // Device>(132231);
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
  Test::impl_test_mult<view_type_a_ls, view_type_b_ll, view_type_c_lr, Device>(Nbig);
  Test::impl_test_mult<view_type_a_ll, view_type_b_ls, view_type_c_lr, Device>(Nbig);
#endif

  return 1;
}

template <class ScalarA, class ScalarB, class ScalarC, class Device>
int test_mult_mv() {
  const int Nbig = std::is_same<ScalarA, Kokkos::Experimental::half_t>::value ? 128 : 1024;
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutLeft, Device> view_type_a_ll;
  typedef Kokkos::View<ScalarB**, Kokkos::LayoutLeft, Device> view_type_b_ll;
  typedef Kokkos::View<ScalarC**, Kokkos::LayoutLeft, Device> view_type_c_ll;
  Test::impl_test_mult_mv<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(0, 5);
  Test::impl_test_mult_mv<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(13, 5);
  Test::impl_test_mult_mv<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(Nbig, 5);
  // Test::impl_test_mult_mv<view_type_a_ll, view_type_b_ll, view_type_c_ll,
  // Device>(132231,5);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutRight, Device> view_type_a_lr;
  typedef Kokkos::View<ScalarB**, Kokkos::LayoutRight, Device> view_type_b_lr;
  typedef Kokkos::View<ScalarC**, Kokkos::LayoutRight, Device> view_type_c_lr;
  Test::impl_test_mult_mv<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(0, 5);
  Test::impl_test_mult_mv<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(13, 5);
  Test::impl_test_mult_mv<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(1024, 5);
  // Test::impl_test_mult_mv<view_type_a_lr, view_type_b_lr, view_type_c_lr,
  // Device>(132231,5);
#endif

#if (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutStride, Device> view_type_a_ls;
  typedef Kokkos::View<ScalarB**, Kokkos::LayoutStride, Device> view_type_b_ls;
  typedef Kokkos::View<ScalarC**, Kokkos::LayoutStride, Device> view_type_c_ls;
  Test::impl_test_mult_mv<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(0, 5);
  Test::impl_test_mult_mv<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(13, 5);
  Test::impl_test_mult_mv<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(Nbig, 5);
  // Test::impl_test_mult_mv<view_type_a_ls, view_type_b_ls, view_type_c_ls,
  // Device>(132231,5);
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
  Test::impl_test_mult_mv<view_type_a_ls, view_type_b_ll, view_type_c_lr, Device>(Nbig, 5);
  Test::impl_test_mult_mv<view_type_a_ll, view_type_b_ls, view_type_c_lr, Device>(Nbig, 5);
#endif

  return 1;
}

#if defined(KOKKOSKERNELS_INST_HALF) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, mult_half) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::mult_half");
  test_mult<Kokkos::Experimental::half_t, Kokkos::Experimental::half_t, Kokkos::Experimental::half_t, TestDevice>();
  Kokkos::Profiling::popRegion();
}
TEST_F(TestCategory, mult_mv_half) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::mult_mv_half");
  test_mult_mv<Kokkos::Experimental::half_t, Kokkos::Experimental::half_t, Kokkos::Experimental::half_t, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, mult_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::mult_float");
  test_mult<float, float, float, TestDevice>();
  Kokkos::Profiling::popRegion();
}
TEST_F(TestCategory, mult_mv_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::mult_mv_float");
  test_mult_mv<float, float, float, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, mult_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::mult_double");
  test_mult<double, double, double, TestDevice>();
  Kokkos::Profiling::popRegion();
}
TEST_F(TestCategory, mult_mv_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::mult_mv_double");
  test_mult_mv<double, double, double, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, mult_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::mult_complex_double");
  test_mult<Kokkos::complex<double>, Kokkos::complex<double>, Kokkos::complex<double>, TestDevice>();
  Kokkos::Profiling::popRegion();
}
TEST_F(TestCategory, mult_mv_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::mult_mv_complex_double");
  test_mult_mv<Kokkos::complex<double>, Kokkos::complex<double>, Kokkos::complex<double>, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_INT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, mult_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::mult_int");
  test_mult<int, int, int, TestDevice>();
  Kokkos::Profiling::popRegion();
}
TEST_F(TestCategory, mult_mv_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::mult_mv_int");
  test_mult_mv<int, int, int, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
TEST_F(TestCategory, mult_double_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::mult_double_int");
  test_mult<double, int, float, TestDevice>();
  Kokkos::Profiling::popRegion();
}
TEST_F(TestCategory, mult_mv_double_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::mult_mv_double_int");
  test_mult_mv<double, int, float, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif
