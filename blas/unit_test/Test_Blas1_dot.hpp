// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosKernels_ArithTraits.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {
template <class ViewTypeA, class ViewTypeB, class Device>
void impl_test_dot(int N) {
  typedef typename ViewTypeA::value_type ScalarA;
  typedef typename ViewTypeB::value_type ScalarB;
  typedef KokkosKernels::ArithTraits<ScalarA> ats;

  TestUtils::view_stride_adapter<ViewTypeA> a("a", N);
  TestUtils::view_stride_adapter<ViewTypeB> b("b", N);

  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);

  {
    ScalarA randStart, randEnd;
    double randMag = std::is_same<ScalarA, Kokkos::Experimental::half_t>::value ? 1.0 : 10.0;
    TestUtils::getRandomBounds(randMag, randStart, randEnd);
    Kokkos::fill_random(a.d_view, rand_pool, randStart, randEnd);
  }
  {
    ScalarB randStart, randEnd;
    double randMag = std::is_same<ScalarB, Kokkos::Experimental::half_t>::value ? 1.0 : 10.0;
    TestUtils::getRandomBounds(randMag, randStart, randEnd);
    Kokkos::fill_random(b.d_view, rand_pool, randStart, randEnd);
  }

  Kokkos::deep_copy(a.h_base, a.d_base);
  Kokkos::deep_copy(b.h_base, b.d_base);

  ScalarA expected_result = 0;
  for (int i = 0; i < N; i++) expected_result += ats::conj(a.h_view(i)) * b.h_view(i);

  ScalarA nonconst_nonconst_result = KokkosBlas::dot(a.d_view, b.d_view);
  double eps                       = TestUtils::selectEpsilonForScalarType<ScalarA>(1e-7, 2e-5, 0.1);
  EXPECT_NEAR_KK(nonconst_nonconst_result, expected_result, eps * expected_result);

  ScalarA const_const_result = KokkosBlas::dot(a.d_view_const, b.d_view_const);
  EXPECT_NEAR_KK(const_const_result, expected_result, eps * expected_result);

  ScalarA nonconst_const_result = KokkosBlas::dot(a.d_view, b.d_view_const);
  EXPECT_NEAR_KK(nonconst_const_result, expected_result, eps * expected_result);

  ScalarA const_nonconst_result = KokkosBlas::dot(a.d_view_const, b.d_view);
  EXPECT_NEAR_KK(const_nonconst_result, expected_result, eps * expected_result);
}

template <class ViewTypeA, class ViewTypeB, class Device>
void impl_test_dot_mv(int N, int K) {
  typedef typename ViewTypeA::value_type ScalarA;
  typedef typename ViewTypeB::value_type ScalarB;
  typedef KokkosKernels::ArithTraits<ScalarA> ats;

  TestUtils::view_stride_adapter<ViewTypeA> a("A", N, K);
  TestUtils::view_stride_adapter<ViewTypeB> b("B", N, K);

  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);

  {
    ScalarA randStart, randEnd;
    double randMag = std::is_same<ScalarA, Kokkos::Experimental::half_t>::value ? 1.0 : 10.0;
    TestUtils::getRandomBounds(randMag, randStart, randEnd);
    Kokkos::fill_random(a.d_view, rand_pool, randStart, randEnd);
  }
  {
    ScalarB randStart, randEnd;
    double randMag = std::is_same<ScalarB, Kokkos::Experimental::half_t>::value ? 1.0 : 10.0;
    TestUtils::getRandomBounds(randMag, randStart, randEnd);
    Kokkos::fill_random(b.d_view, rand_pool, randStart, randEnd);
  }

  Kokkos::deep_copy(a.h_base, a.d_base);
  Kokkos::deep_copy(b.h_base, b.d_base);

  ScalarA* expected_result = new ScalarA[K];
  for (int j = 0; j < K; j++) {
    expected_result[j] = ScalarA();
    for (int i = 0; i < N; i++) expected_result[j] += ats::conj(a.h_view(i, j)) * b.h_view(i, j);
  }

  double eps = TestUtils::selectEpsilonForScalarType<ScalarA>(1e-7, 2e-5, 0.1);

  Kokkos::View<ScalarB*, Kokkos::HostSpace> r("Dot::Result", K);

  KokkosBlas::dot(r, a.d_view, b.d_view);
  Kokkos::fence();
  for (int k = 0; k < K; k++) {
    ScalarA nonconst_nonconst_result = r(k);
    EXPECT_NEAR_KK(nonconst_nonconst_result, expected_result[k], eps * expected_result[k]);
  }

  KokkosBlas::dot(r, a.d_view_const, b.d_view_const);
  Kokkos::fence();
  for (int k = 0; k < K; k++) {
    ScalarA const_const_result = r(k);
    EXPECT_NEAR_KK(const_const_result, expected_result[k], eps * expected_result[k]);
  }

  KokkosBlas::dot(r, a.d_view, b.d_view_const);
  Kokkos::fence();
  for (int k = 0; k < K; k++) {
    ScalarA non_const_const_result = r(k);
    EXPECT_NEAR_KK(non_const_const_result, expected_result[k], eps * expected_result[k]);
  }

  KokkosBlas::dot(r, a.d_view_const, b.d_view);
  Kokkos::fence();
  for (int k = 0; k < K; k++) {
    ScalarA const_non_const_result = r(k);
    EXPECT_NEAR_KK(const_non_const_result, expected_result[k], eps * expected_result[k]);
  }

  delete[] expected_result;
}
}  // namespace Test

template <class ScalarA, class ScalarB, class Device>
int test_dot() {
  const int Nbig = std::is_same<ScalarA, Kokkos::Experimental::half_t>::value ? 128 : 1024;
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutLeft, Device> view_type_a_ll;
  typedef Kokkos::View<ScalarB*, Kokkos::LayoutLeft, Device> view_type_b_ll;
  Test::impl_test_dot<view_type_a_ll, view_type_b_ll, Device>(0);
  Test::impl_test_dot<view_type_a_ll, view_type_b_ll, Device>(13);
  Test::impl_test_dot<view_type_a_ll, view_type_b_ll, Device>(Nbig);
  // Test::impl_test_dot<view_type_a_ll, view_type_b_ll, Device>(132231);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutRight, Device> view_type_a_lr;
  typedef Kokkos::View<ScalarB*, Kokkos::LayoutRight, Device> view_type_b_lr;
  Test::impl_test_dot<view_type_a_lr, view_type_b_lr, Device>(0);
  Test::impl_test_dot<view_type_a_lr, view_type_b_lr, Device>(13);
  Test::impl_test_dot<view_type_a_lr, view_type_b_lr, Device>(Nbig);
  // Test::impl_test_dot<view_type_a_lr, view_type_b_lr, Device>(132231);
#endif

#if (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutStride, Device> view_type_a_ls;
  typedef Kokkos::View<ScalarB*, Kokkos::LayoutStride, Device> view_type_b_ls;
  Test::impl_test_dot<view_type_a_ls, view_type_b_ls, Device>(0);
  Test::impl_test_dot<view_type_a_ls, view_type_b_ls, Device>(13);
  Test::impl_test_dot<view_type_a_ls, view_type_b_ls, Device>(Nbig);
  // Test::impl_test_dot<view_type_a_ls, view_type_b_ls, Device>(132231);
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
  Test::impl_test_dot<view_type_a_ls, view_type_b_ll, Device>(Nbig);
  Test::impl_test_dot<view_type_a_ll, view_type_b_ls, Device>(Nbig);
#endif

  return 1;
}

template <class ScalarA, class ScalarB, class Device>
int test_dot_mv() {
  const int Nbig = std::is_same<ScalarA, Kokkos::Experimental::half_t>::value ? 128 : 1024;
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutLeft, Device> view_type_a_ll;
  typedef Kokkos::View<ScalarB**, Kokkos::LayoutLeft, Device> view_type_b_ll;
  Test::impl_test_dot_mv<view_type_a_ll, view_type_b_ll, Device>(0, 5);
  Test::impl_test_dot_mv<view_type_a_ll, view_type_b_ll, Device>(13, 5);
  Test::impl_test_dot_mv<view_type_a_ll, view_type_b_ll, Device>(Nbig, 5);
  Test::impl_test_dot_mv<view_type_a_ll, view_type_b_ll, Device>(789, 1);
  // Test::impl_test_dot_mv<view_type_a_ll, view_type_b_ll, Device>(132231,5);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutRight, Device> view_type_a_lr;
  typedef Kokkos::View<ScalarB**, Kokkos::LayoutRight, Device> view_type_b_lr;
  Test::impl_test_dot_mv<view_type_a_lr, view_type_b_lr, Device>(0, 5);
  Test::impl_test_dot_mv<view_type_a_lr, view_type_b_lr, Device>(13, 5);
  Test::impl_test_dot_mv<view_type_a_lr, view_type_b_lr, Device>(Nbig, 5);
  Test::impl_test_dot_mv<view_type_a_lr, view_type_b_lr, Device>(789, 1);
  // Test::impl_test_dot_mv<view_type_a_lr, view_type_b_lr, Device>(132231,5);
#endif

// Removing the layout stride test as ViewTypeA a("a", N);
// is invalid since the view constructor needs a stride object!
#if (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutStride, Device> view_type_a_ls;
  typedef Kokkos::View<ScalarB**, Kokkos::LayoutStride, Device> view_type_b_ls;
  Test::impl_test_dot_mv<view_type_a_ls, view_type_b_ls, Device>(0, 5);
  Test::impl_test_dot_mv<view_type_a_ls, view_type_b_ls, Device>(13, 5);
  Test::impl_test_dot_mv<view_type_a_ls, view_type_b_ls, Device>(Nbig, 5);
  Test::impl_test_dot_mv<view_type_a_ls, view_type_b_ls, Device>(789, 1);
  // Test::impl_test_dot_mv<view_type_a_ls, view_type_b_ls, Device>(132231,5);
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
  Test::impl_test_dot_mv<view_type_a_ls, view_type_b_ll, Device>(Nbig, 5);
  Test::impl_test_dot_mv<view_type_a_ll, view_type_b_ls, Device>(Nbig, 5);
#endif

  return 1;
}

#if defined(KOKKOSKERNELS_INST_HALF) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, dot_half) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::dot_half");
  test_dot<Kokkos::Experimental::half_t, Kokkos::Experimental::half_t, TestDevice>();
  Kokkos::Profiling::popRegion();
}
TEST_F(TestCategory, dot_mv_half) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::dot_mv_half");
  test_dot_mv<Kokkos::Experimental::half_t, Kokkos::Experimental::half_t, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, dot_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::dot_float");
  test_dot<float, float, TestDevice>();
  Kokkos::Profiling::popRegion();
}
TEST_F(TestCategory, dot_mv_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::dot_mv_float");
  test_dot_mv<float, float, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, dot_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::dot_double");
  test_dot<double, double, TestDevice>();
  Kokkos::Profiling::popRegion();
}
TEST_F(TestCategory, dot_mv_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::dot_mv_double");
  test_dot_mv<double, double, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, dot_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::dot_complex_double");
  test_dot<Kokkos::complex<double>, Kokkos::complex<double>, TestDevice>();
  Kokkos::Profiling::popRegion();
}
TEST_F(TestCategory, dot_mv_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::dot_mv_complex_double");
  test_dot_mv<Kokkos::complex<double>, Kokkos::complex<double>, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_INT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, dot_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::dot_int");
  test_dot<int, int, TestDevice>();
  Kokkos::Profiling::popRegion();
}
TEST_F(TestCategory, dot_mv_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::dot_mv_int");
  test_dot_mv<int, int, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

/*#if !defined(KOKKOSKERNELS_ETI_ONLY) &&
!defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS) TEST_F( TestCategory,
dot_double_int ) { test_dot<double,int,TestDevice> ();
}
TEST_F( TestCategory, dot_mv_double_int ) {
    test_dot_mv<double,int,TestDevice> ();
}
#endif*/
