//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas2_ger.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {
template <class ViewTypeX, class ViewTypeY, class ViewTypeA, class Device>
void impl_test_ger(int M, int N) {
  typedef typename ViewTypeX::value_type                 ScalarX;
  typedef typename ViewTypeY::value_type                 ScalarY;
  typedef typename ViewTypeA::value_type                 ScalarA;
  typedef          Kokkos::ArithTraits<ScalarA>          KAT_A;
  typedef          multivector_layout_adapter<ViewTypeA> vfA_type;

  ScalarA alpha = 3;
  double  eps   = (std::is_same<typename KAT_A::mag_type, float>::value ? 1e-2 : 5e-10);

  typename vfA_type::BaseType b_A  ("A", M, N);
  ViewTypeX                   x    ("X", M);
  ViewTypeY                   y    ("Y", N);
  ViewTypeA                   org_A("Org_A", M, N);

  ViewTypeA A = vfA_type::view(b_A);

  typedef multivector_layout_adapter<typename ViewTypeA::HostMirror> h_vfA_type;

  typename h_vfA_type::BaseType  h_b_A = Kokkos::create_mirror_view(b_A);
  typename ViewTypeX::HostMirror h_x   = Kokkos::create_mirror_view(x);
  typename ViewTypeY::HostMirror h_y   = Kokkos::create_mirror_view(y);
  typename ViewTypeA::HostMirror h_A   = h_vfA_type::view(h_b_A);

  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);

  if ((M == 1) && (N == 1)) {
    h_x[0] = 2;

    h_y[0] = 3;

    h_b_A(0,0) = 7;

    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);
    Kokkos::deep_copy(b_A, h_b_A);
  }
  else if ((M == 2) && (N == 2)) {
    h_x[0] = 2;
    h_x[1] = 9;

    h_y[0] = -3;
    h_y[1] = 7;

    h_b_A(0,0) = 17;
    h_b_A(0,1) = -43;
    h_b_A(1,0) = 29;
    h_b_A(1,1) = 101;

    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);
    Kokkos::deep_copy(b_A, h_b_A);
  }
  else {
    {
      ScalarX randStart, randEnd;
      Test::getRandomBounds(1.0, randStart, randEnd);
      Kokkos::fill_random(x, rand_pool, randStart, randEnd);
    }
    {
      ScalarY randStart, randEnd;
      Test::getRandomBounds(1.0, randStart, randEnd);
      Kokkos::fill_random(y, rand_pool, randStart, randEnd);
    }
    {
      ScalarA randStart, randEnd;
      Test::getRandomBounds(1.0, randStart, randEnd);
      Kokkos::fill_random(b_A, rand_pool, randStart, randEnd);
    }
  }

  Kokkos::deep_copy(org_A, A);
  auto h_org_A = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), org_A);

  Kokkos::deep_copy(h_x, x);
  Kokkos::deep_copy(h_y, y);
  Kokkos::deep_copy(h_b_A, b_A);

  Kokkos::View<ScalarA**, Kokkos::HostSpace> expected("expected A += alpha * x * y^t", M ,N);
  Kokkos::deep_copy(expected, h_org_A);
  vanillaGER(alpha, h_x, h_y, expected);

  KokkosBlas::ger(alpha, x, y, A);
  Kokkos::deep_copy(h_A, A);

  if ((M <= 2) && (N <= 2)) {
    for (int i(0); i < M; ++i) {
      for (int j(0); j < N; ++j) {
        std::cout << "expected(" << i << "," << j << ") = " << expected(i,j)
		  << "; h_A("    << i << "," << j << ") = " << h_A(i,j)
		  << std::endl;
      }
    }
  }

  int numErrors(0);
  for (int i(0); i < M; ++i) {
    for (int j(0); j < N; ++j) {
      if (KAT_A::abs(expected(i,j) - h_A(i,j)) > KAT_A::abs(eps * expected(i,j))) {
        numErrors++;
      }
    }	 
  }
  std::cout << "A is " << M << " by " << N
            << ", alpha = " << alpha
            << ": M*N = "   << M*N
            << ", numErrors = " << numErrors
            << std::endl;
  EXPECT_EQ(numErrors, 0) << "A is " << M << " by " << N
                          << ", alpha = " << alpha
                          << ": ger incorrect";

}
} // namespace Test

template <class ScalarX, class ScalarY, class ScalarA, class Device>
int test_ger() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarX*, Kokkos::LayoutLeft, Device> view_type_x_ll;
  typedef Kokkos::View<ScalarY*, Kokkos::LayoutLeft, Device> view_type_y_ll;
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutLeft, Device> view_type_a_ll;
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(0, 1024);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(1024, 0);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(13, 13);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(13, 1024);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(50, 40);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(1024, 1024);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(2131, 2131);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarX*, Kokkos::LayoutRight, Device> view_type_x_lr;
  typedef Kokkos::View<ScalarY*, Kokkos::LayoutRight, Device> view_type_y_lr;
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutRight, Device> view_type_a_lr;
  //Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(0, 1024); // EEP
  //Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(1024, 0);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(1, 1);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(2, 2);
  //Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(13, 13);
  //Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(13, 1024);
  //Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(50, 40);
  //Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(1024, 1024);
  //Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(2131, 2131);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTSTRIDE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarX*, Kokkos::LayoutStride, Device> view_type_x_ls;
  typedef Kokkos::View<ScalarY*, Kokkos::LayoutStride, Device> view_type_y_ls;
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutStride, Device> view_type_a_ls;
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(0, 1024);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(1024, 0);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(13, 13);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(13, 1024);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(50, 40);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(1024, 1024);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(2131, 2131);
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
  Test::impl_test_ger<view_type_x_ls, view_type_y_ll, view_type_a_lr, Device>(1024, 1024);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ls, view_type_a_lr, Device>(1024, 1024);
#endif

  return 1;
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_float");
  test_ger<float, float, float, TestExecSpace>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_double");
  test_ger<double, double, double, TestExecSpace>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_complex_double");
  test_ger<Kokkos::complex<double>, Kokkos::complex<double>, Kokkos::complex<double>, TestExecSpace>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_INT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_int");
  test_ger<int, int, int, TestExecSpace>();
  Kokkos::Profiling::popRegion();
}
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && \
    !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
TEST_F(TestCategory, ger_double_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_double_int");
  test_ger<double, int, float, TestExecSpace>();
  Kokkos::Profiling::popRegion();
}
#endif
