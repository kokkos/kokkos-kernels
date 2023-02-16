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
template <class ViewTypeA, class ViewTypeX, class ViewTypeY, class Device>
void impl_test_ger(int M, int N) {
  typedef typename ViewTypeA::value_type ScalarA;
  typedef typename ViewTypeX::value_type ScalarX;
  typedef typename ViewTypeY::value_type ScalarY;
  typedef Kokkos::ArithTraits<ScalarY> KAT_Y;

  typedef multivector_layout_adapter<ViewTypeA> vfA_type;

  ScalarA alpha = 3;
  double  eps   = (std::is_same<typename KAT_Y::mag_type, float>::value ? 1e-2 : 5e-10);

  typename vfA_type::BaseType b_A("A", M, N);
  ViewTypeX x("X", M);
  ViewTypeY y("Y", N);
  ViewTypeA org_A("Org_A", M, N);

  ViewTypeA A                        = vfA_type::view(b_A);
  typename ViewTypeX::const_type c_x = x;
  typename ViewTypeA::const_type c_A = A;

  typedef multivector_layout_adapter<typename ViewTypeA::HostMirror> h_vfA_type;

  typename h_vfA_type::BaseType h_b_A = Kokkos::create_mirror_view(b_A);

  typename ViewTypeA::HostMirror h_A = h_vfA_type::view(h_b_A);
  typename ViewTypeX::HostMirror h_x = Kokkos::create_mirror_view(x);
  typename ViewTypeY::HostMirror h_y = Kokkos::create_mirror_view(y);

  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);

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

  Kokkos::deep_copy(org_A, A);
  auto h_org_A = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), org_A);

  Kokkos::deep_copy(h_x, x);
  Kokkos::deep_copy(h_y, y);
  Kokkos::deep_copy(h_b_A, b_A);

  Kokkos::View<ScalarY*, Kokkos::HostSpace> expected("expected A += alpha * x * y^t", ldy);
  Kokkos::deep_copy(expected, h_org_A);
  vanillaGER(alpha, h_x, h_y, expected);

  KokkosBlas::ger(alpha, x, y, A);
  Kokkos::deep_copy(h_y, y);
  int numErrors = 0;
  for (int i = 0; i < ldy; i++) {
    if (KAT_Y::abs(expected(i) - h_y(i)) > KAT_Y::abs(eps * expected(i)))
      numErrors++;
  }
  EXPECT_EQ(numErrors, 0) << "Nonconst input, " << M << " by " << N
                          << ", alpha = " << alpha
                          << ": ger incorrect";

  Kokkos::deep_copy(y, org_A);
  KokkosBlas::ger(alpha, c_x, y, A);
  Kokkos::deep_copy(h_y, y);
  numErrors = 0;
  for (int i = 0; i < ldy; i++) {
    if (KAT_Y::abs(expected(i) - h_y(i)) > KAT_Y::abs(eps * expected(i)))
      numErrors++;
  }
  EXPECT_EQ(numErrors, 0) << "Const vector input, " << M << " by " << N
                          << ", alpha = " << alpha
                          << ": ger incorrect";

  Kokkos::deep_copy(y, org_A);
  KokkosBlas::ger(alpha, c_x, y, c_A);
  Kokkos::deep_copy(h_y, y);
  numErrors = 0;
  for (int i = 0; i < ldy; i++) {
    if (KAT_Y::abs(expected(i) - h_y(i)) > KAT_Y::abs(eps * expected(i)))
      numErrors++;
  }
  EXPECT_EQ(numErrors, 0) << "Const matrix/vector input, " << M << " by " << N
                          << ", alpha = " << alpha
                          << ": ger incorrect";
}
}  // namespace Test

template <class ScalarA, class ScalarX, class ScalarY, class Device>
int test_ger() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&      \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutLeft, Device> view_type_a_ll;
  typedef Kokkos::View<ScalarX*, Kokkos::LayoutLeft, Device> view_type_b_ll;
  typedef Kokkos::View<ScalarY*, Kokkos::LayoutLeft, Device> view_type_c_ll;
  Test::impl_test_ger<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(0, 1024);
  Test::impl_test_ger<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(1024, 0);
  Test::impl_test_ger<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(13, 13);
  Test::impl_test_ger<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(13, 1024);
  Test::impl_test_ger<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(50, 40);
  Test::impl_test_ger<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(1024, 1024);
  Test::impl_test_ger<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(2131, 2131);

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&       \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutRight, Device> view_type_a_lr;
  typedef Kokkos::View<ScalarX*, Kokkos::LayoutRight, Device> view_type_b_lr;
  typedef Kokkos::View<ScalarY*, Kokkos::LayoutRight, Device> view_type_c_lr;
  Test::impl_test_ger<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(0, 1024);
  Test::impl_test_ger<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(1024, 0);
  Test::impl_test_ger<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(13, 13);
  Test::impl_test_ger<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(13, 1024);
  Test::impl_test_ger<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(50, 40);
  Test::impl_test_ger<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(1024, 1024);
  Test::impl_test_ger<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(2131, 2131);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTSTRIDE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&        \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutStride, Device> view_type_a_ls;
  typedef Kokkos::View<ScalarX*, Kokkos::LayoutStride, Device> view_type_b_ls;
  typedef Kokkos::View<ScalarY*, Kokkos::LayoutStride, Device> view_type_c_ls;
  Test::impl_test_ger<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(0, 1024);
  Test::impl_test_ger<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(1024, 0);
  Test::impl_test_ger<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(13, 13);
  Test::impl_test_ger<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(13, 1024);
  Test::impl_test_ger<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(50, 40);
  Test::impl_test_ger<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(1024, 1024);
  Test::impl_test_ger<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(2131, 2131);
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && \
    !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
  Test::impl_test_ger<view_type_a_ls, view_type_b_ll, view_type_c_lr, Device>(1024, 1024);
  Test::impl_test_ger<view_type_a_ll, view_type_b_ls, view_type_c_lr, Device>(1024, 1024);
#endif

  return 1;
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_float");
  test_ger<float, float, float, TestExecSpace>("N");
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_tran_float");
  test_ger<float, float, float, TestExecSpace>("T");
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&  \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_double");
  test_ger<double, double, double, TestExecSpace>("N");
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_tran_double");
  test_ger<double, double, double, TestExecSpace>("T");
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&          \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_complex_double");
  test_ger<Kokkos::complex<double>, Kokkos::complex<double>,
            Kokkos::complex<double>, TestExecSpace>("N");
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_tran_complex_double");
  test_ger<Kokkos::complex<double>, Kokkos::complex<double>,
            Kokkos::complex<double>, TestExecSpace>("T");
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_conj_complex_double");
  test_ger<Kokkos::complex<double>, Kokkos::complex<double>,
            Kokkos::complex<double>, TestExecSpace>("C");
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_INT) ||   \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_int");
  test_ger<int, int, int, TestExecSpace>("N");
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_tran_int");
  test_ger<int, int, int, TestExecSpace>("T");
  Kokkos::Profiling::popRegion();
}
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && \
    !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
TEST_F(TestCategory, ger_double_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_double_int");
  test_ger<double, int, float, TestExecSpace>("N");
  Kokkos::Profiling::popRegion();

  // Kokkos::Profiling::pushRegion("KokkosBlas::Test::gert_double_int");
  //  test_ger<double,int,float,TestExecSpace> ("T");
  // Kokkos::Profiling::popRegion();
}

#endif
