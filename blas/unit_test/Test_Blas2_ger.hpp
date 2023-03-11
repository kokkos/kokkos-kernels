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
//#include <KokkosKernels_TestUtils.hpp>

namespace Test {

template <class ViewTypeX, class ViewTypeY, class ViewTypeA, class Device>
void impl_test_ger(int M, int N, bool useAnalyticalResults = false) {
  typedef typename ViewTypeX::value_type ScalarX;
  typedef typename ViewTypeY::value_type ScalarY;
  typedef typename ViewTypeA::value_type ScalarA;

  ViewTypeX x("X", M);
  ViewTypeY y("Y", N);
  ViewTypeA A("A", M, N);

  typename ViewTypeX::HostMirror h_x = Kokkos::create_mirror_view(x);
  typename ViewTypeY::HostMirror h_y = Kokkos::create_mirror_view(y);
  typename ViewTypeA::HostMirror h_A = Kokkos::create_mirror_view(A);

  Kokkos::View<ScalarA**, Kokkos::HostSpace> expectedResult("expected A += alpha * x * y^t", M ,N);
  bool expectedResultIsKnown = false;

  ScalarA alpha = 3;

  if (useAnalyticalResults) {
    for (int i = 0; i < M; i++) {
      h_x[i] = sin(i);
    }

    for (int j = 0; j < N; j++) {
      h_y[j] = cos(j);
    }

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        h_A(i,j) = 3 * cos(i) * sin(j);
      }
    }

    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);
    Kokkos::deep_copy(A, h_A);

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        expectedResult(i,j) = 3 * sin(i+j);
      }
    }
    expectedResultIsKnown = true;
  }
  else if ((M == 1) && (N == 1)) {
    h_x[0] = 2;

    h_y[0] = 3;

    h_A(0,0) = 7;

    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);
    Kokkos::deep_copy(A, h_A);

    expectedResult(0,0) = 25;
    expectedResultIsKnown = true;
  }
  else if ((M == 1) && (N == 2)) {
    h_x[0] = 2;

    h_y[0] = 3;
    h_y[1] = 4;

    h_A(0,0) = 7;
    h_A(0,1) = -6;

    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);
    Kokkos::deep_copy(A, h_A);

    expectedResult(0,0) = 25;
    expectedResult(0,1) = 18;
    expectedResultIsKnown = true;
  }
  else if ((M == 2) && (N == 2)) {
    h_x[0] = 2;
    h_x[1] = 9;

    h_y[0] = -3;
    h_y[1] = 7;

    h_A(0,0) = 17;
    h_A(0,1) = -43;
    h_A(1,0) = 29;
    h_A(1,1) = 101;

    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);
    Kokkos::deep_copy(A, h_A);

    expectedResult(0,0) = -1;
    expectedResult(0,1) = -1;
    expectedResult(1,0) = -52;
    expectedResult(1,1) = 290;
    expectedResultIsKnown = true;
  }
  else {
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
      Kokkos::fill_random(A, rand_pool, randStart, randEnd);
    }

    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_y, y);
    Kokkos::deep_copy(h_A, A);
  }

  Kokkos::View<ScalarA**, Kokkos::HostSpace> vanillaResult("vanilla = A + alpha * x * y^t", M ,N);
  {
    KOKKOS_IMPL_DO_NOT_USE_PRINTF( "In Test_Blas2_ger.hpp, computing vanilla A with alpha type = %s\n", typeid(alpha).name() );
    bool testIsGpu = KokkosKernels::Impl::kk_is_gpu_exec_space< typename ViewTypeA::execution_space >();
    bool A_is_lr = std::is_same< typename ViewTypeA::array_layout, Kokkos::LayoutRight >::value;

    if ( testIsGpu && A_is_lr ) {
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          vanillaResult(i,j) = h_A(i,j) + alpha * h_y(j) * h_x(i);
        }
      }
    }
    else {
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          vanillaResult(i,j) = h_A(i,j) + alpha * h_x(i) * h_y(j);
        }
      }
    }
  }

  typedef Kokkos::ArithTraits<ScalarA> KAT_A;
  double eps = (std::is_same<typename KAT_A::mag_type, float>::value ? 2.e-2 : 5e-10);
  if (expectedResultIsKnown) {
    // ******************************************************************
    // Compare vanillaResult against expectedResult
    // ******************************************************************
    int numErrors(0);
    if (useAnalyticalResults) {
      for (int i(0); i < M; ++i) {
        for (int j(0); j < N; ++j) {
          if (KAT_A::abs(expectedResult(i,j) - vanillaResult(i,j)) > KAT_A::abs(eps * expectedResult(i,j))) {
            std::cout << "ERROR, i = " << i
                      << ", j = "      << j
                      << ": expectedResult(i,j) = " << expectedResult(i,j)
                      << ", vanillaResult(i,j) = "  << vanillaResult(i,j)
                      << ", KAT_A::abs(expectedResult(i,j) - vanillaResult(i,j)) = " << KAT_A::abs(expectedResult(i,j) - vanillaResult(i,j))
                      << ", KAT_A::abs(eps * expectedResult(i,j)) = "                << KAT_A::abs(eps * expectedResult(i,j))
                      << std::endl;
            numErrors++;
          }
        } // for j
      } // for i
      EXPECT_EQ(numErrors, 0) << "A is " << M << " by " << N
                              << ": vanilla differs from analytical";
    }
    else {
      for (int i(0); i < M; ++i) {
        for (int j(0); j < N; ++j) {
          if ( expectedResult(i,j) != vanillaResult(i,j) ) {
            std::cout << "ERROR, i = " << i
                      << ", j = "      << j
                      << ": expectedResult(i,j) = " << expectedResult(i,j)
                      << ", vanillaResult(i,j) = "  << vanillaResult(i,j)
                      << std::endl;
            numErrors++;
          }
        } // for j
      } // for i
      EXPECT_EQ(numErrors, 0) << "A is " << M << " by " << N
                              << ": vanilla result is incorrect";
    }
  }
  else {
    // ******************************************************************
    // Copy vanillaResult to expectedResult
    // ******************************************************************
    Kokkos::deep_copy(expectedResult, vanillaResult);
  }
  
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "In Test_Blas2_ger.hpp, right before calling KokkosBlas::ger(): ViewType = %s\n", typeid(ViewTypeA).name() );
  KokkosBlas::ger("T", alpha, x, y, A);
  Kokkos::deep_copy(h_A, A);

  int numErrors(0);
  for (int i(0); i < M; ++i) {
    for (int j(0); j < N; ++j) {
      if (KAT_A::abs(expectedResult(i,j) - h_A(i,j)) > KAT_A::abs(eps * expectedResult(i,j))) {
        std::cout << "ERROR, i = " << i
                  << ", j = "      << j
                  << ": expectedResult(i,j) = " << expectedResult(i,j)
                  << ", h_A(i,j) = "            << h_A(i,j)
                  << ", KAT_A::abs(expectedResult(i,j) - h_A(i,j)) = " << KAT_A::abs(expectedResult(i,j) - h_A(i,j))
                  << ", KAT_A::abs(eps * expectedResult(i,j)) = "      << KAT_A::abs(eps * expectedResult(i,j))
                  << std::endl;
        numErrors++;
      }
    } // for j
  } // for i
  std::cout << "A is " << M << " by " << N
            << ", numErrors = " << numErrors
            << std::endl;
  EXPECT_EQ(numErrors, 0) << "A is " << M << " by " << N
                          << ": ger result is incorrect";

}
} // namespace Test

template <class ScalarX, class ScalarY, class ScalarA, class Device>
int test_ger( const std::string & caseName ) {
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+==========================================================================\n" );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Starting %s ...\n", caseName.c_str() );

#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Starting %s for LAYOUTLEFT ...\n", caseName.c_str() );
  typedef Kokkos::View<ScalarX*, Kokkos::LayoutLeft, Device> view_type_x_ll;
  typedef Kokkos::View<ScalarY*, Kokkos::LayoutLeft, Device> view_type_y_ll;
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutLeft, Device> view_type_a_ll;
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(0, 1024);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(1024, 0);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(13, 13);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(13, 1024);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(13, 1024, true);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(50, 40);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(1024, 1024);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(2131, 2131);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(2131, 2131, true);
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Finished %s for LAYOUTLEFT\n", caseName.c_str() );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Starting %s for LAYOUTRIGHT ...\n", caseName.c_str() );
  typedef Kokkos::View<ScalarX*, Kokkos::LayoutRight, Device> view_type_x_lr;
  typedef Kokkos::View<ScalarY*, Kokkos::LayoutRight, Device> view_type_y_lr;
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutRight, Device> view_type_a_lr;
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(0, 1024);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(1024, 0);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(1, 1);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(2, 2);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(1, 2);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(13, 13);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(13, 1024);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(13, 1024, true);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(50, 40);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(1024, 1024);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(2131, 2131);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(2131, 2131, true);
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Finished %s for LAYOUTRIGHT\n", caseName.c_str() );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTSTRIDE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Starting %s for LAYOUTSTRIDE ...\n", caseName.c_str() );
  typedef Kokkos::View<ScalarX*, Kokkos::LayoutStride, Device> view_type_x_ls;
  typedef Kokkos::View<ScalarY*, Kokkos::LayoutStride, Device> view_type_y_ls;
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutStride, Device> view_type_a_ls;
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(0, 1024);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(1024, 0);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(13, 13);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(13, 1024);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(13, 1024, true);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(50, 40);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(1024, 1024);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(2131, 2131);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(2131, 2131, true);
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Finished %s for LAYOUTSTRIDE\n", caseName.c_str() );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Starting %s for MIXED LAYOUTS ...\n", caseName.c_str() );
  Test::impl_test_ger<view_type_x_ls, view_type_y_ll, view_type_a_lr, Device>(1024, 1024);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ll, view_type_a_lr, Device>(1024, 1024, true);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ls, view_type_a_lr, Device>(1024, 1024);
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Finished %s for MIXED LAYOUTS\n", caseName.c_str() );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
#endif

  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Finished %s\n", caseName.c_str() );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+==========================================================================\n" );

  return 1;
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_float");
  test_ger<float, float, float, TestExecSpace>( "test case ger_float" );
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_double");
  test_ger<double, double, double, TestExecSpace>( "test case ger_double" );
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_complex_double");
  test_ger<Kokkos::complex<double>, Kokkos::complex<double>, Kokkos::complex<double>, TestExecSpace>( "test case ger_complex_double" );
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_complex_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_complex_float");
  test_ger<Kokkos::complex<float>, Kokkos::complex<float>, Kokkos::complex<float>, TestExecSpace>( "test case ger_complex_float" );
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_INT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_int");
  test_ger<int, int, int, TestExecSpace>( "test case ger_int" );
  Kokkos::Profiling::popRegion();
}
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && \
    !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
TEST_F(TestCategory, ger_double_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_double_int");
  test_ger<double, int, float, TestExecSpace>( "test case ger_mixed_types" );
  Kokkos::Profiling::popRegion();
}
#endif
