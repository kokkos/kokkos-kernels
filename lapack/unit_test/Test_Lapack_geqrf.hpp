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

// Only enable this test where KokkosLapack supports geqrf:
// CUDA+CUSOLVER, HIP+ROCSOLVER and HOST+LAPACK
#if (defined(TEST_CUDA_LAPACK_CPP) &&                                       \
     defined(KOKKOSKERNELS_ENABLE_TPL_CUSOLVER)) ||                         \
    (defined(TEST_HIP_LAPACK_CPP) &&                                        \
     defined(KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER)) ||                        \
    (defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK) &&                            \
     (defined(TEST_OPENMP_LAPACK_CPP) || defined(TEST_SERIAL_LAPACK_CPP) || \
      defined(TEST_THREADS_LAPACK_CPP)))

// AquiEEP

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <KokkosLapack_geqrf.hpp>
//#include <KokkosBlas2_gemv.hpp>
//#include <KokkosBlas3_gemm.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {

template <class ViewTypeA, class ViewTypeTau>
void getQR(int const m, int const n,
           typename ViewTypeA::HostMirror const&  // h_A
           ,
           typename ViewTypeTau::HostMirror const&  // h_tau
           ,
           typename ViewTypeA::HostMirror&  // h_Q
           ,
           typename ViewTypeA::HostMirror& h_R,
           typename ViewTypeA::HostMirror&  // h_QR
) {
  using ScalarA = typename ViewTypeA::value_type;

  for (int i(0); i < m; ++i) {
    for (int j(0); j < n; ++j) {
      if constexpr (Kokkos::ArithTraits<ScalarA>::is_complex) {
        h_R(i, j).real() = 0.;
        h_R(i, j).imag() = 0.;
      } else {
        h_R(i, j) = 0.;
      }
    }
  }

  ViewTypeA I("I", m, m);
  typename ViewTypeA::HostMirror h_I = Kokkos::create_mirror_view(I);
  for (int i(0); i < m; ++i) {
    for (int j(0); j < m; ++j) {
      if constexpr (Kokkos::ArithTraits<ScalarA>::is_complex) {
        if (i == j) {
          h_I(i, j).real() = 1.;
        } else {
          h_I(i, j).real() = 0.;
        }
        h_I(i, j).imag() = 0.;
      } else {
        if (i == j) {
          h_I(i, j) = 1.;
        } else {
          h_I(i, j) = 0.;
        }
      }
    }
  }
}

template <class ViewTypeA, class ViewTypeTau, class Device>
void impl_test_geqrf(int m, int n) {
  using ViewTypeInfo = Kokkos::View<int*, Kokkos::LayoutLeft, Device>;
  using execution_space = typename Device::execution_space;
  using ScalarA         = typename ViewTypeA::value_type;
  // using ats             = Kokkos::ArithTraits<ScalarA>;

  execution_space space{};

  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);

  int minMN(std::min(m, n));

  // Create device views
  ViewTypeA    A   ("A", m, n);
  ViewTypeTau  Tau ("Tau", minMN);
  ViewTypeInfo Info("Info", 1);

  // Create host mirrors of device views.
  typename ViewTypeA::HostMirror    h_A     = Kokkos::create_mirror_view(A);
  typename ViewTypeA::HostMirror    h_Aorig = Kokkos::create_mirror_view(A);
  typename ViewTypeTau::HostMirror  h_tau   = Kokkos::create_mirror_view(Tau);
  typename ViewTypeInfo::HostMirror h_info  = Kokkos::create_mirror_view(Info);

  // Initialize data.
  if ((m == 3) && (n == 3)) {
    if constexpr (Kokkos::ArithTraits<ScalarA>::is_complex) {
      h_A(0, 0).real() = 12.;
      h_A(0, 1).real() = -51.;
      h_A(0, 2).real() = 4.;

      h_A(1, 0).real() = 6.;
      h_A(1, 1).real() = 167.;
      h_A(1, 2).real() = -68.;

      h_A(2, 0).real() = -4.;
      h_A(2, 1).real() = 24.;
      h_A(2, 2).real() = -41.;

      for (int i(0); i < m; ++i) {
        for (int j(0); j < n; ++j) {
          h_A(i, j).imag() = 0.;
        }
      }
    } else {
      h_A(0, 0) = 12.;
      h_A(0, 1) = -51.;
      h_A(0, 2) = 4.;

      h_A(1, 0) = 6.;
      h_A(1, 1) = 167.;
      h_A(1, 2) = -68.;

      h_A(2, 0) = -4.;
      h_A(2, 1) = 24.;
      h_A(2, 2) = -41.;
    }

    Kokkos::deep_copy(A, h_A);
  } else {
    Kokkos::fill_random(A, rand_pool,
                        Kokkos::rand<Kokkos::Random_XorShift64<execution_space>,
                                     ScalarA>::max());
    Kokkos::deep_copy(h_A, A);
  }

  Kokkos::deep_copy(h_Aorig, h_A);

#if 1  // def HAVE_KOKKOSKERNELS_DEBUG
  for (int i(0); i < m; ++i) {
    for (int j(0); j < n; ++j) {
      std::cout << "A(" << i << "," << j << ") = " << h_A(i, j) << std::endl;
    }
  }
#endif

  Kokkos::fence();

  // Perform the QR factorization
  try {
    KokkosLapack::geqrf(space, A, Tau, Info);
  } catch (const std::runtime_error& e) {
    std::cout << "KokkosLapack::geqrf(): caught exception '" << e.what() << "'"
              << std::endl;
    FAIL();
    return;
  }

  Kokkos::fence();

  Kokkos::deep_copy(h_info, Info);
  EXPECT_EQ(h_info[0], 0) << "Failed geqrf() test: Info[0] = " << h_info[0];

  // Get the results
  Kokkos::deep_copy(h_A, A);
  Kokkos::deep_copy(h_tau, Tau);

#if 1  // def HAVE_KOKKOSKERNELS_DEBUG
  std::cout << "info[0] = " << h_info[0] << std::endl;
  for (int i(0); i < minMN; ++i) {
    for (int j(0); j < n; ++j) {
      std::cout << "R(" << i << "," << j << ") = " << h_A(i, j) << std::endl;
    }
  }
  for (int i(0); i < minMN; ++i) {
    std::cout << "tau(" << i << ") = " << h_tau[i] << std::endl;
  }
#endif

  ViewTypeA Q("Q", m, m);
  ViewTypeA R("R", m, n);
  ViewTypeA QR("QR", m, n);

  typename ViewTypeA::HostMirror h_Q  = Kokkos::create_mirror_view(Q);
  typename ViewTypeA::HostMirror h_R  = Kokkos::create_mirror_view(R);
  typename ViewTypeA::HostMirror h_QR = Kokkos::create_mirror_view(QR);

  getQR<ViewTypeA, ViewTypeTau>(m, n, h_A, h_tau, h_Q, h_R, h_QR);

#if 1  // def HAVE_KOKKOSKERNELS_DEBUG
  for (int i(0); i < m; ++i) {
    for (int j(0); j < m; ++j) {
      std::cout << "Q(" << i << "," << j << ") = " << h_Q(i, j) << std::endl;
    }
  }
  for (int i(0); i < m; ++i) {
    for (int j(0); j < n; ++j) {
      std::cout << "R(" << i << "," << j << ") = " << h_R(i, j) << std::endl;
    }
  }
  for (int i(0); i < m; ++i) {
    for (int j(0); j < n; ++j) {
      std::cout << "QR(" << i << "," << j << ") = " << h_QR(i, j) << std::endl;
    }
  }
#endif

  if ((m == 3) && (n == 3)) {
  }

  // Dense matrix-matrix multiply: C = beta*C + alpha*op(A)*op(B).
  // void gemm( const execution_space                & space
  //          , const char                             transA[]
  //          , const char                             transB[]
  //          , typename AViewType::const_value_type & alpha
  //          , const AViewType                      & A
  //          , const BViewType                      & B
  //          , typename CViewType::const_value_type & beta
  //          , const CViewType                      & C
  //          );

  // Rank-1 update of a general matrix: A = A + alpha * x * y^{T,H}.
  // void ger( const ExecutionSpace                       & space
  //         , const char                                   trans[]
  //         , const typename AViewType::const_value_type & alpha
  //         , const XViewType                            & x
  //         , const YViewType                            & y
  //         , const AViewType                            & A
  //         );

  // Checking vs ref on CPU, this eps is about 10^-9
  // typedef typename ats::mag_type mag_type;
  // const mag_type eps = 3.0e7 * ats::epsilon();
  bool test_flag = true;
  for (int i = 0; i < n; i++) {
#if 0
    if (ats::abs(h_B(i) - h_X0(i)) > eps) {
      test_flag = false;
      printf(
          "    Error %d, pivot %c, padding %c: result( %.15lf ) !="
          "solution( %.15lf ) at (%d), error=%.15e, eps=%.15e\n",
          N, mode[0], padding[0], ats::abs(h_B(i)), ats::abs(h_X0(i)), int(i),
          ats::abs(h_B(i) - h_X0(i)), eps);
      break;
    }
#endif
  }
  ASSERT_EQ(test_flag, true);
}

}  // namespace Test

template <class Scalar, class Device>
void test_geqrf() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&      \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  using view_type_a_ll   = Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device>;
  using view_type_tau_ll = Kokkos::View<Scalar*, Kokkos::LayoutLeft, Device>;

  Test::impl_test_geqrf<view_type_a_ll, view_type_tau_ll, Device>(3, 3);
#endif
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, geqrf_float) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::geqrf_float");
  test_geqrf<float, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&  \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, geqrf_double) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::geqrf_double");
  test_geqrf<double, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&          \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, geqrf_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::geqrf_complex_double");
  test_geqrf<Kokkos::complex<double>, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&         \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, geqrf_complex_float) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::geqrf_complex_float");
  test_geqrf<Kokkos::complex<float>, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#endif  // CUDA+CUSOLVER or HIP+ROCSOLVER or LAPACK+HOST
