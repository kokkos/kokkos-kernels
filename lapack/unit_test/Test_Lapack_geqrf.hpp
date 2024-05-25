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

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <KokkosBlas2_ger.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosLapack_geqrf.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {

template <class ViewTypeA, class ViewTypeTau>
void getQR(int const m, int const n,
           typename ViewTypeA::HostMirror const& h_A,
           typename ViewTypeTau::HostMirror const& h_tau,
           typename ViewTypeA::HostMirror& h_Q,
           typename ViewTypeA::HostMirror& h_R,
           typename ViewTypeA::HostMirror& h_QR
) {
  using ScalarA = typename ViewTypeA::value_type;

  // Populate h_R
  for (int i(0); i < m; ++i) {
    for (int j(0); j < n; ++j) {
      if ((i <= j) && (i < n)) {
        h_R(i,j) = h_A(i,j);
      }
      else {
        h_R(i,j) = Kokkos::ArithTraits<ScalarA>::zero();
      }
    }
  }

  // Instantiate the identity matrix
  ViewTypeA I("I", m, m);
  typename ViewTypeA::HostMirror h_I = Kokkos::create_mirror_view(I);
  Kokkos::deep_copy(h_I,Kokkos::ArithTraits<ScalarA>::zero());
  for (int i(0); i < m; ++i) {
    if constexpr (Kokkos::ArithTraits<ScalarA>::is_complex) {
      h_I(i,i).real() = 1.;
    } else {
      h_I(i,i) = 1.;
    }
  }

  // Populate h_Q
  int minMN(std::min(m, n));
  ViewTypeTau v("v", m);
  typename ViewTypeTau::HostMirror h_v = Kokkos::create_mirror_view(v);

  ViewTypeA Qk("Qk", m, m);
  typename ViewTypeA::HostMirror h_Qk = Kokkos::create_mirror_view(Qk);

  ViewTypeA auxM("auxM", m, m);
  typename ViewTypeA::HostMirror h_auxM = Kokkos::create_mirror_view(auxM);

  // Q = H(0) H(1) . . . H(min(M,N)-1), where for k=0,1,...,min(m,n)-1:
  //   H(k) = I - Tau(k) * v * v**H, and
  //   v is a vector of size m with:
  //     v(0:k-1) = 0,
  //     v(k)     = 1,
  //     v(k+1:m-1) = A(k+1:m-1,k).
  for (int k(0); k < minMN; ++k) {
    Kokkos::deep_copy(h_v,Kokkos::ArithTraits<ScalarA>::zero());
    h_v[k] = 1.;
    for (int index(k+1); index < minMN; ++index) {
      h_v[index] = h_A(index,k);
    }

    // Rank-1 update of a general matrix: A = A + alpha * x * y^{T,H}.
    // void ger( const char                                   trans[]
    //         , const typename AViewType::const_value_type & alpha
    //         , const XViewType                            & x
    //         , const YViewType                            & y
    //         , const AViewType                            & A
    //         );
    Kokkos::deep_copy(h_Qk, h_I);
    KokkosBlas::ger( "H"
                   , -h_tau[k]
                   , h_v
                   , h_v
                   , h_Qk
                   );

    // Dense matrix-matrix multiply: C = beta*C + alpha*op(A)*op(B).
    // void gemm( const char                             transA[]
    //          , const char                             transB[]
    //          , typename AViewType::const_value_type & alpha
    //          , const AViewType                      & A
    //          , const BViewType                      & B
    //          , typename CViewType::const_value_type & beta
    //          , const CViewType                      & C
    //          );
    if (k == 0) {
      Kokkos::deep_copy(h_Q, h_Qk);
    }
    else {
      Kokkos::deep_copy(h_auxM, Kokkos::ArithTraits<ScalarA>::zero());
      KokkosBlas::gemm( "N"
                      , "N"
                      , 1.
                      , h_Q
                      , h_Qk
                      , 0.
                      , h_auxM
                      );
      Kokkos::deep_copy(h_Q, h_auxM);
    }
  } // for k

  Kokkos::deep_copy(h_QR, Kokkos::ArithTraits<ScalarA>::zero());
  KokkosBlas::gemm( "N"
                  , "N"
                  , 1.
                  , h_Q
                  , h_R
                  , 0.
                  , h_QR
                  );

  // AquiEEP: test Q^H Q = I
}

template <class ViewTypeA, class ViewTypeTau, class Device>
void impl_test_geqrf(int m, int n) {
  using ViewTypeInfo    = Kokkos::View<int*, Kokkos::LayoutLeft, Device>;
  using execution_space = typename Device::execution_space;
  using ScalarA         = typename ViewTypeA::value_type;
  using ats             = Kokkos::ArithTraits<ScalarA>;

  execution_space space{};

  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);

  int minMN(std::min(m, n));

  // Create device views
  ViewTypeA    A    ("A", m, n);
  ViewTypeA    Aorig("Aorig", m, n);
  ViewTypeTau  Tau  ("Tau", minMN);
  ViewTypeInfo Info ("Info", 1);

  // Create host mirrors of device views.
  typename ViewTypeA::HostMirror h_A       = Kokkos::create_mirror_view(A);
  typename ViewTypeA::HostMirror h_Aorig   = Kokkos::create_mirror_view(Aorig);
  typename ViewTypeTau::HostMirror h_tau   = Kokkos::create_mirror_view(Tau);
  typename ViewTypeInfo::HostMirror h_info = Kokkos::create_mirror_view(Info);

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
          h_A(i,j).imag() = 0.;
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

#ifdef HAVE_KOKKOSKERNELS_DEBUG
  for (int i(0); i < m; ++i) {
    for (int j(0); j < n; ++j) {
      std::cout << "Aorig(" << i << "," << j << ") = " << h_A(i,j) << std::endl;
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

#ifdef HAVE_KOKKOSKERNELS_DEBUG
  std::cout << "info[0] = " << h_info[0] << std::endl;
  for (int i(0); i < minMN; ++i) {
    for (int j(0); j < n; ++j) {
      std::cout << "Aoutput(" << i << "," << j << ") = " << std::setprecision(16) << h_A(i,j) << std::endl;
    }
  }
  for (int i(0); i < minMN; ++i) {
    std::cout << "tau(" << i << ") = " << h_tau[i] << std::setprecision(16) << std::endl;
  }
#endif

  const typename Kokkos::ArithTraits<typename ViewTypeA::non_const_value_type>::mag_type absTol(1.e-8);

  if ((m == 3) && (n == 3)) {
    std::vector<std::vector<ScalarA>> refMatrix(m);
    for (int i(0); i < m; ++i) {
      refMatrix[i].resize(n,Kokkos::ArithTraits<ScalarA>::zero());
    }

    std::vector<ScalarA> refTau(m,Kokkos::ArithTraits<ScalarA>::zero());

    if constexpr (Kokkos::ArithTraits<ScalarA>::is_complex) {
      refMatrix[0][0].real() = -14.;
      refMatrix[0][1].real() = -21.;
      refMatrix[0][2].real() = 14.;

      refMatrix[1][0].real() = 0.2307692307692308;
      refMatrix[1][1].real() = -175.;
      refMatrix[1][2].real() = 70.;

      refMatrix[2][0].real() = -0.1538461538461539;
      refMatrix[2][1].real() = 1./18.;
      refMatrix[2][2].real() = -35.;

      refTau[0].real() = 1.857142857142857;
      refTau[1].real() = 1.993846153846154;
      refTau[2].real() = 0.;
    }
    else {
      refMatrix[0][0] = -14.;
      refMatrix[0][1] = -21.;
      refMatrix[0][2] = 14.;

      refMatrix[1][0] = 0.2307692307692308;
      refMatrix[1][1] = -175.;
      refMatrix[1][2] = 70.;

      refMatrix[2][0] = -0.1538461538461539;
      refMatrix[2][1] = 1./18.;
      refMatrix[2][2] = -35.;

      refTau[0] = 1.857142857142857;
      refTau[1] = 1.993846153846154;
      refTau[2] = 0.;
    }

    {
      bool test_flag_A = true;
      for (int i(0); (i < m) && test_flag_A; ++i) {
        for (int j(0); (j < n) && test_flag_A; ++j) {
          if (ats::abs(h_A(i,j) - refMatrix[i][j]) > absTol) {
            test_flag_A = false;
          }
        }
      }
      ASSERT_EQ(test_flag_A, true);
    }

    {
      bool test_flag_tau = true;
      for (int i(0); (i < m) && test_flag_tau; ++i) {
        if (ats::abs(h_tau[i] - refTau[i]) > absTol) {
          test_flag_tau = false;
        }
      }
      ASSERT_EQ(test_flag_tau, true);
    }
  }

  ViewTypeA Q("Q", m, m);
  ViewTypeA R("R", m, n);
  ViewTypeA QR("QR", m, n);

  typename ViewTypeA::HostMirror h_Q  = Kokkos::create_mirror_view(Q);
  typename ViewTypeA::HostMirror h_R  = Kokkos::create_mirror_view(R);
  typename ViewTypeA::HostMirror h_QR = Kokkos::create_mirror_view(QR);

  getQR<ViewTypeA, ViewTypeTau>(m, n, h_A, h_tau, h_Q, h_R, h_QR);

#ifdef HAVE_KOKKOSKERNELS_DEBUG
  for (int i(0); i < m; ++i) {
    for (int j(0); j < m; ++j) {
      std::cout << "Q(" << i << "," << j << ") = " << h_Q(i,j) << std::endl;
    }
  }
  for (int i(0); i < m; ++i) {
    for (int j(0); j < n; ++j) {
      std::cout << "R(" << i << "," << j << ") = " << h_R(i,j) << std::endl;
    }
  }
  for (int i(0); i < m; ++i) {
    for (int j(0); j < n; ++j) {
      std::cout << "QR(" << i << "," << j << ") = " << h_QR(i,j) << std::endl;
    }
  }
#endif

  if ((m == 3) && (n == 3)) {
    std::vector<std::vector<ScalarA>> refQ(m);
    for (int i(0); i < m; ++i) {
      refQ[i].resize(n,Kokkos::ArithTraits<ScalarA>::zero());
    }

    std::vector<std::vector<ScalarA>> refR(m);
    for (int i(0); i < m; ++i) {
      refR[i].resize(n,Kokkos::ArithTraits<ScalarA>::zero());
    }

#if 0
    Q = [ -6/7     69/175   58/175
          -3/7   -158/175   -6/175
           2/7     -6/35    33/35 ]

    R = [ -14   -21   14
           0   -175   70
           0      0  -35 ]
#endif

    if constexpr (Kokkos::ArithTraits<ScalarA>::is_complex) {
      refQ[0][0].real() = -6./7.;
      refQ[0][1].real() = 69./175.;
      refQ[0][2].real() = 58./175.;

      refQ[1][0].real() = -3./7.;
      refQ[1][1].real() = -158./175.;
      refQ[1][2].real() = -6./175.;

      refQ[2][0].real() = 2./7.;
      refQ[2][1].real() = -6./35.;
      refQ[2][2].real() = 33./35.;

      refR[0][0].real() = -14.;
      refR[0][1].real() = -21.;
      refR[0][2].real() = 14.;

      refR[1][1].real() = -175.;
      refR[1][2].real() = 70.;

      refR[2][2].real() = -35.;
    }
    else {
      refQ[0][0] = -6./7.;
      refQ[0][1] = 69./175.;
      refQ[0][2] = 58./175.;

      refQ[1][0] = -3./7.;
      refQ[1][1] = -158./175.;
      refQ[1][2] = -6./175.;

      refQ[2][0] = 2./7.;
      refQ[2][1] = -6./35.;
      refQ[2][2] = 33./35.;

      refR[0][0] = -14.;
      refR[0][1] = -21.;
      refR[0][2] = 14.;

      refR[1][1] = -175.;
      refR[1][2] = 70.;

      refR[2][2] = -35.;
    }

    {
      bool test_flag_Q = true;
      for (int i(0); (i < m) && test_flag_Q; ++i) {
        for (int j(0); (j < n) && test_flag_Q; ++j) {
          if (ats::abs(h_Q(i,j) - refQ[i][j]) > absTol) {
            test_flag_Q = false;
          }
        }
      }
      ASSERT_EQ(test_flag_Q, true);
    }

    {
      bool test_flag_R = true;
      for (int i(0); (i < m) && test_flag_R; ++i) {
        for (int j(0); (j < n) && test_flag_R; ++j) {
          if (ats::abs(h_R(i,j) - refR[i][j]) > absTol) {
            test_flag_R = false;
          }
        }
      }
      ASSERT_EQ(test_flag_R, true);
    }
  }

  {
    bool test_flag_QR = true;
    for (int i(0); (i < m) && test_flag_QR; ++i) {
      for (int j(0); (j < n) && test_flag_QR; ++j) {
        if (ats::abs(h_QR(i,j) - h_Aorig(i,j)) > absTol) {
          std::cout << "m = " << m
                    << ", n = " << n
                    << ", i = " << i
                    << ", j = " << j
                    << ", h_Aorig(i,j) = " << std::setprecision(16) << h_Aorig(i,j)
                    << ", h_QR(i,j) = "    << std::setprecision(16) << h_QR(i,j)
                    << ", |diff| = "       << std::setprecision(16) << ats::abs(h_QR(i,j) - h_Aorig(i,j))
                    << ", absTol = "       << std::setprecision(16) << absTol
                    << std::endl;
          test_flag_QR = false;
        }
      }
    }
    ASSERT_EQ(test_flag_QR, true);
  }
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
  Test::impl_test_geqrf<view_type_a_ll, view_type_tau_ll, Device>(100, 100);
  //Test::impl_test_geqrf<view_type_a_ll, view_type_tau_ll, Device>(100, 70); // AquiEEP
  Test::impl_test_geqrf<view_type_a_ll, view_type_tau_ll, Device>(70, 100);
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
