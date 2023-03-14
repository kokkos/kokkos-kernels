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

namespace Test {

// Code for complex values
template < class ScalarA
         , class X_HostType
         , class Y_HostType
         , class A_HostType
         , class E_HostType
         , typename std::enable_if< std::is_same<ScalarA,Kokkos::complex<float>>::value || std::is_same<ScalarA,Kokkos::complex<double>>::value >::type* = nullptr
         >
void implTestGer_populateAnalyticalValues( const int    M
                                         , const int    N
                                         , const bool   useHermitianOption
                                         , ScalarA    & alpha
                                         , X_HostType & h_x
                                         , Y_HostType & h_y
                                         , A_HostType & h_A
                                         , E_HostType & h_expected
                                         ) {
  alpha.real() =  1.;
  alpha.imag() = -1.;

  for (int i = 0; i < M; i++) {
    h_x[i].real() = sin(i);
    h_x[i].imag() = cos(i);
  }

  for (int j = 0; j < N; j++) {
    h_y[j].real() = cos(j);
    h_y[j].imag() = sin(j);
  }

  if (useHermitianOption) { // Aqui
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        h_A(i,j).real() = -sin(i+j) - sin(i) * sin(j) - cos(i) * cos(j);
        h_A(i,j).imag() = -sin(i+j) - sin(i) * sin(j) + cos(i) * cos(j);
      }
    }
  }
  else {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        h_A(i,j).real() = -sin(i-j) - sin(i) * sin(j) + cos(i) * cos(j);
        h_A(i,j).imag() = -sin(i-j) - sin(i) * sin(j) - cos(i) * cos(j);
      }
    }
  }

  if (useHermitianOption) { // Aqui
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        h_expected(i,j).real() = -2. * sin(i) * sin(j);
        h_expected(i,j).imag() = -2. * sin(i+j);
      }
    }
  }
  else {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        h_expected(i,j).real() =  2. * cos(i) * cos(j);
        h_expected(i,j).imag() = -2. * sin(i-j);
      }
    }
  }
}

// Code for non-complex values
template < class ScalarA
         , class X_HostType
         , class Y_HostType
         , class A_HostType
         , class E_HostType
         , typename std::enable_if< !std::is_same<ScalarA,Kokkos::complex<float>>::value && !std::is_same<ScalarA,Kokkos::complex<double>>::value >::type* = nullptr
         >
void implTestGer_populateAnalyticalValues( const int    M
                                         , const int    N
                                         , const bool   /* useHermitianOption */
                                         , ScalarA    & alpha
                                         , X_HostType & h_x
                                         , Y_HostType & h_y
                                         , A_HostType & h_A
                                         , E_HostType & h_expected
                                         ) {
  alpha = 3;

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

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      h_expected(i,j) = 3 * sin(i+j);
    }
  }
}

// Code for complex values
template < class ScalarA
         , class X_HostType
         , class Y_HostType
         , class A_HostType
         , class E_HostType
         , typename std::enable_if< std::is_same<ScalarA,Kokkos::complex<float>>::value || std::is_same<ScalarA,Kokkos::complex<double>>::value >::type* = nullptr
         >
void implTestGer_populateVanillaValues( const int          M
                                      , const int          N
                                      , const bool         useHermitianOption
                                      , const ScalarA    & alpha
                                      , const X_HostType & h_x
                                      , const Y_HostType & h_y
                                      , const A_HostType & h_A
                                      , const bool         useDifferentOrderOfOperations
                                      , E_HostType       & h_vanilla
                                      ) {
  if (useDifferentOrderOfOperations) {
    if (useHermitianOption) {
      typedef Kokkos::ArithTraits<ScalarA> KAT_A;
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          h_vanilla(i,j) = h_A(i,j) + alpha * KAT_A::conj( h_y(j) ) * h_x(i);
        }
      }
    }
    else {
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          h_vanilla(i,j) = h_A(i,j) + alpha * h_y(j) * h_x(i);
        }
      }
    }
  }
  else {
    if (useHermitianOption) {
      typedef Kokkos::ArithTraits<ScalarA> KAT_A;
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          h_vanilla(i,j) = h_A(i,j) + alpha * h_x(i) * KAT_A::conj( h_y(j) );
        }
      }
    }
    else {
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          h_vanilla(i,j) = h_A(i,j) + alpha * h_x(i) * h_y(j);
        }
      }
    }
  }
}

// Code for non-complex values
template < class ScalarA
         , class X_HostType
         , class Y_HostType
         , class A_HostType
         , class E_HostType
         , typename std::enable_if< !std::is_same<ScalarA,Kokkos::complex<float>>::value && !std::is_same<ScalarA,Kokkos::complex<double>>::value >::type* = nullptr
         >
void implTestGer_populateVanillaValues( const int          M
                                      , const int          N
                                      , const bool         /* useHermitianOption */
                                      , const ScalarA    & alpha
                                      , const X_HostType & h_x
                                      , const Y_HostType & h_y
                                      , const A_HostType & h_A
                                      , const bool         useDifferentOrderOfOperations
                                      , E_HostType       & h_vanilla
                                      ) {
  if (useDifferentOrderOfOperations) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        h_vanilla(i,j) = h_A(i,j) + alpha * h_y(j) * h_x(i);
      }
    }
  }
  else {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        h_vanilla(i,j) = h_A(i,j) + alpha * h_x(i) * h_y(j);
      }
    }
  }
}

// Code for complex values
template < class ScalarA
         , class E_HostType
         , class Eps_Type
         , typename std::enable_if< std::is_same<ScalarA,Kokkos::complex<float>>::value || std::is_same<ScalarA,Kokkos::complex<double>>::value >::type* = nullptr
         >
void implTestGer_compareVanillaExpected( const bool         A_is_lr
                                       , const int          M
                                       , const int          N
                                       , const bool         useAnalyticalResults
                                       , const bool         useHermitianOption
                                       , const ScalarA    & alpha
                                       , const E_HostType & h_vanilla
                                       , const E_HostType & h_expected
                                       , const Eps_Type     eps
                                       ) {
  int numErrorsReal(0);
  int numErrorsImag(0);
  if (useAnalyticalResults) {
    typedef Kokkos::ArithTraits<ScalarA> KAT_A;
    for (int i(0); i < M; ++i) {
      for (int j(0); j < N; ++j) {
        if ( KAT_A::abs(h_expected(i,j).real() - h_vanilla(i,j).real()) > KAT_A::abs(eps * h_expected(i,j).real()) ) {
          if (numErrorsReal == 0) {
            std::cout << "ERROR, i = " << i
                      << ", j = "      << j
                      << ": h_expected(i,j).real() = " << h_expected(i,j).real()
                      << ", h_vanilla(i,j).real() = "  << h_vanilla(i,j).real()
                      << ", KAT_A::abs(h_expected(i,j).real() - h_vanilla(i,j).real()) = " << KAT_A::abs(h_expected(i,j).real() - h_vanilla(i,j).real())
                      << ", KAT_A::abs(eps * h_expected(i,j).real()) = "            << KAT_A::abs(eps * h_expected(i,j).real())
                      << std::endl;
          }
          numErrorsReal++;
        }
        if ( KAT_A::abs(h_expected(i,j).imag() - h_vanilla(i,j).imag()) > KAT_A::abs(eps * h_expected(i,j).imag()) ) {
          if (numErrorsImag == 0) {
            std::cout << "ERROR, i = " << i
                      << ", j = "      << j
                      << ": h_expected(i,j).imag() = " << h_expected(i,j).imag()
                      << ", h_vanilla(i,j).imag() = "  << h_vanilla(i,j).imag()
                      << ", KAT_A::abs(h_expected(i,j).imag() - h_vanilla(i,j).imag()) = " << KAT_A::abs(h_expected(i,j).imag() - h_vanilla(i,j).imag())
                      << ", KAT_A::abs(eps * h_expected(i,j).imag()) = "            << KAT_A::abs(eps * h_expected(i,j).imag())
                      << std::endl;
          }
          numErrorsImag++;
        }
      } // for j
    } // for i
    EXPECT_EQ(numErrorsReal, 0) << "A is " << M << " by " << N
                                << ", A_is_lr = "            << A_is_lr
                                << ", alpha type = "         << typeid(alpha).name()
                                << ", useHermitianOption = " << useHermitianOption
                                << ": vanilla differs too much from analytical on real components"
                                << ", numErrorsReal = " << numErrorsReal;
    EXPECT_EQ(numErrorsImag, 0) << "A is " << M << " by " << N
                                << ", A_is_lr = "            << A_is_lr
                                << ", alpha type = "         << typeid(alpha).name()
                                << ", useHermitianOption = " << useHermitianOption
                                << ": vanilla differs too much from analytical on imag components"
                                << ", numErrorsImag = " << numErrorsImag;
  }
  else {
    for (int i(0); i < M; ++i) {
      for (int j(0); j < N; ++j) {
        if ( h_expected(i,j).real() != h_vanilla(i,j).real() ) {
          if (numErrorsReal == 0) {
            std::cout << "ERROR, i = " << i
                      << ", j = "      << j
                      << ": h_expected(i,j).real() = " << h_expected(i,j).real()
                      << ", h_vanilla(i,j).real() = "  << h_vanilla(i,j).real()
                      << std::endl;
          }
          numErrorsReal++;
        }
        if ( h_expected(i,j).imag() != h_vanilla(i,j).imag() ) {
          if (numErrorsImag == 0) {
            std::cout << "ERROR, i = " << i
                      << ", j = "      << j
                      << ": h_expected(i,j).imag() = " << h_expected(i,j).imag()
                      << ", h_vanilla(i,j).imag() = "  << h_vanilla(i,j).imag()
                      << std::endl;
          }
          numErrorsImag++;
        }
      } // for j
    } // for i
    EXPECT_EQ(numErrorsReal, 0) << "A is " << M << " by " << N
                                << ", A_is_lr = "            << A_is_lr
                                << ", alpha type = "         << typeid(alpha).name()
                                << ", useHermitianOption = " << useHermitianOption
                                << ": vanilla result is incorrect on real components"
                                << ", numErrorsReal = " << numErrorsReal;
    EXPECT_EQ(numErrorsImag, 0) << "A is " << M << " by " << N
                                << ", A_is_lr = "            << A_is_lr
                                << ", alpha type = "         << typeid(alpha).name()
                                << ", useHermitianOption = " << useHermitianOption
                                << ": vanilla result is incorrect on imag components"
                                << ", numErrorsImag = " << numErrorsImag;
  }
}
  
// Code for non-complex values
template < class ScalarA
         , class E_HostType
         , class Eps_Type
         , typename std::enable_if< !std::is_same<ScalarA,Kokkos::complex<float>>::value && !std::is_same<ScalarA,Kokkos::complex<double>>::value >::type* = nullptr
         >
void implTestGer_compareVanillaExpected( const bool         A_is_lr
                                       , const int          M
                                       , const int          N
                                       , const bool         useAnalyticalResults
                                       , const bool         useHermitianOption
                                       , const ScalarA    & alpha
                                       , const E_HostType & h_vanilla
                                       , const E_HostType & h_expected
                                       , const Eps_Type     eps
                                       ) {
  int numErrors(0);
  if (useAnalyticalResults) {
    typedef Kokkos::ArithTraits<ScalarA> KAT_A;
    for (int i(0); i < M; ++i) {
      for (int j(0); j < N; ++j) {
        if (KAT_A::abs(h_expected(i,j) - h_vanilla(i,j)) > KAT_A::abs(eps * h_expected(i,j))) {
          if (numErrors == 0) {
            std::cout << "ERROR, i = " << i
                      << ", j = "      << j
                      << ": h_expected(i,j) = " << h_expected(i,j)
                      << ", h_vanilla(i,j) = "  << h_vanilla(i,j)
                      << ", KAT_A::abs(h_expected(i,j) - h_vanilla(i,j)) = " << KAT_A::abs(h_expected(i,j) - h_vanilla(i,j))
                      << ", KAT_A::abs(eps * h_expected(i,j)) = "            << KAT_A::abs(eps * h_expected(i,j))
                      << std::endl;
          }
          numErrors++;
        }
      } // for j
    } // for i
    EXPECT_EQ(numErrors, 0) << "A is " << M << " by " << N
                            << ", A_is_lr = "            << A_is_lr
                            << ", alpha type = "         << typeid(alpha).name()
                            << ", useHermitianOption = " << useHermitianOption
                            << ": vanilla differs too much from analytical"
                            << ", numErrors = " << numErrors;
  }
  else {
    for (int i(0); i < M; ++i) {
      for (int j(0); j < N; ++j) {
        if ( h_expected(i,j) != h_vanilla(i,j) ) {
          if (numErrors == 0) {
            std::cout << "ERROR, i = " << i
                      << ", j = "      << j
                      << ": h_expected(i,j) = " << h_expected(i,j)
                      << ", h_vanilla(i,j) = "  << h_vanilla(i,j)
                      << std::endl;
          }
          numErrors++;
        }
      } // for j
    } // for i
    EXPECT_EQ(numErrors, 0) << "A is " << M << " by " << N
                            << ", A_is_lr = "            << A_is_lr
                            << ", alpha type = "         << typeid(alpha).name()
                            << ", useHermitianOption = " << useHermitianOption
                            << ": vanilla result is incorrect"
                            << ", numErrors = " << numErrors;
  }
}
  
// Code for complex values
template < class ScalarA
         , class A_HostType
         , class E_HostType
         , class Eps_Type
         , typename std::enable_if< std::is_same<ScalarA,Kokkos::complex<float>>::value || std::is_same<ScalarA,Kokkos::complex<double>>::value >::type* = nullptr
         >
void implTestGer_compareKokkosExpected( const bool         A_is_lr
                                      , const int          M
                                      , const int          N
                                      , const bool         useHermitianOption
                                      , const ScalarA    & alpha
                                      , const A_HostType & h_A
                                      , const E_HostType & h_expected
                                      , const Eps_Type     eps
                                      ) {
  typedef Kokkos::ArithTraits<ScalarA> KAT_A;
  int numErrorsReal(0);
  int numErrorsImag(0);
  for (int i(0); i < M; ++i) {
    for (int j(0); j < N; ++j) {
      if (KAT_A::abs(h_expected(i,j).real() - h_A(i,j).real()) > KAT_A::abs(eps * h_expected(i,j).real())) {
        if (numErrorsReal == 0) {
          std::cout << "ERROR, i = " << i
                    << ", j = "      << j
                    << ": h_expected(i,j).real() = " << h_expected(i,j).real()
                    << ", h_A(i,j).real() = "        << h_A(i,j).real()
                    << ", KAT_A::abs(h_expected(i,j).real() - h_A(i,j).real()) = " << KAT_A::abs(h_expected(i,j).real() - h_A(i,j).real())
                    << ", KAT_A::abs(eps * h_expected(i,j).real()) = "             << KAT_A::abs(eps * h_expected(i,j).real())
                    << std::endl;
        } 
        numErrorsReal++;
      }
      if (KAT_A::abs(h_expected(i,j).imag() - h_A(i,j).imag()) > KAT_A::abs(eps * h_expected(i,j).imag())) {
        if (numErrorsImag == 0) {
          std::cout << "ERROR, i = " << i
                    << ", j = "      << j
                    << ": h_expected(i,j).imag() = " << h_expected(i,j).imag()
                    << ", h_A(i,j).imag() = "        << h_A(i,j).imag()
                    << ", KAT_A::abs(h_expected(i,j).imag() - h_A(i,j).imag()) = " << KAT_A::abs(h_expected(i,j).imag() - h_A(i,j).imag())
                    << ", KAT_A::abs(eps * h_expected(i,j).imag()) = "             << KAT_A::abs(eps * h_expected(i,j).imag())
                    << std::endl;
        }
        numErrorsImag++;
      }
    } // for j
  } // for i
  std::cout << "A is " << M << " by " << N
            << ", A_is_lr = "            << A_is_lr
            << ", alpha type = "         << typeid(alpha).name()
            << ", useHermitianOption = " << useHermitianOption
            << ", numErrorsReal = " << numErrorsReal
            << ", numErrorsImag = " << numErrorsImag
            << std::endl;
  EXPECT_EQ(numErrorsReal, 0) << "A is " << M << " by " << N
                              << ", A_is_lr = "            << A_is_lr
                              << ", alpha type = "         << typeid(alpha).name()
                              << ", useHermitianOption = " << useHermitianOption
                              << ": ger result is incorrect on real components"
                              << ", numErrorsReal = " << numErrorsReal;
  EXPECT_EQ(numErrorsImag, 0) << "A is " << M << " by " << N
                              << ", A_is_lr = "            << A_is_lr
                              << ", alpha type = "         << typeid(alpha).name()
                              << ", useHermitianOption = " << useHermitianOption
                              << ": ger result is incorrect on imag components"
                              << ", numErrorsImag = " << numErrorsImag;
}
  
// Code for non-complex values
template < class ScalarA
         , class A_HostType
         , class E_HostType
         , class Eps_Type
         , typename std::enable_if< !std::is_same<ScalarA,Kokkos::complex<float>>::value && !std::is_same<ScalarA,Kokkos::complex<double>>::value >::type* = nullptr
         >
void implTestGer_compareKokkosExpected( const bool         A_is_lr
                                      , const int          M
                                      , const int          N
                                      , const bool         useHermitianOption
                                      , const ScalarA    & alpha
                                      , const A_HostType & h_A
                                      , const E_HostType & h_expected
                                      , const Eps_Type     eps
                                      ) {
  typedef Kokkos::ArithTraits<ScalarA> KAT_A;
  int numErrors(0);
  for (int i(0); i < M; ++i) {
    for (int j(0); j < N; ++j) {
      if (KAT_A::abs(h_expected(i,j) - h_A(i,j)) > KAT_A::abs(eps * h_expected(i,j))) {
        if (numErrors == 0) {
          std::cout << "ERROR, i = " << i
                    << ", j = "      << j
                    << ": h_expected(i,j) = " << h_expected(i,j)
                    << ", h_A(i,j) = "            << h_A(i,j)
                    << ", KAT_A::abs(h_expected(i,j) - h_A(i,j)) = " << KAT_A::abs(h_expected(i,j) - h_A(i,j))
                    << ", KAT_A::abs(eps * h_expected(i,j)) = "      << KAT_A::abs(eps * h_expected(i,j))
                    << std::endl;
        }
        numErrors++;
      }
    } // for j
  } // for i
  std::cout << "A is " << M << " by " << N
            << ", A_is_lr = "            << A_is_lr
            << ", alpha type = "         << typeid(alpha).name()
            << ", useHermitianOption = " << useHermitianOption
            << ", numErrors = " << numErrors
            << std::endl;
  EXPECT_EQ(numErrors, 0) << "A is " << M << " by " << N
                          << ", A_is_lr = "            << A_is_lr
                          << ", alpha type = "         << typeid(alpha).name()
                          << ", useHermitianOption = " << useHermitianOption
                          << ": ger result is incorrect"
                          << ", numErrors = " << numErrors;
}
  
template <class ViewTypeX, class ViewTypeY, class ViewTypeA, class Device>
void impl_test_ger( const int M
                  , const int N
                  , const bool useAnalyticalResults = false
                  , const bool useHermitianOption   = false
                  ) {
  // ********************************************************************
  // Step 1 of 7: declare main types and variables
  // ********************************************************************
  typedef typename ViewTypeX::value_type ScalarX;
  typedef typename ViewTypeY::value_type ScalarY;
  typedef typename ViewTypeA::value_type ScalarA;

  ViewTypeX x("X", M);
  ViewTypeY y("Y", N);
  ViewTypeA A("A", M, N);

  typename ViewTypeX::HostMirror h_x = Kokkos::create_mirror_view(x);
  typename ViewTypeY::HostMirror h_y = Kokkos::create_mirror_view(y);
  typename ViewTypeA::HostMirror h_A = Kokkos::create_mirror_view(A);

  Kokkos::View<ScalarA**, Kokkos::HostSpace> h_expected("expected A += alpha * x * y^t", M, N);
  bool expectedResultIsKnown = false;

  ScalarA alpha(0.);

  // ********************************************************************
  // Step 2 of 7: populate alpha, h_x, h_y, h_A, h_expected, x, y, A
  // ********************************************************************
  if (useAnalyticalResults) {
    implTestGer_populateAnalyticalValues( M
                                        , N
                                        , useHermitianOption
                                        , alpha
                                        , h_x
                                        , h_y
                                        , h_A
                                        , h_expected
                                        );
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);
    Kokkos::deep_copy(A, h_A);

    expectedResultIsKnown = true;
  }
  else if ((M == 1) && (N == 1)) {
    alpha = 3;

    h_x[0] = 2;

    h_y[0] = 3;

    h_A(0,0) = 7;

    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);
    Kokkos::deep_copy(A, h_A);

    h_expected(0,0) = 25;
    expectedResultIsKnown = true;
  }
  else if ((M == 1) && (N == 2)) {
    alpha = 3;

    h_x[0] = 2;

    h_y[0] = 3;
    h_y[1] = 4;

    h_A(0,0) = 7;
    h_A(0,1) = -6;

    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);
    Kokkos::deep_copy(A, h_A);

    h_expected(0,0) = 25;
    h_expected(0,1) = 18;
    expectedResultIsKnown = true;
  }
  else if ((M == 2) && (N == 2)) {
    alpha = 3;

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

    h_expected(0,0) = -1;
    h_expected(0,1) = -1;
    h_expected(1,0) = -52;
    h_expected(1,1) = 290;
    expectedResultIsKnown = true;
  }
  else {
    alpha = 3;

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

  // ********************************************************************
  // Step 3 of 7: populate h_vanilla
  // ********************************************************************
  Kokkos::View<ScalarA**, Kokkos::HostSpace> h_vanilla("vanilla = A + alpha * x * y^t", M, N);
  bool A_is_lr = std::is_same< typename ViewTypeA::array_layout, Kokkos::LayoutRight >::value;
  {
    KOKKOS_IMPL_DO_NOT_USE_PRINTF( "In Test_Blas2_ger.hpp, computing vanilla A with alpha type = %s\n", typeid(alpha).name() );
    bool useDifferentOrderOfOperations = false;
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
    bool testIsGpu = KokkosKernels::Impl::kk_is_gpu_exec_space< typename ViewTypeA::execution_space >();
    if ( testIsGpu && A_is_lr ) {
      useDifferentOrderOfOperations = true;
    }
#endif
    implTestGer_populateVanillaValues( M
                                     , N
                                     , useHermitianOption
                                     , alpha
                                     , h_x
                                     , h_y
                                     , h_A
                                     , useDifferentOrderOfOperations
                                     , h_vanilla
                                     );
  }
  
  // ********************************************************************
  // Step 4 of 7: set 'eps' (relative comparison threshold) according to current test
  // ********************************************************************
  typedef Kokkos::ArithTraits<ScalarA> KAT_A;
  typedef typename KAT_A::mag_type EPS_TYPE;
  EPS_TYPE eps( 0. );
  {
    eps = (std::is_same<EPS_TYPE, float>::value ? 2.5e-2 : 5e-10);
  }

  // ********************************************************************
  // Step 5 of 7: use h_vanilla and h_expected as appropriate
  // ********************************************************************
  if (expectedResultIsKnown) {
    // ******************************************************************
    // Compare h_vanilla against h_expected
    // ******************************************************************
    implTestGer_compareVanillaExpected( A_is_lr
                                      , M
                                      , N
                                      , useAnalyticalResults
                                      , useHermitianOption
                                      , alpha
                                      , h_vanilla
                                      , h_expected
                                      , eps
                                      );
  }
  else {
    // ******************************************************************
    // Copy h_vanilla to h_expected
    // ******************************************************************
    Kokkos::deep_copy(h_expected, h_vanilla);
  }
  
  // ********************************************************************
  // Step 6 of 7: update h_A with the results computed with KokkosKernels
  // ********************************************************************
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "In Test_Blas2_ger.hpp, right before calling KokkosBlas::ger(): ViewType = %s\n", typeid(ViewTypeA).name() );
  std::string trans = useHermitianOption ? "H" : "T";
  KokkosBlas::ger(trans.c_str(), alpha, x, y, A);
  Kokkos::deep_copy(h_A, A);

  // ********************************************************************
  // Step 7 of 7: compare KokkosKernels results against the expected ones
  // ********************************************************************
  implTestGer_compareKokkosExpected( A_is_lr
                                   , M
                                   , N
                                   , useHermitianOption
                                   , alpha
                                   , h_A
                                   , h_expected
                                   , eps
                                   );
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
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(13, 1024, true, false);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(13, 1024, true, true);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(50, 40);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(1024, 1024);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(2131, 2131);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(2131, 2131, true, false);
  Test::impl_test_ger<view_type_x_ll, view_type_y_ll, view_type_a_ll, Device>(2131, 2131, true, true);
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
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(13, 1024, true, false);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(13, 1024, true, true);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(50, 40);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(1024, 1024);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(2131, 2131);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(2131, 2131, true, false);
  Test::impl_test_ger<view_type_x_lr, view_type_y_lr, view_type_a_lr, Device>(2131, 2131, true, true);
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
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(13, 1024, true, false);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(13, 1024, true, true);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(50, 40);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(1024, 1024);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(2131, 2131);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(2131, 2131, true, false);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ls, view_type_a_ls, Device>(2131, 2131, true, true);
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Finished %s for LAYOUTSTRIDE\n", caseName.c_str() );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Starting %s for MIXED LAYOUTS ...\n", caseName.c_str() );
  Test::impl_test_ger<view_type_x_ls, view_type_y_ll, view_type_a_lr, Device>(1024, 1024);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ll, view_type_a_lr, Device>(1024, 1024, true, false);
  Test::impl_test_ger<view_type_x_ls, view_type_y_ll, view_type_a_lr, Device>(1024, 1024, true, true);
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

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, ger_complex_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::ger_complex_float");
  test_ger<Kokkos::complex<float>, Kokkos::complex<float>, Kokkos::complex<float>, TestExecSpace>( "test case ger_complex_float" );
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
