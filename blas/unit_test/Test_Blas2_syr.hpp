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
#include <KokkosBlas2_syr.hpp>
#include <Kokkos_MathematicalConstants.hpp>

namespace Test {

constexpr double piVal = 3.14159265358979323846;

template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
class SyrTester
{
public:
  SyrTester();

  ~SyrTester();

  void test( const int  N
           , const int  nonConstConstCombinations
           , const bool useAnalyticalResults = false
           , const bool useHermitianOption   = false
           , const bool useUpOption          = false
           );

private:
  typedef Kokkos::View<ScalarX*,  tLayoutX, Device> _ViewTypeX;
  typedef Kokkos::View<ScalarA**, tLayoutA, Device> _ViewTypeA;

  typedef typename _ViewTypeX::HostMirror                      _HostViewTypeX;
  typedef typename _ViewTypeA::HostMirror                      _HostViewTypeA;
  typedef Kokkos::View<ScalarA**, tLayoutA, Kokkos::HostSpace> _ViewTypeExpected;

  typedef Kokkos::ArithTraits<ScalarA> _KAT_A;
  typedef typename _KAT_A::mag_type    _AuxType;

  void populateVariables( ScalarA           & alpha
                        , _HostViewTypeX    & h_x
                        , _HostViewTypeA    & h_A
                        , _ViewTypeExpected & h_expected
                        , _ViewTypeX        & x
                        , _ViewTypeA        & A
                        , bool              & expectedResultIsKnown
                        );

  void makeMatrixSymmetric(_HostViewTypeA & h_A);

  void makeMatrixHermitian(_HostViewTypeA & h_A);
  
  template <class T>
  typename std::enable_if< std::is_same<T,Kokkos::complex<float>>::value || std::is_same<T,Kokkos::complex<double>>::value
                         , void
                         >::type
  populateAnalyticalValues( T                 & alpha
                          , _HostViewTypeX    & h_x
                          , _HostViewTypeA    & h_A
                          , _ViewTypeExpected & h_expected
                          );
  
  template <class T>
  typename std::enable_if< !std::is_same<T,Kokkos::complex<float>>::value && !std::is_same<T,Kokkos::complex<double>>::value
                         , void
                         >::type
  populateAnalyticalValues( T                 & alpha
                          , _HostViewTypeX    & h_x
                          , _HostViewTypeA    & h_A
                          , _ViewTypeExpected & h_expected
                          );

  template <class T>
  typename std::enable_if< std::is_same<T,Kokkos::complex<float>>::value || std::is_same<T,Kokkos::complex<double>>::value
                         , void
                         >::type
  populateVanillaValues( const T              & alpha
                       , const _HostViewTypeX & h_x
                       , const _HostViewTypeA & h_A
                       , _ViewTypeExpected    & h_vanilla
                       );
  
  template <class T>
  typename std::enable_if< !std::is_same<T,Kokkos::complex<float>>::value && !std::is_same<T,Kokkos::complex<double>>::value
                         , void
                         >::type
  populateVanillaValues( const T              & alpha
                       , const _HostViewTypeX & h_x
                       , const _HostViewTypeA & h_A
                       , _ViewTypeExpected    & h_vanilla
                       );
  
  template <class T>
  typename std::enable_if< std::is_same<T,Kokkos::complex<float>>::value || std::is_same<T,Kokkos::complex<double>>::value
                         , void
                         >::type
  compareVanillaExpected( const T                 & alpha
                        , const _ViewTypeExpected & h_vanilla
                        , const _ViewTypeExpected & h_expected
                        );

  template <class T>
  typename std::enable_if< !std::is_same<T,Kokkos::complex<float>>::value && !std::is_same<T,Kokkos::complex<double>>::value
                         , void
                         >::type
  compareVanillaExpected( const T                 & alpha
                        , const _ViewTypeExpected & h_vanilla
                        , const _ViewTypeExpected & h_expected
                        );

  template <class T>
  typename std::enable_if< std::is_same<T,Kokkos::complex<float>>::value || std::is_same<T,Kokkos::complex<double>>::value
                         , void
                         >::type
  compareKokkosExpected( const T                 & alpha
                       , const _HostViewTypeA    & h_A
                       , const _ViewTypeExpected & h_expected
                       );

  template <class T>
  typename std::enable_if< !std::is_same<T,Kokkos::complex<float>>::value && !std::is_same<T,Kokkos::complex<double>>::value
                         , void
                         >::type
  compareKokkosExpected( const T                 & alpha
                       , const _HostViewTypeA    & h_A
                       , const _ViewTypeExpected & h_expected
                       );

  template <class T>
  T shrinkAngleToZeroTwoPiRange(const T input);

  template <class TX>
  void callKkSyrAndCompareAgainstExpected( const ScalarA           & alpha
                                         , TX                      & x
                                         , _ViewTypeA              & A
                                         , const _HostViewTypeA    & h_A
                                         , const _ViewTypeExpected & h_expected
                                         , const std::string       & situation
                                         );

  const bool     _A_is_complex;
  const bool     _A_is_lr;
  const bool     _A_is_ll;
  const bool     _testIsGpu;
  const bool     _vanillaUsesDifferentOrderOfOps;
  const _AuxType _epsAbs;
  const _AuxType _epsRel;
  int            _M;
  int            _N;
  bool           _useAnalyticalResults;
  bool           _useHermitianOption;
  bool           _useUpOption;
  bool           _kkSyrShouldThrowException;
  bool           _kkGerShouldThrowException;
};

template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
SyrTester< ScalarX
         , tLayoutX
         , ScalarA
         , tLayoutA
         , Device
         >::SyrTester()
  : _A_is_complex                  ( std::is_same<ScalarA,Kokkos::complex<float>>::value || std::is_same<ScalarA,Kokkos::complex<double>>::value )
  , _A_is_lr                       ( std::is_same< tLayoutA, Kokkos::LayoutRight >::value )
  , _A_is_ll                       ( std::is_same< tLayoutA, Kokkos::LayoutLeft >::value )
  , _testIsGpu                     ( KokkosKernels::Impl::kk_is_gpu_exec_space< typename Device::execution_space >() )
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
  , _vanillaUsesDifferentOrderOfOps( _A_is_lr && _testIsGpu ) // AquiEEP
#else
  , _vanillaUsesDifferentOrderOfOps( false )
#endif
  , _epsAbs                        (std::is_same<_AuxType, float>::value ? 1.0e-6 : 1.0e-9)
  , _epsRel                        (std::is_same<_AuxType, float>::value ? 5.0e-3 : 1.0e-6)
  , _M                             (-1) 
  , _N                             (-1)
  , _useAnalyticalResults          (false)
  , _useHermitianOption            (false)
  , _useUpOption                   (false)
  , _kkSyrShouldThrowException     (false)
  , _kkGerShouldThrowException     (false)
{
}

template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
SyrTester< ScalarX
         , tLayoutX
         , ScalarA
         , tLayoutA
         , Device
         >::~SyrTester()
{
  // Nothing to do
}

template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
void SyrTester< ScalarX
              , tLayoutX
              , ScalarA
              , tLayoutA
              , Device
              >::test( const int  N
                     , const int  nonConstConstCombinations
                     , const bool useAnalyticalResults
                     , const bool useHermitianOption
                     , const bool useUpOption
                     )
{
  std::cout << "Entering SyrTester::test()... - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - " << std::endl;

  std::cout << "_A_is_complex = "                     << _A_is_complex
            << ", _A_is_lr = "                        << _A_is_lr
            << ", _A_is_ll = "                        << _A_is_ll
            << ", _testIsGpu = "                      << _testIsGpu
            << ", _vanillaUsesDifferentOrderOfOps = " << _vanillaUsesDifferentOrderOfOps
            << ", _epsAbs = "                         << _epsAbs
            << ", _epsRel = "                         << _epsRel
            << ", nonConstConstCombinations = "       << nonConstConstCombinations
            << ", useAnalyticalResults = "            << useAnalyticalResults
            << ", useHermitianOption = "              << useHermitianOption
            << ", useUpOption = "                     << useUpOption
            << std::endl;
  
  // ********************************************************************
  // Step 1 of 7: declare main types and variables
  // ********************************************************************
  _M = N;
  _N = N;
  _useAnalyticalResults = useAnalyticalResults;
  _useHermitianOption   = useHermitianOption;
  _useUpOption          = useUpOption;

#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
  _kkSyrShouldThrowException = false;

  _kkGerShouldThrowException = false;
  if (_A_is_complex && _useHermitianOption) {
    _kkGerShouldThrowException = !_A_is_ll;
  }
#endif

  bool test_x (false);
  bool test_cx(false);
  if (nonConstConstCombinations == 0) {
    test_x = true;
  }
  else if (nonConstConstCombinations == 1) {
    test_cx = true;
  }
  else {
    test_x  = true;
    test_cx = true;
  }

  view_stride_adapter<_ViewTypeX, false> x("X", _M);
  view_stride_adapter<_ViewTypeA, false> A("A", _M, _N);

  view_stride_adapter<_ViewTypeExpected, true> h_expected("expected A += alpha * x * x^{t,h}", _M, _N);
  bool expectedResultIsKnown = false;

  ScalarA alpha(0.);

  // ********************************************************************
  // Step 2 of 7: populate alpha, h_x, h_A, h_expected, x, A
  // ********************************************************************
  this->populateVariables( alpha
                         , x.h_view
                         , A.h_view
                         , h_expected.d_view
                         , x.d_view
                         , A.d_view
                         , expectedResultIsKnown
                         );

  // ********************************************************************
  // Step 3 of 7: populate h_vanilla
  // ********************************************************************
  view_stride_adapter<_ViewTypeExpected, true> h_vanilla("vanilla = A + alpha * x * x^{t,h}", _M, _N);
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "In Test_Blas2_syr.hpp, computing vanilla A with alpha type = %s\n", typeid(alpha).name() );
  this->populateVanillaValues( alpha
                             , x.h_view
                             , A.h_view
                             , h_vanilla.d_view
                             );
  
  // ********************************************************************
  // Step 4 of 7: use h_vanilla and h_expected as appropriate
  // ********************************************************************
  if (expectedResultIsKnown) {
    // ******************************************************************
    // Compare h_vanilla against h_expected
    // ******************************************************************
    this->compareVanillaExpected( alpha
                                , h_vanilla.d_view
                                , h_expected.d_view
                                );
  }
  else {
    // ******************************************************************
    // Copy h_vanilla to h_expected
    // ******************************************************************
    Kokkos::deep_copy(h_expected.d_base, h_vanilla.d_base);
  }
  
  // ********************************************************************
  // Step 5 of 7: test with 'non const x'
  // ********************************************************************
  view_stride_adapter<_ViewTypeA, false> org_A("Org_A", _M, _N); // AquiEEP (see ger as well)
  Kokkos::deep_copy(org_A.d_base, A.d_base);

  if (test_x) {
    this->callKkSyrAndCompareAgainstExpected( alpha
                                            , x.d_view
                                            , A.d_view
                                            , A.h_view
                                            , h_expected.d_view
                                            , "non const x"
                                            );

    if ((_useAnalyticalResults      == false) &&
        (_kkGerShouldThrowException == false)) {
#if 0 // AquiEEP
      this->compareKkSyrAgainstKkGer( alpha
                                    , x.d_view
                                    , A.d_view
                                    , A.h_view
                                    , h_expected.d_view
                                    , "non const x"
                                    );
#endif
    }
  }

  // ********************************************************************
  // Step 6 of 7: test with const x
  // ********************************************************************
  if (test_cx) {
    Kokkos::deep_copy(A.d_base, org_A.d_base);
  
    this->callKkSyrAndCompareAgainstExpected( alpha
                                            , x.d_view_const
                                            , A.d_view
                                            , A.h_view
                                            , h_expected.d_view
                                            , "const x"
                                            );
  }

  // ********************************************************************
  // Step 7 of 7: tests with invalid values on the first input parameter
  // ********************************************************************
  EXPECT_ANY_THROW( KokkosBlas::syr(".", "U", alpha, x.d_view, A.d_view) ) << "Failed test: kk syr should have thrown an exception for mode '.'";
  EXPECT_ANY_THROW( KokkosBlas::syr( "", "U", alpha, x.d_view, A.d_view) ) << "Failed test: kk syr should have thrown an exception for mode ''";
  EXPECT_ANY_THROW( KokkosBlas::syr("T", ".", alpha, x.d_view, A.d_view) ) << "Failed test: kk syr should have thrown an exception for uplo '.'";
  EXPECT_ANY_THROW( KokkosBlas::syr("T",  "", alpha, x.d_view, A.d_view) ) << "Failed test: kk syr should have thrown an exception for uplo ''";

  std::cout << "Leaving SyrTester::test() - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - " << std::endl;
}

template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
void SyrTester< ScalarX
              , tLayoutX
              , ScalarA
              , tLayoutA
              , Device
              >::populateVariables( ScalarA           & alpha
                                  , _HostViewTypeX    & h_x
                                  , _HostViewTypeA    & h_A
                                  , _ViewTypeExpected & h_expected
                                  , _ViewTypeX        & x
                                  , _ViewTypeA        & A
                                  , bool              & expectedResultIsKnown
                                  )
{
  expectedResultIsKnown = false;

  if (_useAnalyticalResults) {
    this->populateAnalyticalValues( alpha
                                  , h_x
                                  , h_A
                                  , h_expected
                                  );
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(A, h_A);

    expectedResultIsKnown = true;
  }
  else if (_N == 1) {
    alpha = 3;

    h_x[0] = 2;

    h_A(0,0) = 7;

    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(A, h_A);

    h_expected(0,0) = 19;
    expectedResultIsKnown = true;
  }
  else if (_N == 2) {
    alpha = 3;

    h_x[0] = -2;
    h_x[1] = 9;

    h_A(0,0) = 17;
    h_A(0,1) = -43;
    h_A(1,0) = -43;
    h_A(1,1) = 101;

    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(A, h_A);

    if (_useUpOption) {
      h_expected(0,0) = 29;
      h_expected(0,1) = -97;
      h_expected(1,0) = -43;
      h_expected(1,1) = 344;
    }
    else {
      h_expected(0,0) = 29;
      h_expected(0,1) = -43;
      h_expected(1,0) = -97;
      h_expected(1,1) = 344;
    }
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
      ScalarA randStart, randEnd;
      Test::getRandomBounds(1.0, randStart, randEnd);
      Kokkos::fill_random(A, rand_pool, randStart, randEnd);
    }

    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_A, A);

    if (_useHermitianOption && _A_is_complex) {
      this->makeMatrixHermitian(h_A);
    }
    else {
      this->makeMatrixSymmetric(h_A);
    }
    Kokkos::deep_copy(A, h_A);
  }

  if (_N <= 2) {
    for (int i(0); i < _M; ++i) {
      for (int j(0); j < _N; ++j) {
        std::cout << "h_origA("   << i << "," << j << ")=" << h_A(i,j)
                  << std::endl;
      }
    }
  }

}

template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
void
SyrTester< ScalarX
         , tLayoutX
         , ScalarA
         , tLayoutA
         , Device
         >::makeMatrixSymmetric(_HostViewTypeA & h_A)
{
  for (int i(0); i < _N; ++i) {
    for (int j(i+1); j < _N; ++j) {
      h_A(i,j) = h_A(j,i);
    }
  }
}

template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
void
SyrTester< ScalarX
         , tLayoutX
         , ScalarA
         , tLayoutA
         , Device
         >::makeMatrixHermitian(_HostViewTypeA & h_A)
{
  for (int i(0); i < _N; ++i) {
    for (int j(i+1); j < _N; ++j) {
      h_A(i,j) = _KAT_A::conj( h_A(j,i) );
    }
  }

  for (int i(0); i < _N; ++i) {
    h_A(i,i) = 0.5 * ( h_A(i,i) + _KAT_A::conj( h_A(i,i) ) );
  }
}
  
// Code for complex values
template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
template <class T>
typename std::enable_if< std::is_same<T,Kokkos::complex<float>>::value || std::is_same<T,Kokkos::complex<double>>::value
                       , void
                       >::type
SyrTester< ScalarX
         , tLayoutX
         , ScalarA
         , tLayoutA
         , Device
         >::populateAnalyticalValues( T                 & alpha
                                    , _HostViewTypeX    & h_x
                                    , _HostViewTypeA    & h_A
                                    , _ViewTypeExpected & h_expected
                                    ) {
  _AuxType auxI(0.);
  _AuxType auxJ(0.);
  _AuxType auxIpJ(0.);
  _AuxType auxImJ(0.);

  alpha.real() =  1.;
  alpha.imag() = -1.;

  for (int i = 0; i < _M; ++i) {
    auxI = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(i) );
    h_x[i].real() = sin(auxI);
    h_x[i].imag() = cos(auxI);
  }

  if (_useHermitianOption) {
    for (int i = 0; i < _M; ++i) {
      auxI = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(i) );
      for (int j = 0; j < _N; ++j) {
        if (( (_useUpOption == true ) && (i <= j) ) ||
            ( (_useUpOption == false) && (i >= j) )) {
            auxJ = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(j) );
            auxIpJ = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(i+j) );
            h_A(i,j).real() = -sin(auxIpJ) - sin(auxI) * sin(auxJ) - cos(auxI) * cos(auxJ);
            h_A(i,j).imag() = -sin(auxIpJ) - sin(auxI) * sin(auxJ) + cos(auxI) * cos(auxJ);
        }
      }
    }
  }
  else {
    for (int i = 0; i < _M; ++i) {
      auxI = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(i) );
      for (int j = 0; j < _N; ++j) {
        if (( (_useUpOption == true ) && (i <= j) ) ||
            ( (_useUpOption == false) && (i >= j) )) {
          auxJ = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(j) );
          auxImJ = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(i-j) );
          h_A(i,j).real() = -sin(auxImJ) - sin(auxI) * sin(auxJ) + cos(auxI) * cos(auxJ);
          h_A(i,j).imag() = -sin(auxImJ) - sin(auxI) * sin(auxJ) - cos(auxI) * cos(auxJ);
        }
      }
    }
  }

  if (_useHermitianOption) {
    for (int i = 0; i < _M; ++i) {
      auxI = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(i) );
      for (int j = 0; j < _N; ++j) {
        if (( (_useUpOption == true ) && (i <= j) ) ||
            ( (_useUpOption == false) && (i >= j) )) {
          auxJ = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(j) );
          auxIpJ = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(i+j) );
          h_expected(i,j).real() = -2. * sin(auxI) * sin(auxJ); // AquiEEP
          h_expected(i,j).imag() = 2. * (cos(auxIpJ) - sin(auxIpJ));
        }
      }
    }
  }
  else {
    for (int i = 0; i < _M; ++i) {
      auxI = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(i) );
      for (int j = 0; j < _N; ++j) {
        if (( (_useUpOption == true ) && (i <= j) ) ||
            ( (_useUpOption == false) && (i >= j) )) {
          auxJ = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(j) );
          auxImJ = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(i-j) );
          h_expected(i,j).real() =  2. * cos(auxI) * cos(auxJ);
          h_expected(i,j).imag() = -2. * sin(auxImJ);
        }
      }
    }
  }
}

// Code for non-complex values
template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
template <class T>
typename std::enable_if< !std::is_same<T,Kokkos::complex<float>>::value && !std::is_same<T,Kokkos::complex<double>>::value
                       , void
                       >::type
SyrTester< ScalarX
         , tLayoutX
         , ScalarA
         , tLayoutA
         , Device
         >::populateAnalyticalValues( T                 & alpha
                                    , _HostViewTypeX    & h_x
                                    , _HostViewTypeA    & h_A
                                    , _ViewTypeExpected & h_expected
                                    ) {
  _AuxType auxI(0.);
  _AuxType auxJ(0.);
  _AuxType auxIpJ(0.);

  alpha = 3;

  for (int i = 0; i < _M; ++i) {
    auxI = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(i) );
    h_x[i] = sin(auxI);
  }

  for (int i = 0; i < _M; ++i) {
    auxI = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(i) );
    for (int j = 0; j < _N; ++j) {
      if (( (_useUpOption == true ) && (i <= j) ) ||
          ( (_useUpOption == false) && (i >= j) )) {
        auxJ = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(j) );
        h_A(i,j) = 3 * cos(auxI) * sin(auxJ);
      }
    }
  }

  for (int i = 0; i < _M; ++i) {
    for (int j = 0; j < _N; ++j) {
      if (( (_useUpOption == true ) && (i <= j) ) ||
          ( (_useUpOption == false) && (i >= j) )) {
        auxIpJ = this->shrinkAngleToZeroTwoPiRange( static_cast<_AuxType>(i+j) );
        h_expected(i,j) = 3 * sin(auxIpJ); // AquiEEP
      }
    }
  }
}

// Code for complex values
template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
template <class T>
typename std::enable_if< std::is_same<T,Kokkos::complex<float>>::value || std::is_same<T,Kokkos::complex<double>>::value
                       , void
                       >::type
SyrTester< ScalarX
         , tLayoutX
         , ScalarA
         , tLayoutA
         , Device
         >::populateVanillaValues( const T              & alpha
                                 , const _HostViewTypeX & h_x
                                 , const _HostViewTypeA & h_A
                                 , _ViewTypeExpected    & h_vanilla
                                 ) {
  if (_vanillaUsesDifferentOrderOfOps) {
    if (_useHermitianOption) {
      for (int i = 0; i < _M; ++i) {
        for (int j = 0; j < _N; ++j) {
          if (( (_useUpOption == true ) && (i <= j) ) ||
              ( (_useUpOption == false) && (i >= j) )) {
            h_vanilla(i,j) = h_A(i,j) + alpha * _KAT_A::conj( h_x(j) ) * h_x(i);
          }
          else {
            h_vanilla(i,j) = h_A(i,j);
          }
        }
      }
    }
    else {
      for (int i = 0; i < _M; ++i) {
        for (int j = 0; j < _N; ++j) {
          if (( (_useUpOption == true ) && (i <= j) ) ||
              ( (_useUpOption == false) && (i >= j) )) {
            h_vanilla(i,j) = h_A(i,j) + alpha * h_x(j) * h_x(i);
          }
          else {
            h_vanilla(i,j) = h_A(i,j);
          }
        }
      }
    }
  }
  else {
    if (_useHermitianOption) {
      for (int i = 0; i < _M; ++i) {
        for (int j = 0; j < _N; ++j) {
          if (( (_useUpOption == true ) && (i <= j) ) ||
              ( (_useUpOption == false) && (i >= j) )) {
            h_vanilla(i,j) = h_A(i,j) + alpha * h_x(i) * _KAT_A::conj( h_x(j) );
          }
          else {
            h_vanilla(i,j) = h_A(i,j);
          }
        }
      }
    }
    else {
      for (int i = 0; i < _M; ++i) {
        for (int j = 0; j < _N; ++j) {
          if (( (_useUpOption == true ) && (i <= j) ) ||
              ( (_useUpOption == false) && (i >= j) )) {
            h_vanilla(i,j) = h_A(i,j) + alpha * h_x(i) * h_x(j);
          }
          else {
            h_vanilla(i,j) = h_A(i,j);
          }
        }
      }
    }
  }
}

// Code for non-complex values
template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
template <class T>
typename std::enable_if< !std::is_same<T,Kokkos::complex<float>>::value && !std::is_same<T,Kokkos::complex<double>>::value
                       , void
                       >::type
SyrTester< ScalarX
         , tLayoutX
         , ScalarA
         , tLayoutA
         , Device
         >::populateVanillaValues( const T              & alpha
                                 , const _HostViewTypeX & h_x
                                 , const _HostViewTypeA & h_A
                                 , _ViewTypeExpected    & h_vanilla
                                 ) {
  if (_vanillaUsesDifferentOrderOfOps) {
    for (int i = 0; i < _M; ++i) {
      for (int j = 0; j < _N; ++j) {
        if (( (_useUpOption == true ) && (i <= j) ) ||
            ( (_useUpOption == false) && (i >= j) )) {
          h_vanilla(i,j) = h_A(i,j) + alpha * h_x(j) * h_x(i);
        }
        else {
          h_vanilla(i,j) = h_A(i,j);
        }
      }
    }
  }
  else {
    for (int i = 0; i < _M; ++i) {
      for (int j = 0; j < _N; ++j) {
        if (( (_useUpOption == true ) && (i <= j) ) ||
            ( (_useUpOption == false) && (i >= j) )) {
          h_vanilla(i,j) = h_A(i,j) + alpha * h_x(i) * h_x(j);
        }
        else {
          h_vanilla(i,j) = h_A(i,j);
        }
      }
    }
  }
}

template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
template <class T>
T SyrTester< ScalarX
           , tLayoutX
           , ScalarA
           , tLayoutA
           , Device
           >::shrinkAngleToZeroTwoPiRange(const T input)
{
  T output(input);
#if 0
  T twoPi( 2. * Kokkos::numbers::pi );
  if (input > 0.) {
    output -= std::floor( input / twoPi ) * twoPi;
  }
  else if (input < 0.) {
    output += std::floor( -input / twoPi ) * twoPi;
  }
#endif
  return output;
}

// Code for complex values
template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
template <class T>
typename std::enable_if< std::is_same<T,Kokkos::complex<float>>::value || std::is_same<T,Kokkos::complex<double>>::value
                       , void
                       >::type
SyrTester< ScalarX
         , tLayoutX
         , ScalarA
         , tLayoutA
         , Device
         >::compareVanillaExpected( const T                 & alpha
                                  , const _ViewTypeExpected & h_vanilla
                                  , const _ViewTypeExpected & h_expected
                                  ) {
  if (_N <= 2) {
    for (int i(0); i < _M; ++i) {
      for (int j(0); j < _N; ++j) {
        std::cout << "h_exp("   << i << "," << j << ")=" << h_expected(i,j)
                  << ", h_van(" << i << "," << j << ")=" << h_vanilla(i,j)
                  << std::endl;
      }
    }
  }

  int maxNumErrorsAllowed( static_cast<double>(_M) * static_cast<double>(_N) * 1.e-3 );

  if (_useAnalyticalResults) {
    int      numErrorsRealAbs   (0);
    int      numErrorsRealRel   (0);
    int      numErrorsImagAbs   (0);
    int      numErrorsImagRel   (0);
    _AuxType diff               (0.);
    _AuxType diffThreshold      (0.);
    bool     errorHappened      (false);
    _AuxType maxErrorRealRel    (0.);
    int      iForMaxErrorRealRel(0);
    int      jForMaxErrorRealRel(0);
    _AuxType maxErrorImagRel    (0.);
    int      iForMaxErrorImagRel(0);
    int      jForMaxErrorImagRel(0);

    for (int i(0); i < _M; ++i) {
      for (int j(0); j < _N; ++j) {
        diff = _KAT_A::abs(h_expected(i,j).real() - h_vanilla(i,j).real());
        errorHappened = false;
        if (h_expected(i,j).real() == 0.) {
          diffThreshold = _KAT_A::abs(_epsAbs);
          if ( diff > diffThreshold ) {
            errorHappened = true;
            numErrorsRealAbs++;
          }
        }
        else {
          _AuxType aux = diff / _KAT_A::abs(h_expected(i,j).real());
          if (maxErrorRealRel < aux) {
            maxErrorRealRel = aux;
            iForMaxErrorRealRel = i;
            jForMaxErrorRealRel = j;
          }

          diffThreshold = _KAT_A::abs(_epsRel * h_expected(i,j).real());
          if ( diff > diffThreshold ) {
            errorHappened = true;
            numErrorsRealRel++;
          }
        }
        if (errorHappened && (numErrorsRealAbs + numErrorsRealRel == 1)) {
          std::cout << "ERROR, i = " << i
                    << ", j = "      << j
                    << ": h_expected(i,j).real() = " << h_expected(i,j).real()
                    << ", h_vanilla(i,j).real() = "  << h_vanilla(i,j).real()
                    << ", _KAT_A::abs(h_expected(i,j).real() - h_vanilla(i,j).real()) = " << diff
                    << ", diffThreshold = "                                               << diffThreshold
                    << std::endl;
        }

        diff = _KAT_A::abs(h_expected(i,j).imag() - h_vanilla(i,j).imag());
        errorHappened = false;
        if (h_expected(i,j).imag() == 0.) {
          diffThreshold = _KAT_A::abs(_epsAbs);
          if ( diff > diffThreshold ) {
            errorHappened = true;
            numErrorsImagAbs++;
          }
        }
        else {
          _AuxType aux = diff / _KAT_A::abs(h_expected(i,j).imag());
          if (maxErrorImagRel < aux) {
            maxErrorImagRel = aux;
            iForMaxErrorImagRel = i;
            jForMaxErrorImagRel = j;
          }

          diffThreshold = _KAT_A::abs(_epsRel * h_expected(i,j).imag());
          if ( diff > diffThreshold ) {
            errorHappened = true;
            numErrorsImagRel++;
          }
        }
        if (errorHappened && (numErrorsImagAbs + numErrorsImagRel == 1)) {
          std::cout << "ERROR, i = " << i
                    << ", j = "      << j
                    << ": h_expected(i,j).imag() = " << h_expected(i,j).imag()
                    << ", h_vanilla(i,j).imag() = "  << h_vanilla(i,j).imag()
                    << ", _KAT_A::abs(h_expected(i,j).imag() - h_vanilla(i,j).imag()) = " << diff
                    << ", diffThreshold = "                                               << diffThreshold
                    << std::endl;
        }
      } // for j
    } // for i

    {
      std::ostringstream msg;
      msg << ", A is " << _M << " by " << _N
          << ", _A_is_lr = "               << _A_is_lr
          << ", _A_is_ll = "               << _A_is_ll
          << ", alpha type = "             << typeid(alpha).name()
          << ", _useHermitianOption = "    << _useHermitianOption
          << ", _useUpOption = "           << _useUpOption
          << ": vanilla differs too much from analytical on real components"
          << ", numErrorsRealAbs = "       << numErrorsRealAbs
          << ", numErrorsRealRel = "       << numErrorsRealRel
          << ", maxErrorRealRel = "        << maxErrorRealRel
          << ", iForMaxErrorRealRel = "    << iForMaxErrorRealRel
          << ", jForMaxErrorRealRel = "    << jForMaxErrorRealRel
          << ", h_expected(i,j).real() = " << ( ((_M > 0) && (_N > 0)) ? h_expected(iForMaxErrorRealRel,jForMaxErrorRealRel).real() : 9.999e+99 )
          << ", h_vanilla(i,j).real() = "  << ( ((_M > 0) && (_N > 0)) ? h_vanilla(iForMaxErrorRealRel,jForMaxErrorRealRel).real() : 9.999e+99 )
          << ", maxNumErrorsAllowed = "    << maxNumErrorsAllowed;

      int numErrorsReal(numErrorsRealAbs + numErrorsRealRel);
      if (numErrorsReal > 0) {
        std::cout<< "WARNING" << msg.str() << std::endl;
      }
      EXPECT_LE(numErrorsReal, maxNumErrorsAllowed) << "Failed test" << msg.str();
    }
    {
      std::ostringstream msg;
      msg << ", A is " << _M << " by " << _N
          << ", _A_is_lr = "               << _A_is_lr
          << ", _A_is_ll = "               << _A_is_ll
          << ", alpha type = "             << typeid(alpha).name()
          << ", _useHermitianOption = "    << _useHermitianOption
          << ", _useUpOption = "           << _useUpOption
          << ": vanilla differs too much from analytical on imag components"
          << ", numErrorsImagAbs = "       << numErrorsImagAbs
          << ", numErrorsImagRel = "       << numErrorsImagRel
          << ", maxErrorImagRel = "        << maxErrorImagRel
          << ", iForMaxErrorImagRel = "    << iForMaxErrorImagRel
          << ", jForMaxErrorImagRel = "    << jForMaxErrorImagRel
          << ", h_expected(i,j).imag() = " << ( ((_M > 0) && (_N > 0)) ? h_expected(iForMaxErrorImagRel,jForMaxErrorImagRel).imag() : 9.999e+99 )
          << ", h_vanilla(i,j).imag() = "  << ( ((_M > 0) && (_N > 0)) ? h_vanilla(iForMaxErrorImagRel,jForMaxErrorImagRel).imag() : 9.999e+99 )
          << ", maxNumErrorsAllowed = "    << maxNumErrorsAllowed;

      int numErrorsImag(numErrorsImagAbs + numErrorsImagRel);
      if (numErrorsImag > 0) {
        std::cout<< "WARNING" << msg.str() << std::endl;
      }
      EXPECT_LE(numErrorsImag, maxNumErrorsAllowed) << "Failed test" << msg.str();
    }
  }
  else {
    int numErrorsReal(0);
    int numErrorsImag(0);

    for (int i(0); i < _M; ++i) {
      for (int j(0); j < _N; ++j) {
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
    EXPECT_EQ(numErrorsReal, 0) << "Failed test"
                                << ", A is " << _M << " by " << _N
                                << ", _A_is_lr = "            << _A_is_lr
                                << ", _A_is_ll = "            << _A_is_ll
                                << ", alpha type = "          << typeid(alpha).name()
                                << ", _useHermitianOption = " << _useHermitianOption
                                << ", _useUpOption = "        << _useUpOption
                                << ": vanilla result is incorrect on real components"
                                << ", numErrorsReal = " << numErrorsReal;
    EXPECT_EQ(numErrorsImag, 0) << "Failed test"
                                << ", A is " << _M << " by " << _N
                                << ", _A_is_lr = "            << _A_is_lr
                                << ", _A_is_ll = "            << _A_is_ll
                                << ", alpha type = "          << typeid(alpha).name()
                                << ", _useHermitianOption = " << _useHermitianOption
                                << ", _useUpOption = "        << _useUpOption
                                << ": vanilla result is incorrect on imag components"
                                << ", numErrorsImag = " << numErrorsImag;
  }
}
  
// Code for non-complex values
template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
template <class T>
typename std::enable_if< !std::is_same<T,Kokkos::complex<float>>::value && !std::is_same<T,Kokkos::complex<double>>::value
                       , void
                       >::type
SyrTester< ScalarX
         , tLayoutX
         , ScalarA
         , tLayoutA
         , Device
         >::compareVanillaExpected( const T                 & alpha
                                  , const _ViewTypeExpected & h_vanilla
                                  , const _ViewTypeExpected & h_expected
                                  ) {
  if (_N <= 2) {
    for (int i(0); i < _M; ++i) {
      for (int j(0); j < _N; ++j) {
        std::cout << "h_exp("   << i << "," << j << ")=" << h_expected(i,j)
                  << ", h_van(" << i << "," << j << ")=" << h_vanilla(i,j)
                  << std::endl;
      }
    }
  }

  int maxNumErrorsAllowed( static_cast<double>(_M) * static_cast<double>(_N) * 1.e-3 );

  if (_useAnalyticalResults) {
    int      numErrorsAbs   (0);
    int      numErrorsRel   (0);
    _AuxType diff           (0.);
    _AuxType diffThreshold  (0.);
    bool     errorHappened  (false);
    _AuxType maxErrorRel    (0.);
    int      iForMaxErrorRel(0);
    int      jForMaxErrorRel(0);

    for (int i(0); i < _M; ++i) {
      for (int j(0); j < _N; ++j) {
        diff = _KAT_A::abs(h_expected(i,j) - h_vanilla(i,j));
        errorHappened = false;
        if (h_expected(i,j) == 0.) {
          diffThreshold = _KAT_A::abs(_epsAbs);
          if (diff > diffThreshold) {
            errorHappened = true;
            numErrorsAbs++;
          }
        }
        else {
          _AuxType aux = diff / _KAT_A::abs(h_expected(i,j));
          if (maxErrorRel < aux) {
            maxErrorRel = aux;
            iForMaxErrorRel = i;
            jForMaxErrorRel = j;
          }

          diffThreshold = _KAT_A::abs(_epsRel * h_expected(i,j));
          if (diff > diffThreshold) {
            errorHappened = true;
            numErrorsRel++;
          }
        }
        if (errorHappened && (numErrorsAbs + numErrorsRel == 1)) {
          std::cout << "ERROR, i = " << i
                    << ", j = "      << j
                    << ": h_expected(i,j) = " << h_expected(i,j)
                    << ", h_vanilla(i,j) = "  << h_vanilla(i,j)
                    << ", _KAT_A::abs(h_expected(i,j) - h_vanilla(i,j)) = " << diff
                    << ", diffThreshold = "                                 << diffThreshold
                    << std::endl;
        }
      } // for j
    } // for i

    {
      std::ostringstream msg;
      msg << ", A is " << _M << " by " << _N
          << ", _A_is_lr = "            << _A_is_lr
          << ", _A_is_ll = "            << _A_is_ll
          << ", alpha type = "          << typeid(alpha).name()
          << ", _useHermitianOption = " << _useHermitianOption
          << ", _useUpOption = "        << _useUpOption
          << ": vanilla differs too much from expected"
          << ", numErrorsAbs = "        << numErrorsAbs
          << ", numErrorsRel = "        << numErrorsRel
          << ", maxErrorRel = "         << maxErrorRel
          << ", iForMaxErrorRel = "     << iForMaxErrorRel
          << ", jForMaxErrorRel = "     << jForMaxErrorRel
          << ", h_expected(i,j) = "     << ( ((_M > 0) && (_N > 0)) ? h_expected(iForMaxErrorRel,jForMaxErrorRel) : 9.999e+99 )
          << ", h_vanilla(i,j) = "      << ( ((_M > 0) && (_N > 0)) ? h_vanilla(iForMaxErrorRel,jForMaxErrorRel) : 9.999e+99 )
          << ", maxNumErrorsAllowed = " << maxNumErrorsAllowed;

      int numErrors(numErrorsAbs + numErrorsRel);
      if (numErrors > 0) {
        std::cout<< "WARNING" << msg.str() << std::endl;
      }
      EXPECT_LE(numErrors, maxNumErrorsAllowed) << "Failed test" << msg.str();
    }
  }
  else {
    int numErrors(0);

    for (int i(0); i < _M; ++i) {
      for (int j(0); j < _N; ++j) {
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
    EXPECT_EQ(numErrors, 0) << "Failed test"
                            << ", A is " << _M << " by " << _N
                            << ", _A_is_lr = "            << _A_is_lr
                            << ", _A_is_ll = "            << _A_is_ll
                            << ", alpha type = "          << typeid(alpha).name()
                            << ", _useHermitianOption = " << _useHermitianOption
                            << ", _useUpOption = "        << _useUpOption
                            << ": vanilla result is incorrect"
                            << ", numErrors = " << numErrors;
  }
}
  
// Code for complex values
template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
template <class T>
typename std::enable_if< std::is_same<T,Kokkos::complex<float>>::value || std::is_same<T,Kokkos::complex<double>>::value
                       , void
                       >::type
SyrTester< ScalarX
         , tLayoutX
         , ScalarA
         , tLayoutA
         , Device
         >::compareKokkosExpected( const T                 & alpha
                                 , const _HostViewTypeA    & h_A
                                 , const _ViewTypeExpected & h_expected
                                 ) {
  if (_N <= 2) {
    for (int i(0); i < _M; ++i) {
      for (int j(0); j < _N; ++j) {
        std::cout << "h_exp(" << i << "," << j << ")=" << h_expected(i,j)
                  << ", h_A(" << i << "," << j << ")=" << h_A(i,j)
                  << std::endl;
      }
    }
  }

  int maxNumErrorsAllowed( static_cast<double>(_M) * static_cast<double>(_N) * 1.e-3 );

  int      numErrorsRealAbs   (0);
  int      numErrorsRealRel   (0);
  int      numErrorsImagAbs   (0);
  int      numErrorsImagRel   (0);
  _AuxType diff               (0.);
  _AuxType diffThreshold      (0.);
  bool     errorHappened      (false);
  _AuxType maxErrorRealRel    (0.);
  int      iForMaxErrorRealRel(0);
  int      jForMaxErrorRealRel(0);
  _AuxType maxErrorImagRel    (0.);
  int      iForMaxErrorImagRel(0);
  int      jForMaxErrorImagRel(0);
  for (int i(0); i < _M; ++i) {
    for (int j(0); j < _N; ++j) {
      diff = _KAT_A::abs(h_expected(i,j).real() - h_A(i,j).real());
      errorHappened = false;
      if (h_expected(i,j).real() == 0.) {
        diffThreshold = _KAT_A::abs(_epsAbs);
        if (diff > diffThreshold) {
          errorHappened = true;
          numErrorsRealAbs++;
        }
      }
      else {
        _AuxType aux = diff / _KAT_A::abs(h_expected(i,j).real());
        if (maxErrorRealRel < aux) {
          maxErrorRealRel = aux;
          iForMaxErrorRealRel = i;
          jForMaxErrorRealRel = j;
        }

        diffThreshold = _KAT_A::abs(_epsRel * h_expected(i,j).real());
        if (diff > diffThreshold) {
          errorHappened = true;
          numErrorsRealRel++;
        }
      }
      if (errorHappened && (numErrorsRealAbs + numErrorsRealRel == 1)) {
        std::cout << "ERROR, i = " << i
                  << ", j = "      << j
                  << ": h_expected(i,j).real() = " << h_expected(i,j).real()
                  << ", h_A(i,j).real() = "        << h_A(i,j).real()
                  << ", _KAT_A::abs(h_expected(i,j).real() - h_A(i,j).real()) = " << diff
                  << ", diffThreshold = "                                         << diffThreshold
                  << std::endl;
      }

      diff = _KAT_A::abs(h_expected(i,j).imag() - h_A(i,j).imag());
      errorHappened = false;
      if (h_expected(i,j).imag() == 0.) {
        diffThreshold = _KAT_A::abs(_epsAbs);
        if (diff > diffThreshold) {
          errorHappened = true;
          numErrorsImagAbs++;
        }
      }
      else {
        _AuxType aux = diff / _KAT_A::abs(h_expected(i,j).imag());
        if (maxErrorImagRel < aux) {
          maxErrorImagRel = aux;
          iForMaxErrorImagRel = i;
          jForMaxErrorImagRel = j;
        }

        diffThreshold = _KAT_A::abs(_epsRel * h_expected(i,j).imag());
        if (diff > diffThreshold) {
          errorHappened = true;
          numErrorsImagRel++;
        }
      }
      if (errorHappened && (numErrorsImagAbs + numErrorsImagRel == 1)) {
        std::cout << "ERROR, i = " << i
                  << ", j = "      << j
                  << ": h_expected(i,j).imag() = " << h_expected(i,j).imag()
                  << ", h_A(i,j).imag() = "        << h_A(i,j).imag()
                  << ", _KAT_A::abs(h_expected(i,j).imag() - h_A(i,j).imag()) = " << diff
                  << ", diffThreshold = "                                         << diffThreshold
                  << std::endl;
      }
    } // for j
  } // for i

  std::cout << "A is " << _M << " by " << _N
            << ", _A_is_lr = "               << _A_is_lr
            << ", _A_is_ll = "               << _A_is_ll
            << ", alpha type = "             << typeid(alpha).name()
            << ", _useHermitianOption = "    << _useHermitianOption
            << ", _useUpOption = "           << _useUpOption
            << ", numErrorsRealAbs = "       << numErrorsRealAbs
            << ", numErrorsRealRel = "       << numErrorsRealRel
            << ", maxErrorRealRel = "        << maxErrorRealRel
            << ", iForMaxErrorRealRel = "    << iForMaxErrorRealRel
            << ", jForMaxErrorRealRel = "    << jForMaxErrorRealRel
            << ", h_expected(i,j).real() = " << ( ((_M > 0) && (_N > 0)) ? h_expected(iForMaxErrorRealRel,jForMaxErrorRealRel).real() : 9.999e+99 )
            << ", h_A(i,j).real() = "        << ( ((_M > 0) && (_N > 0)) ? h_A(iForMaxErrorRealRel,jForMaxErrorRealRel).real() : 9.999e+99 )
            << ", numErrorsImagAbs = "       << numErrorsImagAbs
            << ", numErrorsImagRel = "       << numErrorsImagRel
            << ", maxErrorImagRel = "        << maxErrorImagRel
            << ", iForMaxErrorImagRel = "    << iForMaxErrorImagRel
            << ", jForMaxErrorImagRel = "    << jForMaxErrorImagRel
            << ", h_expected(i,j).imag() = " << ( ((_M > 0) && (_N > 0)) ? h_expected(iForMaxErrorImagRel,jForMaxErrorImagRel).imag() : 9.999e+99 )
            << ", h_A(i,j).imag() = "        << ( ((_M > 0) && (_N > 0)) ? h_A(iForMaxErrorImagRel,jForMaxErrorImagRel).imag() : 9.999e+99 )
            << ", maxNumErrorsAllowed = "    << maxNumErrorsAllowed
            << std::endl;
  if ((_M == 2131) && (_N == 2131)) {
    std::cout << "Information"
              << ": A is " << _M << " by " << _N
              << ", _A_is_lr = "              << _A_is_lr
              << ", _A_is_ll = "              << _A_is_ll
              << ", alpha type = "            << typeid(alpha).name()
              << ", _useHermitianOption = "   << _useHermitianOption
              << ", _useUpOption = "          << _useUpOption
              << ", h_expected(11, 2119) = (" << h_expected(11,2119).real() << ", " << h_expected(11,2119).imag() << ")"
              << ", h_A(11, 2119) = ("        << h_A(11,2119).real()        << ", " << h_A(11,2119).imag()        << ")"
              << std::endl;
    std::cout << "Information"
              << ": A is " << _M << " by " << _N
              << ", _A_is_lr = "               << _A_is_lr
              << ", _A_is_ll = "               << _A_is_ll
              << ", alpha type = "             << typeid(alpha).name()
              << ", _useHermitianOption = "    << _useHermitianOption
              << ", _useUpOption = "           << _useUpOption
              << ", h_expected(710, 1065) = (" << h_expected(710,1065).real() << ", " << h_expected(710,1065).imag() << ")"
              << ", h_A(710, 1065) = ("        << h_A(710,1065).real()        << ", " << h_A(710,1065).imag()        << ")"
              << std::endl;
  }

  {
    std::ostringstream msg;
    msg << ", A is " << _M << " by " << _N
        << ", _A_is_lr = "               << _A_is_lr
        << ", _A_is_ll = "               << _A_is_ll
        << ", alpha type = "             << typeid(alpha).name()
        << ", _useHermitianOption = "    << _useHermitianOption
        << ", _useUpOption = "           << _useUpOption
        << ": syr result is incorrect on real components"
        << ", numErrorsRealAbs = "       << numErrorsRealAbs
        << ", numErrorsRealRel = "       << numErrorsRealRel
        << ", maxErrorRealRel = "        << maxErrorRealRel
        << ", iForMaxErrorRealRel = "    << iForMaxErrorRealRel
        << ", jForMaxErrorRealRel = "    << jForMaxErrorRealRel
        << ", h_expected(i,j).real() = " << ( ((_M > 0) && (_N > 0)) ? h_expected(iForMaxErrorRealRel,jForMaxErrorRealRel).real() : 9.999e+99 )
        << ", h_A(i,j).real() = "        << ( ((_M > 0) && (_N > 0)) ? h_A(iForMaxErrorRealRel,jForMaxErrorRealRel).real() : 9.999e+99 )
        << ", maxNumErrorsAllowed = "    << maxNumErrorsAllowed;

    int numErrorsReal(numErrorsRealAbs + numErrorsRealRel);
    if (numErrorsReal > 0) {
      std::cout<< "WARNING" << msg.str() << std::endl;
    }
    EXPECT_LE(numErrorsReal, maxNumErrorsAllowed) << "Failed test" << msg.str();
  }
  {
    std::ostringstream msg;
    msg << ", A is " << _M << " by " << _N
        << ", _A_is_lr = "               << _A_is_lr
        << ", _A_is_ll = "               << _A_is_ll
        << ", alpha type = "             << typeid(alpha).name()
        << ", _useHermitianOption = "    << _useHermitianOption
        << ", _useUpOption = "           << _useUpOption
        << ": syr result is incorrect on imag components"
        << ", numErrorsImagAbs = "       << numErrorsImagAbs
        << ", numErrorsImagRel = "       << numErrorsImagRel
        << ", maxErrorImagRel = "        << maxErrorImagRel
        << ", iForMaxErrorImagRel = "    << iForMaxErrorImagRel
        << ", jForMaxErrorImagRel = "    << jForMaxErrorImagRel
        << ", h_expected(i,j).imag() = " << ( ((_M > 0) && (_N > 0)) ? h_expected(iForMaxErrorImagRel,jForMaxErrorImagRel).imag() : 9.999e+99 )
        << ", h_A(i,j).imag() = "        << ( ((_M > 0) && (_N > 0)) ? h_A(iForMaxErrorImagRel,jForMaxErrorImagRel).imag() : 9.999e+99 )
        << ", maxNumErrorsAllowed = "    << maxNumErrorsAllowed;

    int numErrorsImag(numErrorsImagAbs + numErrorsImagRel);
    if (numErrorsImag > 0) {
      std::cout<< "WARNING" << msg.str() << std::endl;
    }
    EXPECT_LE(numErrorsImag, maxNumErrorsAllowed) << "Failed test" << msg.str();
  }
}
  
// Code for non-complex values
template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
template <class T>
typename std::enable_if< !std::is_same<T,Kokkos::complex<float>>::value && !std::is_same<T,Kokkos::complex<double>>::value
                       , void
                       >::type
SyrTester< ScalarX
         , tLayoutX
         , ScalarA
         , tLayoutA
         , Device
         >::compareKokkosExpected( const T                 & alpha
                                 , const _HostViewTypeA    & h_A
                                 , const _ViewTypeExpected & h_expected
                                 ) {
  if (_N <= 2) {
    for (int i(0); i < _M; ++i) {
      for (int j(0); j < _N; ++j) {
        std::cout << "h_exp(" << i << "," << j << ")=" << h_expected(i,j)
                  << ", h_A(" << i << "," << j << ")=" << h_A(i,j)
                  << std::endl;
      }
    }
  }

  int maxNumErrorsAllowed( static_cast<double>(_M) * static_cast<double>(_N) * 1.e-3 );

  int      numErrorsAbs   (0);
  int      numErrorsRel   (0);
  _AuxType diff           (0.);
  _AuxType diffThreshold  (0.);
  bool     errorHappened  (false);
  _AuxType maxErrorRel    (0.);
  int      iForMaxErrorRel(0);
  int      jForMaxErrorRel(0);
  for (int i(0); i < _M; ++i) {
    for (int j(0); j < _N; ++j) {
      diff = _KAT_A::abs(h_expected(i,j) - h_A(i,j));
      errorHappened = false;
      if (h_expected(i,j) == 0.) {
        diffThreshold = _KAT_A::abs(_epsAbs);
        if (diff > diffThreshold) {
          errorHappened = true;
          numErrorsAbs++;
        }
      }
      else {
        _AuxType aux = diff / _KAT_A::abs(h_expected(i,j));
        if (maxErrorRel < aux) {
          maxErrorRel = aux;
          iForMaxErrorRel = i;
          jForMaxErrorRel = j;
        }

        diffThreshold = _KAT_A::abs(_epsRel * h_expected(i,j));
        if (diff > diffThreshold) {
          errorHappened = true;
          numErrorsRel++;
        }
      }
      if (errorHappened && (numErrorsAbs + numErrorsRel == 1)) {
        std::cout << "ERROR, i = " << i
                  << ", j = "      << j
                  << ": h_expected(i,j) = " << h_expected(i,j)
                  << ", h_A(i,j) = "        << h_A(i,j)
                  << ", _KAT_A::abs(h_expected(i,j) - h_A(i,j)) = " << diff
                  << ", diffThreshold = "                           << diffThreshold
                  << std::endl;
      }
    } // for j
  } // for i
  std::cout << "A is " << _M << " by " << _N
            << ", _A_is_lr = "            << _A_is_lr
            << ", _A_is_ll = "            << _A_is_ll
            << ", alpha type = "          << typeid(alpha).name()
            << ", _useHermitianOption = " << _useHermitianOption
            << ", _useUpOption = "        << _useUpOption
            << ", numErrorsAbs = "        << numErrorsAbs
            << ", numErrorsRel = "        << numErrorsRel
            << ", maxErrorRel = "         << maxErrorRel
            << ", iForMaxErrorRel = "     << iForMaxErrorRel
            << ", jForMaxErrorRel = "     << jForMaxErrorRel
            << ", h_expected(i,j) = "     << ( ((_M > 0) && (_N > 0)) ? h_expected(iForMaxErrorRel,jForMaxErrorRel) : 9.999e+99 )
            << ", h_A(i,j) = "            << ( ((_M > 0) && (_N > 0)) ? h_A(iForMaxErrorRel,jForMaxErrorRel) : 9.999e+99 )
            << ", maxNumErrorsAllowed = " << maxNumErrorsAllowed
            << std::endl;
  {
    std::ostringstream msg;
    msg << ", A is " << _M << " by " << _N
        << ", _A_is_lr = "            << _A_is_lr
        << ", _A_is_ll = "            << _A_is_ll
        << ", alpha type = "          << typeid(alpha).name()
        << ", _useHermitianOption = " << _useHermitianOption
        << ", _useUpOption = "        << _useUpOption
        << ": syr result is incorrect"
        << ", numErrorsAbs = "        << numErrorsAbs
        << ", numErrorsRel = "        << numErrorsRel
        << ", maxErrorRel = "         << maxErrorRel
        << ", iForMaxErrorRel = "     << iForMaxErrorRel
        << ", jForMaxErrorRel = "     << jForMaxErrorRel
        << ", h_expected(i,j) = "     << ( ((_M > 0) && (_N > 0)) ? h_expected(iForMaxErrorRel,jForMaxErrorRel) : 9.999e+99 )
        << ", h_A(i,j) = "            << ( ((_M > 0) && (_N > 0)) ? h_A(iForMaxErrorRel,jForMaxErrorRel) : 9.999e+99 )
        << ", maxNumErrorsAllowed = " << maxNumErrorsAllowed;

    int numErrors(numErrorsAbs + numErrorsRel);
    if (numErrors > 0) {
      std::cout<< "WARNING" << msg.str() << std::endl;
    }
    EXPECT_LE(numErrors, maxNumErrorsAllowed) << "Failed test" << msg.str();
  }
}

template <class ScalarX, class tLayoutX, class ScalarA, class tLayoutA, class Device>
template <class TX>
void SyrTester< ScalarX
              , tLayoutX
              , ScalarA
              , tLayoutA
              , Device
              >::callKkSyrAndCompareAgainstExpected( const ScalarA           & alpha
                                                   , TX                      & x
                                                   , _ViewTypeA              & A
                                                   , const _HostViewTypeA    & h_A
                                                   , const _ViewTypeExpected & h_expected
                                                   , const std::string       & situation
                                                   )
{
  std::cout << "In Test_Blas2_syr, '" << situation << "', alpha = " << alpha << std::endl;
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "In Test_Blas2_syr.hpp, right before calling KokkosBlas::syr(): ViewTypeA = %s, _kkSyrShouldThrowException=%d\n", typeid(_ViewTypeA).name(), _kkSyrShouldThrowException );
  std::string mode = _useHermitianOption ? "H" : "T";
  std::string uplo = _useUpOption ? "U" : "L";
  bool gotStdException    (false);
  bool gotUnknownException(false);
  try {
    KokkosBlas::syr(mode.c_str(), uplo.c_str(), alpha, x, A);
  }
  catch( const std::exception& e ) {
    std::cout << "In Test_Blas2_syr, '" << situation << "': caught exception, e.what() = " << e.what() << std::endl;
    gotStdException = true;
  }
  catch( ... ) {
    std::cout << "In Test_Blas2_syr, '" << situation << "': caught unknown exception" << std::endl;
    gotUnknownException = true;
  }

  EXPECT_EQ(gotUnknownException, false) << "Failed test, '" << situation << "': unknown exception should not have happened";

  EXPECT_EQ(gotStdException, _kkSyrShouldThrowException) << "Failed test, '" << situation << "': kk syr() should"
                                                         << (_kkSyrShouldThrowException ? " " : " not ")
                                                         << "have thrown a std::exception";

  if (( gotStdException     == false ) &&
      ( gotUnknownException == false )) {
    Kokkos::deep_copy(h_A, A);

    this->compareKokkosExpected( alpha
                               , h_A
                               , h_expected
                               );
  }
}

} // namespace Test

template <class ScalarX, class ScalarA, class Device>
int test_syr( const std::string & caseName ) {
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+==========================================================================\n" );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Starting %s ...\n", caseName.c_str() );

  bool xBool = std::is_same<ScalarX, float>::value ||
               std::is_same<ScalarX, double>::value ||
               std::is_same<ScalarX, Kokkos::complex<float>>::value ||
               std::is_same<ScalarX, Kokkos::complex<double>>::value;
  bool aBool = std::is_same<ScalarA, float>::value ||
               std::is_same<ScalarA, double>::value ||
               std::is_same<ScalarA, Kokkos::complex<float>>::value ||
               std::is_same<ScalarA, Kokkos::complex<double>>::value;
  bool useAnalyticalResults = xBool && aBool;

#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Starting %s for LAYOUTLEFT ...\n", caseName.c_str() );

  if (true) {
    Test::SyrTester<ScalarX, Kokkos::LayoutLeft, ScalarA, Kokkos::LayoutLeft, Device> tester;
    tester.test(0, 0);
    tester.test(1, 0);
    tester.test(2, 0);
    tester.test(13, 0);
    tester.test(1024, 0);

    if (useAnalyticalResults) {
      //tester.test(1024, 0 , true, false, false);
      //tester.test(1024, 0 , true, false, true);
      //tester.test(1024, 0 , true, true, false);
      //tester.test(1024, 0 , true, true, true);
    }

    tester.test(2,  0 , false, false, true);
    tester.test(50, 0 , false, false, true);
    tester.test(2,  0 , false, true, false);
    tester.test(50, 0 , false, true, false);
    tester.test(2,  0 , false, true, true);
    tester.test(50, 0 , false, true, true);

    tester.test(50, 4 );
    tester.test(2131, 0);
  }

  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Finished %s for LAYOUTLEFT\n", caseName.c_str() );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Starting %s for LAYOUTRIGHT ...\n", caseName.c_str() );

  if (true) {
    Test::SyrTester<ScalarX, Kokkos::LayoutRight, ScalarA, Kokkos::LayoutRight, Device> tester;
    tester.test(0, 0);
    tester.test(1, 0);
    tester.test(2, 0);
    tester.test(13, 0);
    tester.test(1024, 0);

    if (useAnalyticalResults) {
      //tester.test(1024, 0, true, false, false);
      //tester.test(1024, 0, true, false, true);
      //tester.test(1024, 0, true, true, false);
      //tester.test(1024, 0, true, true, true);
    }

    tester.test(2,  0, false, false, true);
    tester.test(50, 0, false, false, true);
    tester.test(2,  0, false, true, false);
    tester.test(50, 0, false, true, false);
    tester.test(2,  0, false, true, true);
    tester.test(50, 0, false, true, true);

    tester.test(50, 4);
    tester.test(2131, 0);
  }

  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Finished %s for LAYOUTRIGHT\n", caseName.c_str() );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTSTRIDE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Starting %s for LAYOUTSTRIDE ...\n", caseName.c_str() );

  if (true) {
    Test::SyrTester<ScalarX, Kokkos::LayoutStride, ScalarA, Kokkos::LayoutStride, Device> tester;
    tester.test(0, 0);
    tester.test(1, 0);
    tester.test(2, 0);
    tester.test(13, 0);
    tester.test(1024, 0);

    if (useAnalyticalResults) {
      //tester.test(1024, 0, true, false, false);
      //tester.test(1024, 0, true, false, true);
      //tester.test(1024, 0, true, true, false);
      //tester.test(1024, 0, true, true, true);
    }

    tester.test(2,  0, false, false, true);
    tester.test(50, 0, false, false, true);
    tester.test(2,  0, false, true, false);
    tester.test(50, 0, false, true, false);
    tester.test(2,  0, false, true, true);
    tester.test(50, 0, false, true, true);

    tester.test(50, 4);
    tester.test(2131, 0);
  }

  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Finished %s for LAYOUTSTRIDE\n", caseName.c_str() );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Starting %s for MIXED LAYOUTS ...\n", caseName.c_str() );

  if (true) {
    Test::SyrTester<ScalarX, Kokkos::LayoutStride, ScalarA, Kokkos::LayoutRight, Device> tester;
    tester.test(1, 0);
    tester.test(2, 0);
    tester.test(1024, 0);

    if (useAnalyticalResults) {
      //tester.test(1024, 0, true, false, true);
      //tester.test(1024, 0, true, true, true);
    }

    tester.test(2,  0, false, false, true);
    tester.test(50, 0, false, false, true);
    tester.test(2,  0, false, true, true);
    tester.test(50, 0, false, true, true);
  }

  if (true) {
    Test::SyrTester<ScalarX, Kokkos::LayoutLeft, ScalarA, Kokkos::LayoutRight, Device> tester;
    tester.test(1024, 0);
  }
  
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Finished %s for MIXED LAYOUTS\n", caseName.c_str() );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+--------------------------------------------------------------------------\n" );
#endif

  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Finished %s\n", caseName.c_str() );
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "+==========================================================================\n" );

  return 1;
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, syr_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::syr_float");
  test_syr<float, float, TestExecSpace>( "test case syr_float" );
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, syr_complex_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::syr_complex_float");
  test_syr<Kokkos::complex<float>, Kokkos::complex<float>, TestExecSpace>( "test case syr_complex_float" );
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, syr_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::syr_double");
  test_syr<double, double, TestExecSpace>( "test case syr_double" );
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, syr_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::syr_complex_double");
  test_syr<Kokkos::complex<double>, Kokkos::complex<double>, TestExecSpace>( "test case syr_complex_double" );
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_INT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, syr_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::syr_int");
  test_syr<int, int, TestExecSpace>( "test case syr_int" );
  Kokkos::Profiling::popRegion();
}
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && \
    !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
TEST_F(TestCategory, syr_double_int) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::syr_double_int");
  test_syr<int, float, TestExecSpace>( "test case syr_mixed_types" );
  Kokkos::Profiling::popRegion();
}
#endif
