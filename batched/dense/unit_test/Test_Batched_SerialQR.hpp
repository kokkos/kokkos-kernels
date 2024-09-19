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
/// \author Luc Berger-Vergiat (lberg@sandia.gov)
/// \author Cameron Smith (smithc11@rpi.edu)

#include "gtest/gtest.h"
#include <random>

#include <KokkosBatched_QR_Decl.hpp>      // KokkosBatched::QR
#include "KokkosBatched_ApplyQ_Decl.hpp"  // KokkosBatched::ApplyQ
#include "KokkosBatched_QR_FormQ_Serial_Internal.hpp"
#include <KokkosBatched_Util.hpp>  // KokkosBlas::Algo
#include <Kokkos_Core.hpp>

template <class MatricesType, class IndexViewType, class TauViewType, class TmpViewType>
struct qrFunctor {
  using Scalar = typename MatricesType::value_type;

  int maxMatSize;
  MatricesType As;
  IndexViewType numRows;
  IndexViewType numCols;
  IndexViewType offsetA;
  TauViewType taus;
  TmpViewType ws;
  MatricesType Qs;
  IndexViewType offsetQ;

  qrFunctor(const int maxMatSize_, MatricesType As_, IndexViewType numRows_, IndexViewType numCols_,
            IndexViewType offsetA_, TauViewType taus_, TmpViewType ws_, MatricesType Qs_, IndexViewType offsetQ_)
      : maxMatSize(maxMatSize_),
        As(As_),
        numRows(numRows_),
        numCols(numCols_),
        offsetA(offsetA_),
        taus(taus_),
        ws(ws_),
        Qs(Qs_),
        offsetQ(offsetQ_) {}

  KOKKOS_FUNCTION
  void operator()(const int matIdx) const {
    Kokkos::View<Scalar**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> A(&As(offsetA(matIdx)), numRows(matIdx),
                                                                      numCols(matIdx));
    Kokkos::View<Scalar*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> tau(&taus(matIdx * maxMatSize),
                                                                       Kokkos::min(numRows(matIdx), numCols(matIdx)));
    Kokkos::View<Scalar*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> w(&ws(matIdx * maxMatSize),
                                                                     Kokkos::min(numRows(matIdx), numCols(matIdx)));
    Kokkos::View<Scalar**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> Q(&Qs(offsetQ(matIdx)), numRows(matIdx),
                                                                      numRows(matIdx));

    KokkosBatched::SerialQR<KokkosBlas::Algo::QR::Unblocked>::invoke(A, tau, w);

    // Store identity in Q
    for (int idx = 0; idx < numRows(matIdx); ++idx) {
      Q(matIdx, matIdx) = Kokkos::ArithTraits<Scalar>::one();
    }

    // Call ApplyQ on Q
    KokkosBatched::SerialApplyQ<Side::Left, Trans::NoTranspose, Algo::ApplyQ::Unblocked>::invoke(A, tau, Q, w);
  }
};

template <class Device, class Scalar, class AlgoTagType>
void test_QR_square() {
  // Analytical test with a rectangular matrix
  //     [12, -51,   4]              [150, -69,  58]        [14,  21, -14]
  // A = [ 6, 167, -68]    Q = 1/175 [ 75, 158,  -6]    R = [ 0, 175, -70]
  //     [-4,  24, -41]              [-50,  30, 165]        [ 0,   0, -35]
  //
  // Expected outputs:
  //                         [ -5,  -3]
  // tau = [5/8, 10/18]  A = [1/2,   5]
  //                         [  0, 1/3]
  //

  using MatrixViewType    = Kokkos::View<Scalar**>;
  using ColVectorViewType = Kokkos::View<Scalar*>;
  using ColWorkViewType   = Kokkos::View<Scalar*>;

  const Scalar tol = 10 * Kokkos::ArithTraits<Scalar>::eps();
  constexpr int m = 3, n = 3;

  MatrixViewType A("A", m, n), B("B", m, n), Q("Q", m, m);
  ColVectorViewType t("t", n);
  ColWorkViewType w("w", n);

  typename MatrixViewType::HostMirror A_h = Kokkos::create_mirror_view(A);
  A_h(0, 0)                               = 12;
  A_h(0, 1)                               = -51;
  A_h(0, 2)                               = 4;
  A_h(1, 0)                               = 6;
  A_h(1, 1)                               = 167;
  A_h(1, 2)                               = -68;
  A_h(2, 0)                               = -4;
  A_h(2, 1)                               = 24;
  A_h(2, 2)                               = -41;

  Kokkos::deep_copy(A, A_h);
  Kokkos::deep_copy(B, A_h);

  Kokkos::parallel_for(
      "serialQR", 1, KOKKOS_LAMBDA(int) {
        // compute the QR factorization of A and store the results in A and t
        // (tau) - see the lapack dgeqp3(...) documentation:
        // www.netlib.org/lapack/explore-html-3.6.1/dd/d9a/group__double_g_ecomputational_ga1b0500f49e03d2771b797c6e88adabbb.html
        KokkosBatched::SerialQR<AlgoTagType>::invoke(A, t, w);
      });

  Kokkos::fence();
  Kokkos::deep_copy(A_h, A);
  typename ColVectorViewType::HostMirror tau = Kokkos::create_mirror_view(t);
  Kokkos::deep_copy(tau, t);

  Test::EXPECT_NEAR_KK_REL(A_h(0, 0), static_cast<Scalar>(-14), tol);
  Test::EXPECT_NEAR_KK_REL(A_h(0, 1), static_cast<Scalar>(-21), tol);
  Test::EXPECT_NEAR_KK_REL(A_h(0, 2), static_cast<Scalar>(14), tol);
  Test::EXPECT_NEAR_KK_REL(A_h(1, 0), static_cast<Scalar>(6. / 26.), tol);
  Test::EXPECT_NEAR_KK_REL(A_h(1, 1), static_cast<Scalar>(-175), tol);
  Test::EXPECT_NEAR_KK_REL(A_h(1, 2), static_cast<Scalar>(70), tol);
  Test::EXPECT_NEAR_KK_REL(A_h(2, 0), static_cast<Scalar>(-4.0 / 26.0), tol);
  // Test::EXPECT_NEAR_KK_REL(A_h(2, 1),   35.0, tol);      // Analytical expression too painful to compute...
  Test::EXPECT_NEAR_KK_REL(A_h(2, 2), static_cast<Scalar>(35), tol);

  Test::EXPECT_NEAR_KK_REL(tau(0), static_cast<Scalar>(7. / 13.), tol);
  // Test::EXPECT_NEAR_KK_REL(tau(1), 25. / 32., tol);      // Analytical expression too painful to compute...
  Test::EXPECT_NEAR_KK_REL(tau(2), static_cast<Scalar>(1. / 2.), tol);

  Kokkos::parallel_for(
      "serialApplyQ", 1, KOKKOS_LAMBDA(int) {
        KokkosBatched::SerialApplyQ<Side::Left, Trans::Transpose, Algo::ApplyQ::Unblocked>::invoke(A, t, B, w);
      });
  typename MatrixViewType::HostMirror B_h = Kokkos::create_mirror_view(B);
  Kokkos::deep_copy(B_h, B);

  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-14), B_h(0, 0), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-21), B_h(0, 1), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(14), B_h(0, 2), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0), B_h(1, 0), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-175), B_h(1, 1), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(70), B_h(1, 2), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0), B_h(2, 0), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0), B_h(2, 1), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(35), B_h(2, 2), tol);

  Kokkos::parallel_for(
      "serialFormQ", 1, KOKKOS_LAMBDA(int) {
        KokkosBatched::SerialQR_FormQ_Internal::invoke(m, n, A.data(), A.stride(0), A.stride(1), t.data(), t.stride(0),
                                                       Q.data(), Q.stride(0), Q.stride(1), w.data());
      });
  typename MatrixViewType::HostMirror Q_h = Kokkos::create_mirror_view(Q);
  Kokkos::deep_copy(Q_h, Q);

  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-6. / 7.), Q_h(0, 0), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(69. / 175.), Q_h(0, 1), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-58. / 175.), Q_h(0, 2), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-3. / 7.), Q_h(1, 0), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-158. / 175.), Q_h(1, 1), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(6. / 175.), Q_h(1, 2), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(2. / 7.), Q_h(2, 0), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-6. / 35.), Q_h(2, 1), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-33. / 35.), Q_h(2, 2), tol);

  Kokkos::parallel_for(
      "serialApplyQ", 1, KOKKOS_LAMBDA(int) {
        KokkosBatched::SerialApplyQ<Side::Left, Trans::Transpose, Algo::ApplyQ::Unblocked>::invoke(A, t, Q, w);
      });
  Kokkos::deep_copy(Q_h, Q);

  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(1.), Q_h(0, 0), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(1.), Q_h(1, 1), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(1.), Q_h(2, 2), tol);

  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), Q_h(0, 1), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), Q_h(0, 2), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), Q_h(1, 0), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), Q_h(1, 2), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), Q_h(2, 0), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), Q_h(2, 1), tol);
}

template <class Device, class Scalar, class AlgoTagType>
void test_QR_rectangular() {
  // Analytical test with a rectangular matrix
  //     [3,  5]        [-0.60, -0.64,  0.48]        [-5, -3]
  // A = [4,  0]    Q = [-0.80, -0.48, -0.36]    R = [ 0,  5]
  //     [0, -3]        [ 0.00, -0.60, -0.80]        [ 0,  0]
  //
  // Expected outputs:
  //                         [ -5,  -3]
  // tau = [5/8, 10/18]  A = [1/2,   5]
  //                         [  0, 1/3]
  //

  using MatrixViewType    = Kokkos::View<Scalar**>;
  using ColVectorViewType = Kokkos::View<Scalar*>;
  using ColWorkViewType   = Kokkos::View<Scalar*>;

  const Scalar tol = 10 * Kokkos::ArithTraits<Scalar>::eps();
  constexpr int m = 3, n = 2;

  MatrixViewType A("A", m, n), B("B", m, n), Q("Q", m, m);
  ColVectorViewType t("t", n);
  ColWorkViewType w("w", n);

  typename MatrixViewType::HostMirror A_h = Kokkos::create_mirror_view(A);
  A_h(0, 0)                               = 3;
  A_h(0, 1)                               = 5;
  A_h(1, 0)                               = 4;
  A_h(1, 1)                               = 0;
  A_h(2, 0)                               = 0;
  A_h(2, 1)                               = -3;

  Kokkos::deep_copy(A, A_h);
  Kokkos::deep_copy(B, A_h);

  Kokkos::parallel_for(
      "serialQR", 1, KOKKOS_LAMBDA(int) {
        // compute the QR factorization of A and store the results in A and t
        // (tau) - see the lapack dgeqp3(...) documentation:
        // www.netlib.org/lapack/explore-html-3.6.1/dd/d9a/group__double_g_ecomputational_ga1b0500f49e03d2771b797c6e88adabbb.html
        KokkosBatched::SerialQR<AlgoTagType>::invoke(A, t, w);
      });

  Kokkos::fence();
  Kokkos::deep_copy(A_h, A);
  typename ColVectorViewType::HostMirror tau = Kokkos::create_mirror_view(t);
  Kokkos::deep_copy(tau, t);

  Test::EXPECT_NEAR_KK_REL(A_h(0, 0), static_cast<Scalar>(-5.0), tol);
  Test::EXPECT_NEAR_KK_REL(A_h(0, 1), static_cast<Scalar>(-3.0), tol);
  Test::EXPECT_NEAR_KK_REL(A_h(1, 0), static_cast<Scalar>(0.5), tol);
  Test::EXPECT_NEAR_KK_REL(A_h(1, 1), static_cast<Scalar>(5.0), tol);
  Test::EXPECT_NEAR_KK(A_h(2, 0), static_cast<Scalar>(0.), tol);
  Test::EXPECT_NEAR_KK_REL(A_h(2, 1), static_cast<Scalar>(1. / 3.), tol);

  Test::EXPECT_NEAR_KK_REL(tau(0), 5. / 8., tol);
  Test::EXPECT_NEAR_KK_REL(tau(1), 10. / 18., tol);

  Kokkos::parallel_for(
      "serialApplyQ", 1, KOKKOS_LAMBDA(int) {
        KokkosBatched::SerialApplyQ<Side::Left, Trans::Transpose, Algo::ApplyQ::Unblocked>::invoke(A, t, B, w);
      });
  typename MatrixViewType::HostMirror B_h = Kokkos::create_mirror_view(B);
  Kokkos::deep_copy(B_h, B);

  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-5.0), B_h(0, 0), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-3.0), B_h(0, 1), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), B_h(1, 0), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(5.0), B_h(1, 1), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), B_h(2, 0), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), B_h(2, 1), tol);

  Kokkos::parallel_for(
      "serialFormQ", 1, KOKKOS_LAMBDA(int) {
        KokkosBatched::SerialQR_FormQ_Internal::invoke(m, n, A.data(), A.stride(0), A.stride(1), t.data(), t.stride(0),
                                                       Q.data(), Q.stride(0), Q.stride(1), w.data());
      });
  typename MatrixViewType::HostMirror Q_h = Kokkos::create_mirror_view(Q);
  Kokkos::deep_copy(Q_h, Q);

  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-0.60), Q_h(0, 0), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(0.64), Q_h(0, 1), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(0.48), Q_h(0, 2), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-0.80), Q_h(1, 0), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-0.48), Q_h(1, 1), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-0.36), Q_h(1, 2), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), Q_h(2, 0), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(-0.60), Q_h(2, 1), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(0.80), Q_h(2, 2), tol);

  Kokkos::parallel_for(
      "serialApplyQ", 1, KOKKOS_LAMBDA(int) {
        KokkosBatched::SerialApplyQ<Side::Left, Trans::Transpose, Algo::ApplyQ::Unblocked>::invoke(A, t, Q, w);
      });
  Kokkos::deep_copy(Q_h, Q);

  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(1.0), Q_h(0, 0), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(1.0), Q_h(1, 1), tol);
  Test::EXPECT_NEAR_KK_REL(static_cast<Scalar>(1.0), Q_h(2, 2), tol);

  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), Q_h(0, 1), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), Q_h(0, 2), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), Q_h(1, 0), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), Q_h(1, 2), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), Q_h(2, 0), tol);
  Test::EXPECT_NEAR_KK(static_cast<Scalar>(0.), Q_h(2, 1), tol);
}

template <class Device, class Scalar, class AlgoTagType>
void test_QR_batch() {
  // Generate a batch of matrices
  // Compute QR factorization
  // Verify that R is triangular
  // Verify that Q is unitary
  // Check that Q*R = A

  using ExecutionSpace = typename Device::execution_space;

  constexpr int numMat     = 314;
  constexpr int maxMatSize = 100;
  Kokkos::View<int*> numRows("rows in matrix", numMat);
  Kokkos::View<int*> numCols("cols in matrix", numMat);
  Kokkos::View<int*> offsetA("matrix offset", numMat + 1);
  Kokkos::View<int*> offsetQ("matrix offset", numMat + 1);
  Kokkos::View<Scalar*> tau("tau", maxMatSize * numMat);
  Kokkos::View<Scalar*> tmp("work buffer", maxMatSize * numMat);

  typename Kokkos::View<int*>::HostMirror numRows_h = Kokkos::create_mirror_view(numRows);
  typename Kokkos::View<int*>::HostMirror numCols_h = Kokkos::create_mirror_view(numCols);
  typename Kokkos::View<int*>::HostMirror offsetA_h = Kokkos::create_mirror_view(offsetA);
  typename Kokkos::View<int*>::HostMirror offsetQ_h = Kokkos::create_mirror_view(offsetQ);

  std::mt19937 gen;
  gen.seed(27182818);
  std::uniform_int_distribution<int> distrib(1, maxMatSize);

  offsetA_h(0) = 0;
  offsetQ_h(0) = 0;
  int a = 0, b = 0;
  for (int matIdx = 0; matIdx < numMat; ++matIdx) {
    a = distrib(gen);
    b = distrib(gen);

    numRows_h(matIdx)     = Kokkos::max(a, b);
    numCols_h(matIdx)     = Kokkos::min(a, b);
    offsetA_h(matIdx + 1) = offsetA_h(matIdx) + a * b;
    offsetQ_h(matIdx + 1) = offsetQ_h(matIdx) + numRows_h(matIdx) * numRows_h(matIdx);
  }

  Kokkos::deep_copy(numRows, numRows_h);
  Kokkos::deep_copy(numCols, numCols_h);
  Kokkos::deep_copy(offsetA, offsetA_h);
  Kokkos::deep_copy(offsetQ, offsetQ_h);

  const int numVals = offsetA_h(numMat);
  Kokkos::View<Scalar*> mats("matrices", numVals);
  Kokkos::View<Scalar*> Qs("Q matrices", offsetQ_h(numMat));

  Kokkos::Random_XorShift64_Pool<ExecutionSpace> rand_pool(2718);
  constexpr double max_val = 1000;
  {
    Scalar randStart, randEnd;
    Test::getRandomBounds(max_val, randStart, randEnd);
    Kokkos::fill_random(ExecutionSpace{}, mats, rand_pool, randStart, randEnd);
  }

  qrFunctor myFunc(maxMatSize, mats, numRows, numCols, offsetA, tau, tmp, Qs, offsetQ);
  Kokkos::parallel_for("KokkosBatched::test_QR_batch", Kokkos::RangePolicy<ExecutionSpace>(0, numMat), myFunc);
}

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F(TestCategory, serial_qr_square_analytic_float) {
  typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
  test_QR_square<TestDevice, float, AlgoTagType>();
}
TEST_F(TestCategory, serial_qr_rectangular_analytic_float) {
  typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
  test_QR_rectangular<TestDevice, float, AlgoTagType>();
}
TEST_F(TestCategory, serial_qr_batch_float) {
  typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
  test_QR_batch<TestDevice, float, AlgoTagType>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, serial_qr_square_analytic_double) {
  typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
  test_QR_square<TestDevice, double, AlgoTagType>();
}
TEST_F(TestCategory, serial_qr_rectangular_analytic_double) {
  typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
  test_QR_rectangular<TestDevice, double, AlgoTagType>();
}
TEST_F(TestCategory, serial_qr_batch_double) {
  typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
  test_QR_batch<TestDevice, double, AlgoTagType>();
}
#endif

// #if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
// TEST_F(TestCategory, batched_scalar_serial_qr_scomplex) {
//   typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
//   test_QR_rectangular<TestDevice, Kokkos::complex<float>, AlgoTagType>();
// }
// #endif

// #if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
// TEST_F(TestCategory, batched_scalar_serial_qr_dcomplex) {
//   typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
//   test_QR_rectangular<TestDevice, Kokkos::complex<double>, AlgoTagType>();
// }
// #endif
