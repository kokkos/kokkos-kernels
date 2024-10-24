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

template <class MatricesType, class TauViewType, class TmpViewType, class ErrorViewType>
struct qrFunctor {
  using Scalar = typename MatricesType::value_type;

  MatricesType As;
  TauViewType taus;
  TmpViewType ws;
  MatricesType Qs, Bs;
  ErrorViewType global_error;

  qrFunctor(MatricesType As_, TauViewType taus_, TmpViewType ws_, MatricesType Qs_, MatricesType Bs_,
            ErrorViewType global_error_)
      : As(As_), taus(taus_), ws(ws_), Qs(Qs_), Bs(Bs_), global_error(global_error_) {}

  KOKKOS_FUNCTION
  void operator()(const int matIdx) const {
    auto A   = Kokkos::subview(As, matIdx, Kokkos::ALL, Kokkos::ALL);
    auto tau = Kokkos::subview(taus, matIdx, Kokkos::ALL);
    auto w   = Kokkos::subview(ws, matIdx, Kokkos::ALL);
    auto Q   = Kokkos::subview(Qs, matIdx, Kokkos::ALL, Kokkos::ALL);
    auto B   = Kokkos::subview(Bs, matIdx, Kokkos::ALL, Kokkos::ALL);

    const Scalar SC_one = Kokkos::ArithTraits<Scalar>::one();
    const Scalar tol    = Kokkos::ArithTraits<Scalar>::eps() * 10;

    int error = 0;

    KokkosBatched::SerialQR<KokkosBlas::Algo::QR::Unblocked>::invoke(A, tau, w);

    // Store identity in Q
    for (int idx = 0; idx < Q.extent_int(0); ++idx) {
      Q(idx, idx) = SC_one;
    }

    // Call ApplyQ on Q
    KokkosBatched::SerialApplyQ<Side::Left, Trans::NoTranspose, Algo::ApplyQ::Unblocked>::invoke(A, tau, Q, w);

    // Now apply Q' to Q
    KokkosBatched::SerialApplyQ<Side::Left, Trans::Transpose, Algo::ApplyQ::Unblocked>::invoke(A, tau, Q, w);

    // At this point Q stores Q'Q
    // which should be the identity matrix
    for (int rowIdx = 0; rowIdx < Q.extent_int(0); ++rowIdx) {
      for (int colIdx = 0; colIdx < Q.extent_int(1); ++colIdx) {
        if (rowIdx == colIdx) {
          if (Kokkos::abs(Q(rowIdx, colIdx) - SC_one) > tol) {
            error += 1;
            Kokkos::printf("Q(%d, %d)=%e instead of 1.0.\n", rowIdx, colIdx, Q(rowIdx, colIdx));
          }
        } else {
          if (Kokkos::abs(Q(rowIdx, colIdx)) > tol) {
            error += 1;
            Kokkos::printf("Q(%d, %d)=%e instead of 0.0.\n", rowIdx, colIdx, Q(rowIdx, colIdx));
          }
        }
      }
    }
    Kokkos::atomic_add(&global_error(), error);

    // Apply Q' to B which holds a copy of the orginal A
    // Afterwards B should hold a copy of R and be zero below its diagonal
    KokkosBatched::SerialApplyQ<Side::Left, Trans::Transpose, Algo::ApplyQ::Unblocked>::invoke(A, tau, B, w);
    for (int rowIdx = 0; rowIdx < B.extent_int(0); ++rowIdx) {
      for (int colIdx = 0; colIdx < B.extent_int(1); ++colIdx) {
        if (rowIdx <= colIdx) {
          if (Kokkos::abs(B(rowIdx, colIdx) - A(rowIdx, colIdx)) > tol * Kokkos::abs(A(rowIdx, colIdx))) {
            error += 1;
            Kokkos::printf("B(%d, %d)=%e instead of %e.\n", rowIdx, colIdx, B(rowIdx, colIdx), A(rowIdx, colIdx));
          }
        } else {
          if (Kokkos::abs(B(rowIdx, colIdx)) > 1000 * tol) {
            error += 1;
            Kokkos::printf("B(%d, %d)=%e instead of 0.0.\n", rowIdx, colIdx, B(rowIdx, colIdx));
          }
        }
      }
    }
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

  Test::EXPECT_NEAR_KK_REL(tau(0), static_cast<Scalar>(5. / 8.), tol);
  Test::EXPECT_NEAR_KK_REL(tau(1), static_cast<Scalar>(10. / 18.), tol);

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

  {  // Square matrix case
    constexpr int numMat     = 314;
    constexpr int maxMatSize = 36;
    Kokkos::View<Scalar**, ExecutionSpace> tau("tau", numMat, maxMatSize);
    Kokkos::View<Scalar**, ExecutionSpace> tmp("work buffer", numMat, maxMatSize);
    Kokkos::View<Scalar***, ExecutionSpace> As("A matrices", numMat, maxMatSize, maxMatSize);
    Kokkos::View<Scalar***, ExecutionSpace> Bs("B matrices", numMat, maxMatSize, maxMatSize);
    Kokkos::View<Scalar***, ExecutionSpace> Qs("Q matrices", numMat, maxMatSize, maxMatSize);
    Kokkos::View<int, ExecutionSpace> global_error("global number of error");

    Kokkos::Random_XorShift64_Pool<ExecutionSpace> rand_pool(2718);
    constexpr double max_val = 1000;
    {
      Scalar randStart, randEnd;
      Test::getRandomBounds(max_val, randStart, randEnd);
      Kokkos::fill_random(ExecutionSpace{}, As, rand_pool, randStart, randEnd);
    }
    Kokkos::deep_copy(Bs, As);

    qrFunctor myFunc(As, tau, tmp, Qs, Bs, global_error);
    Kokkos::parallel_for("KokkosBatched::test_QR_batch", Kokkos::RangePolicy<ExecutionSpace>(0, numMat), myFunc);

    typename Kokkos::View<int, ExecutionSpace>::HostMirror global_error_h = Kokkos::create_mirror_view(global_error);
    Kokkos::deep_copy(global_error_h, global_error);
    EXPECT_EQ(global_error_h(), 0);
  }

  {  // Rectangular matrix case
    constexpr int numMat  = 314;
    constexpr int numRows = 42;
    constexpr int numCols = 36;
    Kokkos::View<Scalar**, ExecutionSpace> tau("tau", numMat, numCols);
    Kokkos::View<Scalar**, ExecutionSpace> tmp("work buffer", numMat, numCols);
    Kokkos::View<Scalar***, ExecutionSpace> As("A matrices", numMat, numRows, numCols);
    Kokkos::View<Scalar***, ExecutionSpace> Bs("B matrices", numMat, numRows, numCols);
    Kokkos::View<Scalar***, ExecutionSpace> Qs("Q matrices", numMat, numRows, numRows);
    Kokkos::View<int, ExecutionSpace> global_error("global number of error");

    Kokkos::Random_XorShift64_Pool<ExecutionSpace> rand_pool(2718);
    constexpr double max_val = 1000;
    {
      Scalar randStart, randEnd;
      Test::getRandomBounds(max_val, randStart, randEnd);
      Kokkos::fill_random(ExecutionSpace{}, As, rand_pool, randStart, randEnd);
    }
    Kokkos::deep_copy(Bs, As);

    qrFunctor myFunc(As, tau, tmp, Qs, Bs, global_error);
    Kokkos::parallel_for("KokkosBatched::test_QR_batch", Kokkos::RangePolicy<ExecutionSpace>(0, numMat), myFunc);

    typename Kokkos::View<int, ExecutionSpace>::HostMirror global_error_h = Kokkos::create_mirror_view(global_error);
    Kokkos::deep_copy(global_error_h, global_error);
    EXPECT_EQ(global_error_h(), 0);
  }
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
