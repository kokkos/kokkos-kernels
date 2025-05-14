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
#include "KokkosBatched_Householder_Decl.hpp"
#include "KokkosBatched_ApplyHouseholder_Decl.hpp"

namespace Test {

template <class Device, class Scalar>
void test_Householder_analytic_real() {
  using ExecutionSpace = typename Device::execution_space;
  using vec_type = Kokkos::View<Scalar*, ExecutionSpace>;

  const Scalar tol = 20 * Kokkos::ArithTraits<Scalar>::eps();

  vec_type vec("vector to reflect", 4), reflector("reflector", 4), tau("tau", 1);
  auto reflector_h = Kokkos::create_mirror_view(reflector);
  reflector_h(0) = 4; reflector_h(1) = 2; reflector_h(2) = 2; reflector_h(3) = 1;
  Kokkos::deep_copy(reflector, reflector_h);
  Kokkos::deep_copy(vec, reflector_h);

  auto tau_h = Kokkos::create_mirror_view(tau);
  SerialHouseholder<KokkosBatched::Side::Left>::invoke(reflector_h, tau_h);

  // x = [4, 2, 2, 1]
  // output = [-5, 2/9, 2/9, 1/9]
  Test::EXPECT_NEAR_KK_REL(reflector_h(0), -5.0, tol);
  Test::EXPECT_NEAR_KK_REL(reflector_h(1),  2.0 /  9.0, tol);
  Test::EXPECT_NEAR_KK_REL(reflector_h(2),  2.0 /  9.0, tol);
  Test::EXPECT_NEAR_KK_REL(reflector_h(3),  1.0 /  9.0, tol);
  Test::EXPECT_NEAR_KK_REL(tau_h(0), 90.0 / 162.0, tol);

  vec_type workspace("workspace", 1);
  auto u = Kokkos::subview(reflector, Kokkos::pair<int, int>(1, 4));
  KokkosBatched::SerialApplyHouseholder<KokkosBatched::Side::Left>::invoke(u, tau, vec, workspace);

  auto vec_h = Kokkos::create_mirror_view(vec);
  Kokkos::deep_copy(vec_h, vec);
  Test::EXPECT_NEAR_KK_REL(vec_h(0), reflector_h(0), tol);
  Test::EXPECT_NEAR_KK(vec_h(1),  0.0, tol);
  Test::EXPECT_NEAR_KK(vec_h(2),  0.0, tol);
  Test::EXPECT_NEAR_KK(vec_h(3),  0.0, tol);
}

template <class Device, class Scalar>
void test_Householder_analytic_cplx() {
  using ExecutionSpace = typename Device::execution_space;
  using vec_type = Kokkos::View<Scalar*, ExecutionSpace>;

  // const Scalar zero = Kokkos::ArithTraits<Scalar>::zero();
  const Scalar tol = 20 * Kokkos::ArithTraits<Scalar>::eps();

  vec_type vec("vector to reflect", 4), reflector("reflector", 4), tau("tau", 1);
  auto reflector_h = Kokkos::create_mirror(vec);
  reflector_h(0) = Kokkos::complex(4., 3.);
  reflector_h(1) = Kokkos::complex(2., 5.);
  reflector_h(2) = Kokkos::complex(2., 5.);
  reflector_h(3) = Kokkos::complex(1., 4.);
  Kokkos::deep_copy(reflector, reflector_h);
  Kokkos::deep_copy(vec, reflector_h);

  auto tau_h = Kokkos::create_mirror_view(tau);
  SerialHouseholder<KokkosBatched::Side::Left>::invoke(reflector_h, tau_h);

  // x = [4+3i, 2+5i, 2+5i, 1+4i]
  // 1 / (x(0) + ||x||) -> (14-3i) / (14*14 + 3*3) = (14-3i) / 205
  // output = [-10, (43 + 64i) / 205, (43 + 64i) / 205, (26 + 53i) / 205]
  Test::EXPECT_NEAR_KK_REL(reflector_h(0), Scalar(-10.0,  0.0), tol);
  Test::EXPECT_NEAR_KK_REL(reflector_h(1), Scalar( 43.0, 64.0) / 205.0, tol);
  Test::EXPECT_NEAR_KK_REL(reflector_h(2), Scalar( 43.0, 64.0) / 205.0, tol);
  Test::EXPECT_NEAR_KK_REL(reflector_h(3), Scalar( 26.0, 53.0) / 205.0, tol);
  Test::EXPECT_NEAR_KK_REL(tau_h(0), Scalar( 1.4,  -0.3) / 2.05, tol);

  std::cout << "Reflector vector: {" << reflector_h(0) << ",  " << reflector_h(1)
	    << ",  " << reflector_h(2) << ",  " << reflector_h(3) << "}" << std::endl;
  std::cout << "tau=" << tau_h(0) << std::endl;

  vec_type workspace("workspace", 1);
  auto u = Kokkos::subview(reflector, Kokkos::pair<int, int>(1, 4));
  KokkosBatched::SerialApplyHouseholder<KokkosBatched::Side::Left>::invoke(u, tau, vec, workspace);

  auto vec_h = Kokkos::create_mirror_view(vec);
  Kokkos::deep_copy(vec_h, vec);
  // Test::EXPECT_NEAR_KK_REL(vec_h(0), reflector_h(0), tol);
  // Test::EXPECT_NEAR_KK(vec_h(1),  zero, tol);
  // Test::EXPECT_NEAR_KK(vec_h(2),  zero, tol);
  // Test::EXPECT_NEAR_KK(vec_h(3),  zero, tol);

  std::cout << "Reflected vector: {" << vec_h(0) << ",  " << vec_h(1) << ",  " << vec_h(2) << ",  " << vec_h(3) << "}" << std::endl;
}

}  // namespace Test


#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F(TestCategory, serial_householder_float) {
  ::Test::test_Householder_analytic_real<TestDevice, float>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, serial_householder_double) {
  ::Test::test_Householder_analytic_real<TestDevice, double>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
TEST_F(TestCategory, serial_householder_cplx_float) {
  ::Test::test_Householder_analytic_cplx<TestDevice, Kokkos::complex<float>>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
TEST_F(TestCategory, serial_householder_cplx_double) {
  ::Test::test_Householder_analytic_cplx<TestDevice, Kokkos::complex<double>>();
}
#endif
