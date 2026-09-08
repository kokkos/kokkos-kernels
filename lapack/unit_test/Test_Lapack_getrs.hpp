// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef TEST_LAPACK_GETRS_HPP
#define TEST_LAPACK_GETRS_HPP

#if (defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK) || defined(KOKKOSKERNELS_ENABLE_TPL_ACCELERATE)) && \
    (defined(TEST_OPENMP_LAPACK_CPP) || defined(TEST_SERIAL_LAPACK_CPP) || defined(TEST_THREADS_LAPACK_CPP))

#include <gtest/gtest.h>
#include <KokkosLapack_getrf.hpp>
#include <KokkosLapack_getrs.hpp>
#include <KokkosKernels_ArithTraits.hpp>

namespace Test {

template <class Scalar, class Device>
void test_getrs_host() {
  using ViewDevice = Kokkos::Device<typename Device::execution_space, typename Device::memory_space>;
  using Matrix = Kokkos::View<Scalar**, Kokkos::LayoutLeft, ViewDevice>;
  using Pivot = Kokkos::View<int*, Kokkos::LayoutLeft, ViewDevice>;
  using AT = KokkosKernels::ArithTraits<Scalar>;
  using Mag = typename AT::mag_type;
  typename Device::execution_space space;
  constexpr int n = 3;
  // LayoutLeft with padding exercises leading dimensions independently of N.
  Matrix a(Kokkos::view_alloc("LU", Kokkos::AllowPadding), n, n);
  Matrix original("original", n, n), saved("saved LU", n, n);
  Pivot piv("pivots", n), saved_piv("saved pivots", n), info("info", 2);
  const double values[3][3] = {{0, 2, 1}, {4, 1, -1}, {2, 3, 5}};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      Scalar value = Scalar(values[i][j]);
      if constexpr (AT::is_complex) value += Scalar(0, (i - j) * 0.25);
      original(i, j) = a(i, j) = value;
    }
  }
  KokkosLapack::getrf(space, a, piv, info);
  space.fence();
  ASSERT_EQ(info(0), 0);
  ASSERT_NE(piv(0), 1);
  Kokkos::deep_copy(saved, a);
  Kokkos::deep_copy(saved_piv, piv);
  typename Matrix::const_type ca = a;
  typename Pivot::const_type cp = piv;
  for (int nrhs : {1, 2}) {
    Kokkos::View<Scalar*, Device> storage("padded RHS", (n + 2) * nrhs);
    Kokkos::LayoutLeft layout(n, nrhs);
    layout.stride = n + 2;
    Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        rhs(storage.data(), layout);
    ASSERT_EQ(rhs.stride(1), n + 2);
    Matrix expected("solution", n, nrhs);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < nrhs; ++j) {
        expected(i, j) = Scalar(i + j + 1);
        if constexpr (AT::is_complex) expected(i, j) += Scalar(0, 0.5 * (j + 1));
      }
    for (const char* trans : {"N", "T", "C", "n", "t", "c"}) {
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < nrhs; ++j) {
          Scalar value = 0;
          for (int k = 0; k < n; ++k) {
            Scalar entry = (trans[0] == 'N' || trans[0] == 'n') ? original(i, k) : original(k, i);
            if (trans[0] == 'C' || trans[0] == 'c') entry = AT::conj(entry);
            value += entry * expected(k, j);
          }
          rhs(i, j) = value;
        }
      Kokkos::deep_copy(info, -7);
      if (trans[0] == 'N') KokkosLapack::getrs(space, trans, a, piv, rhs, info);
      else KokkosLapack::getrs(trans, ca, cp, rhs, info);
      space.fence();
      ASSERT_EQ(info(0), 0);
      EXPECT_EQ(info(1), -7);
      for (int i = 0; i < n; ++i) {
        EXPECT_EQ(piv(i), saved_piv(i));
        for (int j = 0; j < n; ++j) EXPECT_EQ(a(i, j), saved(i, j));
        for (int j = 0; j < nrhs; ++j)
          EXPECT_LE(AT::abs(rhs(i, j) - expected(i, j)), Mag(100) * AT::epsilon());
      }
    }
  }
}
}  // namespace Test

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, getrs_float) { ::Test::test_getrs_host<float, TestDevice>(); }
#endif
#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, getrs_double) { ::Test::test_getrs_host<double, TestDevice>(); }
#endif
#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, getrs_complex_float) { ::Test::test_getrs_host<Kokkos::complex<float>, TestDevice>(); }
#endif
#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, getrs_complex_double) { ::Test::test_getrs_host<Kokkos::complex<double>, TestDevice>(); }
#endif

#endif  // Host LAPACK backend
#endif  // TEST_LAPACK_GETRS_HPP
