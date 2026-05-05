// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosLapack_potrs.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {

template <class AViewType, class BViewType, class Device>
void impl_test_potrs(int N) {
  using ScalarType = typename AViewType::value_type;
  using AT = KokkosKernels::ArithTraits<ScalarType>;
  using MagnitudeType = typename AT::mag_type;

  const MagnitudeType eps = AT::epsilon() * 1000;
  const MagnitudeType max_val = 10.0;

  view_stride_adapter<AViewType> A("A", N, N); // TODO: Adjust dimensions
  view_stride_adapter<BViewType> B("B", N, N); // TODO: Adjust dimensions
  const char uplo[] = "N"; // TODO: Set appropriate value
  int n = 0; // TODO: Set appropriate value
  int nrhs = 0; // TODO: Set appropriate value
  int lda = 0; // TODO: Set appropriate value
  int ldb = 0; // TODO: Set appropriate value

  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);

  // TODO: Initialize input views with random data
  // Example:
  // {
  //   ScalarType randStart, randEnd;
  //   Test::getRandomBounds(max_val, randStart, randEnd);
  //   Kokkos::fill_random(x.d_view, rand_pool, randStart, randEnd);
  // }

  // TODO: Copy input data to host for verification
  // Example: Kokkos::deep_copy(x.h_base, x.d_base);

  // Call your function
  KokkosLapack::potrs(uplo, n, nrhs, A.d_view, lda, B.d_view, ldb);

  // TODO: Copy results back to host
  // Example: Kokkos::deep_copy(y.h_base, y.d_base);

  // TODO: Add your verification logic here
  // Example:
  // for (int i = 0; i < N; i++) {
  //   EXPECT_NEAR_KK(/* expected value */, y.h_view(i), eps);
  // }
}

// TODO: You may also need to add a multivector test

}  // namespace Test

template <class Scalar, class Device>
void test_potrs(int N)
{
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || \
  (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  {
    using AViewType = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, Device>;
    using BViewType = Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device>;

    Test::impl_test_potrs<AViewType, BViewType, Device>(N);
  }
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || \
  (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  {
    using AViewType = Kokkos::View<const Scalar**, Kokkos::LayoutRight, Device>;
    using BViewType = Kokkos::View<Scalar**, Kokkos::LayoutRight, Device>;

    Test::impl_test_potrs<AViewType, BViewType, Device>(N);
  }
#endif
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, potrs_float) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::potrs_float");
  test_potrs<float, TestDevice>(1024);
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, potrs_double) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::potrs_double");
  test_potrs<double, TestDevice>(1024);
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, potrs_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::potrs_complex_double");
  test_potrs<Kokkos::complex<double>, TestDevice>(1024);
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, potrs_complex_float) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::potrs_complex_float");
  test_potrs<Kokkos::complex<float>, TestDevice>(1024);
  Kokkos::Profiling::popRegion();
}
#endif
