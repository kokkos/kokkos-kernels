#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosBlas1_rotg.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {

template <class Scalar>
void impl_test_rotg(Scalar a, Scalar b) {
  using KATS = typename Kokkos::ArithTraits<Scalar>;
  using mag_type = typename KATS::mag_type;

  const mag_type eps = 100*KATS::epsilon();
  mag_type c;
  Scalar s;
  KokkosBlas::rotg(a, b, c, s);

  // c**2 + s**2 = 1
  EXPECT_NEAR_KK(c*c + s*s, KATS::one(), eps);
}  // impl_test_rotg
}  // namespace Test

template <class Scalar>
void test_rotg() {
#if (!defined(KOKKOSKERNELS_ETI_ONLY) &&      \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  Test::impl_test_rotg<Scalar>(0.0, 0.0);
  Test::impl_test_rotg<Scalar>(1.0, 0.0);
  Test::impl_test_rotg<Scalar>(0.0, 1.0);
  Test::impl_test_rotg<Scalar>(3.0, 2.0);
#endif
}

#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, rotg_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::rotg_double");
  test_rotg<double>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, rotg_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::rotg_float");
  test_rotg<Kokkos::complex<float>>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, rotg_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::rotg_complex_double");
  test_rotg<Kokkos::complex<double>>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, rotg_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::rotg_complex_float");
  test_rotg<float>();
  Kokkos::Profiling::popRegion();
}
#endif

