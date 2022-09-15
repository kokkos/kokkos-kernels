#include <KokkosBlas1_rotg.hpp>

namespace Test {
template <class Scalar>
void test_rotg_impl(const Scalar a_in, const Scalar b_in) {
  using magnitude_type     = typename Kokkos::ArithTraits<Scalar>::mag_type;
  const magnitude_type eps = Kokkos::ArithTraits<Scalar>::eps();
  const Scalar zero        = Kokkos::ArithTraits<Scalar>::zero();

  // Initialize inputs/outputs
  Scalar a = a_in;
  Scalar b = b_in;
  magnitude_type c = Kokkos::ArithTraits<magnitude_type>::zero();
  Scalar s = zero;

  KokkosBlas::rotg(a, b, c, s);

  // Check that a*c - b*s == 0
  // and a == sqrt(a*a + b*b)
  EXPECT_NEAR_KK(a_in * s - b_in * c, zero, 10 * eps);
  EXPECT_NEAR_KK(Kokkos::sqrt(a_in * a_in + b_in * b_in), a, 10 * eps);
}
}  // namespace Test

template <class Scalar, class ExecutionSpace>
int test_rotg() {
  const Scalar zero = Kokkos::ArithTraits<Scalar>::zero();
  const Scalar one  = Kokkos::ArithTraits<Scalar>::one();
  const Scalar two  = one + one;

  Test::test_rotg_impl(one, zero);
  Test::test_rotg_impl(one / two, one / two);
  Test::test_rotg_impl(2.1 * one, 1.3 * one);

  return 1;
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, rotg_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::rotg");
  test_rotg<float, TestExecSpace>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&  \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, rotg_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::rotg");
  test_rotg<double, TestExecSpace>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&         \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, rotg_complex_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::rotg");
  test_rotg<Kokkos::complex<float>, TestExecSpace>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&          \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, rotg_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::rotg");
  test_rotg<Kokkos::complex<double>, TestExecSpace>();
  Kokkos::Profiling::popRegion();
}
#endif
