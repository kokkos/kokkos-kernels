// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// only enable this test where KokkosLapack supports gesv:
// CUDA+CUSOLVER, HIP+ROCSOLVER and HOST+LAPACK
#if (defined(TEST_CUDA_LAPACK_CPP) && defined(KOKKOSKERNELS_ENABLE_TPL_CUSOLVER)) ||               \
    (defined(TEST_HIP_LAPACK_CPP) && defined(KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER)) ||               \
    ((defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK) || defined(KOKKOSKERNELS_ENABLE_TPL_ACCELERATE)) && \
     (defined(TEST_OPENMP_LAPACK_CPP) || defined(TEST_SERIAL_LAPACK_CPP) || defined(TEST_THREADS_LAPACK_CPP)))

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <KokkosLapack_getrf.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas3_trsm.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {

template <class MatrixType>
struct copy_L_plus_I {
  using ATS = KokkosKernels::ArithTraits<typename MatrixType::non_const_value_type>;

  MatrixType m_A, m_L;

  copy_L_plus_I(const MatrixType &A, const MatrixType &L) : m_A(A), m_L(L) {
    static_assert(MatrixType::rank() == 2);
  }

  KOKKOS_FUNCTION void operator() (const int idx) const {
    const int colIdx = idx / m_A.extent(0);
    const int rowIdx = idx % m_A.extent(0);

    if(colIdx == rowIdx) {
      m_L(rowIdx, colIdx) = ATS::one();
    } else if(colIdx < rowIdx) {
      m_L(rowIdx, colIdx) = m_A(rowIdx, colIdx);
    }
  }
};

template <class AMatrixType>
void impl_test_getrf_sym() {
  using Device   = typename AMatrixType::device_type;
  using IpivType = Kokkos::View<int*, Device>;
  using InfoType = Kokkos::View<int*, Device>;
  using scalar_type    = typename AMatrixType::non_const_value_type;
  using ExecutionSpace = typename Device::execution_space;

  const scalar_type zero = KokkosKernels::ArithTraits<scalar_type>::zero();
  const scalar_type one  = KokkosKernels::ArithTraits<scalar_type>::one();
  const scalar_type two  = one + one;

  AMatrixType A("matrix A", 4, 4);
  auto A_h = Kokkos::create_mirror_view(A);
  A_h(0, 0) =  two; A_h(0, 1) = -one; A_h(0, 2) = zero; A_h(0, 3) = zero;
  A_h(1, 0) = -one; A_h(1, 1) =  two; A_h(1, 2) = -one; A_h(1, 3) = zero;
  A_h(2, 0) = zero; A_h(2, 1) = -one; A_h(2, 2) =  two; A_h(2, 3) = -one;
  A_h(3, 0) = zero; A_h(3, 1) = zero; A_h(3, 2) = -one; A_h(3, 3) =  two;
  Kokkos::deep_copy(A, A_h);

  IpivType ipiv("LU pivots", 4);
  InfoType info("LU info", 1);

  KokkosLapack::getrf(ExecutionSpace(), A, ipiv, info);
  Kokkos::deep_copy(A_h, A);

  const scalar_type three = one + two;
  const scalar_type four  = two + two;
  const scalar_type five  = two + three;

  auto tol  = 10*KokkosKernels::ArithTraits<scalar_type>::eps();

  EXPECT_NEAR_KK_REL(A_h(0, 0),  two, tol);
  EXPECT_NEAR_KK_REL(A_h(0, 1), -one, tol);
  EXPECT_NEAR_KK_REL(A_h(0, 2), zero, tol);
  EXPECT_NEAR_KK_REL(A_h(0, 3), zero, tol);

  EXPECT_NEAR_KK_REL(A_h(1, 0), -one / two, tol);
  EXPECT_NEAR_KK_REL(A_h(1, 1), three / two, tol);
  EXPECT_NEAR_KK_REL(A_h(1, 2), -one, tol);
  EXPECT_NEAR_KK_REL(A_h(1, 3), zero, tol);

  EXPECT_NEAR_KK_REL(A_h(2, 0), zero, tol);
  EXPECT_NEAR_KK_REL(A_h(2, 1), -two / three, tol);
  EXPECT_NEAR_KK_REL(A_h(2, 2), four / three, tol);
  EXPECT_NEAR_KK_REL(A_h(2, 3), -one, tol);

  EXPECT_NEAR_KK_REL(A_h(3, 0), zero, tol);
  EXPECT_NEAR_KK_REL(A_h(3, 1), zero, tol);
  EXPECT_NEAR_KK_REL(A_h(3, 2), -three / four, tol);
  EXPECT_NEAR_KK_REL(A_h(3, 3), five / four, tol);

}  // impl_test_getrf_sym

template <class AMatrixType>
void impl_test_getrf_unsym() {
  using Device   = typename AMatrixType::device_type;
  using IpivType = Kokkos::View<int*, Device>;
  using InfoType = Kokkos::View<int*, Device>;
  using scalar_type    = typename AMatrixType::non_const_value_type;
  using ExecutionSpace = typename Device::execution_space;

  AMatrixType A("matrix A", 3, 3);
  auto A_h = Kokkos::create_mirror_view(A);
  A_h(0, 0) = scalar_type(0); A_h(0, 1) = scalar_type(5); A_h(0, 2) = scalar_type(22. / 3);
  A_h(1, 0) = scalar_type(4); A_h(1, 1) = scalar_type(2); A_h(1, 2) = scalar_type(1);
  A_h(2, 0) = scalar_type(2); A_h(2, 1) = scalar_type(7); A_h(2, 2) = scalar_type(9);
  Kokkos::deep_copy(A, A_h);

  IpivType ipiv("LU pivots", 3);
  InfoType info("LU info", 1);

  KokkosLapack::getrf(ExecutionSpace(), A, ipiv, info);
  Kokkos::deep_copy(A_h, A);
  auto ipiv_h = Kokkos::create_mirror_view(ipiv);
  Kokkos::deep_copy(ipiv_h, ipiv);

  auto tol  = 10*KokkosKernels::ArithTraits<scalar_type>::eps();

  ASSERT_EQ(ipiv_h(0), 2);
  ASSERT_EQ(ipiv_h(1), 3);
  ASSERT_EQ(ipiv_h(2), 3);

  EXPECT_NEAR_KK_REL(A_h(0, 0), scalar_type(4), tol);
  EXPECT_NEAR_KK_REL(A_h(0, 1), scalar_type(2), tol);
  EXPECT_NEAR_KK_REL(A_h(0, 2), scalar_type(1), tol);

  EXPECT_NEAR_KK_REL(A_h(1, 0), scalar_type(1. / 2), tol);
  EXPECT_NEAR_KK_REL(A_h(1, 1), scalar_type(6), tol);
  EXPECT_NEAR_KK_REL(A_h(1, 2), scalar_type(17. / 2), tol);

  EXPECT_NEAR_KK_REL(A_h(2, 0), scalar_type(0), tol);
  EXPECT_NEAR_KK_REL(A_h(2, 1), scalar_type(5. / 6), tol);
  EXPECT_NEAR_KK_REL(A_h(2, 2), scalar_type(1. / 4), tol);

}  // impl_test_getrf_unsym

template <class AMatrixType>
void impl_test_getrf(const int m, const int n) {
  using Device   = typename AMatrixType::device_type;
  using IpivType = Kokkos::View<int*, Device>;
  using InfoType = Kokkos::View<int*, Device>;
  using scalar_type    = typename AMatrixType::non_const_value_type;
  using ExecutionSpace = typename Device::execution_space;

  const int minMN = Kokkos::min(m, n);

  AMatrixType A("matrix A", m, n), Aref("copy of A for reference", m, n), LU("LU product", m, n);

  IpivType ipiv("LU pivots", minMN);
  InfoType info("LU info", 1);

  Kokkos::Random_XorShift64_Pool<ExecutionSpace> rand_pool(13718);
  Kokkos::fill_random(A, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<ExecutionSpace>, scalar_type>::max());
  Kokkos::deep_copy(Aref, A);

  std::cout << "Aref\n"
	    << "   [" << Aref(0, 0) << ", " << Aref(0, 1) << ", " << Aref(0, 2) << ", " << Aref(0, 3) << "]\n"
	    << "   [" << Aref(1, 0) << ", " << Aref(1, 1) << ", " << Aref(1, 2) << ", " << Aref(1, 3) << "]\n"
	    << "   [" << Aref(2, 0) << ", " << Aref(2, 1) << ", " << Aref(2, 2) << ", " << Aref(2, 3) << "]\n"
	    << "   [" << Aref(3, 0) << ", " << Aref(3, 1) << ", " << Aref(3, 2) << ", " << Aref(3, 3) << "]" << std::endl;

  KokkosLapack::getrf(ExecutionSpace(), A, ipiv, info);

  std::cout << "A\n"
	    << "   [" << A(0, 0) << ", " << A(0, 1) << ", " << A(0, 2) << ", " << A(0, 3) << "]\n"
	    << "   [" << A(1, 0) << ", " << A(1, 1) << ", " << A(1, 2) << ", " << A(1, 3) << "]\n"
	    << "   [" << A(2, 0) << ", " << A(2, 1) << ", " << A(2, 2) << ", " << A(2, 3) << "]\n"
	    << "   [" << A(3, 0) << ", " << A(3, 1) << ", " << A(3, 2) << ", " << A(3, 3) << "]" << std::endl;
  std::cout << "ipiv: [" << ipiv(0) << ", " << ipiv(1) << ", " << ipiv(2) << ",  " << ipiv(3) << "]" << std::endl;

  copy_L_plus_I my_func(A, LU);
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, m * n), my_func);
  KokkosBlas::trsm(ExecutionSpace(), "R", "U", "N", "N", 1.0, A, LU);

  std::cout << "LU\n"
	    << "   [" << LU(0, 0) << ", " << LU(0, 1) << ", " << LU(0, 2) << ", " << LU(0, 3) << "]\n"
	    << "   [" << LU(1, 0) << ", " << LU(1, 1) << ", " << LU(1, 2) << ", " << LU(1, 3) << "]\n"
	    << "   [" << LU(2, 0) << ", " << LU(2, 1) << ", " << LU(2, 2) << ", " << LU(2, 3) << "]\n"
	    << "   [" << LU(3, 0) << ", " << LU(3, 1) << ", " << LU(3, 2) << ", " << LU(3, 3) << "]" << std::endl;
}

}  // namespace Test

template <class Scalar, class Device>
void test_getrf() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  using view_type_a = Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device>;

  Test::impl_test_getrf_sym<view_type_a>();
  Test::impl_test_getrf_unsym<view_type_a>();

  Test::impl_test_getrf<view_type_a>(4, 4);
  // Test::impl_test_getrf<view_type_a>(100, 70);
  // Test::impl_test_getrf<view_type_a>(70, 100);
#endif
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, getrf_float) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::getrf_float");
  test_getrf<float, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, getrf_double) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::getrf_double");
  test_getrf<double, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, getrf_complex_float) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::getrf_complex_float");
  test_getrf<Kokkos::complex<float>, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, getrf_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::getrf_complex_double");
  test_getrf<Kokkos::complex<double>, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#endif  // CUDA+CUSOLVER or HIP+ROCSOLVER or LAPACK+HOST
