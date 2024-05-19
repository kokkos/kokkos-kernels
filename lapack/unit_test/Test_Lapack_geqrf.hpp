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

// Only enable this test where KokkosLapack supports geqrf:
// CUDA+CUSOLVER, HIP+ROCSOLVER and HOST+LAPACK
#if (defined(TEST_CUDA_LAPACK_CPP) &&                                       \
      defined(KOKKOSKERNELS_ENABLE_TPL_CUSOLVER)) ||                        \
    (defined(TEST_HIP_LAPACK_CPP) &&                                        \
     defined(KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER)) ||                        \
    (defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK) &&                            \
     (defined(TEST_OPENMP_LAPACK_CPP) || defined(TEST_SERIAL_LAPACK_CPP) || \
      defined(TEST_THREADS_LAPACK_CPP)))

// AquiEEP

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <KokkosLapack_geqrf.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {

template <class ViewTypeA, class ViewTypeTW, class Device>
void impl_test_geqrf(int m, int n) {
  using execution_space = typename Device::execution_space;
  using ScalarA         = typename ViewTypeA::value_type;
  //using ats             = Kokkos::ArithTraits<ScalarA>;

  execution_space space{};

  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);

  int lwork(1);
  if (std::min(m,n) != 0) {
    lwork = n;
  }

  // Create device views
  ViewTypeA  A   ("A", m, n);
  ViewTypeTW Tau ("Tau", std::min(m,n));
  ViewTypeTW Work("Work", lwork);

  // Create host mirrors of device views.
  typename ViewTypeTW::HostMirror h_tau  = Kokkos::create_mirror_view(Tau);
  typename ViewTypeTW::HostMirror h_work = Kokkos::create_mirror(Work);

  // Initialize data.
  if ((m == 3) && (n == 3)) {
  }
  else {
    Kokkos::fill_random( A
                       , rand_pool
                       , Kokkos::rand<Kokkos::Random_XorShift64<execution_space>, ScalarA>::max()
                       );
  }

  Kokkos::fence();

  // Deep copy device view to host view.
  //Kokkos::deep_copy(h_X0, X0);

  // Allocate IPIV view on host
  using ViewTypeP = Kokkos::View<int*, Kokkos::LayoutLeft, execution_space>;
  ViewTypeP ipiv;
  int Nt = n;
  ipiv = ViewTypeP("IPIV", Nt);

  // Solve.
  try {
    KokkosLapack::geqrf(space, A, Tau, Work);
  }
  catch (const std::runtime_error& error) {
    return;
  }
  Kokkos::fence();

  // Get the solution vector.
  //Kokkos::deep_copy(h_B, B);

  // Checking vs ref on CPU, this eps is about 10^-9
  //typedef typename ats::mag_type mag_type;
  //const mag_type eps = 3.0e7 * ats::epsilon();
  bool test_flag     = true;
  for (int i = 0; i < n; i++) {
#if 0
    if (ats::abs(h_B(i) - h_X0(i)) > eps) {
      test_flag = false;
      printf(
          "    Error %d, pivot %c, padding %c: result( %.15lf ) !="
          "solution( %.15lf ) at (%d), error=%.15e, eps=%.15e\n",
          N, mode[0], padding[0], ats::abs(h_B(i)), ats::abs(h_X0(i)), int(i),
          ats::abs(h_B(i) - h_X0(i)), eps);
      break;
    }
#endif
  }
  ASSERT_EQ(test_flag, true);
}

}  // namespace Test

template <class Scalar, class Device>
void test_geqrf() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&      \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  using view_type_a_ll = Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device>;
  using view_type_tw_ll = Kokkos::View<Scalar*, Kokkos::LayoutLeft, Device>;

  Test::impl_test_geqrf<view_type_a_ll, view_type_tw_ll, Device>(3, 3);
#endif
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, geqrf_float) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::geqrf_float");
  test_geqrf<float, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&  \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, geqrf_double) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::geqrf_double");
  test_geqrf<double, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&          \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, geqrf_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::geqrf_complex_double");
  test_geqrf<Kokkos::complex<double>, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&         \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, geqrf_complex_float) {
  Kokkos::Profiling::pushRegion("KokkosLapack::Test::geqrf_complex_float");
  test_geqrf<Kokkos::complex<float>, TestDevice>();
  Kokkos::Profiling::popRegion();
}
#endif

#endif  // CUDA+CUSOLVER or HIP+ROCSOLVER or LAPACK+HOST
