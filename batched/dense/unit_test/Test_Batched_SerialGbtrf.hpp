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
/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBatched_Getrf.hpp>
#include <KokkosBatched_Gbtrf.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include "Test_Batched_DenseUtils.hpp"

namespace Test {
namespace Gbtrf {

template <typename DeviceType, typename ABViewType, typename PivViewType, typename AlgoTagType>
struct Functor_BatchedSerialGbtrf {
  using execution_space = typename DeviceType::execution_space;
  ABViewType m_ab;
  PivViewType m_ipiv;
  int m_kl, m_ku;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGbtrf(const ABViewType &ab, const PivViewType &ipiv, int kl, int ku)
      : m_ab(ab), m_ipiv(ipiv), m_kl(kl), m_ku(ku) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k, int &info) const {
    auto sub_ab   = Kokkos::subview(m_ab, k, Kokkos::ALL(), Kokkos::ALL());
    auto sub_ipiv = Kokkos::subview(m_ipiv, k, Kokkos::ALL());

    info += KokkosBatched::SerialGbtrf<AlgoTagType>::invoke(sub_ab, sub_ipiv, m_kl, m_ku);
  }

  inline int run() {
    using value_type = typename ABViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialGbtrf");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    int info_sum                      = 0;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space> policy(0, m_ab.extent(0));
    Kokkos::parallel_reduce(name.c_str(), policy, *this, info_sum);
    Kokkos::Profiling::popRegion();
    return info_sum;
  }
};

template <typename DeviceType, typename AViewType, typename PivViewType, typename AlgoTagType>
struct Functor_BatchedSerialGetrf {
  using execution_space = typename DeviceType::execution_space;
  AViewType m_a;
  PivViewType m_ipiv;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGetrf(const AViewType &a, const PivViewType &ipiv) : m_a(a), m_ipiv(ipiv) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    auto sub_a    = Kokkos::subview(m_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto sub_ipiv = Kokkos::subview(m_ipiv, k, Kokkos::ALL());

    KokkosBatched::SerialGetrf<KokkosBatched::Algo::Getrf::Unblocked>::invoke(sub_a, sub_ipiv);
  }

  void run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialGbtrf");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::RangePolicy<execution_space> policy(0, m_a.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
  }
};

/// \brief Implementation details of batched gbtrf analytical test
///
/// \param Nb [in] Batch size of matrices
/// \param BlkSize [in] Block size of matrix A
///        4x4 more general matrix
///        which satisfies PA = LU
///        P = [[0, 0, 1, 0],
///             [1, 0, 0, 0],
///             [0, 1, 0, 0],
///             [0, 0, 0, 1]]
///        A: [[1, -3, -2,  0],
///            [-1, 1, -3, -2],
///            [2, -1,  1, -3],
///            [0,  2, -1,  1]]
///        L = [[1,       0, 0, 0],
///             [0.5,     1, 0, 0],
///             [-0.5, -0.2, 1, 0],
///             [0,    -0.8, 1, 1]]
///        U =  [[2, -1,      1,  -3. ],
///              [0, -2.5, -2.5,  1.5 ],
///              [0,  0,     -3,  -3.2],
///              [0,  0,      0,   5.4]]
///        Note P is obtained by piv = [2 2 2 3]
///        We compare the non-diagnoal elements of L only, which is
///        NL = [[0,       0, 0, 0],
///              [0.5,     0, 0, 0],
///              [-0.5, -0.2, 0, 0],
///              [0,    -0.8, 1, 0]]
template <typename DeviceType, typename ScalarType, typename LayoutType, typename AlgoTagType>
void impl_test_batched_gbtrf_analytical(const int Nb) {
  using ats           = typename Kokkos::ArithTraits<ScalarType>;
  using RealType      = typename ats::mag_type;
  using View2DType    = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType    = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using PivView2DType = Kokkos::View<int **, LayoutType, DeviceType>;

  const int BlkSize = 4, kl = 2, ku = 2;
  const int ldab = 2 * kl + ku + 1;
  View3DType A("A", Nb, BlkSize, BlkSize), NL("NL", Nb, BlkSize, BlkSize), NL_ref("NL_ref", Nb, BlkSize, BlkSize),
      U("U", Nb, BlkSize, BlkSize), U_ref("U_ref", Nb, BlkSize, BlkSize);

  View3DType AB("AB", Nb, ldab, BlkSize);

  PivView2DType ipiv("ipiv", Nb, BlkSize), ipiv_ref("ipiv_ref", Nb, BlkSize);

  auto h_A        = Kokkos::create_mirror_view(A);
  auto h_NL_ref   = Kokkos::create_mirror_view(NL_ref);
  auto h_U_ref    = Kokkos::create_mirror_view(U_ref);
  auto h_ipiv_ref = Kokkos::create_mirror_view(ipiv_ref);
  for (int ib = 0; ib < Nb; ib++) {
    h_A(ib, 0, 0) = 1;
    h_A(ib, 0, 1) = -3;
    h_A(ib, 0, 2) = -2;
    h_A(ib, 0, 3) = 0;
    h_A(ib, 1, 0) = -1;
    h_A(ib, 1, 1) = 1;
    h_A(ib, 1, 2) = -3;
    h_A(ib, 1, 3) = -2;
    h_A(ib, 2, 0) = 2;
    h_A(ib, 2, 1) = -1;
    h_A(ib, 2, 2) = 1;
    h_A(ib, 2, 3) = -3;
    h_A(ib, 3, 0) = 0;
    h_A(ib, 3, 1) = 2;
    h_A(ib, 3, 2) = -1;
    h_A(ib, 3, 3) = 1;

    h_U_ref(ib, 0, 0) = 2;
    h_U_ref(ib, 0, 1) = -1;
    h_U_ref(ib, 0, 2) = 1;
    h_U_ref(ib, 0, 3) = -3;
    h_U_ref(ib, 1, 1) = -2.5;
    h_U_ref(ib, 1, 2) = -2.5;
    h_U_ref(ib, 1, 3) = 1.5;
    h_U_ref(ib, 2, 2) = -3;
    h_U_ref(ib, 2, 3) = -3.2;
    h_U_ref(ib, 3, 3) = 5.4;

    h_NL_ref(ib, 1, 0) = 0.5;
    h_NL_ref(ib, 2, 0) = -0.5;
    h_NL_ref(ib, 2, 1) = -0.2;
    h_NL_ref(ib, 3, 1) = -0.8;
    h_NL_ref(ib, 3, 2) = 1.0;

    h_ipiv_ref(ib, 0) = 2;
    h_ipiv_ref(ib, 1) = 2;
    h_ipiv_ref(ib, 2) = 2;
    h_ipiv_ref(ib, 3) = 3;
  }

  Kokkos::deep_copy(A, h_A);

  // Convert into banded storage
  full_to_banded(A, AB, kl, ku);

  // gbtrf to factorize matrix A = P * L * U
  auto info = Functor_BatchedSerialGbtrf<DeviceType, View3DType, PivView2DType, AlgoTagType>(AB, ipiv, kl, ku).run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // Extract matrix U and L from AB
  // first convert it to the full matrix (stored in A)
  banded_to_full<View3DType, View3DType>(AB, A, kl, ku);

  // Copy upper triangular components to U
  create_triangular_matrix<View3DType, View3DType, KokkosBatched::Uplo::Upper, KokkosBatched::Diag::NonUnit>(A, U);

  // Apply pivot at host
  auto h_ipiv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ipiv);
  Kokkos::deep_copy(h_A, A);
  for (int ib = 0; ib < Nb; ib++) {
    for (int j = 0; j < BlkSize - 1; j++) {
      for (int i = j + 1; i < BlkSize - 1; i++) {
        Kokkos::kokkos_swap(h_A(ib, i, j), h_A(ib, h_ipiv(ib, i), j));
      }
    }
  }
  Kokkos::deep_copy(A, h_A);

  // Copy non-diagonal lower triangular components to NL
  create_triangular_matrix<View3DType, View3DType, KokkosBatched::Uplo::Lower, KokkosBatched::Diag::NonUnit>(A, NL, -1);

  // Check if U, NL and ipiv have expected values
  auto h_U  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U);
  auto h_NL = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), NL);

  RealType eps = 1.0e1 * ats::epsilon();
  for (int ib = 0; ib < Nb; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      EXPECT_EQ(h_ipiv(ib, i), h_ipiv_ref(ib, i));
      for (int j = 0; j < BlkSize; j++) {
        EXPECT_NEAR_KK(h_U(ib, i, j), h_U_ref(ib, i, j), eps);
        EXPECT_NEAR_KK(h_NL(ib, i, j), h_NL_ref(ib, i, j), eps);
      }
    }
  }
}

/// \brief Implementation details of batched gbtrf test
///
/// \param N [in] Batch size of RHS (banded matrix can also be batched matrix)
/// \param k [in] Number of superdiagonals or subdiagonals of matrix A
/// \param BlkSize [in] Block size of matrix A
template <typename DeviceType, typename ScalarType, typename LayoutType, typename AlgoTagType>
void impl_test_batched_gbtrf(const int Nb, const int BlkSize) {
  using ats           = typename Kokkos::ArithTraits<ScalarType>;
  using RealType      = typename ats::mag_type;
  using View2DType    = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType    = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using PivView2DType = Kokkos::View<int **, LayoutType, DeviceType>;

  const int kl = 2, ku = 2;
  const int ldab = 2 * kl + ku + 1;
  View3DType A("A", Nb, BlkSize, BlkSize), LU("LU", Nb, BlkSize, BlkSize), NL("NL", Nb, BlkSize, BlkSize),
      NL_ref("NL_ref", Nb, BlkSize, BlkSize), U("U", Nb, BlkSize, BlkSize), U_ref("U_ref", Nb, BlkSize, BlkSize);

  View3DType AB("AB", Nb, ldab, BlkSize), AB_upper("AB_upper", Nb, kl + ku + 1, BlkSize);
  PivView2DType ipiv("ipiv", Nb, BlkSize), ipiv_ref("ipiv_ref", Nb, BlkSize);

  // Create a random matrix A and make it Positive Definite Symmetric
  using execution_space = typename DeviceType::execution_space;
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);
  ScalarType randStart, randEnd;

  // Initialize LU with random matrix
  KokkosKernels::Impl::getRandomBounds(1.0, randStart, randEnd);
  Kokkos::fill_random(A, rand_pool, randStart, randEnd);

  // Make the matrix Positive Definite Symmetric and Diagonal dominant
  random_to_pds(A, LU);
  Kokkos::deep_copy(A, ScalarType(0.0));

  full_to_banded(LU, AB, kl, ku);  // In banded storage
  banded_to_full(AB, A, kl, ku);   // In full storage

  Kokkos::deep_copy(LU, A);  // for getrf

  // gbtrf to factorize matrix A = P * L * U
  Functor_BatchedSerialGbtrf<DeviceType, View3DType, PivView2DType, AlgoTagType>(AB, ipiv, kl, ku).run();

  // Extract matrix U and L from AB
  // first convert it to the full matrix (stored in A)
  banded_to_full<View3DType, View3DType>(AB, A, kl, ku);

  // Copy upper triangular components to U
  create_triangular_matrix<View3DType, View3DType, KokkosBatched::Uplo::Upper, KokkosBatched::Diag::NonUnit>(A, U);

  // Apply pivot at host
  auto h_ipiv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ipiv);
  auto h_A    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  for (int ib = 0; ib < Nb; ib++) {
    for (int j = 0; j < BlkSize - 1; j++) {
      for (int i = j + 1; i < BlkSize - 1; i++) {
        Kokkos::kokkos_swap(h_A(ib, i, j), h_A(ib, h_ipiv(ib, i), j));
      }
    }
  }
  Kokkos::deep_copy(A, h_A);

  // Copy non-diagonal lower triangular components to NL
  create_triangular_matrix<View3DType, View3DType, KokkosBatched::Uplo::Lower, KokkosBatched::Diag::NonUnit>(A, NL, -1);

  // Reference is made by getrf
  // getrf to factorize matrix A = P * L * U
  Functor_BatchedSerialGetrf<DeviceType, View3DType, PivView2DType, AlgoTagType>(LU, ipiv_ref).run();

  // Copy non-diagonal lower triangular components to NL
  create_triangular_matrix<View3DType, View3DType, KokkosBatched::Uplo::Lower, KokkosBatched::Diag::NonUnit>(LU, NL_ref,
                                                                                                             -1);

  // Copy upper triangular components to U_ref
  create_triangular_matrix<View3DType, View3DType, KokkosBatched::Uplo::Upper, KokkosBatched::Diag::NonUnit>(LU, U_ref);

  // Check if U, NL and ipiv have expected values
  auto h_U        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U);
  auto h_U_ref    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U_ref);
  auto h_NL       = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), NL);
  auto h_NL_ref   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), NL_ref);
  auto h_ipiv_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ipiv_ref);

  RealType eps = 1.0e3 * ats::epsilon();
  for (int ib = 0; ib < Nb; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      EXPECT_EQ(h_ipiv(ib, i), h_ipiv_ref(ib, i));
      for (int j = 0; j < BlkSize; j++) {
        EXPECT_NEAR_KK(h_U(ib, i, j), h_U_ref(ib, i, j), eps);
        EXPECT_NEAR_KK(h_NL(ib, i, j), h_NL_ref(ib, i, j), eps);
      }
    }
  }
}

}  // namespace Gbtrf
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename AlgoTagType>
int test_batched_gbtrf() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    Test::Gbtrf::impl_test_batched_gbtrf_analytical<DeviceType, ScalarType, LayoutType, AlgoTagType>(1);
    Test::Gbtrf::impl_test_batched_gbtrf_analytical<DeviceType, ScalarType, LayoutType, AlgoTagType>(2);
    // for (int i = 5; i < 10; i++) {
    //   Test::Gbtrf::impl_test_batched_gbtrf<DeviceType, ScalarType, LayoutType, AlgoTagType>(1, i);
    //   Test::Gbtrf::impl_test_batched_gbtrf<DeviceType, ScalarType, LayoutType, AlgoTagType>(2, i);
    // }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    Test::Gbtrf::impl_test_batched_gbtrf_analytical<DeviceType, ScalarType, LayoutType, AlgoTagType>(1);
    Test::Gbtrf::impl_test_batched_gbtrf_analytical<DeviceType, ScalarType, LayoutType, AlgoTagType>(2);
    // for (int i = 5; i < 10; i++) {
    //   Test::Gbtrf::impl_test_batched_gbtrf<DeviceType, ScalarType, LayoutType, AlgoTagType>(1, i);
    //   Test::Gbtrf::impl_test_batched_gbtrf<DeviceType, ScalarType, LayoutType, AlgoTagType>(2, i);
    // }
  }
#endif

  return 0;
}

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F(TestCategory, test_batched_gbtrf_float) {
  using algo_tag_type = typename KokkosBatched::Algo::Gbtrf::Unblocked;

  test_batched_gbtrf<TestDevice, float, algo_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, test_batched_gbtrf_double) {
  using algo_tag_type = typename KokkosBatched::Algo::Gbtrf::Unblocked;

  test_batched_gbtrf<TestDevice, double, algo_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
TEST_F(TestCategory, test_batched_gbtrf_fcomplex) {
  using algo_tag_type = typename KokkosBatched::Algo::Gbtrf::Unblocked;

  test_batched_gbtrf<TestDevice, Kokkos::complex<float>, algo_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
TEST_F(TestCategory, test_batched_gbtrf_dcomplex) {
  using algo_tag_type = typename KokkosBatched::Algo::Gbtrf::Unblocked;

  test_batched_gbtrf<TestDevice, Kokkos::complex<double>, algo_tag_type>();
}
#endif
