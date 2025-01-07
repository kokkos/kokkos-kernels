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
#include <KokkosBatched_Gemm_Decl.hpp>
#include "Test_Batched_DenseUtils.hpp"

using namespace KokkosBatched;

namespace Test {
namespace Getrf {

template <typename DeviceType, typename AViewType, typename PivViewType, typename AlgoTagType>
struct Functor_BatchedSerialGetrf {
  using execution_space = typename DeviceType::execution_space;
  AViewType _a;
  PivViewType _ipiv;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGetrf(const AViewType &a, const PivViewType &ipiv) : _a(a), _ipiv(ipiv) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k, int &info) const {
    auto aa   = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto ipiv = Kokkos::subview(_ipiv, k, Kokkos::ALL());

    info += KokkosBatched::SerialGetrf<AlgoTagType>::invoke(aa, ipiv);
  }

  inline int run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialGetrf");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    int info_sum                      = 0;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space> policy(0, _a.extent(0));
    Kokkos::parallel_reduce(name.c_str(), policy, *this, info_sum);
    Kokkos::Profiling::popRegion();
    return info_sum;
  }
};

template <typename DeviceType, typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
struct Functor_BatchedSerialGemm {
  using execution_space = typename DeviceType::execution_space;
  AViewType _a;
  BViewType _b;
  CViewType _c;
  ScalarType _alpha, _beta;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGemm(const ScalarType alpha, const AViewType &a, const BViewType &b, const ScalarType beta,
                            const CViewType &c)
      : _alpha(alpha), _a(a), _b(b), _beta(beta), _c(c) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto bb = Kokkos::subview(_b, k, Kokkos::ALL(), Kokkos::ALL());
    auto cc = Kokkos::subview(_c, k, Kokkos::ALL(), Kokkos::ALL());

    KokkosBatched::SerialGemm<Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Unblocked>::invoke(_alpha, aa, bb,
                                                                                                     _beta, cc);
  }

  inline void run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialGetrf");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::RangePolicy<execution_space> policy(0, _a.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
  }
};

template <typename DeviceType, typename ScalarType, typename LayoutType, typename AlgoTagType>
/// \brief Implementation details of batched getrf test
///        LU factorization with partial pivoting
///        A = [[1. 0. 0. 0.]
///             [0. 1. 0. 0.]
///             [0. 0. 1. 0.]
///             [0. 0. 0. 1.]]
///        LU = [[1. 0. 0. 0.]
///              [0. 1. 0. 0.]
///              [0. 0. 1. 0.]
///              [0. 0. 0. 1.]]
///        piv = [0 1 2 3]
///
/// \param N [in] Batch size of RHS (banded matrix can also be batched matrix)
/// \param k [in] Number of superdiagonals or subdiagonals of matrix A
/// \param BlkSize [in] Block size of matrix A
void impl_test_batched_getrf_analytical(const int N) {
  using ats            = typename Kokkos::ArithTraits<ScalarType>;
  using RealType       = typename ats::mag_type;
  using RealView2DType = Kokkos::View<RealType **, LayoutType, DeviceType>;

  using View3DType    = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using PivView2DType = Kokkos::View<int **, LayoutType, DeviceType>;

  constexpr int BlkSize = 4;
  View3DType A("A", N, BlkSize, BlkSize), A_reconst("A_reconst", N, BlkSize, BlkSize), NL("NL", N, BlkSize, BlkSize),
      L("L", N, BlkSize, BlkSize), U("U", N, BlkSize, BlkSize), LU("LU", N, BlkSize, BlkSize),
      I("I", N, BlkSize, BlkSize);
  RealView2DType ones(Kokkos::view_alloc("ones", Kokkos::WithoutInitializing), N, BlkSize);
  PivView2DType ipiv("ipiv", N, BlkSize), ipiv_ref("ipiv_ref", N, BlkSize);

  auto h_A        = Kokkos::create_mirror_view(A);
  auto h_ipiv_ref = Kokkos::create_mirror_view(ipiv_ref);
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      h_ipiv_ref(ib, i) = i;
      for (int j = 0; j < BlkSize; j++) {
        h_A(ib, i, j) = i == j ? 1.0 : 0.0;
      }
    }
  }

  Kokkos::deep_copy(ipiv_ref, h_ipiv_ref);
  Kokkos::deep_copy(LU, h_A);
  Kokkos::deep_copy(ones, RealType(1.0));

  create_diagonal_matrix(ones, I);
  Kokkos::fence();

  // getrf to factorize matrix A = P * L * U
  auto info = Functor_BatchedSerialGetrf<DeviceType, View3DType, PivView2DType, AlgoTagType>(LU, ipiv).run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // Reconstruct L and D from Factorized matrix A
  // Copy non-diagonal lower triangular components to NL
  create_triangular_matrix<View3DType, View3DType, KokkosBatched::Uplo::Lower>(LU, NL, -1);

  // Copy upper triangular components to U
  create_triangular_matrix<View3DType, View3DType, KokkosBatched::Uplo::Upper>(LU, U);

  // Copy I to L
  Kokkos::deep_copy(L, I);

  // Matrix matrix addition by Gemm
  // NL + I by NL * I + L (==I) (result stored in L)
  Functor_BatchedSerialGemm<DeviceType, ScalarType, View3DType, View3DType, View3DType>(1.0, NL, I, 1.0, L).run();

  // LU = L * U
  Functor_BatchedSerialGemm<DeviceType, ScalarType, View3DType, View3DType, View3DType>(1.0, L, U, 0.0, LU).run();

  Kokkos::fence();

  // permute A by ipiv
  auto h_ipiv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ipiv);
  for (int ib = 0; ib < N; ib++) {
    // Permute A by pivot vector
    for (int i = 0; i < BlkSize; i++) {
      for (int j = 0; j < BlkSize; j++) {
        Kokkos::kokkos_swap(h_A(ib, h_ipiv(ib, i), j), h_A(ib, i, j));
      }
    }
  }

  // A stores permuted A
  Kokkos::deep_copy(A, h_A);

  // Check if piv = [0 1 2 3]
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      EXPECT_EQ(h_ipiv(ib, i), h_ipiv_ref(ib, i));
    }
  }

  // this eps is about 10^-14
  RealType eps = 1.0e3 * ats::epsilon();

  auto h_LU = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), LU);

  // Check if LU = A (permuted)
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      for (int j = 0; j < BlkSize; j++) {
        EXPECT_NEAR_KK(h_LU(ib, i, j), h_A(ib, i, j), eps);
      }
    }
  }
}

template <typename DeviceType, typename ScalarType, typename LayoutType, typename AlgoTagType>
/// \brief Implementation details of batched getrf test
///        LU factorization with partial pivoting
///
/// \param N [in] Batch size of RHS (banded matrix can also be batched matrix)
/// \param k [in] Number of superdiagonals or subdiagonals of matrix A
/// \param BlkSize [in] Block size of matrix A
void impl_test_batched_getrf(const int N, const int BlkSize) {
  using ats            = typename Kokkos::ArithTraits<ScalarType>;
  using RealType       = typename ats::mag_type;
  using RealView2DType = Kokkos::View<RealType **, LayoutType, DeviceType>;

  using View3DType    = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using PivView2DType = Kokkos::View<int **, LayoutType, DeviceType>;

  View3DType A("A", N, BlkSize, BlkSize), A_reconst("A_reconst", N, BlkSize, BlkSize), NL("NL", N, BlkSize, BlkSize),
      L("L", N, BlkSize, BlkSize), U("U", N, BlkSize, BlkSize), LU("LU", N, BlkSize, BlkSize),
      I("I", N, BlkSize, BlkSize);
  RealView2DType ones(Kokkos::view_alloc("ones", Kokkos::WithoutInitializing), N, BlkSize);
  PivView2DType ipiv("ipiv", N, BlkSize);

  using execution_space = typename DeviceType::execution_space;
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);
  ScalarType randStart, randEnd;

  // Initialize A_reconst with random matrix
  KokkosKernels::Impl::getRandomBounds(1.0, randStart, randEnd);
  Kokkos::fill_random(A, rand_pool, randStart, randEnd);
  Kokkos::deep_copy(LU, A);

  // Unit matrix I
  Kokkos::deep_copy(ones, RealType(1.0));
  create_diagonal_matrix(ones, I);

  Kokkos::fence();

  // getrf to factorize matrix A = P * L * U
  auto info = Functor_BatchedSerialGetrf<DeviceType, View3DType, PivView2DType, AlgoTagType>(LU, ipiv).run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // Reconstruct L and D from Factorized matrix A
  // Copy non-diagonal lower triangular components to NL
  create_triangular_matrix<View3DType, View3DType, KokkosBatched::Uplo::Lower>(LU, NL, -1);

  // Copy upper triangular components to U
  create_triangular_matrix<View3DType, View3DType, KokkosBatched::Uplo::Upper>(LU, U);

  // Copy I to L
  Kokkos::deep_copy(L, I);

  // Matrix matrix addition by Gemm
  // NL + I by NL * I + L (==I) (result stored in L)
  Functor_BatchedSerialGemm<DeviceType, ScalarType, View3DType, View3DType, View3DType>(1.0, NL, I, 1.0, L).run();

  // LU = L * U
  Functor_BatchedSerialGemm<DeviceType, ScalarType, View3DType, View3DType, View3DType>(1.0, L, U, 0.0, LU).run();

  Kokkos::fence();

  // permute A by ipiv
  auto h_ipiv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ipiv);
  auto h_A    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  for (int ib = 0; ib < N; ib++) {
    // Permute A by pivot vector
    for (int i = 0; i < BlkSize; i++) {
      for (int j = 0; j < BlkSize; j++) {
        Kokkos::kokkos_swap(h_A(ib, h_ipiv(ib, i), j), h_A(ib, i, j));
      }
    }
  }

  // this eps is about 10^-14
  RealType eps = 1.0e3 * ats::epsilon();

  auto h_LU = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), LU);
  // Check if LU = A (permuted)
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      for (int j = 0; j < BlkSize; j++) {
        EXPECT_NEAR_KK(h_LU(ib, i, j), h_A(ib, i, j), eps);
      }
    }
  }
}

}  // namespace Getrf
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename AlgoTagType>
int test_batched_getrf() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    Test::Getrf::impl_test_batched_getrf_analytical<DeviceType, ScalarType, LayoutType, AlgoTagType>(1);
    Test::Getrf::impl_test_batched_getrf_analytical<DeviceType, ScalarType, LayoutType, AlgoTagType>(2);
    for (int i = 0; i < 10; i++) {
      Test::Getrf::impl_test_batched_getrf<DeviceType, ScalarType, LayoutType, AlgoTagType>(1, i);
      Test::Getrf::impl_test_batched_getrf<DeviceType, ScalarType, LayoutType, AlgoTagType>(2, i);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    Test::Getrf::impl_test_batched_getrf_analytical<DeviceType, ScalarType, LayoutType, AlgoTagType>(1);
    Test::Getrf::impl_test_batched_getrf_analytical<DeviceType, ScalarType, LayoutType, AlgoTagType>(2);
    for (int i = 0; i < 10; i++) {
      Test::Getrf::impl_test_batched_getrf<DeviceType, ScalarType, LayoutType, AlgoTagType>(1, i);
      Test::Getrf::impl_test_batched_getrf<DeviceType, ScalarType, LayoutType, AlgoTagType>(2, i);
    }
  }
#endif

  return 0;
}
