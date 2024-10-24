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

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Laswp.hpp"
#include "Test_Batched_DenseUtils.hpp"

using namespace KokkosBatched;

namespace Test {
namespace Laswp {

template <typename DeviceType, typename PivViewType, typename AViewType, typename ArgDirect>
struct Functor_BatchedSerialLaswp {
  using execution_space = typename DeviceType::execution_space;
  PivViewType m_ipiv;
  AViewType m_a;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialLaswp(const PivViewType &ipiv, const AViewType &a) : m_ipiv(ipiv), m_a(a) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const ArgDirect &, const int k, int &info) const {
    auto sub_ipiv = Kokkos::subview(m_ipiv, k, Kokkos::ALL);
    if constexpr (AViewType::rank == 3) {
      auto sub_a = Kokkos::subview(m_a, k, Kokkos::ALL, Kokkos::ALL);
      info += KokkosBatched::SerialLaswp<ArgDirect>::invoke(sub_ipiv, sub_a);
    } else {
      auto sub_a = Kokkos::subview(m_a, k, Kokkos::ALL);
      info += KokkosBatched::SerialLaswp<ArgDirect>::invoke(sub_ipiv, sub_a);
    }
  }

  inline int run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialLaswp");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    int info_sum                      = 0;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space, ArgDirect> policy(0, m_a.extent(0));
    Kokkos::parallel_reduce(name.c_str(), policy, *this, info_sum);
    Kokkos::Profiling::popRegion();
    return info_sum;
  }
};

/// \brief Implementation details of batched laswp analytical test
///        Confirm A = Ref (permuted), where
///        A0: [[4],
///             [1],
///             [2]]
///        p0: [1, 2, 0]
///                  Initial    0<->1       1<->2     2<->0
///        Forward:  [4,1,2] -> [1,4,2] -> [1,2,4] -> [4,2,1]
///                  Initial    2<->0       1<->2     0<->1
///        Backward: [4,1,2] -> [2,1,4] -> [2,4,1] -> [4,2,1]
///
///        A1: [[4, 1, 5],
///             [2, 3, 7],
///             [6, 0, 8]]
///        p1: [1, 2, 0]
///                  Initial        0<->1          1<->2          2<->0
///        Forward:  [[4, 1, 5], -> [[2, 3, 7], -> [[2, 3, 7], -> [[4, 1, 5],
///                   [2, 3, 7],     [4, 1, 5],     [6, 0, 8],     [6, 0, 8],
///                   [6, 0, 8]]     [6, 0, 8]]     [4, 1, 5]]     [2, 3, 7]]
///                  Initial        2<->0          1<->2          0<->1
///        Backward: [[4, 1, 5], -> [[6, 0, 8], -> [[6, 0, 8], -> [[4, 1, 5],
///                   [2, 3, 7],     [2, 3, 7],      4, 1, 5],     [6, 0, 8],
///                   [6, 0, 8]]     [4, 1, 5]]     [2, 3, 7]]     [2, 3, 7]]
///
///        A2: [[5, 1],
///             [2, 4],
///             [3, 0]]
///        p2: [2, 0, 1]
///                  Initial     0<->2        1<->0       2<->1
///        Forward:  [[5, 1], -> [[3, 0],  -> [[2, 4], -> [[2, 4],
///                   [2, 4],     [2, 4]       [3, 0],     [5, 1],
///                   [3, 0]]     [5, 1]]      [5, 1]]     [3, 0]]
///                  Initial     2<->1        1<->0       0<->2
///        Backward: [[5, 1], -> [[5, 1],  -> [[3, 0], -> [[2, 4],
///                   [2, 4],     [3, 0],      [5, 1],     [5, 1],
///                   [3, 0]]     [2, 4]]      [2, 4]]     [3, 0]]
///
/// \param N [in] Batch size of matrices
template <typename DeviceType, typename ScalarType, typename LayoutType, typename ArgDirect>
void impl_test_batched_laswp_analytical(const int N) {
  using ats           = typename Kokkos::ArithTraits<ScalarType>;
  using RealType      = typename ats::mag_type;
  using View2DType    = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType    = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using PivView2DType = Kokkos::View<int **, LayoutType, DeviceType>;

  View2DType A0("A0", N, 3), Ref0("Ref0", N, 3), A0_identity("A0_identity", N, 3), Ref0_identity("Ref0_identity", N, 3);
  View3DType A1("A1", N, 3, 3), Ref1("Ref1", N, 3, 3), A1_identity("A1_identity", N, 3, 3),
      Ref1_identity("Ref1_identity", N, 3, 3);
  View3DType A2("A2", N, 3, 2), Ref2("Ref2", N, 3, 2), A2_identity("A2_identity", N, 3, 2),
      Ref2_identity("Ref2_identity", N, 3, 2);
  PivView2DType ipiv0("ipiv0", N, 3), ipiv1("ipiv1", N, 3), ipiv2("ipiv2", N, 3);

  // Initialize A_reconst with random matrix
  auto h_A0   = Kokkos::create_mirror_view(A0);
  auto h_A1   = Kokkos::create_mirror_view(A1);
  auto h_A2   = Kokkos::create_mirror_view(A2);
  auto h_Ref0 = Kokkos::create_mirror_view(Ref0);
  auto h_Ref1 = Kokkos::create_mirror_view(Ref1);
  auto h_Ref2 = Kokkos::create_mirror_view(Ref2);

  for (int ib = 0; ib < N; ib++) {
    h_A0(ib, 0) = 4.0;
    h_A0(ib, 1) = 1.0;
    h_A0(ib, 2) = 2.0;

    h_Ref0(ib, 0) = 4.0;
    h_Ref0(ib, 1) = 2.0;
    h_Ref0(ib, 2) = 1.0;

    h_A1(ib, 0, 0) = 4.0;
    h_A1(ib, 0, 1) = 1.0;
    h_A1(ib, 0, 2) = 5.0;
    h_A1(ib, 1, 0) = 2.0;
    h_A1(ib, 1, 1) = 3.0;
    h_A1(ib, 1, 2) = 7.0;
    h_A1(ib, 2, 0) = 6.0;
    h_A1(ib, 2, 1) = 0.0;
    h_A1(ib, 2, 2) = 8.0;

    h_Ref1(ib, 0, 0) = 4.0;
    h_Ref1(ib, 0, 1) = 1.0;
    h_Ref1(ib, 0, 2) = 5.0;
    h_Ref1(ib, 1, 0) = 6.0;
    h_Ref1(ib, 1, 1) = 0.0;
    h_Ref1(ib, 1, 2) = 8.0;
    h_Ref1(ib, 2, 0) = 2.0;
    h_Ref1(ib, 2, 1) = 3.0;
    h_Ref1(ib, 2, 2) = 7.0;

    h_A2(ib, 0, 0) = 5.0;
    h_A2(ib, 0, 1) = 1.0;
    h_A2(ib, 1, 0) = 2.0;
    h_A2(ib, 1, 1) = 4.0;
    h_A2(ib, 2, 0) = 3.0;
    h_A2(ib, 2, 1) = 0.0;

    h_Ref2(ib, 0, 0) = 2.0;
    h_Ref2(ib, 0, 1) = 4.0;
    h_Ref2(ib, 1, 0) = 5.0;
    h_Ref2(ib, 1, 1) = 1.0;
    h_Ref2(ib, 2, 0) = 3.0;
    h_Ref2(ib, 2, 1) = 0.0;
  }
  Kokkos::deep_copy(A0, h_A0);
  Kokkos::deep_copy(A1, h_A1);
  Kokkos::deep_copy(A2, h_A2);

  // Copy A to Ref_identity
  Kokkos::deep_copy(Ref0_identity, A0);
  Kokkos::deep_copy(Ref1_identity, A1);
  Kokkos::deep_copy(Ref2_identity, A2);

  // Permute ipiv
  auto h_ipiv0 = Kokkos::create_mirror_view(ipiv0);
  auto h_ipiv1 = Kokkos::create_mirror_view(ipiv1);
  auto h_ipiv2 = Kokkos::create_mirror_view(ipiv2);

  for (int ib = 0; ib < N; ib++) {
    h_ipiv0(ib, 0) = 1;
    h_ipiv0(ib, 1) = 2;
    h_ipiv0(ib, 2) = 0;

    h_ipiv1(ib, 0) = 1;
    h_ipiv1(ib, 1) = 2;
    h_ipiv1(ib, 2) = 0;

    h_ipiv2(ib, 0) = 2;
    h_ipiv2(ib, 1) = 0;
    h_ipiv2(ib, 2) = 1;
  }

  Kokkos::deep_copy(ipiv0, h_ipiv0);
  Kokkos::deep_copy(ipiv1, h_ipiv1);
  Kokkos::deep_copy(ipiv2, h_ipiv2);

  auto info0 = Functor_BatchedSerialLaswp<DeviceType, PivView2DType, View2DType, ArgDirect>(ipiv0, A0).run();
  auto info1 = Functor_BatchedSerialLaswp<DeviceType, PivView2DType, View3DType, ArgDirect>(ipiv1, A1).run();
  auto info2 = Functor_BatchedSerialLaswp<DeviceType, PivView2DType, View3DType, ArgDirect>(ipiv2, A2).run();

  Kokkos::fence();
  EXPECT_EQ(info0, 0);
  EXPECT_EQ(info1, 0);
  EXPECT_EQ(info2, 0);

  // Copy permuted A to A_identity which is permuted back to original A
  Kokkos::deep_copy(A0_identity, A0);
  Kokkos::deep_copy(A1_identity, A1);
  Kokkos::deep_copy(A2_identity, A2);

  using InvDirect =
      typename std::conditional_t<std::is_same_v<ArgDirect, Direct::Forward>, Direct::Backward, Direct::Forward>;

  // Permute b_identity in inverse direction to get original b
  info0 = Functor_BatchedSerialLaswp<DeviceType, PivView2DType, View2DType, InvDirect>(ipiv0, A0_identity).run();
  info1 = Functor_BatchedSerialLaswp<DeviceType, PivView2DType, View3DType, InvDirect>(ipiv1, A1_identity).run();
  info2 = Functor_BatchedSerialLaswp<DeviceType, PivView2DType, View3DType, InvDirect>(ipiv2, A2_identity).run();

  Kokkos::fence();
  EXPECT_EQ(info0, 0);
  EXPECT_EQ(info1, 0);
  EXPECT_EQ(info2, 0);

  // this eps is about 10^-14
  RealType eps = 1.0e3 * ats::epsilon();

  Kokkos::deep_copy(h_A0, A0);
  Kokkos::deep_copy(h_A1, A1);
  Kokkos::deep_copy(h_A2, A2);
  auto h_A0_identity = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A0_identity);
  auto h_A1_identity = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A1_identity);
  auto h_A2_identity = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A2_identity);

  auto h_Ref0_identity = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ref0_identity);
  auto h_Ref1_identity = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ref1_identity);
  auto h_Ref2_identity = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ref2_identity);
  // Check if A is permuted correctly and A_identity is restored
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < 3; i++) {
      EXPECT_NEAR_KK(h_A0(ib, i), h_Ref0(ib, i), eps);
      EXPECT_NEAR_KK(h_A0_identity(ib, i), h_Ref0_identity(ib, i), eps);
      for (int j = 0; j < 2; j++) {
        EXPECT_NEAR_KK(h_A2(ib, i, j), h_Ref2(ib, i, j), eps);
        EXPECT_NEAR_KK(h_A2_identity(ib, i, j), h_Ref2_identity(ib, i, j), eps);
      }
      for (int j = 0; j < 3; j++) {
        EXPECT_NEAR_KK(h_A1(ib, i, j), h_Ref1(ib, i, j), eps);
        EXPECT_NEAR_KK(h_A1_identity(ib, i, j), h_Ref1_identity(ib, i, j), eps);
      }
    }
  }
}

/// \brief Implementation details of batched laswp test on vectors
///        Apply pivot to vector
///
/// \param N [in] Batch size of vectors
/// \param BlkSize [in] Length of vector b
template <typename DeviceType, typename ScalarType, typename LayoutType, typename ArgDirect>
void impl_test_batched_laswp_vector(const int N, const int BlkSize) {
  using ats           = typename Kokkos::ArithTraits<ScalarType>;
  using RealType      = typename ats::mag_type;
  using View2DType    = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using PivView2DType = Kokkos::View<int **, LayoutType, DeviceType>;

  View2DType b("b", N, BlkSize), Ref("Ref", N, BlkSize), b_identity("b_identity", N, BlkSize),
      Ref_identity("Ref_identity", N, BlkSize);
  PivView2DType ipiv("ipiv", N, BlkSize);

  // Initialize A_reconst with random matrix
  using execution_space = typename DeviceType::execution_space;
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);
  ScalarType randStart, randEnd;

  KokkosKernels::Impl::getRandomBounds(1.0, randStart, randEnd);
  Kokkos::fill_random(b, rand_pool, randStart, randEnd);
  Kokkos::deep_copy(Ref, b);  // This Ref is used to store permuted b
  Kokkos::deep_copy(Ref_identity, b);

  // Permute ipiv
  auto h_ipiv = Kokkos::create_mirror_view(ipiv);
  std::vector<int> ipiv_vec(BlkSize);
  for (int i = 0; i < BlkSize; i++) {
    ipiv_vec[i] = i;
  }
  auto rng = std::default_random_engine{};
  std::shuffle(ipiv_vec.begin(), ipiv_vec.end(), rng);
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      h_ipiv(ib, i) = ipiv_vec[i];
    }
  }
  Kokkos::deep_copy(ipiv, h_ipiv);

  auto info = Functor_BatchedSerialLaswp<DeviceType, PivView2DType, View2DType, ArgDirect>(ipiv, b).run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // Copy permuted b to b_identity which is permuted back to original b
  Kokkos::deep_copy(b_identity, b);

  // Permute b_identity in inverse direction to get original b
  using InvDirect =
      typename std::conditional_t<std::is_same_v<ArgDirect, Direct::Forward>, Direct::Backward, Direct::Forward>;
  Functor_BatchedSerialLaswp<DeviceType, PivView2DType, View2DType, InvDirect>(ipiv, b_identity).run();

  // Make a reference
  auto h_Ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ref);
  for (int ib = 0; ib < N; ib++) {
    if constexpr (std::is_same_v<ArgDirect, Direct::Forward>) {
      // Permute Ref by forward pivoting
      for (int i = 0; i < BlkSize; i++) {
        if (h_ipiv(ib, i) != i) {
          Kokkos::kokkos_swap(h_Ref(ib, h_ipiv(ib, i)), h_Ref(ib, i));
        }
      }
    } else {
      // Permute Ref by backward pivoting
      for (int i = (BlkSize - 1); i >= 0; --i) {
        if (h_ipiv(ib, i) != i) {
          Kokkos::kokkos_swap(h_Ref(ib, h_ipiv(ib, i)), h_Ref(ib, i));
        }
      }
    }
  }

  // this eps is about 10^-14
  RealType eps = 1.0e3 * ats::epsilon();

  auto h_b            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  auto h_b_identity   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_identity);
  auto h_Ref_identity = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ref_identity);
  // Check b is permuted correctly and b_identity is restored
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      EXPECT_NEAR_KK(h_b(ib, i), h_Ref(ib, i), eps);
      EXPECT_NEAR_KK(h_b_identity(ib, i), h_Ref_identity(ib, i), eps);
    }
  }
}

/// \brief Implementation details of batched laswp test on matrices
///        Apply pivot to matrix
///
/// \param N [in] Batch size of vectors
/// \param BlkSize [in] Row size of matrix A
template <typename DeviceType, typename ScalarType, typename LayoutType, typename ArgDirect>
void impl_test_batched_laswp_matrix(const int N, const int BlkSize) {
  using ats           = typename Kokkos::ArithTraits<ScalarType>;
  using RealType      = typename ats::mag_type;
  using View3DType    = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using PivView2DType = Kokkos::View<int **, LayoutType, DeviceType>;

  // In order for the tests on non-square matrices, we fix the column size to 5
  // and scan with the row size
  constexpr int M = 5;
  View3DType A("A", N, BlkSize, M), Ref("Ref", N, BlkSize, M), A_identity("A_identity", N, BlkSize, M),
      Ref_identity("Ref_identity", N, BlkSize, M);
  PivView2DType ipiv("ipiv", N, BlkSize);

  // Initialize A_reconst with random matrix
  using execution_space = typename DeviceType::execution_space;
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);
  ScalarType randStart, randEnd;

  KokkosKernels::Impl::getRandomBounds(1.0, randStart, randEnd);
  Kokkos::fill_random(A, rand_pool, randStart, randEnd);
  Kokkos::deep_copy(Ref, A);  // This Ref is used to store permuted A
  Kokkos::deep_copy(Ref_identity, A);

  // Permute ipiv
  auto h_ipiv = Kokkos::create_mirror_view(ipiv);
  std::vector<int> ipiv_vec(BlkSize);
  for (int i = 0; i < BlkSize; i++) {
    ipiv_vec[i] = i;
  }
  auto rng = std::default_random_engine{};
  std::shuffle(ipiv_vec.begin(), ipiv_vec.end(), rng);
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      h_ipiv(ib, i) = ipiv_vec[i];
    }
  }
  Kokkos::deep_copy(ipiv, h_ipiv);

  auto info = Functor_BatchedSerialLaswp<DeviceType, PivView2DType, View3DType, ArgDirect>(ipiv, A).run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // Copy permuted A to A_identity which is permuted back to original A
  Kokkos::deep_copy(A_identity, A);

  // Permute A_identity in inverse direction to get original A
  using InvDirect =
      typename std::conditional_t<std::is_same_v<ArgDirect, Direct::Forward>, Direct::Backward, Direct::Forward>;
  info = Functor_BatchedSerialLaswp<DeviceType, PivView2DType, View3DType, InvDirect>(ipiv, A_identity).run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // permute Ref by ipiv
  auto h_Ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ref);
  for (int ib = 0; ib < N; ib++) {
    if constexpr (std::is_same_v<ArgDirect, Direct::Forward>) {
      // Permute Ref by forward pivoting
      for (int i = 0; i < BlkSize; i++) {
        if (h_ipiv(ib, i) != i) {
          for (int j = 0; j < M; j++) {
            Kokkos::kokkos_swap(h_Ref(ib, h_ipiv(ib, i), j), h_Ref(ib, i, j));
          }
        }
      }
    } else {
      // Permute Ref by backward pivoting
      for (int i = (BlkSize - 1); i >= 0; --i) {
        if (h_ipiv(ib, i) != i) {
          for (int j = 0; j < M; j++) {
            Kokkos::kokkos_swap(h_Ref(ib, h_ipiv(ib, i), j), h_Ref(ib, i, j));
          }
        }
      }
    }
  }

  // this eps is about 10^-14
  RealType eps = 1.0e3 * ats::epsilon();

  auto h_A            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  auto h_A_identity   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_identity);
  auto h_Ref_identity = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ref_identity);
  // Check A is permuted correctly and A_identity is restored
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      for (int j = 0; j < M; j++) {
        EXPECT_NEAR_KK(h_A(ib, i, j), h_Ref(ib, i, j), eps);
        EXPECT_NEAR_KK(h_A_identity(ib, i, j), h_Ref_identity(ib, i, j), eps);
      }
    }
  }
}

}  // namespace Laswp
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename ArgDirect>
int test_batched_laswp() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    Test::Laswp::impl_test_batched_laswp_analytical<DeviceType, ScalarType, LayoutType, ArgDirect>(1);
    Test::Laswp::impl_test_batched_laswp_analytical<DeviceType, ScalarType, LayoutType, ArgDirect>(2);
    for (int i = 0; i < 10; i++) {
      Test::Laswp::impl_test_batched_laswp_vector<DeviceType, ScalarType, LayoutType, ArgDirect>(1, i);
      Test::Laswp::impl_test_batched_laswp_vector<DeviceType, ScalarType, LayoutType, ArgDirect>(2, i);
      Test::Laswp::impl_test_batched_laswp_matrix<DeviceType, ScalarType, LayoutType, ArgDirect>(1, i);
      Test::Laswp::impl_test_batched_laswp_matrix<DeviceType, ScalarType, LayoutType, ArgDirect>(2, i);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    Test::Laswp::impl_test_batched_laswp_analytical<DeviceType, ScalarType, LayoutType, ArgDirect>(1);
    Test::Laswp::impl_test_batched_laswp_analytical<DeviceType, ScalarType, LayoutType, ArgDirect>(2);
    for (int i = 0; i < 10; i++) {
      Test::Laswp::impl_test_batched_laswp_vector<DeviceType, ScalarType, LayoutType, ArgDirect>(1, i);
      Test::Laswp::impl_test_batched_laswp_vector<DeviceType, ScalarType, LayoutType, ArgDirect>(2, i);
      Test::Laswp::impl_test_batched_laswp_matrix<DeviceType, ScalarType, LayoutType, ArgDirect>(1, i);
      Test::Laswp::impl_test_batched_laswp_matrix<DeviceType, ScalarType, LayoutType, ArgDirect>(2, i);
    }
  }
#endif

  return 0;
}

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F(TestCategory, test_batched_laswp_f_float) { test_batched_laswp<TestDevice, float, Direct::Forward>(); }
TEST_F(TestCategory, test_batched_laswp_b_float) { test_batched_laswp<TestDevice, float, Direct::Backward>(); }
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, test_batched_laswp_f_double) { test_batched_laswp<TestDevice, double, Direct::Forward>(); }
TEST_F(TestCategory, test_batched_laswp_b_double) { test_batched_laswp<TestDevice, double, Direct::Backward>(); }
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
TEST_F(TestCategory, test_batched_laswp_f_fcomplex) {
  test_batched_laswp<TestDevice, Kokkos::complex<float>, Direct::Forward>();
}
TEST_F(TestCategory, test_batched_laswp_b_fcomplex) {
  test_batched_laswp<TestDevice, Kokkos::complex<float>, Direct::Backward>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
TEST_F(TestCategory, test_batched_laswp_f_dcomplex) {
  test_batched_laswp<TestDevice, Kokkos::complex<double>, Direct::Forward>();
}
TEST_F(TestCategory, test_batched_laswp_b_dcomplex) {
  test_batched_laswp<TestDevice, Kokkos::complex<double>, Direct::Backward>();
}
#endif
