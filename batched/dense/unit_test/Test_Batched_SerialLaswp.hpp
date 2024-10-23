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

template <typename D>
struct ParamTag {
  using direct = D;
};

template <typename DeviceType, typename PivViewType, typename AViewType, typename ParamTagType>
struct Functor_BatchedSerialLaswp {
  using execution_space = typename DeviceType::execution_space;
  PivViewType m_ipiv;
  AViewType m_a;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialLaswp(const PivViewType &ipiv, const AViewType &a) : m_ipiv(ipiv), m_a(a) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const ParamTagType &, const int k, int &info) const {
    auto sub_ipiv = Kokkos::subview(m_ipiv, k, Kokkos::ALL());
    if constexpr (AViewType::rank == 3) {
      auto sub_a = Kokkos::subview(m_a, k, Kokkos::ALL(), Kokkos::ALL());
      info += KokkosBatched::SerialLaswp<typename ParamTagType::direct>::invoke(sub_ipiv, sub_a);
    } else {
      auto sub_a = Kokkos::subview(m_a, k, Kokkos::ALL());
      info += KokkosBatched::SerialLaswp<typename ParamTagType::direct>::invoke(sub_ipiv, sub_a);
    }
  }

  inline int run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialLaswp");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    int info_sum                      = 0;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space, ParamTagType> policy(0, m_a.extent(0));
    Kokkos::parallel_reduce(name.c_str(), policy, *this, info_sum);
    Kokkos::Profiling::popRegion();
    return info_sum;
  }
};

template <typename DeviceType, typename ScalarType, typename LayoutType, typename ParamTagType>
/// \brief Implementation details of batched laswp test on vectors
///        Apply pivot to vector
///
/// \param N [in] Batch size of vectors
/// \param BlkSize [in] Length of vector b
void impl_test_batched_laswp_vector(const int N, const int BlkSize) {
  using ats           = typename Kokkos::ArithTraits<ScalarType>;
  using RealType      = typename ats::mag_type;
  using View2DType    = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using PivView2DType = Kokkos::View<int **, LayoutType, DeviceType>;

  View2DType b("b", N, BlkSize), Ref("Ref", N, BlkSize);
  PivView2DType ipiv("ipiv", N, BlkSize);

  // Initialize A_reconst with random matrix
  using execution_space = typename DeviceType::execution_space;
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);
  ScalarType randStart, randEnd;

  KokkosKernels::Impl::getRandomBounds(1.0, randStart, randEnd);
  Kokkos::fill_random(b, rand_pool, randStart, randEnd);
  Kokkos::deep_copy(Ref, b);

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

  auto info = Functor_BatchedSerialLaswp<DeviceType, PivView2DType, View2DType, ParamTagType>(ipiv, b).run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // Make a reference
  using direct = typename ParamTagType::direct;
  auto h_Ref   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ref);
  for (int ib = 0; ib < N; ib++) {
    if constexpr (std::is_same_v<direct, Direct::Forward>) {
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

  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  // Check b is permuted correctly
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      EXPECT_NEAR_KK(h_b(ib, i), h_Ref(ib, i), eps);
    }
  }
}

template <typename DeviceType, typename ScalarType, typename LayoutType, typename ParamTagType>
/// \brief Implementation details of batched laswp test on matrices
///        Apply pivot to matrix
///
/// \param N [in] Batch size of vectors
/// \param BlkSize [in] Row size of matrix A
void impl_test_batched_laswp_matrix(const int N, const int BlkSize) {
  using ats           = typename Kokkos::ArithTraits<ScalarType>;
  using RealType      = typename ats::mag_type;
  using View3DType    = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using PivView2DType = Kokkos::View<int **, LayoutType, DeviceType>;

  // In order for the tests on non-square matrices, we fix the column size to 5
  // and scan with the row size
  constexpr int M = 5;
  View3DType A("A", N, BlkSize, M), Ref("Ref", N, BlkSize, M);
  PivView2DType ipiv("ipiv", N, BlkSize);

  // Initialize A_reconst with random matrix
  using execution_space = typename DeviceType::execution_space;
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);
  ScalarType randStart, randEnd;

  KokkosKernels::Impl::getRandomBounds(1.0, randStart, randEnd);
  Kokkos::fill_random(A, rand_pool, randStart, randEnd);
  Kokkos::deep_copy(Ref, A);

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

  auto info = Functor_BatchedSerialLaswp<DeviceType, PivView2DType, View3DType, ParamTagType>(ipiv, A).run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // permute Ref by ipiv
  using direct = typename ParamTagType::direct;
  auto h_Ref   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ref);
  for (int ib = 0; ib < N; ib++) {
    if constexpr (std::is_same_v<direct, Direct::Forward>) {
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

  auto h_A = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  // Check b is permuted correctly
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      for (int j = 0; j < M; j++) {
        EXPECT_NEAR_KK(h_A(ib, i, j), h_Ref(ib, i, j), eps);
      }
    }
  }
}

}  // namespace Laswp
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename ParamTagType>
int test_batched_laswp() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    for (int i = 0; i < 10; i++) {
      Test::Laswp::impl_test_batched_laswp_vector<DeviceType, ScalarType, LayoutType, ParamTagType>(1, i);
      Test::Laswp::impl_test_batched_laswp_vector<DeviceType, ScalarType, LayoutType, ParamTagType>(2, i);
      Test::Laswp::impl_test_batched_laswp_matrix<DeviceType, ScalarType, LayoutType, ParamTagType>(1, i);
      Test::Laswp::impl_test_batched_laswp_matrix<DeviceType, ScalarType, LayoutType, ParamTagType>(2, i);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    for (int i = 0; i < 10; i++) {
      Test::Laswp::impl_test_batched_laswp_vector<DeviceType, ScalarType, LayoutType, ParamTagType>(1, i);
      Test::Laswp::impl_test_batched_laswp_vector<DeviceType, ScalarType, LayoutType, ParamTagType>(2, i);
      Test::Laswp::impl_test_batched_laswp_matrix<DeviceType, ScalarType, LayoutType, ParamTagType>(1, i);
      Test::Laswp::impl_test_batched_laswp_matrix<DeviceType, ScalarType, LayoutType, ParamTagType>(2, i);
    }
  }
#endif

  return 0;
}

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F(TestCategory, test_batched_laswp_f_float) {
  using param_tag_type = ::Test::Laswp::ParamTag<Direct::Forward>;

  test_batched_laswp<TestDevice, float, param_tag_type>();
}
TEST_F(TestCategory, test_batched_laswp_b_float) {
  using param_tag_type = ::Test::Laswp::ParamTag<Direct::Backward>;

  test_batched_laswp<TestDevice, float, param_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, test_batched_laswp_f_double) {
  using param_tag_type = ::Test::Laswp::ParamTag<Direct::Forward>;

  test_batched_laswp<TestDevice, double, param_tag_type>();
}
TEST_F(TestCategory, test_batched_laswp_b_double) {
  using param_tag_type = ::Test::Laswp::ParamTag<Direct::Backward>;

  test_batched_laswp<TestDevice, double, param_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
TEST_F(TestCategory, test_batched_laswp_f_fcomplex) {
  using param_tag_type = ::Test::Laswp::ParamTag<Direct::Forward>;

  test_batched_laswp<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_laswp_b_fcomplex) {
  using param_tag_type = ::Test::Laswp::ParamTag<Direct::Backward>;

  test_batched_laswp<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
TEST_F(TestCategory, test_batched_laswp_f_dcomplex) {
  using param_tag_type = ::Test::Laswp::ParamTag<Direct::Forward>;

  test_batched_laswp<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_laswp_b_dcomplex) {
  using param_tag_type = ::Test::Laswp::ParamTag<Direct::Backward>;

  test_batched_laswp<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
#endif
