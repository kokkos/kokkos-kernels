// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

#include <concepts>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBatched_Symv.hpp>
#include <KokkosBatched_Sypmv.hpp>
#include "Test_Batched_DenseUtils.hpp"

namespace Test {
namespace Sypmv {

template <bool Conj, typename U, typename M>
struct ParamTag {
  static constexpr bool conj = Conj;
  using uplo                 = U;
  using mode                 = M;
};

template <typename DeviceType, typename ScalarType, typename APViewType, typename XViewType, typename YViewType,
          typename ParamTagType>
struct Functor_BatchedSypmv {
  using execution_space      = typename DeviceType::execution_space;
  using member_type          = typename Kokkos::TeamPolicy<execution_space>::member_type;
  using ArgMode              = typename ParamTagType::mode;
  using ArgUplo              = typename ParamTagType::uplo;
  static constexpr bool conj = ParamTagType::conj;

  APViewType m_ap;
  XViewType m_x;
  YViewType m_y;
  ScalarType m_alpha, m_beta;

  Functor_BatchedSypmv(const ScalarType alpha, const APViewType &ap, const XViewType &x, const ScalarType beta,
                       const YViewType &y)
      : m_ap(ap), m_x(x), m_y(y), m_alpha(alpha), m_beta(beta) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type &member, int &info) const {
    const int k = member.league_rank();

    auto sub_ap = Kokkos::subview(m_ap, k, Kokkos::ALL());
    auto sub_x  = Kokkos::subview(m_x, k, Kokkos::ALL());
    auto sub_y  = Kokkos::subview(m_y, k, Kokkos::ALL());

    if constexpr (std::same_as<ArgMode, KokkosBatched::Mode::Serial>) {
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        info += KokkosBatched::SerialSypmv<ArgUplo, conj>::invoke(m_alpha, sub_ap, sub_x, m_beta, sub_y);
      });
    } else if constexpr (std::same_as<ArgMode, KokkosBatched::Mode::Team>) {
      info +=
          KokkosBatched::TeamSypmv<member_type, ArgUplo, conj>::invoke(member, m_alpha, sub_ap, sub_x, m_beta, sub_y);
    } else if constexpr (std::same_as<ArgMode, KokkosBatched::Mode::TeamVector>) {
      info += KokkosBatched::TeamVectorSypmv<member_type, ArgUplo, conj>::invoke(member, m_alpha, sub_ap, sub_x, m_beta,
                                                                                 sub_y);
    }
  }

  inline int run() {
    using value_type        = typename APViewType::non_const_value_type;
    std::string name_region = std::same_as<ArgMode, KokkosBatched::Mode::Serial> ? "KokkosBatched::Test::SerialSypmv"
                              : std::same_as<ArgMode, KokkosBatched::Mode::Team>
                                  ? "KokkosBatched::Test::TeamSypmv"
                                  : "KokkosBatched::Test::TeamVectorSypmv";
    const std::string name_value_type = KokkosKernels::ArithTraits<value_type>::name();
    std::string name                  = name_region + name_value_type;
    int info_sum                      = 0;
    Kokkos::Profiling::pushRegion(name.c_str());
    const int league_size = m_ap.extent_int(0);
    Kokkos::TeamPolicy<execution_space> policy(league_size, Kokkos::AUTO);
    Kokkos::parallel_reduce(name.c_str(), policy, *this, info_sum);
    Kokkos::Profiling::popRegion();
    return info_sum;
  }
};

template <typename DeviceType, typename ScalarType, typename AViewType, typename XViewType, typename YViewType,
          typename ParamTagType>
struct Functor_BatchedSymv {
  using execution_space = typename DeviceType::execution_space;
  using member_type     = typename Kokkos::TeamPolicy<execution_space>::member_type;
  using ArgMode         = typename ParamTagType::mode;
  using ArgUplo         = typename ParamTagType::uplo;
  using ArgTrans =
      std::conditional_t<ParamTagType::conj, KokkosBatched::Trans::ConjTranspose, KokkosBatched::Trans::Transpose>;
  AViewType m_A;
  XViewType m_x;
  YViewType m_y;
  ScalarType m_alpha, m_beta;

  Functor_BatchedSymv(const ScalarType alpha, const AViewType &A, const XViewType &x, const ScalarType beta,
                      const YViewType &y)
      : m_A(A), m_x(x), m_y(y), m_alpha(alpha), m_beta(beta) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type &member, int &info) const {
    const int k = member.league_rank();

    auto sub_A = Kokkos::subview(m_A, k, Kokkos::ALL(), Kokkos::ALL());
    auto sub_x = Kokkos::subview(m_x, k, Kokkos::ALL());
    auto sub_y = Kokkos::subview(m_y, k, Kokkos::ALL());

    if constexpr (std::same_as<ArgMode, KokkosBatched::Mode::Serial>) {
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        info += KokkosBatched::SerialSymv<ArgUplo, ArgTrans>::invoke(m_alpha, sub_A, sub_x, m_beta, sub_y);
      });
    } else if constexpr (std::same_as<ArgMode, KokkosBatched::Mode::Team>) {
      info +=
          KokkosBatched::TeamSymv<member_type, ArgUplo, ArgTrans>::invoke(member, m_alpha, sub_A, sub_x, m_beta, sub_y);
    } else if constexpr (std::same_as<ArgMode, KokkosBatched::Mode::TeamVector>) {
      info += KokkosBatched::TeamVectorSymv<member_type, ArgUplo, ArgTrans>::invoke(member, m_alpha, sub_A, sub_x,
                                                                                    m_beta, sub_y);
    }
  }

  inline int run() {
    using value_type        = typename AViewType::non_const_value_type;
    std::string name_region = std::same_as<ArgMode, KokkosBatched::Mode::Serial> ? "KokkosBatched::Test::SerialSypmv"
                              : std::same_as<ArgMode, KokkosBatched::Mode::Team>
                                  ? "KokkosBatched::Test::TeamSypmv"
                                  : "KokkosBatched::Test::TeamVectorSypmv";
    const std::string name_value_type = KokkosKernels::ArithTraits<value_type>::name();
    std::string name                  = name_region + name_value_type;
    int info_sum                      = 0;
    Kokkos::Profiling::pushRegion(name.c_str());
    const int league_size = m_A.extent_int(0);
    Kokkos::TeamPolicy<execution_space> policy(league_size, Kokkos::AUTO);
    Kokkos::parallel_reduce(name.c_str(), policy, *this, info_sum);
    Kokkos::Profiling::popRegion();
    return info_sum;
  }
};

/// \brief Implementation details of batched sypmv analytical test
///        to confirm y := alpha*op( A )*x + beta*y is computed correctly
///        alpha = 1.5, beta = 1.2
///        4x4 matrix (upper)
///        U: [[1, -3, -2,  0],
///            [0,  3, -1, -2],
///            [0,  0,  9,  5],
///            [0,  0,  0, 27]]
///        U_packed: [1, -3, 3, -2, -1, 9, 0, -2, 5, 27]
///        x: [1, 2, 3, 4], y: [5, 6, 7, 8]
///        Ref: [-10.5  -4.8  72.9 188.1]
///
///        4x4 matrix (lower)
///        L: [[ 1  0  0  0]
///            [ 3  3  0  0]
///            [ 2  1  9  0]
///            [ 0  2 -5 27]]
///        L_packed:  [1, 3, 2, 0, 3, 1, 2, 9, -5, 27]
///        x: [1, 2, 3, 4], y: [5, 6, 7, 8]
///        Ref: [-10.5  -4.8  72.9 188.1]
///
/// \param Nb [in] Batch size of matrices
template <typename DeviceType, typename ScalarType, typename LayoutType, typename ParamTagType>
void impl_test_batched_sypmv_analytical(const std::size_t Nb) {
  using ats               = typename KokkosKernels::ArithTraits<ScalarType>;
  using RealType          = typename ats::mag_type;
  using View2DType        = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using StridedView2DType = Kokkos::View<ScalarType **, Kokkos::LayoutStride, DeviceType>;
  using ArgUplo           = typename ParamTagType::uplo;

  const std::size_t BlkSize     = 4;
  const std::size_t packed_size = BlkSize * (BlkSize + 1) / 2;
  View2DType ap("ap", Nb, packed_size), x("x", Nb, BlkSize), y("y", Nb, BlkSize), y_ref("y_ref", Nb, BlkSize);

  const std::size_t incx = 2;
  // Testing incx argument with strided views
  Kokkos::LayoutStride layout{Nb, incx, BlkSize, Nb * incx};
  StridedView2DType x_s("x_s", layout), y_s("y_s", layout);

  // Only filling x2, A2 and deep_copy from its subview
  auto h_ap    = Kokkos::create_mirror_view(ap);
  auto h_x     = Kokkos::create_mirror_view(x);
  auto h_y     = Kokkos::create_mirror_view(y);
  auto h_y_ref = Kokkos::create_mirror_view(y_ref);

  for (std::size_t ib = 0; ib < Nb; ib++) {
    if (std::is_same_v<ArgUplo, KokkosBatched::Uplo::Upper>) {
      h_ap(ib, 0) = 1;
      h_ap(ib, 1) = -3;
      h_ap(ib, 2) = 3;
      h_ap(ib, 3) = -2;
      h_ap(ib, 4) = -1;
      h_ap(ib, 5) = 9;
      h_ap(ib, 6) = 0;
      h_ap(ib, 7) = -2;
      h_ap(ib, 8) = 5;
      h_ap(ib, 9) = 27;

      h_y_ref(ib, 0) = -10.5;
      h_y_ref(ib, 1) = -4.8;
      h_y_ref(ib, 2) = 72.9;
      h_y_ref(ib, 3) = 188.1;
    } else {
      h_ap(ib, 0) = 1;
      h_ap(ib, 1) = 3;
      h_ap(ib, 2) = 2;
      h_ap(ib, 3) = 0;
      h_ap(ib, 4) = 3;
      h_ap(ib, 5) = 1;
      h_ap(ib, 6) = 2;
      h_ap(ib, 7) = 9;
      h_ap(ib, 8) = -5;
      h_ap(ib, 9) = 27;

      h_y_ref(ib, 0) = 25.5;
      h_y_ref(ib, 1) = 37.2;
      h_y_ref(ib, 2) = 24.9;
      h_y_ref(ib, 3) = 155.1;
    }

    for (std::size_t j = 0; j < BlkSize; j++) {
      h_x(ib, j) = static_cast<ScalarType>(j + 1);
      h_y(ib, j) = static_cast<ScalarType>(j + 5);
    }
  }

  Kokkos::deep_copy(ap, h_ap);
  Kokkos::deep_copy(x, h_x);
  Kokkos::deep_copy(y, h_y);

  // Deep copy to strided views
  Kokkos::deep_copy(x_s, x);
  Kokkos::deep_copy(y_s, y);

  const ScalarType alpha = 1.5, beta = 1.2;

  auto info = Functor_BatchedSypmv<DeviceType, ScalarType, View2DType, View2DType, View2DType, ParamTagType>(alpha, ap,
                                                                                                             x, beta, y)
                  .run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // With strided views
  info = Functor_BatchedSypmv<DeviceType, ScalarType, View2DType, StridedView2DType, StridedView2DType, ParamTagType>(
             alpha, ap, x_s, beta, y_s)
             .run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  RealType eps = 1.0e1 * ats::epsilon();
  Kokkos::deep_copy(h_y, y);

  // Check if y := alpha*op(A)*x + beta*y is computed correctly
  for (std::size_t ib = 0; ib < Nb; ib++) {
    for (std::size_t i = 0; i < BlkSize; i++) {
      EXPECT_NEAR_KK(h_y(ib, i), h_y_ref(ib, i), eps);
    }
  }

  // Testing for strided views, reusing y
  Kokkos::deep_copy(y, y_s);
  Kokkos::deep_copy(h_y, y);

  // Check if y := alpha*op(A)*x + beta*y is computed correctly
  for (std::size_t ib = 0; ib < Nb; ib++) {
    for (std::size_t i = 0; i < BlkSize; i++) {
      EXPECT_NEAR_KK(h_y(ib, i), h_y_ref(ib, i), eps);
    }
  }
}

/// \brief Implementation details of batched sypmv test
///
/// \param[in] Nb Batch size of matrices
/// \param[in] BlkSize Block size of matrix A
template <typename DeviceType, typename ScalarType, typename LayoutType, typename ParamTagType>
void impl_test_batched_sypmv(const std::size_t Nb, const std::size_t BlkSize) {
  using ats        = typename KokkosKernels::ArithTraits<ScalarType>;
  using RealType   = typename ats::mag_type;
  using View2DType = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using ArgUplo    = typename ParamTagType::uplo;
  using ArgTrans =
      std::conditional_t<ParamTagType::conj, KokkosBatched::Trans::ConjTranspose, KokkosBatched::Trans::Transpose>;

  // Dense Matrix
  View3DType A("A", Nb, BlkSize, BlkSize), A_sym("A_sym", Nb, BlkSize, BlkSize);

  // Packed Matrix
  const std::size_t packed_size = BlkSize * (BlkSize + 1) / 2;
  View2DType ap("ap", Nb, packed_size);

  // Vector x
  View2DType x("x", Nb, BlkSize), y("y", Nb, BlkSize), y_ref("y_ref", Nb, BlkSize);

  // Create a random matrix A and x
  using execution_space = typename DeviceType::execution_space;
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);
  ScalarType randStart, randEnd;

  KokkosKernels::Impl::getRandomBounds(1.0, randStart, randEnd);
  Kokkos::fill_random(A, rand_pool, randStart, randEnd);
  Kokkos::fill_random(x, rand_pool, randStart, randEnd);
  Kokkos::fill_random(y, rand_pool, randStart, randEnd);

  // Upper or lower triangular part of A
  dense_to_symmetric<ArgUplo, ArgTrans>(A, A_sym);
  dense_to_packed<ArgUplo>(A_sym, ap);

  // Set reference y to the initial y values
  Kokkos::deep_copy(y_ref, y);

  const ScalarType alpha = 1.5, beta = 2.5;
  auto info = Functor_BatchedSypmv<DeviceType, ScalarType, View2DType, View2DType, View2DType, ParamTagType>(alpha, ap,
                                                                                                             x, beta, y)
                  .run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // Make references with Symv
  Functor_BatchedSymv<DeviceType, ScalarType, View3DType, View2DType, View2DType, ParamTagType>(alpha, A_sym, x, beta,
                                                                                                y_ref)
      .run();

  RealType eps = 1.0e1 * ats::epsilon();
  auto h_y     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y);
  auto h_y_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y_ref);

  // Check if y:= alpha * op(ap) * x + beta * y
  for (std::size_t ib = 0; ib < Nb; ib++) {
    for (std::size_t i = 0; i < BlkSize; i++) {
      EXPECT_NEAR_KK(h_y(ib, i), h_y_ref(ib, i), eps);
    }
  }
}

}  // namespace Sypmv
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename ParamTagType, typename LayoutType>
void test_batched_sypmv_layout() {
  Test::Sypmv::impl_test_batched_sypmv_analytical<DeviceType, ScalarType, LayoutType, ParamTagType>(1);
  Test::Sypmv::impl_test_batched_sypmv_analytical<DeviceType, ScalarType, LayoutType, ParamTagType>(2);
  for (int i = 0; i < 10; i++) {
    Test::Sypmv::impl_test_batched_sypmv<DeviceType, ScalarType, LayoutType, ParamTagType>(1, i);
    Test::Sypmv::impl_test_batched_sypmv<DeviceType, ScalarType, LayoutType, ParamTagType>(2, i);
  }
}

template <typename DeviceType, typename ScalarType, typename ParamTagType>
int test_batched_sypmv() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    test_batched_sypmv_layout<DeviceType, ScalarType, ParamTagType, LayoutType>();
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    test_batched_sypmv_layout<DeviceType, ScalarType, ParamTagType, LayoutType>();
  }
#endif

  return 0;
}

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F(TestCategory, test_batched_serial_sypmv_f_l_float) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Serial>;
  test_batched_sypmv<TestDevice, float, param_tag_type>();
}
TEST_F(TestCategory, test_batched_serial_sypmv_f_u_float) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Serial>;
  test_batched_sypmv<TestDevice, float, param_tag_type>();
}
TEST_F(TestCategory, test_batched_serial_sypmv_t_l_float) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Serial>;
  test_batched_sypmv<TestDevice, float, param_tag_type>();
}
TEST_F(TestCategory, test_batched_serial_sypmv_t_u_float) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Serial>;
  test_batched_sypmv<TestDevice, float, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_f_l_float) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Team>;
  test_batched_sypmv<TestDevice, float, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_f_u_float) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Team>;
  test_batched_sypmv<TestDevice, float, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_t_l_float) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Team>;
  test_batched_sypmv<TestDevice, float, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_t_u_float) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Team>;
  test_batched_sypmv<TestDevice, float, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_f_l_float) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, float, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_f_u_float) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, float, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_t_l_float) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, float, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_t_u_float) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, float, param_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, test_batched_serial_sypmv_f_l_double) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Serial>;
  test_batched_sypmv<TestDevice, double, param_tag_type>();
}
TEST_F(TestCategory, test_batched_serial_sypmv_f_u_double) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Serial>;
  test_batched_sypmv<TestDevice, double, param_tag_type>();
}
TEST_F(TestCategory, test_batched_serial_sypmv_t_l_double) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Serial>;
  test_batched_sypmv<TestDevice, double, param_tag_type>();
}
TEST_F(TestCategory, test_batched_serial_sypmv_t_u_double) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Serial>;
  test_batched_sypmv<TestDevice, double, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_f_l_double) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Team>;
  test_batched_sypmv<TestDevice, double, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_f_u_double) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Team>;
  test_batched_sypmv<TestDevice, double, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_t_l_double) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Team>;
  test_batched_sypmv<TestDevice, double, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_t_u_double) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Team>;
  test_batched_sypmv<TestDevice, double, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_f_l_double) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, double, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_f_u_double) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, double, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_t_l_double) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, double, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_t_u_double) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, double, param_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
TEST_F(TestCategory, test_batched_serial_sypmv_f_l_fcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Serial>;
  test_batched_sypmv<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_serial_sypmv_f_u_fcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Serial>;
  test_batched_sypmv<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_serial_sypmv_t_l_fcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Serial>;
  test_batched_sypmv<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_serial_sypmv_t_u_fcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Serial>;
  test_batched_sypmv<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_f_l_fcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Team>;
  test_batched_sypmv<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_f_u_fcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Team>;
  test_batched_sypmv<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_t_l_fcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Team>;
  test_batched_sypmv<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_t_u_fcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Team>;
  test_batched_sypmv<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_f_l_fcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_f_u_fcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_t_l_fcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_t_u_fcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
TEST_F(TestCategory, test_batched_serial_sypmv_f_l_dcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Serial>;
  test_batched_sypmv<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_serial_sypmv_f_u_dcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Serial>;
  test_batched_sypmv<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_serial_sypmv_t_l_dcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Serial>;
  test_batched_sypmv<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_serial_sypmv_t_u_dcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Serial>;
  test_batched_sypmv<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_f_l_dcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Team>;
  test_batched_sypmv<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_f_u_dcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Team>;
  test_batched_sypmv<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_t_l_dcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::Team>;
  test_batched_sypmv<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_sypmv_t_u_dcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::Team>;
  test_batched_sypmv<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_f_l_dcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_f_u_dcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_t_l_dcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<false, Uplo::Lower, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_team_vector_sypmv_t_u_dcomplex) {
  using param_tag_type = ::Test::Sypmv::ParamTag<true, Uplo::Upper, Mode::TeamVector>;
  test_batched_sypmv<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
#endif
