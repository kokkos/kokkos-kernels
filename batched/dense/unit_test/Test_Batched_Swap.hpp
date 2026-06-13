// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBatched_Swap.hpp>
#include "Test_Batched_DenseUtils.hpp"

namespace Test {
namespace Swap {

template <typename DeviceType, typename XViewType, typename YViewType, typename ArgMode>
struct Functor_BatchedSwap {
  using execution_space = typename DeviceType::execution_space;
  using member_type     = typename Kokkos::TeamPolicy<execution_space>::member_type;
  XViewType m_x;
  YViewType m_y;

  Functor_BatchedSwap(const XViewType &x, const YViewType &y) : m_x(x), m_y(y) {}

  KOKKOS_INLINE_FUNCTION void operator()(const member_type &member, int &info) const {
    const int k = member.league_rank();
    auto sub_x  = Kokkos::subview(m_x, k, Kokkos::ALL());
    auto sub_y  = Kokkos::subview(m_y, k, Kokkos::ALL());
    if constexpr (std::is_same_v<ArgMode, KokkosBatched::Mode::Serial>) {
      Kokkos::single(Kokkos::PerTeam(member), [&]() { info += KokkosBatched::SerialSwap::invoke(sub_x, sub_y); });
    } else if constexpr (std::is_same_v<ArgMode, KokkosBatched::Mode::Team>) {
      info += KokkosBatched::TeamSwap<member_type>::invoke(member, sub_x, sub_y);
    } else if constexpr (std::is_same_v<ArgMode, KokkosBatched::Mode::TeamVector>) {
      info += KokkosBatched::TeamVectorSwap<member_type>::invoke(member, sub_x, sub_y);
    }
  }

  inline int run() {
    using value_type        = typename XViewType::non_const_value_type;
    std::string name_region = std::is_same_v<ArgMode, KokkosBatched::Mode::Serial> ? "KokkosBatched::Test::SerialSwap"
                              : std::is_same_v<ArgMode, KokkosBatched::Mode::Team>
                                  ? "KokkosBatched::Test::TeamSwap"
                                  : "KokkosBatched::Test::TeamVectorSwap";
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    int info_sum                      = 0;
    Kokkos::Profiling::pushRegion(name.c_str());
    const int league_size = m_x.extent_int(0);

    Kokkos::TeamPolicy<execution_space> policy(league_size, Kokkos::AUTO);
    Kokkos::parallel_reduce(name.c_str(), policy, *this, info_sum);

    Kokkos::Profiling::popRegion();
    return info_sum;
  }
};

/// \brief Implementation details of batched swap analytical test
/// x = [1, 3, 5], y = [2, 4, 6]
///
/// \tparam DeviceType Kokkos device type
/// \tparam ScalarType Kokkos scalar type
/// \tparam LayoutType1 Kokkos layout type for the views
/// \tparam LayoutType2 Kokkos layout type for the strided views
/// \tparam ArgMode: one of Mode::Serial, Mode::Team, Mode::TeamVector
///
/// \param[in] Nb Batch size of vectors
template <typename DeviceType, typename ScalarType, typename LayoutType1, typename LayoutType2, typename ArgMode>
void impl_test_batched_swap_analytical(const std::size_t Nb) {
  using ats             = typename KokkosKernels::ArithTraits<ScalarType>;
  using RealType        = typename ats::mag_type;
  using XViewType       = Kokkos::View<ScalarType **, LayoutType1, DeviceType>;
  using YViewType       = Kokkos::View<ScalarType **, LayoutType2, DeviceType>;
  using StridedViewType = Kokkos::View<ScalarType **, Kokkos::LayoutStride, DeviceType>;

  const std::size_t N = 3;
  XViewType x("x", Nb, N), x_ref("x_ref", Nb, N);
  YViewType y("y", Nb, N), y_ref("y_ref", Nb, N);

  const std::size_t incx = 2;
  // Testing incx argument with strided views
  Kokkos::LayoutStride layout{Nb, incx, N, Nb * incx};
  StridedViewType x_s("x_s", layout), y_s("y_s", layout);

  auto h_x = Kokkos::create_mirror_view(x);
  auto h_y = Kokkos::create_mirror_view(y);

  constexpr bool is_complex = KokkosKernels::ArithTraits<ScalarType>::is_complex;

  for (std::size_t ib = 0; ib < Nb; ib++) {
    if constexpr (is_complex) {
      h_x(ib, 0) = ScalarType(1, 7);
      h_x(ib, 1) = ScalarType(3, 9);
      h_x(ib, 2) = ScalarType(5, 11);

      h_y(ib, 0) = ScalarType(2, 8);
      h_y(ib, 1) = ScalarType(4, 10);
      h_y(ib, 2) = ScalarType(6, 12);
    } else {
      h_x(ib, 0) = ScalarType(1);
      h_x(ib, 1) = ScalarType(3);
      h_x(ib, 2) = ScalarType(5);

      h_y(ib, 0) = ScalarType(2);
      h_y(ib, 1) = ScalarType(4);
      h_y(ib, 2) = ScalarType(6);
    }
  }

  Kokkos::deep_copy(x, h_x);
  Kokkos::deep_copy(y, h_y);

  // Deep copy to strided views
  Kokkos::deep_copy(x_s, x);
  Kokkos::deep_copy(y_s, y);

  // Reference results after swap
  Kokkos::deep_copy(x_ref, y);
  Kokkos::deep_copy(y_ref, x);

  auto info = Functor_BatchedSwap<DeviceType, XViewType, YViewType, ArgMode>(x, y).run();
  EXPECT_EQ(info, 0);

  // With strided views
  info = Functor_BatchedSwap<DeviceType, StridedViewType, StridedViewType, ArgMode>(x_s, y_s).run();
  EXPECT_EQ(info, 0);

  RealType eps = 1.0e1 * ats::epsilon();
  Kokkos::deep_copy(h_x, x);
  Kokkos::deep_copy(h_y, y);
  auto h_x_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_ref);
  auto h_y_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y_ref);

  for (std::size_t ib = 0; ib < Nb; ib++) {
    for (std::size_t i = 0; i < N; i++) {
      KK_EXPECT_NEAR(h_x(ib, i), h_x_ref(ib, i), eps);
      KK_EXPECT_NEAR(h_y(ib, i), h_y_ref(ib, i), eps);
    }
  }

  // Testing for strided views, reusing x and y
  Kokkos::deep_copy(x, x_s);
  Kokkos::deep_copy(y, y_s);
  Kokkos::deep_copy(h_x, x);
  Kokkos::deep_copy(h_y, y);
  for (std::size_t ib = 0; ib < Nb; ib++) {
    for (std::size_t i = 0; i < N; i++) {
      KK_EXPECT_NEAR(h_x(ib, i), h_x_ref(ib, i), eps);
      KK_EXPECT_NEAR(h_y(ib, i), h_y_ref(ib, i), eps);
    }
  }
}

/// \brief Implementation details of batched swap test
///
/// \tparam DeviceType Kokkos device type
/// \tparam ScalarType Kokkos scalar type
/// \tparam LayoutType1 Kokkos layout type for the X views
/// \tparam LayoutType2 Kokkos layout type for the Y views
/// \tparam ArgMode: one of Mode::Serial, Mode::Team, Mode::TeamVector
///
/// \param[in] Nb Batch size of vectors
/// \param[in] N Length of the vector x
template <typename DeviceType, typename ScalarType, typename LayoutType1, typename LayoutType2, typename ArgMode>
void impl_test_batched_swap(const std::size_t Nb, const std::size_t N) {
  using ats             = typename KokkosKernels::ArithTraits<ScalarType>;
  using RealType        = typename ats::mag_type;
  using XViewType       = Kokkos::View<ScalarType **, LayoutType1, DeviceType>;
  using YViewType       = Kokkos::View<ScalarType **, LayoutType2, DeviceType>;
  using StridedViewType = Kokkos::View<ScalarType **, Kokkos::LayoutStride, DeviceType>;

  XViewType x("x", Nb, N), x_ref("x_ref", Nb, N);
  YViewType y("y", Nb, N), y_ref("y_ref", Nb, N);

  const std::size_t incx = 2;
  // Testing incx argument with strided views
  Kokkos::LayoutStride layout{Nb, incx, N, Nb * incx};
  StridedViewType x_s("x_s", layout), y_s("y_s", layout);

  // Create random x
  using execution_space = typename DeviceType::execution_space;
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);
  ScalarType randStart, randEnd;

  KokkosKernels::Impl::getRandomBounds(1.0, randStart, randEnd);
  Kokkos::fill_random(x, rand_pool, randStart, randEnd);
  Kokkos::fill_random(y, rand_pool, randStart, randEnd);

  Kokkos::fence();

  // Deep copy to strided views
  Kokkos::deep_copy(x_s, x);
  Kokkos::deep_copy(y_s, y);

  // Reference results after swap
  Kokkos::deep_copy(x_ref, y);
  Kokkos::deep_copy(y_ref, x);

  auto info = Functor_BatchedSwap<DeviceType, XViewType, YViewType, ArgMode>(x, y).run();
  EXPECT_EQ(info, 0);

  // With strided views
  info = Functor_BatchedSwap<DeviceType, StridedViewType, StridedViewType, ArgMode>(x_s, y_s).run();
  EXPECT_EQ(info, 0);

  Kokkos::fence();

  RealType eps = 1.0e1 * ats::epsilon();
  auto h_x     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  auto h_y     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y);
  auto h_x_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_ref);
  auto h_y_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y_ref);

  // Check if swap is correct
  for (std::size_t ib = 0; ib < Nb; ib++) {
    for (std::size_t i = 0; i < N; i++) {
      if (Kokkos::abs(h_x(ib, i) - h_x_ref(ib, i)) > eps || Kokkos::abs(h_y(ib, i) - h_y_ref(ib, i)) > eps) {
        std::string layout1 = std::is_same_v<LayoutType1, Kokkos::LayoutLeft> ? "LayoutLeft" : "LayoutRight";
        std::string layout2 = std::is_same_v<LayoutType2, Kokkos::LayoutLeft> ? "LayoutLeft" : "LayoutRight";
        std::cout << "Error at batch " << ib << " / " << Nb << ", index " << i << " / " << N << ": "
                  << "h_x = " << h_x(ib, i) << ", h_x_ref = " << h_x_ref(ib, i) << ", "
                  << "h_y = " << h_y(ib, i) << ", h_y_ref = " << h_y_ref(ib, i) << ", "
                  << "layout1 = " << layout1 << ", layout2 = " << layout2 << std::endl;
      }
      KK_EXPECT_NEAR(h_x(ib, i), h_x_ref(ib, i), eps);
      KK_EXPECT_NEAR(h_y(ib, i), h_y_ref(ib, i), eps);
    }
  }

  // Testing for strided views, reusing x and y
  Kokkos::deep_copy(x, x_s);
  Kokkos::deep_copy(y, y_s);
  Kokkos::deep_copy(h_x, x);
  Kokkos::deep_copy(h_y, y);
  for (std::size_t ib = 0; ib < Nb; ib++) {
    for (std::size_t i = 0; i < N; i++) {
      if (Kokkos::abs(h_x(ib, i) - h_x_ref(ib, i)) > eps || Kokkos::abs(h_y(ib, i) - h_y_ref(ib, i)) > eps) {
        std::string layout1 = std::is_same_v<LayoutType1, Kokkos::LayoutLeft> ? "LayoutLeft" : "LayoutRight";
        std::string layout2 = std::is_same_v<LayoutType2, Kokkos::LayoutLeft> ? "LayoutLeft" : "LayoutRight";
        std::cout << "Error with strided views at batch " << ib << " / " << Nb << ", index " << i << " / " << N << ": "
                  << "h_x = " << h_x(ib, i) << ", h_x_ref = " << h_x_ref(ib, i) << ", "
                  << "h_y = " << h_y(ib, i) << ", h_y_ref = " << h_y_ref(ib, i) << ", "
                  << "layout1 = " << layout1 << ", layout2 = " << layout2 << std::endl;
      }
      KK_EXPECT_NEAR(h_x(ib, i), h_x_ref(ib, i), eps);
      KK_EXPECT_NEAR(h_y(ib, i), h_y_ref(ib, i), eps);
    }
  }
}

}  // namespace Swap
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename ArgMode>
int test_batched_swap() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    Test::Swap::impl_test_batched_swap_analytical<DeviceType, ScalarType, LayoutType, Kokkos::LayoutLeft, ArgMode>(1);
    Test::Swap::impl_test_batched_swap_analytical<DeviceType, ScalarType, LayoutType, Kokkos::LayoutLeft, ArgMode>(2);
    Test::Swap::impl_test_batched_swap_analytical<DeviceType, ScalarType, LayoutType, Kokkos::LayoutRight, ArgMode>(1);
    Test::Swap::impl_test_batched_swap_analytical<DeviceType, ScalarType, LayoutType, Kokkos::LayoutRight, ArgMode>(2);

    for (int ib = 0; ib < 5; ib++) {
      for (int i = 0; i < 10; i++) {
        Test::Swap::impl_test_batched_swap<DeviceType, ScalarType, LayoutType, Kokkos::LayoutLeft, ArgMode>(ib, i);
        Test::Swap::impl_test_batched_swap<DeviceType, ScalarType, LayoutType, Kokkos::LayoutRight, ArgMode>(ib, i);
      }
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    Test::Swap::impl_test_batched_swap_analytical<DeviceType, ScalarType, LayoutType, Kokkos::LayoutLeft, ArgMode>(1);
    Test::Swap::impl_test_batched_swap_analytical<DeviceType, ScalarType, LayoutType, Kokkos::LayoutLeft, ArgMode>(2);
    Test::Swap::impl_test_batched_swap_analytical<DeviceType, ScalarType, LayoutType, Kokkos::LayoutRight, ArgMode>(1);
    Test::Swap::impl_test_batched_swap_analytical<DeviceType, ScalarType, LayoutType, Kokkos::LayoutRight, ArgMode>(2);

    for (int ib = 0; ib < 5; ib++) {
      for (int i = 0; i < 10; i++) {
        Test::Swap::impl_test_batched_swap<DeviceType, ScalarType, LayoutType, Kokkos::LayoutLeft, ArgMode>(ib, i);
        Test::Swap::impl_test_batched_swap<DeviceType, ScalarType, LayoutType, Kokkos::LayoutRight, ArgMode>(ib, i);
      }
    }
  }
#endif

  return 0;
}

#if defined(KOKKOSKERNELS_INST_FLOAT)
// Serial
TEST_F(TestCategory, test_batched_serial_swap_float) {
  test_batched_swap<TestDevice, float, KokkosBatched::Mode::Serial>();
}
// Team
TEST_F(TestCategory, test_batched_team_swap_float) {
  test_batched_swap<TestDevice, float, KokkosBatched::Mode::Team>();
}
// TeamVector
TEST_F(TestCategory, test_batched_teamvector_swap_float) {
  test_batched_swap<TestDevice, float, KokkosBatched::Mode::TeamVector>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
// Serial
TEST_F(TestCategory, test_batched_serial_swap_double) {
  test_batched_swap<TestDevice, double, KokkosBatched::Mode::Serial>();
}
// Team
TEST_F(TestCategory, test_batched_team_swap_double) {
  test_batched_swap<TestDevice, double, KokkosBatched::Mode::Team>();
}
// TeamVector
TEST_F(TestCategory, test_batched_teamvector_swap_double) {
  test_batched_swap<TestDevice, double, KokkosBatched::Mode::TeamVector>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
// Serial
TEST_F(TestCategory, test_batched_serial_swap_fcomplex) {
  test_batched_swap<TestDevice, Kokkos::complex<float>, KokkosBatched::Mode::Serial>();
}
// Team
TEST_F(TestCategory, test_batched_team_swap_fcomplex) {
  test_batched_swap<TestDevice, Kokkos::complex<float>, KokkosBatched::Mode::Team>();
}
// TeamVector
TEST_F(TestCategory, test_batched_teamvector_swap_fcomplex) {
  test_batched_swap<TestDevice, Kokkos::complex<float>, KokkosBatched::Mode::TeamVector>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
// Serial
TEST_F(TestCategory, test_batched_serial_swap_dcomplex) {
  test_batched_swap<TestDevice, Kokkos::complex<double>, KokkosBatched::Mode::Serial>();
}
// Team
TEST_F(TestCategory, test_batched_team_swap_dcomplex) {
  test_batched_swap<TestDevice, Kokkos::complex<double>, KokkosBatched::Mode::Team>();
}
// TeamVector
TEST_F(TestCategory, test_batched_teamvector_swap_dcomplex) {
  test_batched_swap<TestDevice, Kokkos::complex<double>, KokkosBatched::Mode::TeamVector>();
}
#endif
