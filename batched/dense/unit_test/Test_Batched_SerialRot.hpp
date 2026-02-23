// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBatched_Rot.hpp>
#include "Test_Batched_DenseUtils.hpp"

namespace Test {
namespace Rot {

template <typename T>
struct ParamTag {
  using trans = T;
};

template <typename DeviceType, typename XViewType, typename YViewType, typename CType, typename SType,
          typename ParamTagType>
struct Functor_BatchedSerialRot {
  using execution_space = typename DeviceType::execution_space;
  XViewType m_x;
  YViewType m_y;
  CType m_c;
  SType m_s;

  Functor_BatchedSerialRot(const XViewType &x, const YViewType &y, const CType c, const SType s)
      : m_x(x), m_y(y), m_c(c), m_s(s) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k, int &info) const {
    auto sub_x = Kokkos::subview(m_x, k, Kokkos::ALL());
    auto sub_y = Kokkos::subview(m_y, k, Kokkos::ALL());

    info += KokkosBatched::SerialRot<typename ParamTagType::trans>::invoke(sub_x, sub_y, m_c, m_s);
  }

  inline int run() {
    using value_type = typename XViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialRot");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    int info_sum                      = 0;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space> policy(0, m_x.extent(0));
    Kokkos::parallel_reduce(name.c_str(), policy, *this, info_sum);
    Kokkos::Profiling::popRegion();
    return info_sum;
  }
};

/// \brief Implementation details of batched rot analytical test
///        to confirm x := c*x + s*y and y := c*y - s*x are computed correctly
///        c = 0.6, s = 0.8
///        x: [1, 2, 3, 4]
///        y: [5, 6, 7, 8]
///        x_ref: [4.6, 6.0, 7.4, 8.8]
///        y_ref: [2.2, 2.0, 1.8, 1.6]
///
/// \param Nb [in] Batch size of vectors
template <typename DeviceType, typename ScalarType, typename LayoutType, typename ParamTagType>
void impl_test_batched_rot_analytical(const std::size_t Nb) {
  using ats               = typename KokkosKernels::ArithTraits<ScalarType>;
  using RealType          = typename ats::mag_type;
  using View2DType        = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using StridedView2DType = Kokkos::View<ScalarType **, Kokkos::LayoutStride, DeviceType>;

  const std::size_t N = 4;
  View2DType x("x", Nb, N), y("y", Nb, N);
  View2DType x_ref("x_ref", Nb, N), y_ref("y_ref", Nb, N);

  const std::size_t incx = 2;
  // Testing incx argument with strided views
  Kokkos::LayoutStride layout{Nb, incx, N, Nb * incx};
  StridedView2DType x_s("x_s", layout), y_s("y_s", layout);

  auto h_x     = Kokkos::create_mirror_view(x);
  auto h_y     = Kokkos::create_mirror_view(y);
  auto h_x_ref = Kokkos::create_mirror_view(x_ref);
  auto h_y_ref = Kokkos::create_mirror_view(y_ref);

  for (std::size_t ib = 0; ib < Nb; ib++) {
    h_x(ib, 0) = ScalarType(1);
    h_x(ib, 1) = ScalarType(2);
    h_x(ib, 2) = ScalarType(3);
    h_x(ib, 3) = ScalarType(4);

    h_y(ib, 0) = ScalarType(5);
    h_y(ib, 1) = ScalarType(6);
    h_y(ib, 2) = ScalarType(7);
    h_y(ib, 3) = ScalarType(8);

    // x_ref(i) = c*x(i) + s*y(i) with c = 0.6, s = 0.8
    // y_ref(i) = c*y(i) - s*x(i)
    // Note: for real s, both Trans::Transpose and Trans::ConjTranspose
    //       give the same result since conj(s) = s.
    h_x_ref(ib, 0) = ScalarType(4.6);
    h_x_ref(ib, 1) = ScalarType(6.0);
    h_x_ref(ib, 2) = ScalarType(7.4);
    h_x_ref(ib, 3) = ScalarType(8.8);

    h_y_ref(ib, 0) = ScalarType(2.2);
    h_y_ref(ib, 1) = ScalarType(2.0);
    h_y_ref(ib, 2) = ScalarType(1.8);
    h_y_ref(ib, 3) = ScalarType(1.6);
  }

  Kokkos::deep_copy(x, h_x);
  Kokkos::deep_copy(y, h_y);

  // Deep copy to strided views
  Kokkos::deep_copy(x_s, x);
  Kokkos::deep_copy(y_s, y);

  const RealType c   = 0.6;
  const ScalarType s = ScalarType(0.8);

  auto info = Functor_BatchedSerialRot<DeviceType, View2DType, View2DType, RealType, ScalarType, ParamTagType>(x, y, c,
                                                                                                               s)
                  .run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // With strided views
  info = Functor_BatchedSerialRot<DeviceType, StridedView2DType, StridedView2DType, RealType, ScalarType,
                                  ParamTagType>(x_s, y_s, c, s)
             .run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  RealType eps = 1.0e1 * ats::epsilon();
  auto h_x_out   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
  auto h_y_out   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  auto h_x_s_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_s);
  auto h_y_s_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y_s);

  // Check if x := c*x + s*y and y := c*y - op(s)*x
  for (std::size_t ib = 0; ib < Nb; ib++) {
    for (std::size_t i = 0; i < N; i++) {
      EXPECT_NEAR_KK(h_x_out(ib, i), h_x_ref(ib, i), eps);
      EXPECT_NEAR_KK(h_y_out(ib, i), h_y_ref(ib, i), eps);
      EXPECT_NEAR_KK(h_x_s_out(ib, i), h_x_ref(ib, i), eps);
      EXPECT_NEAR_KK(h_y_s_out(ib, i), h_y_ref(ib, i), eps);
    }
  }
}

/// \brief Implementation details of batched rot test
///
/// \param Nb [in] Batch size of vectors
/// \param N [in] Length of vectors x and y
template <typename DeviceType, typename ScalarType, typename LayoutType, typename ParamTagType>
void impl_test_batched_rot(const std::size_t Nb, const std::size_t N) {
  using ats               = typename KokkosKernels::ArithTraits<ScalarType>;
  using RealType          = typename ats::mag_type;
  using View2DType        = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using StridedView2DType = Kokkos::View<ScalarType **, Kokkos::LayoutStride, DeviceType>;

  View2DType x("x", Nb, N), y("y", Nb, N);
  View2DType x_ref("x_ref", Nb, N), y_ref("y_ref", Nb, N);

  const std::size_t incx = 2;
  // Testing incx argument with strided views
  Kokkos::LayoutStride layout{Nb, incx, N, Nb * incx};
  StridedView2DType x_s("x_s", layout), y_s("y_s", layout);

  // Create random x and y
  using execution_space = typename DeviceType::execution_space;
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);
  ScalarType randStart, randEnd;

  KokkosKernels::Impl::getRandomBounds(1.0, randStart, randEnd);
  Kokkos::fill_random(x, rand_pool, randStart, randEnd);
  Kokkos::fill_random(y, rand_pool, randStart, randEnd);

  // Save copies for reference
  Kokkos::deep_copy(x_ref, x);
  Kokkos::deep_copy(y_ref, y);

  // Deep copy to strided views
  Kokkos::deep_copy(x_s, x);
  Kokkos::deep_copy(y_s, y);

  const RealType c_val = 0.6;
  ScalarType s_val;
  if constexpr (ats::is_complex) {
    s_val = ScalarType(0.6, 0.8);
  } else {
    s_val = ScalarType(0.8);
  }

  // Run rot on (x, y)
  auto info =
      Functor_BatchedSerialRot<DeviceType, View2DType, View2DType, RealType, ScalarType, ParamTagType>(x, y, c_val,
                                                                                                       s_val)
          .run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // With strided views
  auto info_s = Functor_BatchedSerialRot<DeviceType, StridedView2DType, StridedView2DType, RealType, ScalarType,
                                         ParamTagType>(x_s, y_s, c_val, s_val)
                    .run();

  Kokkos::fence();
  EXPECT_EQ(info_s, 0);

  // Make a reference at host
  auto h_x_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_ref);
  auto h_y_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y_ref);

  // Note: ConjTranspose corresponds to zrot where conj(s) is used
  const bool is_conj = std::is_same_v<typename ParamTagType::trans, Trans::ConjTranspose>;
  for (std::size_t ib = 0; ib < Nb; ib++) {
    for (std::size_t i = 0; i < N; i++) {
      auto s_applied = is_conj ? KokkosKernels::ArithTraits<ScalarType>::conj(s_val) : s_val;
      auto temp      = c_val * h_x_ref(ib, i) + s_val * h_y_ref(ib, i);
      h_y_ref(ib, i) = c_val * h_y_ref(ib, i) - s_applied * h_x_ref(ib, i);
      h_x_ref(ib, i) = temp;
    }
  }

  RealType eps = 1.0e1 * ats::epsilon();

  auto h_x   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
  auto h_y   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  auto h_x_s = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_s);
  auto h_y_s = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y_s);

  // Check if x := c*x + s*y and y := c*y - op(s)*x
  for (std::size_t ib = 0; ib < Nb; ib++) {
    for (std::size_t i = 0; i < N; i++) {
      EXPECT_NEAR_KK(h_x(ib, i), h_x_ref(ib, i), eps);
      EXPECT_NEAR_KK(h_y(ib, i), h_y_ref(ib, i), eps);
      EXPECT_NEAR_KK(h_x_s(ib, i), h_x_ref(ib, i), eps);
      EXPECT_NEAR_KK(h_y_s(ib, i), h_y_ref(ib, i), eps);
    }
  }
}

}  // namespace Rot
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename ParamTagType>
int test_batched_rot() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    Test::Rot::impl_test_batched_rot_analytical<DeviceType, ScalarType, LayoutType, ParamTagType>(1);
    Test::Rot::impl_test_batched_rot_analytical<DeviceType, ScalarType, LayoutType, ParamTagType>(2);
    for (int i = 0; i < 10; i++) {
      Test::Rot::impl_test_batched_rot<DeviceType, ScalarType, LayoutType, ParamTagType>(1, i);
      Test::Rot::impl_test_batched_rot<DeviceType, ScalarType, LayoutType, ParamTagType>(2, i);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    Test::Rot::impl_test_batched_rot_analytical<DeviceType, ScalarType, LayoutType, ParamTagType>(1);
    Test::Rot::impl_test_batched_rot_analytical<DeviceType, ScalarType, LayoutType, ParamTagType>(2);
    for (int i = 0; i < 10; i++) {
      Test::Rot::impl_test_batched_rot<DeviceType, ScalarType, LayoutType, ParamTagType>(1, i);
      Test::Rot::impl_test_batched_rot<DeviceType, ScalarType, LayoutType, ParamTagType>(2, i);
    }
  }
#endif

  return 0;
}

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F(TestCategory, test_batched_rot_t_float) {
  using param_tag_type = ::Test::Rot::ParamTag<Trans::Transpose>;
  test_batched_rot<TestDevice, float, param_tag_type>();
}
TEST_F(TestCategory, test_batched_rot_c_float) {
  using param_tag_type = ::Test::Rot::ParamTag<Trans::ConjTranspose>;
  test_batched_rot<TestDevice, float, param_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, test_batched_rot_t_double) {
  using param_tag_type = ::Test::Rot::ParamTag<Trans::Transpose>;
  test_batched_rot<TestDevice, double, param_tag_type>();
}
TEST_F(TestCategory, test_batched_rot_c_double) {
  using param_tag_type = ::Test::Rot::ParamTag<Trans::ConjTranspose>;
  test_batched_rot<TestDevice, double, param_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
TEST_F(TestCategory, test_batched_rot_t_fcomplex) {
  using param_tag_type = ::Test::Rot::ParamTag<Trans::Transpose>;
  test_batched_rot<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_rot_c_fcomplex) {
  using param_tag_type = ::Test::Rot::ParamTag<Trans::ConjTranspose>;
  test_batched_rot<TestDevice, Kokkos::complex<float>, param_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
TEST_F(TestCategory, test_batched_rot_t_dcomplex) {
  using param_tag_type = ::Test::Rot::ParamTag<Trans::Transpose>;
  test_batched_rot<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
TEST_F(TestCategory, test_batched_rot_c_dcomplex) {
  using param_tag_type = ::Test::Rot::ParamTag<Trans::ConjTranspose>;
  test_batched_rot<TestDevice, Kokkos::complex<double>, param_tag_type>();
}
#endif
