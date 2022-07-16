// Note: Luc Berger-Vergiat 04/14/21
//       This tests uses KOKKOS_LAMBDA so we need
//       to make sure that these are enabled in
//       the CUDA backend before including this test.
#if !defined(TEST_CUDA_BLAS_CPP) || defined(KOKKOS_ENABLE_CUDA_LAMBDA)

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas2_serial_gemv.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas_util.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {

template <class ScalarA, class ScalarX, class ScalarY, class Device,
          class ScalarCoef = void>
class SerialGEMVTest {
 private:
  using random_pool_type =
      Kokkos::Random_XorShift64_Pool<typename Device::execution_space>;

  // ScalarCoef==void default behavior is to derive alpha/beta scalar types
  // from A and X scalar types
  using ScalarType = typename std::conditional<
      !std::is_void<ScalarCoef>::value, ScalarCoef,
      typename std::common_type<ScalarA, ScalarX>::type>::type;

 public:
  static void run(const char *mode) { run_layouts(mode); }

 private:
  static void run_layouts(const char *mode) {
    // Note: all layouts listed here are subview'ed to test Kokkos::LayoutStride
#ifdef KOKKOSKERNELS_TEST_LAYOUTLEFT
    run_view_types<Kokkos::LayoutLeft>(mode);
#endif
#ifdef KOKKOSKERNELS_TEST_LAYOUTRIGHT
    run_view_types<Kokkos::LayoutRight>(mode);
#endif
#if defined(KOKKOSKERNELS_TEST_LAYOUTLEFT) && \
    defined(KOKKOSKERNELS_TEST_LAYOUTRIGHT)
    using A_t = typename Kokkos::View<ScalarA **, Kokkos::LayoutRight, Device>;
    using x_t = typename Kokkos::View<ScalarX *, Kokkos::LayoutLeft, Device>;
    using y_t = typename Kokkos::View<ScalarY *, Kokkos::LayoutRight, Device>;
    run_sizes<A_t, x_t, y_t>(mode);
#endif
  }

  template <typename Layout>
  static void run_view_types(const char *mode) {
    typedef Kokkos::View<ScalarA **, Layout, Device> view_type_A;
    typedef Kokkos::View<ScalarX *, Layout, Device> view_type_x;
    typedef Kokkos::View<ScalarY *, Layout, Device> view_type_y;
    run_sizes<view_type_A, view_type_x, view_type_y>(mode);
  }

  template <class ViewAType, class ViewXType, class ViewYType>
  static void run_sizes(const char *mode) {
    // zero cases
    run_size<ViewAType, ViewXType, ViewYType>(mode, 0, 0);
    run_size<ViewAType, ViewXType, ViewYType>(mode, 0, 4);
    run_size<ViewAType, ViewXType, ViewYType>(mode, 4, 0);
    // small block sizes
    for (int n = 1; n <= 16; ++n) {
      run_size<ViewAType, ViewXType, ViewYType>(mode, n, n);
    }
    // other cases
    run_size<ViewAType, ViewXType, ViewYType>(mode, 1024, 1);
    run_size<ViewAType, ViewXType, ViewYType>(mode, 1024, 13);
    run_size<ViewAType, ViewXType, ViewYType>(mode, 1024, 124);
  }

  template <class ViewTypeA, class ViewTypeX, class ViewTypeY>
  static void run_size(const char *mode, int N, int M) {
    using A_layout = typename ViewTypeA::array_layout;
    using x_layout = typename ViewTypeX::array_layout;
    using y_layout = typename ViewTypeY::array_layout;
    static_assert(!std::is_same<A_layout, Kokkos::LayoutStride>::value, "");
    static_assert(!std::is_same<x_layout, Kokkos::LayoutStride>::value, "");
    static_assert(!std::is_same<y_layout, Kokkos::LayoutStride>::value, "");

    const char trans = mode[0];
    const auto Nt    = trans == 'N' ? N : M;
    const auto Mt    = trans == 'N' ? M : N;

    // 1. run on regular (non-strided) views
    ViewTypeA A1("A1", Nt, Mt);
    ViewTypeX x1("X1", M);
    ViewTypeY y1("Y1", N);
    run_views(trans, A1, x1, y1);

    // 2. run on strided subviews (enforced by adding extra rank on both sides)
    // TODO: use multivector_layout_adapter from Kokkos_TestUtils.hpp ?
    //       it does NOT [always] produce strided subviews yet: fix it ?
    typedef Kokkos::View<ScalarA ****, A_layout, Device> BaseTypeA;
    typedef Kokkos::View<ScalarX ***, x_layout, Device> BaseTypeX;
    typedef Kokkos::View<ScalarY ***, y_layout, Device> BaseTypeY;

    BaseTypeA b_A("A", 2, Nt, Mt, 2);
    BaseTypeX b_x("X", 2, M, 2);
    BaseTypeY b_y("Y", 2, N, 2);
    auto A = Kokkos::subview(b_A, 0, Kokkos::ALL(), Kokkos::ALL(), 0);
    auto x = Kokkos::subview(b_x, 0, Kokkos::ALL(), 0);
    auto y = Kokkos::subview(b_y, 0, Kokkos::ALL(), 0);

    // make sure it's actually LayoutStride there
    static_assert(std::is_same<typename decltype(A)::array_layout,
                               Kokkos::LayoutStride>::value,
                  "");
    static_assert(std::is_same<typename decltype(x)::array_layout,
                               Kokkos::LayoutStride>::value,
                  "");
    static_assert(std::is_same<typename decltype(y)::array_layout,
                               Kokkos::LayoutStride>::value,
                  "");
    run_views(trans, A, x, y);
  }

  template <class ViewTypeA, class ViewTypeX, class ViewTypeY>
  static void run_views(const char trans, ViewTypeA A, ViewTypeX x,
                        ViewTypeY y) {
    ScalarType a = 3;
    ScalarType b = 5;
    double eps   = (std::is_same<ScalarY, float>::value || std::is_same<ScalarY, Kokkos::complex<float>>::value)
                     ? 2 * 1e-5
                     : 1e-7;

    // fill in input views
    random_pool_type rand_pool(13718);
    Kokkos::fill_random(A, rand_pool, ScalarA(10));
    Kokkos::fill_random(x, rand_pool, ScalarX(10));
    Kokkos::fill_random(y, rand_pool, ScalarY(10));
    Kokkos::fence();

    // backup initial y values because it gets updated in calculation
    Kokkos::View<ScalarY *, Kokkos::LayoutLeft, Device> org_y("Org_Y",
                                                              y.extent(0));
    Kokkos::deep_copy(org_y, y);

    Kokkos::View<ScalarY *, Device> ref_y("Y_reference", y.extent(0));
    Kokkos::deep_copy(ref_y, y);
    get_expected_result(trans, a, A, x, b, ref_y);

    // 1. check non-consts
    Kokkos::deep_copy(y, org_y);
    KokkosBlas::Experimental::gemv(trans, a, A, x, b, y);
    EXPECT_NEAR_KK_REL_1DVIEW(y, ref_y, eps);

    // 2. check const x
    Kokkos::deep_copy(y, org_y);
    typename ViewTypeX::const_type c_x = x;
    KokkosBlas::Experimental::gemv(trans, a, A, c_x, b, y);
    EXPECT_NEAR_KK_REL_1DVIEW(y, ref_y, eps);

    // 3. check const A and x
    Kokkos::deep_copy(y, org_y);
    typename ViewTypeA::const_type c_A = A;
    KokkosBlas::Experimental::gemv(trans, a, c_A, c_x, b, y);
    EXPECT_NEAR_KK_REL_1DVIEW(y, ref_y, eps);
  }

  template <class ViewTypeA, class ViewTypeX, class ViewTypeY>
  static void get_expected_result(const char trans, ScalarType a,
                                     ViewTypeA A, ViewTypeX x, ScalarType b,
                                     ViewTypeY y) {
    auto h_A = Kokkos::create_mirror_view(A);
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_y = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(h_A, A);
    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_y, y);

    vanillaGEMV(trans, a, h_A, h_x, b, h_y);

    Kokkos::deep_copy(y, h_y);
  }
};

}  // namespace Test

#define TEST_CASE4(NAME, SCALAR_A, SCALAR_X, SCALAR_Y, SCALAR_COEF)     \
  TEST_F(TestCategory, serial_gemv_nt_##NAME) {                         \
    ::Test::SerialGEMVTest<SCALAR_A, SCALAR_X, SCALAR_Y, TestExecSpace, \
                           SCALAR_COEF>::run("N");                      \
  }                                                                     \
  TEST_F(TestCategory, serial_gemv_t_##NAME) {                          \
    ::Test::SerialGEMVTest<SCALAR_A, SCALAR_X, SCALAR_Y, TestExecSpace, \
                           SCALAR_COEF>::run("T");                      \
  }

#define TEST_CASE2(NAME, SCALAR, SCALAR_COEF) \
  TEST_CASE4(NAME, SCALAR, SCALAR, SCALAR, SCALAR_COEF)
#define TEST_CASE(NAME, SCALAR) TEST_CASE2(NAME, SCALAR, SCALAR)

#ifdef KOKKOSKERNELS_TEST_FLOAT
TEST_CASE(float, float)
#endif

#ifdef KOKKOSKERNELS_TEST_DOUBLE
TEST_CASE(double, double)
#endif

#ifdef KOKKOSKERNELS_TEST_COMPLEX_DOUBLE
TEST_CASE(complex_double, Kokkos::complex<double>)
#endif

#ifdef KOKKOSKERNELS_TEST_COMPLEX_FLOAT
TEST_CASE(complex_float, Kokkos::complex<float>)
#endif

#ifdef KOKKOSKERNELS_TEST_INT
TEST_CASE(int, int)
#endif

#ifdef KOKKOSKERNELS_TEST_ALL_TYPES
// test mixed scalar types (void -> default alpha/beta)
TEST_CASE4(mixed, double, int, float, void)

// test arbitrary double alpha/beta with complex<double> values
TEST_CASE2(alphabeta, Kokkos::complex<double>, double)
#endif

#endif  // Check for lambda availability on CUDA backend
