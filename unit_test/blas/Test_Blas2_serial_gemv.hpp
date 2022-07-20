#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas2_serial_gemv.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas_util.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {

template <class AType, class XType, class YType, class ScalarType>
struct RefGEMVOp {
  RefGEMVOp(char trans_, ScalarType alpha_, AType A_, XType x_,
            ScalarType beta_, YType y_)
      : trans(trans_), alpha(alpha_), beta(beta_), A(A_), x(x_), y(y_) {}

  template <typename TeamMember>
  KOKKOS_INLINE_FUNCTION void operator()(
      const TeamMember & /* member */) const {
    vanillaGEMV(trans, alpha, A, x, beta, y);
  }

 private:
  // parameters
  char trans;
  ScalarType alpha;
  ScalarType beta;
  // data
  AType A;
  XType x;
  YType y;
};

template <class AType, class XType, class YType, class ScalarType,
          class AlgoTag>
struct SerialGEMVOp {
  SerialGEMVOp(char trans_, ScalarType alpha_, AType A_, XType x_,
               ScalarType beta_, YType y_)
      : trans(trans_), alpha(alpha_), beta(beta_), A(A_), x(x_), y(y_) {}

  template <typename TeamMember>
  KOKKOS_INLINE_FUNCTION void operator()(
      const TeamMember & /* member */) const {
    KokkosBlas::Experimental::gemv<AlgoTag>(trans, alpha, A, x, beta, y);
  }

 private:
  // parameters
  char trans;
  ScalarType alpha;
  ScalarType beta;
  // data
  AType A;
  XType x;
  YType y;
};

template <class ScalarA, class ScalarX, class ScalarY, class Device,
          class ScalarCoef = void>
class SerialGEMVTest {
 private:
  using char_type = decltype('x');

  // ScalarCoef==void default behavior is to derive alpha/beta scalar types
  // from A and X scalar types
  using ScalarType = typename std::conditional<
      !std::is_void<ScalarCoef>::value, ScalarCoef,
      typename std::common_type<ScalarA, ScalarX>::type>::type;

 public:
  static void run(const char *mode) {
    run_layouts<KokkosBlas::Algo::Gemv::Unblocked>(mode);
    run_layouts<KokkosBlas::Algo::Gemv::Blocked>(mode);
#ifdef __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__
    // TODO: run_layouts<KokkosBlas::Algo::Gemv::CompactMKL>(mode);
#endif
  }

 private:
  template <class AlgoTag>
  static void run_layouts(const char *mode) {
    // Note: all layouts listed here are subview'ed to test Kokkos::LayoutStride
#ifdef KOKKOSKERNELS_TEST_LAYOUTLEFT
    run_view_types<AlgoTag, Kokkos::LayoutLeft>(mode);
#endif
#ifdef KOKKOSKERNELS_TEST_LAYOUTRIGHT
    run_view_types<AlgoTag, Kokkos::LayoutRight>(mode);
#endif
#if defined(KOKKOSKERNELS_TEST_LAYOUTLEFT) && \
    defined(KOKKOSKERNELS_TEST_LAYOUTRIGHT)
    using A_t = typename Kokkos::View<ScalarA **, Kokkos::LayoutRight, Device>;
    using x_t = typename Kokkos::View<ScalarX *, Kokkos::LayoutLeft, Device>;
    using y_t = typename Kokkos::View<ScalarY *, Kokkos::LayoutRight, Device>;
    run_sizes<AlgoTag, A_t, x_t, y_t>(mode);
#endif
  }

  template <class AlgoTag, class Layout>
  static void run_view_types(const char *mode) {
    typedef Kokkos::View<ScalarA **, Layout, Device> view_type_A;
    typedef Kokkos::View<ScalarX *, Layout, Device> view_type_x;
    typedef Kokkos::View<ScalarY *, Layout, Device> view_type_y;
    run_sizes<AlgoTag, view_type_A, view_type_x, view_type_y>(mode);
  }

  template <class AlgoTag, class ViewAType, class ViewXType, class ViewYType>
  static void run_sizes(const char *mode) {
    // zero cases
    run_size<AlgoTag, ViewAType, ViewXType, ViewYType>(mode, 0, 0);
    run_size<AlgoTag, ViewAType, ViewXType, ViewYType>(mode, 0, 4);
    run_size<AlgoTag, ViewAType, ViewXType, ViewYType>(mode, 4, 0);
    // small block sizes
    for (int n = 1; n <= 16; ++n) {
      run_size<AlgoTag, ViewAType, ViewXType, ViewYType>(mode, n, n);
    }
    // other cases
    run_size<AlgoTag, ViewAType, ViewXType, ViewYType>(mode, 1024, 1);
    run_size<AlgoTag, ViewAType, ViewXType, ViewYType>(mode, 1024, 13);
    run_size<AlgoTag, ViewAType, ViewXType, ViewYType>(mode, 1024, 124);
  }

  template <class AlgoTag, class ViewTypeA, class ViewTypeX, class ViewTypeY>
  static void run_size(const char *mode, int N, int M) {
    using A_layout = typename ViewTypeA::array_layout;
    using x_layout = typename ViewTypeX::array_layout;
    using y_layout = typename ViewTypeY::array_layout;
    static_assert(!std::is_same<A_layout, Kokkos::LayoutStride>::value, "");
    static_assert(!std::is_same<x_layout, Kokkos::LayoutStride>::value, "");
    static_assert(!std::is_same<y_layout, Kokkos::LayoutStride>::value, "");

    const char_type trans = mode[0];
    const auto Nt         = trans == (char)'N' ? N : M;
    const auto Mt         = trans == (char)'N' ? M : N;

    // 1. run on regular (non-strided) views
    ViewTypeA A1("A1", Nt, Mt);
    ViewTypeX x1("X1", M);
    ViewTypeY y1("Y1", N);
    run_views<AlgoTag>(trans, A1, x1, y1);

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
    run_views<AlgoTag>(trans, A, x, y);
  }

  template <class AlgoTag, class ViewTypeA, class ViewTypeX, class ViewTypeY>
  static void run_views(const char trans, ViewTypeA A, ViewTypeX x,
                        ViewTypeY y) {
    Kokkos::TeamPolicy<Device> teams(1, 1);  // just run on device
    fill_inputs(A, x, y);
    ScalarType alpha = 3;  // TODO: test also with zero alpha/beta ?
    ScalarType beta  = 5;

    // get reference results
    Kokkos::View<ScalarY *, Device> y_ref("Y_ref", y.extent(0));
    Kokkos::deep_copy(y_ref, y);
    RefGEMVOp<ViewTypeA, ViewTypeX, ViewTypeY, ScalarType> gemv_ref(
        trans, alpha, A, x, beta, y_ref);
    Kokkos::parallel_for(teams, gemv_ref);

    // 1. check non-consts
    run_case<AlgoTag>(trans, alpha, A, x, beta, y, y_ref);

    // 2. check const x
    typename ViewTypeX::const_type c_x = x;
    run_case<AlgoTag>(trans, alpha, A, c_x, beta, y, y_ref);

    // 3. check const A and x
    typename ViewTypeA::const_type c_A = A;
    run_case<AlgoTag>(trans, alpha, c_A, c_x, beta, y, y_ref);
  }

  template <class AlgoTag, class ViewTypeA, class ViewTypeX, class ViewTypeY,
            class ViewTypeYRef, class ScalarType>
  static void run_case(const char trans, ScalarType alpha, ViewTypeA A,
                       ViewTypeX x, ScalarType beta, ViewTypeY y,
                       ViewTypeYRef y_ref) {
    // run on original y view (not to alter the test)
    // but backup it and restore, so it can be reused
    Kokkos::View<ScalarY *, Device> y_backup("Y2", y.extent(0));
    Kokkos::deep_copy(y_backup, y);

    SerialGEMVOp<ViewTypeA, ViewTypeX, ViewTypeY, ScalarType, AlgoTag> gemv_op(
        trans, alpha, A, x, beta, y);
    Kokkos::parallel_for(Kokkos::TeamPolicy<Device>(1, 1), gemv_op);

    const double eps = epsilon<ScalarY>();
    EXPECT_NEAR_KK_REL_1DVIEW(y, y_ref, eps);
    Kokkos::deep_copy(y, y_backup);
  }

  //----- utilities -----//

  template <class Scalar>
  static double epsilon() {
    return (std::is_same<Scalar, float>::value ||
            std::is_same<Scalar, Kokkos::complex<float>>::value)
               ? 2 * 1e-5
               : 1e-7;
  }

  template <class ViewTypeA, class ViewTypeX, class ViewTypeY>
  static void fill_inputs(ViewTypeA A, ViewTypeX x, ViewTypeY y) {
    using A_scalar_type = typename ViewTypeA::non_const_value_type;
    using x_scalar_type = typename ViewTypeX::non_const_value_type;
    using y_scalar_type = typename ViewTypeY::non_const_value_type;
    using exec_space    = typename Device::execution_space;
    Kokkos::Random_XorShift64_Pool<exec_space> rand_pool(13718);
    Kokkos::fill_random(A, rand_pool, A_scalar_type(10));
    Kokkos::fill_random(x, rand_pool, x_scalar_type(10));
    Kokkos::fill_random(y, rand_pool, y_scalar_type(10));
    Kokkos::fence();
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