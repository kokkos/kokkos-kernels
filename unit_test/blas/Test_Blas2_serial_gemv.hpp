#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas2_serial_gemv.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas_util.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {

template <typename ValueType, int length = KokkosBatched::DefaultVectorLength<
                                  ValueType, TestExecSpace>::value>
using simd_vector =
    KokkosBatched::Vector<KokkosBatched::SIMD<ValueType>, length>;

// Note: vanillaGEMV is called on device here - alternatively one can move
//       _strided_ data using safe_device_to_host_deep_copy() etc.
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

// fill regular view with random values
template <class ViewType, class PoolType,
          class ScalarType = typename ViewType::non_const_value_type>
void fill_random_view(ViewType A, PoolType &rand_pool,
                      const ScalarType max_val = 10.0) {
  Kokkos::fill_random(A, rand_pool, max_val);
  Kokkos::fence();
}

// fill rank-1 view of SIMD vectors with random values
template <class ValueType, int VecLength, class Layout, class... Props,
          class PoolType>
void fill_random_view(
    Kokkos::View<
        KokkosBatched::Vector<KokkosBatched::SIMD<ValueType>, VecLength> *,
        Layout, Props...>
        x,
    PoolType &rand_pool, const ValueType max_val = 10.0) {
  // the view can be strided and have Vector<SIMD> values, so randoms
  // are generated in a plain, linear view first and then copied
  using device_type = typename decltype(x)::device_type;
  Kokkos::View<ValueType *, device_type> rnd("random_vals",
                                             x.extent(0) * VecLength);
  Kokkos::fill_random(rnd, rand_pool, static_cast<ValueType>(10));
  using size_type = decltype(x.extent(0));
  for (size_type i = 0; i < x.extent(0); ++i) {
    x(i).loadUnaligned(&rnd(i * VecLength));
  }
}

// fill rank-2 view of SIMD vectors with random values
template <class ValueType, int VecLength, class Layout, class... Props,
          class PoolType>
static void fill_random_view(
    Kokkos::View<
        KokkosBatched::Vector<KokkosBatched::SIMD<ValueType>, VecLength> **,
        Layout, Props...>
        A,
    PoolType &rand_pool, const ValueType max_val = 10.0) {
  // the view can be strided and have Vector<SIMD> values, so randoms
  // are generated in a plain, linear view first and then copied
  using device_type = typename decltype(A)::device_type;
  Kokkos::View<ValueType *, device_type> rnd(
      "random_vals", A.extent(0) * A.extent(1) * VecLength);
  Kokkos::fill_random(rnd, rand_pool, static_cast<ValueType>(10));
  using size_type = decltype(A.extent(0));
  size_type idx   = 0;
  for (size_type i = 0; i < A.extent(0); ++i) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      A(i, j).loadUnaligned(&rnd(idx));
      idx += VecLength;
    }
  }
}

//
template <class ScalarA, class ScalarX, class ScalarY, class Device,
          class ScalarCoef = void>
struct SerialGEMVTestBase {
  // ScalarCoef==void default behavior is to derive alpha/beta scalar types
  // from A and X scalar types
  using ScalarType = typename std::conditional<
      !std::is_void<ScalarCoef>::value, ScalarCoef,
      typename std::common_type<ScalarA, ScalarX>::type>::type;

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

    const auto trans      = mode[0];
    const bool transposed = trans == (char)'T' || trans == (char)'C';
    const auto Nt         = transposed ? M : N;
    const auto Mt         = transposed ? N : M;

    // 1. run on regular (non-strided) views
    ViewTypeA A1("A1", Nt, Mt);
    ViewTypeX x1("X1", M);
    ViewTypeY y1("Y1", N);
    run_views<AlgoTag>(trans, A1, x1, y1);

    // 2. run on strided subviews (enforced by adding extra rank on both sides)
    // Note: strided views are not supported by MKL routines
    if (!std::is_same<AlgoTag, KokkosBlas::Algo::Gemv::CompactMKL>::value) {
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
    RefGEMVOp<ViewTypeA, ViewTypeX, decltype(y_ref), ScalarType> gemv_ref(
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

    const double eps = epsilon(ScalarY{});
    EXPECT_NEAR_KK_REL_1DVIEW(y, y_ref, eps);
    Kokkos::deep_copy(y, y_backup);
  }

  //----- utilities -----//

  // GEMV tolerance for scalar types
  static double epsilon(float) { return 2 * 1e-5; }
  static double epsilon(double) { return 1e-7; }
  static double epsilon(int) { return 0; }
  // tolerance for derived types
  template <class ScalarType>
  static double epsilon(Kokkos::complex<ScalarType>) {
    return epsilon(ScalarType{});
  }
  template <class ScalarType, int VecLen>
  static double epsilon(simd_vector<ScalarType, VecLen>) {
    return epsilon(ScalarType{});
  }

  template <class ViewTypeA, class ViewTypeX, class ViewTypeY>
  static void fill_inputs(ViewTypeA A, ViewTypeX x, ViewTypeY y) {
    using exec_space = typename Device::execution_space;
    Kokkos::Random_XorShift64_Pool<exec_space> rand_pool(13718);
    fill_random_view(A, rand_pool);
    fill_random_view(x, rand_pool);
    fill_random_view(y, rand_pool);
  }
};

template <class ScalarA, class ScalarX, class ScalarY, class Device,
          class ScalarCoef = void>
struct SerialGEMVTest {
  static void run(const char *mode) {
    using base =
        SerialGEMVTestBase<ScalarA, ScalarX, ScalarY, Device, ScalarCoef>;
    base::template run_layouts<KokkosBlas::Algo::Gemv::Unblocked>(mode);
    base::template run_layouts<KokkosBlas::Algo::Gemv::Blocked>(mode);
  }
};

// Special handling of Vector<SIMD<T>> (instead of plain scalars)
// Note: MKL compact routines don't allow mixed scalar types
template <class ScalarType, int VecLen, class Device, class ScalarCoef>
struct SerialGEMVTest<simd_vector<ScalarType, VecLen>,
                      simd_vector<ScalarType, VecLen>,
                      simd_vector<ScalarType, VecLen>, Device, ScalarCoef> {
  static void run(const char *mode) {
    using vector_type = simd_vector<ScalarType, VecLen>;
    using base = SerialGEMVTestBase<vector_type, vector_type, vector_type,
                                    Device, ScalarCoef>;
    // run all usual, non-vector tests
    base::template run_layouts<KokkosBlas::Algo::Gemv::Unblocked>(mode);
    base::template run_layouts<KokkosBlas::Algo::Gemv::Blocked>(mode);
    // run vector tests
#ifdef __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__
    base::template run_layouts<KokkosBlas::Algo::Gemv::CompactMKL>(mode);
#endif
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
  }                                                                     \
  TEST_F(TestCategory, serial_gemv_ct_##NAME) {                         \
    ::Test::SerialGEMVTest<SCALAR_A, SCALAR_X, SCALAR_Y, TestExecSpace, \
                           SCALAR_COEF>::run("C");                      \
  }

#define TEST_CASE2(NAME, SCALAR, SCALAR_COEF) \
  TEST_CASE4(NAME, SCALAR, SCALAR, SCALAR, SCALAR_COEF)
#define TEST_CASE(NAME, SCALAR) TEST_CASE2(NAME, SCALAR, SCALAR)

#ifdef KOKKOSKERNELS_TEST_FLOAT
TEST_CASE(float, float)
// MKL vector types
#ifdef __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__
using simd_float_sse    = ::Test::simd_vector<float, 4>;
using simd_float_avx    = ::Test::simd_vector<float, 8>;
using simd_float_avx512 = ::Test::simd_vector<float, 16>;
TEST_CASE2(mkl_float_sse, simd_float_sse, float)
TEST_CASE2(mkl_float_avx, simd_float_avx, float)
TEST_CASE2(mkl_float_avx512, simd_float_avx512, float)
#endif
#endif

#ifdef KOKKOSKERNELS_TEST_DOUBLE
TEST_CASE(double, double)
// MKL vector types
#ifdef __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__
using simd_double_sse    = ::Test::simd_vector<double, 2>;
using simd_double_avx    = ::Test::simd_vector<double, 4>;
using simd_double_avx512 = ::Test::simd_vector<double, 8>;
TEST_CASE2(mkl_double_sse, simd_double_sse, double)
TEST_CASE2(mkl_double_avx, simd_double_avx, double)
TEST_CASE2(mkl_double_avx512, simd_double_avx512, double)
#endif
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
