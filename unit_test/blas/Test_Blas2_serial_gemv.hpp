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

// Simplify ETI ifdefs
// TODO: Maybe define this for all tests globally ?
//     + maybe this logic should be in CMake for concisiety
#if !defined(KOKKOSKERNELS_ETI_ONLY) && \
    !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)

#define KOKKOSKERNELS_TEST_MIXED_TYPES
#define KOKKOSKERNELS_TEST_LAYOUTLEFT
#define KOKKOSKERNELS_TEST_LAYOUTRIGHT
#define KOKKOSKERNELS_TEST_LAYOUTSTRIDE
#define KOKKOSKERNELS_TEST_FLOAT
#define KOKKOSKERNELS_TEST_DOUBLE
#define KOKKOSKERNELS_TEST_INT
#define KOKKOSKERNELS_TEST_COMPLEX_FLOAT
#define KOKKOSKERNELS_TEST_COMPLEX_DOUBLE

#else

#ifdef KOKKOSKERNELS_INST_LAYOUTLEFT
#define KOKKOSKERNELS_TEST_LAYOUTLEFT
#endif
#ifdef KOKKOSKERNELS_INST_LAYOUTRIGHT
#define KOKKOSKERNELS_TEST_LAYOUTRIGHT
#endif
#ifdef KOKKOSKERNELS_INST_LAYOUTSTRIDE
#define KOKKOSKERNELS_TEST_LAYOUTSTRIDE
#endif
#ifdef KOKKOSKERNELS_INST_FLOAT
#define KOKKOSKERNELS_TEST_FLOAT
#endif
#ifdef KOKKOSKERNELS_INST_DOUBLE
#define KOKKOSKERNELS_TEST_DOUBLE
#endif
#ifdef KOKKOSKERNELS_INST_INT
#define KOKKOSKERNELS_TEST_INT
#endif
#ifdef KOKKOSKERNELS_INST_COMPLEX_FLOAT
#define KOKKOSKERNELS_TEST_COMPLEX_FLOAT
#endif
#ifdef KOKKOSKERNELS_INST_COMPLEX_DOUBLE
#define KOKKOSKERNELS_TEST_COMPLEX_DOUBLE
#endif

#endif

namespace Test {
template <class ViewTypeA, class ViewTypeX, class ViewTypeY, class Device>
void impl_test_serial_gemv(const char *mode, int N, int M) {
  typedef typename ViewTypeA::value_type ScalarA;
  typedef typename ViewTypeX::value_type ScalarX;
  typedef typename ViewTypeY::value_type ScalarY;

  typedef multivector_layout_adapter<ViewTypeA> vfA_type;
  typedef Kokkos::View<
      ScalarX * [2],
      typename std::conditional<std::is_same<typename ViewTypeX::array_layout,
                                             Kokkos::LayoutStride>::value,
                                Kokkos::LayoutRight, Kokkos::LayoutLeft>::type,
      Device>
      BaseTypeX;
  typedef Kokkos::View<
      ScalarY * [2],
      typename std::conditional<std::is_same<typename ViewTypeY::array_layout,
                                             Kokkos::LayoutStride>::value,
                                Kokkos::LayoutRight, Kokkos::LayoutLeft>::type,
      Device>
      BaseTypeY;

  ScalarA a  = 3;
  ScalarX b  = 5;
  double eps = (std::is_same<ScalarY, float>::value ||
                std::is_same<ScalarY, Kokkos::complex<float>>::value)
                   ? 2 * 1e-5
                   : 1e-7;

  const auto Nt = mode[0] == 'N' ? N : M;
  const auto Mt = mode[0] == 'N' ? M : N;
  typename vfA_type::BaseType b_A("A", Nt, Mt);
  BaseTypeX b_x("X", M);
  BaseTypeY b_y("Y", N);
  BaseTypeY b_org_y("Org_Y", N);

  ViewTypeA A                        = vfA_type::view(b_A);
  ViewTypeX x                        = Kokkos::subview(b_x, Kokkos::ALL(), 0);
  ViewTypeY y                        = Kokkos::subview(b_y, Kokkos::ALL(), 0);
  typename ViewTypeX::const_type c_x = x;
  typename ViewTypeA::const_type c_A = A;

  typedef multivector_layout_adapter<typename ViewTypeA::HostMirror> h_vfA_type;

  typename h_vfA_type::BaseType h_b_A  = Kokkos::create_mirror_view(b_A);
  typename BaseTypeX::HostMirror h_b_x = Kokkos::create_mirror_view(b_x);
  typename BaseTypeY::HostMirror h_b_y = Kokkos::create_mirror_view(b_y);

  typename ViewTypeA::HostMirror h_A = h_vfA_type::view(h_b_A);
  typename ViewTypeX::HostMirror h_x = Kokkos::subview(h_b_x, Kokkos::ALL(), 0);
  typename ViewTypeY::HostMirror h_y = Kokkos::subview(h_b_y, Kokkos::ALL(), 0);

  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
      13718);

  Kokkos::fill_random(b_x, rand_pool, ScalarX(10));
  Kokkos::fill_random(b_y, rand_pool, ScalarY(10));
  Kokkos::fill_random(b_A, rand_pool, ScalarA(10));

  Kokkos::deep_copy(b_org_y, b_y);

  Kokkos::deep_copy(h_b_x, b_x);
  Kokkos::deep_copy(h_b_y, b_y);
  Kokkos::deep_copy(h_b_A, b_A);

  ScalarY expected_result = 0;
  if (mode[0] != 'N' && mode[0] != 'T' && mode[0] != 'C') {
    throw std::runtime_error("incorrect matrix mode letter !");
  }
  typedef Kokkos::Details::ArithTraits<ScalarA> ATV;
  for (int i = 0; i < N; i++) {
    ScalarY y_i = ScalarY();
    for (int j = 0; j < M; j++) {
      const auto a_val = mode[0] == 'C'
                             ? ATV::conj(h_A(j, i))
                             : (mode[0] == 'T' ? h_A(j, i) : h_A(i, j));
      y_i += a_val * h_x(j);
    }
    expected_result += (b * h_y(i) + a * y_i) * (b * h_y(i) + a * y_i);
  }

  char trans = mode[0];

  KokkosBlas::Experimental::gemv(trans, a, A, x, b, y);

  ScalarY nonconst_nonconst_result = KokkosBlas::dot(y, y);
  EXPECT_NEAR_KK(nonconst_nonconst_result, expected_result,
                 eps * expected_result);

  Kokkos::deep_copy(b_y, b_org_y);

  KokkosBlas::Experimental::gemv(trans, a, A, c_x, b, y);

  ScalarY const_nonconst_result = KokkosBlas::dot(y, y);
  EXPECT_NEAR_KK(const_nonconst_result, expected_result, eps * expected_result);

  Kokkos::deep_copy(b_y, b_org_y);

  KokkosBlas::Experimental::gemv(trans, a, c_A, c_x, b, y);

  ScalarY const_const_result = KokkosBlas::dot(y, y);
  EXPECT_NEAR_KK(const_const_result, expected_result, eps * expected_result);
}

template <typename DeviceType, typename ViewAType, typename ViewBType,
          typename ViewCType, typename ScalarType, typename TransType,
          typename AlgoTagType>
struct Functor_TestSerialGemv {
  ViewAType _a;
  ViewBType _b;
  ViewCType _c;

  ScalarType _alpha, _beta;

  KOKKOS_INLINE_FUNCTION
  Functor_TestSerialGemv(const ScalarType alpha, const ViewAType &a,
                         const ViewBType &b, const ScalarType beta,
                         const ViewCType &c)
      : _a(a), _b(b), _c(c), _alpha(alpha), _beta(beta) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const TransType &, const int k) const {
    auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto bb = Kokkos::subview(_b, k, Kokkos::ALL(), 0);
    auto cc = Kokkos::subview(_c, k, Kokkos::ALL(), 0);

    KokkosBlas::SerialGemv<TransType, AlgoTagType>::invoke(_alpha, aa, bb,
                                                           _beta, cc);
  }

  inline void run() {
    typedef typename ViewAType::value_type value_type;
    std::string name_region("KokkosBlas::Test::SerialGemv");
    const std::string name_value_type =
        (std::is_same<typename ViewBType::value_type, value_type>::value &&
         std::is_same<typename ViewCType::value_type, value_type>::value)
            ? Test::value_type_name<value_type>()
            : "::Mixed(" + Test::value_type_name<value_type>() +
                  Test::value_type_name<typename ViewBType::value_type>() +
                  Test::value_type_name<typename ViewCType::value_type>() + ")";
    std::string name = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<DeviceType, TransType> policy(0, _c.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
  }
};

template <typename DeviceType, typename LayoutType, typename ValueAType,
          typename ValueBType, typename ValueCType, typename ScalarType,
          typename TransType, typename AlgoTagType>
void impl_test_batched_serial_gemv(const int N, const int BlkSize) {
  typedef Kokkos::View<ValueAType ***, LayoutType, DeviceType> ViewAType;
  typedef Kokkos::View<ValueBType ***, LayoutType, DeviceType> ViewBType;
  typedef Kokkos::View<ValueCType ***, LayoutType, DeviceType> ViewCType;
  typedef Kokkos::Details::ArithTraits<ValueCType> c_ats;

  /// randomized input testing views
  ScalarType alpha = 1.5, beta = 3.0;

  ViewAType a0("a0", N, BlkSize, BlkSize), a1("a1", N, BlkSize, BlkSize);
  ViewBType b0("b0", N, BlkSize, 1), b1("b1", N, BlkSize, 1);
  ViewCType c0("c0", N, BlkSize, 1), c1("c1", N, BlkSize, 1);

  Kokkos::Random_XorShift64_Pool<typename DeviceType::execution_space> random(
      13718);
  Kokkos::fill_random(a0, random, ValueAType(1.0));
  Kokkos::fill_random(b0, random, ValueBType(1.0));
  Kokkos::fill_random(c0, random, ValueCType(1.0));

  Kokkos::fence();

  Kokkos::deep_copy(a1, a0);
  Kokkos::deep_copy(b1, b0);
  Kokkos::deep_copy(c1, c0);

  /// test body
  Functor_TestSerialGemv<DeviceType, decltype(a0), decltype(b0), decltype(c0),
                         ScalarType, TransType,
                         KokkosBlas::Algo::Gemv::Unblocked>(alpha, a0, b0, beta,
                                                            c0)
      .run();
  Functor_TestSerialGemv<DeviceType, decltype(a0), decltype(b0), decltype(c0),
                         ScalarType, TransType, AlgoTagType>(alpha, a1, b1,
                                                             beta, c1)
      .run();

  Kokkos::fence();

  /// for comparison send it to host
  typename ViewCType::HostMirror c0_host = Kokkos::create_mirror_view(c0);
  typename ViewCType::HostMirror c1_host = Kokkos::create_mirror_view(c1);

  Kokkos::deep_copy(c0_host, c0);
  Kokkos::deep_copy(c1_host, c1);

  /// check c0 = c1 ; this eps is about 10^-14
  typedef typename c_ats::mag_type mag_type;
  mag_type sum(1), diff(0);
  const mag_type eps = 1.0e3 * c_ats::epsilon();

  for (int k = 0; k < N; ++k)
    for (int i = 0; i < BlkSize; ++i)
      for (int j = 0; j < 1; ++j) {
        sum += c_ats::abs(c0_host(k, i, j));
        diff += c_ats::abs(c0_host(k, i, j) - c1_host(k, i, j));
      }
  EXPECT_NEAR_KK(diff / sum, 0, eps);
}

}  // namespace Test

template <class ScalarA, class ScalarX, class ScalarY, class Device>
int test_serial_gemv(const char *mode) {
#ifdef KOKKOSKERNELS_TEST_LAYOUTLEFT
  typedef Kokkos::View<ScalarA **, Kokkos::LayoutLeft, Device> view_type_a_ll;
  typedef Kokkos::View<ScalarX *, Kokkos::LayoutLeft, Device> view_type_b_ll;
  typedef Kokkos::View<ScalarY *, Kokkos::LayoutLeft, Device> view_type_c_ll;
  Test::impl_test_serial_gemv<view_type_a_ll, view_type_b_ll, view_type_c_ll,
                              Device>(mode, 0, 1024);
  Test::impl_test_serial_gemv<view_type_a_ll, view_type_b_ll, view_type_c_ll,
                              Device>(mode, 13, 1024);
  Test::impl_test_serial_gemv<view_type_a_ll, view_type_b_ll, view_type_c_ll,
                              Device>(mode, 124, 124);
#endif

#ifdef KOKKOSKERNELS_TEST_LAYOUTRIGHT
  typedef Kokkos::View<ScalarA **, Kokkos::LayoutRight, Device> view_type_a_lr;
  typedef Kokkos::View<ScalarX *, Kokkos::LayoutRight, Device> view_type_b_lr;
  typedef Kokkos::View<ScalarY *, Kokkos::LayoutRight, Device> view_type_c_lr;
  Test::impl_test_serial_gemv<view_type_a_lr, view_type_b_lr, view_type_c_lr,
                              Device>(mode, 0, 1024);
  Test::impl_test_serial_gemv<view_type_a_lr, view_type_b_lr, view_type_c_lr,
                              Device>(mode, 13, 1024);
  Test::impl_test_serial_gemv<view_type_a_lr, view_type_b_lr, view_type_c_lr,
                              Device>(mode, 124, 124);
#endif

#ifdef KOKKOSKERNELS_TEST_LAYOUTSTRIDE
  typedef Kokkos::View<ScalarA **, Kokkos::LayoutStride, Device> view_type_a_ls;
  typedef Kokkos::View<ScalarX *, Kokkos::LayoutStride, Device> view_type_b_ls;
  typedef Kokkos::View<ScalarY *, Kokkos::LayoutStride, Device> view_type_c_ls;
  Test::impl_test_serial_gemv<view_type_a_ls, view_type_b_ls, view_type_c_ls,
                              Device>(mode, 0, 1024);
  Test::impl_test_serial_gemv<view_type_a_ls, view_type_b_ls, view_type_c_ls,
                              Device>(mode, 13, 1024);
  Test::impl_test_serial_gemv<view_type_a_ls, view_type_b_ls, view_type_c_ls,
                              Device>(mode, 124, 124);
#endif

#ifdef KOKKOSKERNELS_TEST_MIXED_TYPES
  Test::impl_test_serial_gemv<view_type_a_ls, view_type_b_ll, view_type_c_lr,
                              Device>(mode, 124, 124);
  Test::impl_test_serial_gemv<view_type_a_ll, view_type_b_ls, view_type_c_lr,
                              Device>(mode, 124, 124);
#endif

  return 0;
}

template <typename DeviceType, typename ValueAType, typename ValueBType,
          typename ValueCType, typename ScalarType, typename TransType,
          typename AlgoType = KokkosBlas::Algo::Gemv::Blocked>
int test_batched_serial_gemv() {
#ifdef KOKKOSKERNELS_TEST_LAYOUTLEFT
  {
    Test::impl_test_batched_serial_gemv<DeviceType, Kokkos::LayoutLeft,
                                        ValueAType, ValueBType, ValueCType,
                                        ScalarType, TransType, AlgoType>(0, 10);
    for (int i = 0; i < 10; ++i) {
      // printf("Testing: LayoutLeft,  Blksize %d\n", i);
      Test::impl_test_batched_serial_gemv<DeviceType, Kokkos::LayoutLeft,
                                          ValueAType, ValueBType, ValueCType,
                                          ScalarType, TransType, AlgoType>(1024,
                                                                           i);
    }
  }
#endif
#ifdef KOKKOSKERNELS_TEST_LAYOUTRIGHT
  {
    Test::impl_test_batched_serial_gemv<DeviceType, Kokkos::LayoutRight,
                                        ValueAType, ValueBType, ValueCType,
                                        ScalarType, TransType, AlgoType>(0, 10);
    for (int i = 0; i < 10; ++i) {
      // printf("Testing: LayoutRight, Blksize %d\n", i);
      Test::impl_test_batched_serial_gemv<DeviceType, Kokkos::LayoutRight,
                                          ValueAType, ValueBType, ValueCType,
                                          ScalarType, TransType, AlgoType>(1024,
                                                                           i);
    }
  }
#endif

  return 0;
}

#define TEST_CASE4(NAME, SCALAR_A, SCALAR_X, SCALAR_Y, SCALAR_COEF)          \
  TEST_F(TestCategory, serial_gemv_nt_##NAME) {                              \
    /* FIXME: implement arbitrary SCALAR_COEF for alpha/beta type */         \
    test_serial_gemv<SCALAR_A, SCALAR_X, SCALAR_Y, TestExecSpace>("N");      \
  }                                                                          \
  TEST_F(TestCategory, serial_gemv_t_##NAME) {                               \
    /* FIXME: implement arbitrary SCALAR_COEF for alpha/beta type */         \
    test_serial_gemv<SCALAR_A, SCALAR_X, SCALAR_Y, TestExecSpace>("T");      \
  }                                                                          \
  TEST_F(TestCategory, serial_gemv_nt2_##NAME) {                             \
    test_batched_serial_gemv<TestExecSpace, SCALAR_A, SCALAR_X, SCALAR_Y,    \
                             SCALAR_COEF, KokkosBlas::Trans::NoTranspose>(); \
  }                                                                          \
  TEST_F(TestCategory, serial_gemv_t2_##NAME) {                              \
    test_batched_serial_gemv<TestExecSpace, SCALAR_A, SCALAR_X, SCALAR_Y,    \
                             SCALAR_COEF, KokkosBlas::Trans::Transpose>();   \
  }

#define _COMMON_TYPE3(T1, T2, T3) std::common_type<T1, T2, T3>::type

#define TEST_CASE3(NAME, SCALAR_A, SCALAR_X, SCALAR_Y) \
  TEST_CASE4(NAME, SCALAR_A, SCALAR_X, SCALAR_Y,       \
             _COMMON_TYPE3(SCALAR_A, SCALAR_X, SCALAR_Y))

#define TEST_CASE2(NAME, SCALAR, SCALAR_COEF) \
  TEST_CASE4(NAME, SCALAR, SCALAR, SCALAR, SCALAR_COEF)
#define TEST_CASE(NAME, SCALAR)  TEST_CASE2(NAME, SCALAR, SCALAR)

#ifdef KOKKOSKERNELS_TEST_FLOAT
TEST_CASE(nt_float, float)
#endif

#ifdef KOKKOSKERNELS_TEST_DOUBLE
TEST_CASE(nt_double, double)
#endif

#ifdef KOKKOSKERNELS_TEST_COMPLEX_DOUBLE
TEST_CASE(nt_complex_double, Kokkos::complex<double>)
#endif

#ifdef KOKKOSKERNELS_TEST_COMPLEX_FLOAT
TEST_CASE(nt_complex_float, Kokkos::complex<float>)
#endif

#ifdef KOKKOSKERNELS_TEST_INT
TEST_CASE(nt_complex_int, int)
#endif

#ifdef KOKKOSKERNELS_TEST_MIXED_TYPES
// test mixed scalar types
TEST_CASE3(nt_mixed, double, int, float)

// test arbitrary double alpha/beta with complex<double> values
TEST_CASE2(nt_alphabeta, Kokkos::complex<double>, double)
#endif

#endif  // Check for lambda availability on CUDA backend
