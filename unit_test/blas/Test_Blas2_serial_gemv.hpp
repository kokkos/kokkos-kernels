#include <KokkosKernels_TestUtils.hpp>  // for ETI test guards
// Note: include serial gemv before util so it knows if CompactMKL is available
#include <KokkosBlas2_serial_gemv.hpp>
#include <Test_Blas2_gemv_util.hpp>

namespace Test {

template <class AType, class XType, class YType, class ScalarType,
          class AlgoTag>
KK_DEFINE_BLAS2_GEMV_TEST_OP_CLASS(SerialGEMVOp)
template <typename TeamMember>
KOKKOS_INLINE_FUNCTION void operator()(const TeamMember& /* member */) const {
  KokkosBlas::Experimental::gemv<AlgoTag>(params::trans, params::alpha,
                                          params::A, params::x, params::beta,
                                          params::y);
}
KK_END_BLAS2_GEMV_TEST_OP_CLASS

struct SerialGemvFactory {
  template <class AlgoTag, class ViewTypeA, class ViewTypeX, class ViewTypeY,
            class Device, class ScalarType>
  using functor_type =
      SerialGEMVOp<ViewTypeA, ViewTypeX, ViewTypeY, ScalarType, AlgoTag>;

  using algorithms = std::tuple<KokkosBlas::Algo::Gemv::Unblocked,
                                KokkosBlas::Algo::Gemv::Blocked
#ifdef __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__
                                ,
                                KokkosBlas::Algo::Gemv::CompactMKL
#endif
                                >;

  // block testing of CompackMKL on non-vector scalars
  // (they are not supported by the implementation)
  template <class AlgoTag, class ScalarA, class ScalarX, class ScalarY,
            class Device, class ScalarCoef>
  static constexpr bool allow_algorithm =
      !std::is_same<AlgoTag, KokkosBlas::Algo::Gemv::CompactMKL>::value ||
      (KokkosBatched::is_vector<ScalarA>::value &&
       KokkosBatched::is_vector<ScalarX>::value &&
       KokkosBatched::is_vector<ScalarY>::value);
};

}  // namespace Test

#define TEST_SERIAL_CASE4(N, A, X, Y, SC) \
  TEST_CASE4(serial, SerialGemvFactory, N, A, X, Y, SC)
#define TEST_SERIAL_CASE2(N, S, SC) \
  TEST_CASE2(serial, SerialGemvFactory, N, S, SC)
#define TEST_SERIAL_CASE(N, S) TEST_CASE(serial, SerialGemvFactory, N, S)

#ifdef KOKKOSKERNELS_TEST_FLOAT
TEST_SERIAL_CASE(float, float)
// MKL vector types
#ifdef __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__
using simd_float_sse    = ::Test::simd_vector<float, 4>;
using simd_float_avx    = ::Test::simd_vector<float, 8>;
using simd_float_avx512 = ::Test::simd_vector<float, 16>;
TEST_SERIAL_CASE2(mkl_float_sse, simd_float_sse, float)
TEST_SERIAL_CASE2(mkl_float_avx, simd_float_avx, float)
TEST_SERIAL_CASE2(mkl_float_avx512, simd_float_avx512, float)
#endif
#endif

#ifdef KOKKOSKERNELS_TEST_DOUBLE
TEST_SERIAL_CASE(double, double)
// MKL vector types
#ifdef __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__
using simd_double_sse    = ::Test::simd_vector<double, 2>;
using simd_double_avx    = ::Test::simd_vector<double, 4>;
using simd_double_avx512 = ::Test::simd_vector<double, 8>;
TEST_SERIAL_CASE2(mkl_double_sse, simd_double_sse, double)
TEST_SERIAL_CASE2(mkl_double_avx, simd_double_avx, double)
TEST_SERIAL_CASE2(mkl_double_avx512, simd_double_avx512, double)
#endif
#endif

#ifdef KOKKOSKERNELS_TEST_COMPLEX_DOUBLE
TEST_SERIAL_CASE(complex_double, Kokkos::complex<double>)
#endif

#ifdef KOKKOSKERNELS_TEST_COMPLEX_FLOAT
TEST_SERIAL_CASE(complex_float, Kokkos::complex<float>)
#endif

#ifdef KOKKOSKERNELS_TEST_INT
TEST_SERIAL_CASE(int, int)
#endif

#ifdef KOKKOSKERNELS_TEST_ALL_TYPES
// test mixed scalar types (void -> default alpha/beta)
TEST_SERIAL_CASE4(mixed, double, int, float, void)

// test arbitrary double alpha/beta with complex<double> values
TEST_SERIAL_CASE2(alphabeta, Kokkos::complex<double>, double)
#endif

#undef TEST_SERIAL_CASE4
#undef TEST_SERIAL_CASE2
#undef TEST_SERIAL_CASE
