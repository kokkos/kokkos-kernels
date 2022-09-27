// Note: Luc Berger-Vergiat 04/14/21
//       This tests uses KOKKOS_LAMBDA so we need
//       to make sure that these are enabled in
//       the CUDA backend before including this test.
#if !defined(TEST_CUDA_BLAS_CPP) || defined(KOKKOS_ENABLE_CUDA_LAMBDA)

#include <KokkosBlas_util.hpp>
#include <KokkosKernels_TestUtils.hpp>  // for test/inst guards
// Note: include serial gemv before util so it knows if CompactMKL is available
#include <Test_Blas2_gemv_util.hpp>
#include <KokkosBlas2_gemv.hpp>

namespace Test {

template <class AType, class XType, class YType, class ScalarType,
          class AlgoTag>
struct ThreadVectorGEMVOp : public GemvOpBase<AType, XType, YType, ScalarType> {
  using params = GemvOpBase<AType, XType, YType, ScalarType>;

  ThreadVectorGEMVOp(char trans_, ScalarType alpha_, AType A_, XType x_,
                     ScalarType beta_, YType y_)
      : params(trans_, alpha_, A_, x_, beta_, y_) {}

  template <typename TeamMember>
  KOKKOS_INLINE_FUNCTION void operator()(const TeamMember& member) const {
    KokkosBlas::Experimental::Gemv<KokkosBlas::Mode::ThreadVector,
                                   AlgoTag>::invoke(member, params::trans,
                                                    params::alpha, params::A,
                                                    params::x, params::beta,
                                                    params::y);
  }
};

struct ThreadVectorGemvFactory {
  template <class AlgoTag, class ViewTypeA, class ViewTypeX, class ViewTypeY,
            class Device, class ScalarType>
  using functor_type =
      ThreadVectorGEMVOp<ViewTypeA, ViewTypeX, ViewTypeY, ScalarType, AlgoTag>;

  // no Blocked implementation
  using algorithms = std::tuple<KokkosBlas::Algo::Gemv::Unblocked>;
};

}  // namespace Test

#define TEST_THREADVECTOR_CASE4(N, A, X, Y, SC) \
  TEST_GEMV_CASE4(threadvector, ThreadVectorGemvFactory, N, A, X, Y, SC)
#define TEST_THREADVECTOR_CASE2(N, S, SC) \
  TEST_GEMV_CASE2(threadvector, ThreadVectorGemvFactory, N, S, SC)
#define TEST_THREADVECTOR_CASE(N, S) \
  TEST_GEMV_CASE(threadvector, ThreadVectorGemvFactory, N, S)

#ifdef KOKKOSKERNELS_TEST_FLOAT
TEST_THREADVECTOR_CASE(float, float)
#endif

#ifdef KOKKOSKERNELS_TEST_DOUBLE
TEST_THREADVECTOR_CASE(double, double)
#endif

#ifdef KOKKOSKERNELS_TEST_COMPLEX_DOUBLE
TEST_THREADVECTOR_CASE(complex_double, Kokkos::complex<double>)
#endif

#ifdef KOKKOSKERNELS_TEST_COMPLEX_FLOAT
TEST_THREADVECTOR_CASE(complex_float, Kokkos::complex<float>)
#endif

#ifdef KOKKOSKERNELS_TEST_INT
TEST_THREADVECTOR_CASE(int, int)
#endif

#ifdef KOKKOSKERNELS_TEST_ALL_TYPES
// test mixed scalar types (void -> default alpha/beta)
TEST_THREADVECTOR_CASE4(mixed, double, int, float, void)

// test arbitrary double alpha/beta with complex<double> values
TEST_THREADVECTOR_CASE2(alphabeta, Kokkos::complex<double>, double)
#endif

#undef TEST_THREADVECTOR_CASE4
#undef TEST_THREADVECTOR_CASE2
#undef TEST_THREADVECTOR_CASE

#endif  // Check for lambda availability on CUDA backend
