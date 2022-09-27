/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Test_Blas3_gemm_util.hpp"

namespace Test {
namespace Gemm {

template <typename DeviceType, typename ViewTypeA, typename ViewTypeB,
          typename ViewTypeC, typename ScalarType, typename ParamTagType,
          typename AlgoTagType>
struct Functor_TestBlasSerialGemm {
  ViewTypeA _a;
  ViewTypeB _b;
  ViewTypeC _c;

  ScalarType _alpha, _beta;

  KOKKOS_INLINE_FUNCTION
  Functor_TestBlasSerialGemm(const ScalarType alpha, const ViewTypeA &a,
                             const ViewTypeB &b, const ScalarType beta,
                             const ViewTypeC &c)
      : _a(a), _b(b), _c(c), _alpha(alpha), _beta(beta) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const ParamTagType &, const int k) const {
    auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto bb = Kokkos::subview(_b, k, Kokkos::ALL(), Kokkos::ALL());
    auto cc = Kokkos::subview(_c, k, Kokkos::ALL(), Kokkos::ALL());

    KokkosBlas::SerialGemm<typename ParamTagType::transA,
                           typename ParamTagType::transB,
                           AlgoTagType>::invoke(_alpha, aa, bb, _beta, cc);
  }

  inline void run() {
    typedef typename ViewTypeC::value_type value_type;
    std::string name_region("KokkosBlas::Test::SerialGemm");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<DeviceType, ParamTagType> policy(0, _c.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
  }
};

template <typename DeviceType, typename ViewTypeA, typename ViewTypeB,
          typename ViewTypeC, typename ScalarType, typename ParamTagType,
          typename AlgoTagType>
void impl_test_blas_gemm(const int N, const int dimM, const int dimN,
                         const int dimK) {
  using execution_space = typename DeviceType::execution_space;
  using transA          = typename ParamTagType::transA;
  using transB          = typename ParamTagType::transB;
  using value_type_a    = typename ViewTypeA::value_type;
  using value_type_b    = typename ViewTypeB::value_type;
  using value_type_c    = typename ViewTypeC::value_type;
  using ats_c           = Kokkos::Details::ArithTraits<value_type_c>;

  const auto transposed_A = !std::is_same<transA, Trans::NoTranspose>::value;
  const auto transposed_B = !std::is_same<transB, Trans::NoTranspose>::value;

  const int matAdim1 = transposed_A ? dimM : dimK;
  const int matAdim2 = transposed_A ? dimK : dimM;
  const int matBdim1 = transposed_B ? dimK : dimN;
  const int matBdim2 = transposed_B ? dimN : dimK;
  const int matCdim1 = dimM;
  const int matCdim2 = dimN;

  /// randomized input testing views
  ScalarType alpha = ScalarType(1.5);
  ScalarType beta  = ScalarType(3.0);

  ViewTypeA a_expected("a_expected", N, matAdim1, matAdim2);
  ViewTypeA a_actual("a_actual", N, matAdim1, matAdim2);
  ViewTypeB b_expected("b_expected", N, matBdim1, matBdim2);
  ViewTypeB b_actual("b_actual", N, matBdim1, matBdim2);
  ViewTypeC c_expected("c_expected", N, matCdim1, matCdim2);
  ViewTypeC c_actual("c_actual", N, matCdim1, matCdim2);

  Kokkos::Random_XorShift64_Pool<execution_space> random(13718);

  Kokkos::fill_random(a_expected, random, value_type_a(1.0));
  Kokkos::fill_random(b_expected, random, value_type_b(1.0));
  Kokkos::fill_random(c_expected, random, value_type_c(1.0));

  Kokkos::fence();

  Kokkos::deep_copy(a_actual, a_expected);
  Kokkos::deep_copy(b_actual, b_expected);
  Kokkos::deep_copy(c_actual, c_expected);

  Functor_BatchedVanillaGEMM<ViewTypeA, ViewTypeB, ViewTypeC, execution_space>
      vgemm;
  vgemm.A_t   = transposed_A;
  vgemm.B_t   = transposed_B;
  vgemm.A_c   = std::is_same<transA, Trans::ConjTranspose>::value;
  vgemm.B_c   = std::is_same<transB, Trans::ConjTranspose>::value;
  vgemm.A     = a_expected;
  vgemm.B     = b_expected;
  vgemm.C     = c_expected;
  vgemm.alpha = alpha;
  vgemm.beta  = beta;
  vgemm.run();  // Compute c_expected
  Functor_TestBlasSerialGemm<DeviceType, ViewTypeA, ViewTypeB, ViewTypeC,
                             ScalarType, ParamTagType, AlgoTagType>(
      alpha, a_actual, b_actual, beta, c_actual)
      .run();

  typename ViewTypeC::HostMirror c_expected_host =
      Kokkos::create_mirror_view(c_expected);
  typename ViewTypeC::HostMirror c_actual_host =
      Kokkos::create_mirror_view(c_actual);

  // Copy to host for comparison
  Kokkos::deep_copy(c_expected_host, c_expected);
  Kokkos::deep_copy(c_actual_host, c_actual);

  Kokkos::fence();

  // check c_expected = c_actual
  // std::conditional<, float,
  using mag_type = typename ats_c::mag_type;
  mag_type sum(1), diff(0);

  mag_type eps = ats_c::epsilon();

  eps *=
      std::is_same<value_type_c, Kokkos::Experimental::half_t>::value ||
              std::is_same<value_type_c, Kokkos::Experimental::bhalf_t>::value
          ? 4
          : 1e3;

  for (int k = 0; k < N; ++k)
    for (int i = 0; i < matCdim1; ++i)
      for (int j = 0; j < matCdim2; ++j) {
        sum += ats_c::abs(c_expected_host(k, i, j));
        diff += ats_c::abs(c_expected_host(k, i, j) - c_actual_host(k, i, j));
      }
  EXPECT_NEAR_KK(diff / sum, 0, eps);
}

template <typename DeviceType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC, typename ScalarType, typename ParamTagType,
          typename AlgoTagType>
int test_blas_gemm() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    typedef Kokkos::View<ValueTypeA ***, Kokkos::LayoutLeft, DeviceType>
        ViewTypeA;
    typedef Kokkos::View<ValueTypeB ***, Kokkos::LayoutLeft, DeviceType>
        ViewTypeB;
    typedef Kokkos::View<ValueTypeC ***, Kokkos::LayoutLeft, DeviceType>
        ViewTypeC;
    Test::Gemm::impl_test_blas_gemm<DeviceType, ViewTypeA, ViewTypeB, ViewTypeC,
                                    ScalarType, ParamTagType, AlgoTagType>(
        0, 10, 10, 10);
    for (int i = 0; i < 10; ++i) {
      // printf("Testing: LayoutLeft,  Blksize %d\n", i);
      Test::Gemm::impl_test_blas_gemm<DeviceType, ViewTypeA, ViewTypeB,
                                      ViewTypeC, ScalarType, ParamTagType,
                                      AlgoTagType>(1024, i, i, i);
    }
    for (int i = 0; i < 10; ++i) {
      // printf("Testing: LayoutLeft,  Blksize %d\n", i);
      int dimM = i;
      int dimN = 2 * i;
      int dimK = 3 * i;
      Test::Gemm::impl_test_blas_gemm<DeviceType, ViewTypeA, ViewTypeB,
                                      ViewTypeC, ScalarType, ParamTagType,
                                      AlgoTagType>(1024, dimM, dimN, dimK);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    typedef Kokkos::View<ValueTypeA ***, Kokkos::LayoutRight, DeviceType>
        ViewTypeA;
    typedef Kokkos::View<ValueTypeB ***, Kokkos::LayoutRight, DeviceType>
        ViewTypeB;
    typedef Kokkos::View<ValueTypeC ***, Kokkos::LayoutRight, DeviceType>
        ViewTypeC;
    Test::Gemm::impl_test_blas_gemm<DeviceType, ViewTypeA, ViewTypeB, ViewTypeC,
                                    ScalarType, ParamTagType, AlgoTagType>(
        0, 10, 10, 10);
    for (int i = 0; i < 10; ++i) {
      // printf("Testing: LayoutRight, Blksize %d\n", i);
      Test::Gemm::impl_test_blas_gemm<DeviceType, ViewTypeA, ViewTypeB,
                                      ViewTypeC, ScalarType, ParamTagType,
                                      AlgoTagType>(1024, i, i, i);
    }
    for (int i = 0; i < 10; ++i) {
      // printf("Testing: LayoutLeft,  Blksize %d\n", i);
      int dimM = i;
      int dimN = 2 * i;
      int dimK = 3 * i;
      Test::Gemm::impl_test_blas_gemm<DeviceType, ViewTypeA, ViewTypeB,
                                      ViewTypeC, ScalarType, ParamTagType,
                                      AlgoTagType>(1024, dimM, dimN, dimK);
    }
  }
#endif

  return 0;
}

#define TEST_SERIAL_CASE2(NAME, VALUE, SCALAR) \
  TEST_GEMM_CASE(serial, NAME, test_blas_gemm, VALUE, VALUE, VALUE, SCALAR)
#define TEST_SERIAL_CASE(NAME, VALUE) TEST_SERIAL_CASE2(NAME, VALUE, VALUE)

#if defined(KOKKOS_BHALF_T_IS_FLOAT)
TEST_SERIAL_CASE(bhalf_bhalf, ::Test::bhalfScalarType)
#endif

#if defined(KOKKOS_HALF_T_IS_FLOAT)
TEST_SERIAL_CASE(half_half, ::Test::halfScalarType)
#endif

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_SERIAL_CASE(float_float, float)
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_SERIAL_CASE(double_double, double)
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
TEST_SERIAL_CASE(dcomplex_dcomplex, Kokkos::complex<double>)
// TEST_F( TestCategory, blas_scalar_serial_gemm_ct_nt_dcomplex_dcomplex ) {
//   typedef ::Test::Gemm::ParamTag<Trans::ConjTranspose,Trans::NoTranspose>
//   param_tag_type; typedef Algo::Gemm::Blocked algo_tag_type;
//   test_blas_gemm<TestExecSpace,Kokkos::complex<double>,Kokkos::complex<double>,param_tag_type,algo_tag_type>();
// }
// TEST_F( TestCategory, blas_scalar_serial_gemm_nt_ct_dcomplex_dcomplex ) {
//   typedef ::Test::Gemm::ParamTag<Trans::NoTranspose,Trans::ConjTranspose>
//   param_tag_type; typedef Algo::Gemm::Blocked algo_tag_type;
//   test_blas_gemm<TestExecSpace,Kokkos::complex<double>,Kokkos::complex<double>,param_tag_type,algo_tag_type>();
// }
TEST_SERIAL_CASE2(dcomplex_double, Kokkos::complex<double>, double)
// TEST_F( TestCategory, blas_scalar_serial_gemm_ct_nt_dcomplex_double ) {
//   typedef ::Test::Gemm::ParamTag<Trans::ConjTranspose,Trans::NoTranspose>
//   param_tag_type; typedef Algo::Gemm::Blocked algo_tag_type;
//   test_blas_gemm<TestExecSpace,Kokkos::complex<double>,double,param_tag_type,algo_tag_type>();
// }
// TEST_F( TestCategory, blas_scalar_serial_gemm_nt_ct_dcomplex_double ) {
//   typedef ::Test::Gemm::ParamTag<Trans::NoTranspose,Trans::ConjTranspose>
//   param_tag_type; typedef Algo::Gemm::Blocked algo_tag_type;
//   test_blas_gemm<TestExecSpace,Kokkos::complex<double>,double,param_tag_type,algo_tag_type>();
// }
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
TEST_SERIAL_CASE(fcomplex_fcomplex, Kokkos::complex<float>)
TEST_SERIAL_CASE2(fcomplex_float, Kokkos::complex<float>, float)
#endif

#if defined(KOKKOSKERNELS_INST_FLOAT) && defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_GEMM_CASE(serial, mixed_scalars, test_blas_gemm, float, int, double,
               double)
#endif

}  // namespace Gemm
}  // namespace Test
