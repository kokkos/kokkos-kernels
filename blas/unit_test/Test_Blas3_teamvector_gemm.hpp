/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Test_Blas3_gemm_util.hpp"

namespace Test {
namespace Gemm {

template <typename DeviceType, typename ViewType, typename ScalarType,
          typename ParamTagType, typename AlgoTagType>
struct Functor_TestBlasTeamVector {
  ViewType _a, _b, _c;

  ScalarType _alpha, _beta;

  KOKKOS_INLINE_FUNCTION
  Functor_TestBlasTeamVector(const ScalarType alpha, const ViewType &a,
                             const ViewType &b, const ScalarType beta,
                             const ViewType &c)
      : _a(a), _b(b), _c(c), _alpha(alpha), _beta(beta) {}

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const ParamTagType &,
                                         const MemberType &member) const {
    const int k = member.league_rank();

    auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto bb = Kokkos::subview(_b, k, Kokkos::ALL(), Kokkos::ALL());
    auto cc = Kokkos::subview(_c, k, Kokkos::ALL(), Kokkos::ALL());

    KokkosBlas::TeamVectorGemm<typename ParamTagType::transA,
                               typename ParamTagType::transB,
                               AlgoTagType>::invoke(member, _alpha, aa, bb,
                                                    _beta, cc);
  }

  inline void run() {
    typedef typename ViewType::value_type value_type;
    std::string name_region("KokkosBlas::Test::TeamVector");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());
    const int league_size = _c.extent(0);
    Kokkos::TeamPolicy<DeviceType, ParamTagType> policy(league_size,
                                                        Kokkos::AUTO);
    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
  }
};

template <typename DeviceType, typename ViewType, typename ScalarType,
          typename ParamTagType, typename AlgoTagType>
void impl_test_blas_teamvectorgemm(const int N, const int dimM, const int dimN,
                                   const int dimK) {
  using transA          = typename ParamTagType::transA;
  using transB          = typename ParamTagType::transB;
  using execution_space = typename DeviceType::execution_space;
  using value_type      = typename ViewType::value_type;
  using ats             = Kokkos::Details::ArithTraits<value_type>;

  const auto transposed_A = !std::is_same<transA, Trans::NoTranspose>::value;
  const auto transposed_B = !std::is_same<transB, Trans::NoTranspose>::value;

  const int matAdim1 = transposed_A ? dimM : dimK;
  const int matAdim2 = transposed_A ? dimK : dimM;
  const int matBdim1 = transposed_B ? dimK : dimN;
  const int matBdim2 = transposed_B ? dimN : dimK;
  const int matCdim1 = dimM;
  const int matCdim2 = dimN;

  /// randomized input testing views
  ScalarType alpha = ScalarType(1.5), beta = ScalarType(3.0);

  ViewType a_expected("a_expected", N, matAdim1, matAdim2),
      a_actual("a_actual", N, matAdim1, matAdim2),
      b_expected("b_expected", N, matBdim1, matBdim2),
      b_actual("b_actual", N, matBdim1, matBdim2),
      c_expected("c_expected", N, matCdim1, matCdim2),
      c_actual("c_actual", N, matCdim1, matCdim2);

  Kokkos::Random_XorShift64_Pool<typename DeviceType::execution_space> random(
      13718);

  Kokkos::fill_random(a_expected, random, value_type(1.0));
  Kokkos::fill_random(b_expected, random, value_type(1.0));
  Kokkos::fill_random(c_expected, random, value_type(1.0));

  Kokkos::fence();

  Kokkos::deep_copy(a_actual, a_expected);
  Kokkos::deep_copy(b_actual, b_expected);
  Kokkos::deep_copy(c_actual, c_expected);

  // Functor_TestBlasTeamVector<DeviceType,ViewType,ScalarType,
  //   ParamTagType,Algo::Gemm::Unblocked>(alpha, a_expected, b_expected,
  //   beta, c_expected).run();
  Functor_BatchedVanillaGEMM<ViewType, ViewType, ViewType, execution_space>
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

  Functor_TestBlasTeamVector<DeviceType, ViewType, ScalarType, ParamTagType,
                             AlgoTagType>(alpha, a_actual, b_actual, beta,
                                          c_actual)
      .run();

  Kokkos::fence();

  typename ViewType::HostMirror c_expected_host =
      Kokkos::create_mirror_view(c_expected);
  typename ViewType::HostMirror c_actual_host =
      Kokkos::create_mirror_view(c_actual);

  // Copy to host for comparison
  Kokkos::deep_copy(c_expected_host, c_expected);
  Kokkos::deep_copy(c_actual_host, c_actual);

  using mag_type = typename ats::mag_type;
  mag_type sum(1), diff(0);

  mag_type eps = ats::epsilon();

  eps *= std::is_same<value_type, Kokkos::Experimental::half_t>::value ||
                 std::is_same<value_type, Kokkos::Experimental::bhalf_t>::value
             ? 4
             : 1e3;

  for (int k = 0; k < N; ++k)
    for (int i = 0; i < matCdim1; ++i)
      for (int j = 0; j < matCdim2; ++j) {
        sum += ats::abs(c_expected_host(k, i, j));
        diff += ats::abs(c_expected_host(k, i, j) - c_actual_host(k, i, j));
      }
  EXPECT_NEAR_KK(diff / sum, 0, eps);
}

template <typename DeviceType, typename ValueType, typename ScalarType,
          typename ParamTagType, typename AlgoTagType>
typename std::enable_if<
    !std::is_same<AlgoTagType, Algo::Gemm::Unblocked>::value, int>::type
test_blas_teamvectorgemm() {
  // skip algorithms not supported by TeamVectorGemm
  return 0;
}

template <typename DeviceType, typename ValueType, typename ScalarType,
          typename ParamTagType, typename AlgoTagType>
typename std::enable_if<std::is_same<AlgoTagType, Algo::Gemm::Unblocked>::value,
                        int>::type
test_blas_teamvectorgemm() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    typedef Kokkos::View<ValueType ***, Kokkos::LayoutLeft, DeviceType>
        ViewType;
    impl_test_blas_teamvectorgemm<DeviceType, ViewType, ScalarType,
                                  ParamTagType, AlgoTagType>(0, 10, 10, 10);
    for (int i = 0; i < 10; ++i) {
      // printf("Testing: LayoutLeft,  Blksize %d\n", i);
      impl_test_blas_teamvectorgemm<DeviceType, ViewType, ScalarType,
                                    ParamTagType, AlgoTagType>(1024, i, i, i);
    }
    for (int i = 0; i < 10; ++i) {
      // printf("Testing: LayoutLeft,  Blksize %d\n", i);
      int dimM = i;
      int dimN = 2 * i;
      int dimK = 3 * i;
      impl_test_blas_teamvectorgemm<DeviceType, ViewType, ScalarType,
                                    ParamTagType, AlgoTagType>(1024, dimM, dimN,
                                                               dimK);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    typedef Kokkos::View<ValueType ***, Kokkos::LayoutRight, DeviceType>
        ViewType;
    impl_test_blas_teamvectorgemm<DeviceType, ViewType, ScalarType,
                                  ParamTagType, AlgoTagType>(0, 10, 10, 10);
    for (int i = 0; i < 10; ++i) {
      // printf("Testing: LayoutRight, Blksize %d\n", i);
      impl_test_blas_teamvectorgemm<DeviceType, ViewType, ScalarType,
                                    ParamTagType, AlgoTagType>(1024, i, i, i);
    }
    for (int i = 0; i < 10; ++i) {
      // printf("Testing: LayoutLeft,  Blksize %d\n", i);
      int dimM = i;
      int dimN = 2 * i;
      int dimK = 3 * i;
      impl_test_blas_teamvectorgemm<DeviceType, ViewType, ScalarType,
                                    ParamTagType, AlgoTagType>(1024, dimM, dimN,
                                                               dimK);
    }
  }
#endif

  return 0;
}

#define TEST_TEAMVECTOR_CASE2(NAME, VALUE, SCALAR) \
  TEST_GEMM_CASE(team_vector, NAME, test_blas_teamvectorgemm, VALUE, SCALAR)
#define TEST_TEAMVECTOR_CASE(NAME, VALUE) \
  TEST_TEAMVECTOR_CASE2(NAME, VALUE, VALUE)

#if defined(KOKKOS_BHALF_T_IS_FLOAT)
TEST_TEAMVECTOR_CASE(bhalf_bhalf, ::Test::bhalfScalarType)
#endif

#if defined(KOKKOS_HALF_T_IS_FLOAT)
TEST_TEAMVECTOR_CASE(half_half, ::Test::halfScalarType)
#endif

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_TEAMVECTOR_CASE(float_float, float)
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_TEAMVECTOR_CASE(double_double, double)
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
TEST_TEAMVECTOR_CASE(dcomplex_dcomplex, Kokkos::complex<double>)
TEST_TEAMVECTOR_CASE2(dcomplex_double, Kokkos::complex<double>, double)
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
TEST_TEAMVECTOR_CASE(fcomplex_fcomplex, Kokkos::complex<float>)
TEST_TEAMVECTOR_CASE2(fcomplex_float, Kokkos::complex<float>, float)
#endif

#undef TEST_TEAMVECTOR_CASE
#undef TEST_TEAMVECTOR_CASE2

}  // namespace Gemm
}  // namespace Test
