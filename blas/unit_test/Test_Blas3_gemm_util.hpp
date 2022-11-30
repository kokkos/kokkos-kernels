#ifndef TEST_BLAS2_GEMM_UTIL_HPP
#define TEST_BLAS2_GEMM_UTIL_HPP

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

#include "KokkosBlas3_gemm.hpp"
#include "KokkosKernels_TestUtils.hpp"

namespace Test {
namespace Gemm {

using KokkosBlas::Algo;
using KokkosBlas::Trans;

template <typename TA, typename TB>
struct ParamTag {
  typedef TA transA;
  typedef TB transB;
};

#define TEST_GEMM_ALGO(NAME, FUNC, TRANS_A, TRANS_B, VALUE, SCALAR)            \
  TEST_F(TestCategory, batched_scalar_##NAME) {                                \
    typedef ::Test::Gemm::ParamTag<TRANS_A, TRANS_B> param_tag_type;           \
    FUNC<TestExecSpace, VALUE, SCALAR, param_tag_type, Algo::Gemm::Blocked>(); \
    FUNC<TestExecSpace, VALUE, SCALAR, param_tag_type,                         \
         Algo::Gemm::Unblocked>();                                             \
  }

#define TEST_GEMM_CASE(PREFIX, NAME, FUNC, VALUE, SCALAR)              \
  TEST_GEMM_ALGO(PREFIX##_gemm_nt_nt_##NAME, FUNC, Trans::NoTranspose, \
                 Trans::NoTranspose, VALUE, SCALAR)                    \
  TEST_GEMM_ALGO(PREFIX##_gemm_t_nt_##NAME, FUNC, Trans::Transpose,    \
                 Trans::NoTranspose, VALUE, SCALAR)                    \
  TEST_GEMM_ALGO(PREFIX##_gemm_nt_t_##NAME, FUNC, Trans::NoTranspose,  \
                 Trans::Transpose, VALUE, SCALAR)                      \
  TEST_GEMM_ALGO(PREFIX##_gemm_t_t_##NAME, FUNC, Trans::Transpose,     \
                 Trans::Transpose, VALUE, SCALAR)

}  // namespace Gemm
}  // namespace Test

#endif  // TEST_BLAS2_GEMM_UTIL_HPP