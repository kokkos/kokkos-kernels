/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

#include "KokkosBatched_Vector.hpp"

#include "KokkosKernels_TestUtils.hpp"

using namespace KokkosBatched::Experimental;

namespace Test {

  template<typename VectorTagType,int VectorLength>
  void impl_test_batched_vector_math() {
    /// random data initialization
    typedef Vector<VectorTagType,VectorLength> vector_type;
    
    typedef typename vector_type::value_type value_type;    
    const int vector_length = vector_type::vector_length;
    
    typedef Kokkos::Details::ArithTraits<value_type> ats;
    typedef typename ats::mag_type mag_type;

    vector_type a, b, aref, bref;
    const value_type one(1), two(2);
    const mag_type eps = 1.0e3 * ats::epsilon();

    Random<value_type> random;
    for (int iter=0;iter<100;++iter) {
      for (int k=0;k<vector_length;++k) {
        aref[k] = (random.value() + one)/two;
        bref[k] = (random.value() + one)/two;
      }

      {

#undef CHECK
#define CHECK(op)                                               \
        {                                                       \
          mag_type diff = 0;                                    \
          a = op(aref);                                         \
          for (int i=0;i<vector_length;++i)                     \
            EXPECT_NEAR_KK( a[i], std::op(aref[i]), eps*a[i]);  \
        }
        
        CHECK(sqrt);
        CHECK(cbrt);
        CHECK(log);
        CHECK(exp);
        CHECK(sin);
        CHECK(cos);
        CHECK(tan);
        CHECK(sinh);
        CHECK(cosh);
        CHECK(tanh);
        CHECK(asin);
        CHECK(acos);
        CHECK(atan);

#undef CHECK
#define CHECK(op)                                                       \
        {                                                               \
          a = pow(aref,bref);                                           \
          for (int i=0;i<vector_length;++i)                             \
            EXPECT_NEAR_KK( a[i], std::pow(aref[i], bref[i]), eps*a[i] ); \
        }                                                               \
        
        CHECK(pow);
        CHECK(atan2);
        
#undef CHECK
#define CHECK(op)                                                       \
        {                                                               \
          mag_type beta = random.value();                               \
          a = op(aref,beta);                                            \
          for (int i=0;i<vector_length;++i)                             \
            EXPECT_NEAR_KK( a[i], std::op(aref[i], beta), eps*a[i] );   \
        }

        CHECK(pow);
        CHECK(atan2);

#undef CHECK
#define CHECK(op)                                                       \
        {                                                               \
          value_type alpha = random.value();                            \
          a = op(alpha,bref);                                           \
          for (int i=0;i<vector_length;++i)                             \
            EXPECT_NEAR_KK( a[i], std::op(alpha, bref[i]), eps*a[i] );  \
        }
        
        CHECK(pow);
        CHECK(atan2);
#undef CHECK

      } // end test body
    } // end for
  } // impl
} // namespace

template<typename DeviceType,typename VectorTagType,int VectorLength>
int test_batched_vector_math() {
  static_assert(Kokkos::Impl::SpaceAccessibility<DeviceType,Kokkos::HostSpace >::accessible,
                "vector datatype is only tested on host space");
  Test::impl_test_batched_vector_math<VectorTagType,VectorLength>();
  
  return 0;
}


///
/// SIMD
///

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F( TestCategory, batched_vector_math_simd_float8 ) {
  test_batched_vector_math<TestExecSpace,SIMD<float>,8>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F( TestCategory, batched_vector_math_simd_double4 ) {
  test_batched_vector_math<TestExecSpace,SIMD<double>,4>();
}
#endif

// #if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
// TEST_F( TestCategory, batched_vector_math_simd_dcomplex2 ) {
//   test_batched_vector_math<TestExecSpace,SIMD<Kokkos::complex<double> >,2>();
// }
// #endif
