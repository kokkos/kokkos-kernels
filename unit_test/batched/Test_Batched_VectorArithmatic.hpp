/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

#include "KokkosBatched_Vector.hpp"

#include "KokkosKernels_TestUtils.hpp"

using namespace KokkosBatched::Experimental;

namespace Test {

  template<typename VectorTagType,int VectorLength>
  void impl_test_batched_vector_arithmatic() {
    /// random data initialization
    typedef Vector<VectorTagType,VectorLength> vector_type;

    typedef typename vector_type::value_type value_type;    
    const int vector_length = vector_type::vector_length;
    
    typedef Kokkos::Details::ArithTraits<value_type> ats;
    typedef typename ats::mag_type mag_type;

    vector_type a, b, c;
    value_type alpha;
    mag_type beta;
    const value_type zero(0);

    Random<value_type> a_random;
    Random<mag_type> b_random;
    for (int iter=0;iter<100;++iter) {
      for (int k=0;k<vector_length;++k) {
        a[k] = a_random.value();
        b[k] = a_random.value();
        c[k] = zero;
      }
      alpha = a_random.value();
      beta  = b_random.value();

      const mag_type eps = 1.0e3 * ats::epsilon();

      {
        /// test : vec + vec
        c = a + b;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], a[k]+b[k], eps*c[k]);      
      
        /// test : value + vec
        c = alpha + b;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], alpha+b[k], eps*c[k]);      
      
        /// test : vec + value
        c = b + alpha;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], b[k] + alpha, eps*c[k]);      

        /// test : vec + mag
        c = a + beta;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], a[k] + beta, eps*c[k]);      

        /// test : mag + vec
        c = beta + a;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], beta + a[k], eps*c[k]);      
      }
      {
        /// test : vec - vec
        c = a - b;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], a[k]-b[k], eps*c[k]);      
      
        /// test : value - vec
        c = alpha - b;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], alpha-b[k], eps*c[k]);      
      
        /// test : vec + value
        c = b - alpha;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], b[k]-alpha, eps*c[k]);      

        /// test : vec - mag
        c = a - beta;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], a[k] - beta, eps*c[k]);      

        /// test : mag - vec
        c = beta - a;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], beta - a[k], eps*c[k]);      
      }
      {
        /// test : vec * vec
        c = a * b;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], a[k]*b[k], eps*c[k]);      
      
        /// test : value * vec
        c = alpha * b;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], alpha*b[k], eps*c[k]);      
      
        /// test : vec + value
        c = b * alpha;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], b[k]*alpha, eps*c[k]);      

        /// test : vec * mag
        c = a * beta;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], a[k] * beta, eps*c[k]);      

        /// test : mag * vec
        c = beta * a;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], beta * a[k], eps*c[k]);      
      }
      {
        /// test : vec / vec
        c = a / b;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], a[k]/b[k], eps*c[k]);      
        
        /// test : value / vec
        c = alpha / b;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], alpha/b[k], eps*c[k]);      
        
        /// test : vec / value
        c = b / alpha;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], b[k]/alpha, eps*c[k]);      
        
        /// test : mag / vec
        c = beta / a;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], beta/a[k], eps*c[k]);      
        
        /// test : vec / value
        c = a / beta;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], a[k]/beta, eps*c[k]);      
      }
      {
        /// test : vec  -vec
        c = -a;
        for (int k=0;k<vector_length;++k) 
          EXPECT_NEAR_KK( c[k], -a[k], eps*c[k]);      
      }
    }    
  }
}

template<typename DeviceType,typename VectorTagType,int VectorLength>
int test_batched_vector_arithmatic() {
  static_assert(Kokkos::Impl::SpaceAccessibility<DeviceType,Kokkos::HostSpace >::accessible,
                "vector datatype is only tested on host space");
  Test::impl_test_batched_vector_arithmatic<VectorTagType,VectorLength>();
  
  return 0;
}


///
/// SIMD
///

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F( TestCategory, batched_vector_arithmatic_simd_float3 ) {
  test_batched_vector_arithmatic<TestExecSpace,SIMD<float>,3>();
}
TEST_F( TestCategory, batched_vector_arithmatic_simd_float8 ) {
  test_batched_vector_arithmatic<TestExecSpace,SIMD<float>,8>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F( TestCategory, batched_vector_arithmatic_simd_double3 ) {
  test_batched_vector_arithmatic<TestExecSpace,SIMD<double>,3>();
}
TEST_F( TestCategory, batched_vector_arithmatic_simd_double4 ) {
  test_batched_vector_arithmatic<TestExecSpace,SIMD<double>,4>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
TEST_F( TestCategory, batched_vector_arithmatic_simd_scomplex3 ) {
  test_batched_vector_arithmatic<TestExecSpace,SIMD<Kokkos::complex<float> >,3>();
}
TEST_F( TestCategory, batched_vector_arithmatic_simd_scomplex4 ) {
  test_batched_vector_arithmatic<TestExecSpace,SIMD<Kokkos::complex<float> >,4>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
TEST_F( TestCategory, batched_vector_arithmatic_simd_dcomplex3 ) {
  test_batched_vector_arithmatic<TestExecSpace,SIMD<Kokkos::complex<double> >,3>();
}
TEST_F( TestCategory, batched_vector_arithmatic_simd_dcomplex2 ) {
  test_batched_vector_arithmatic<TestExecSpace,SIMD<Kokkos::complex<double> >,2>();
}
#endif

///
/// AVX
///

#if defined(__AVX__) || defined(__AVX2__)
#if defined(KOKKOSKERNELS_INST_FLOAT)
// TEST_F( TestCategory, batched_vector_arithmatic_avx_float8 ) {
//   typedef VectorTag<AVX<float,TestExecSpace>, 8> vector_tag_type;
//   test_batched_vector_arithmatic<TestExecSpace,vector_tag_type>();
// }
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F( TestCategory, batched_vector_arithmatic_avx_double4 ) {
  test_batched_vector_arithmatic<TestExecSpace,AVX<double>,4>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
TEST_F( TestCategory, batched_vector_arithmatic_avx_dcomplex2 ) {
  test_batched_vector_arithmatic<TestExecSpace,AVX<Kokkos::complex<double> >,2>();
}
#endif
#endif

///
/// AVX 512
///

#if defined(__AVX512F__)
#if defined(KOKKOSKERNELS_INST_FLOAT)
// TEST_F( TestCategory, batched_vector_arithmatic_avx_float16 ) {
//   typedef VectorTag<AVX<float,TestExecSpace>, 16> vector_tag_type;
//   test_batched_vector_arithmatic<TestExecSpace,vector_tag_type>();
// }
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F( TestCategory, batched_vector_arithmatic_avx_double8 ) {
  test_batched_vector_arithmatic<TestExecSpace,AVX<double>,8>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
TEST_F( TestCategory, batched_vector_arithmatic_avx_dcomplex4 ) {
  typedef AVX<Kokkos::complex<double>, 4> vector_tag_type;
  test_batched_vector_arithmatic<TestExecSpace,AVX<Kokkos::complex<double> >,4>();
}
#endif
#endif
