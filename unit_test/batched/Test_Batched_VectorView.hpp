/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

#include "KokkosBatched_Vector.hpp"

#include "KokkosKernels_TestUtils.hpp"

using namespace KokkosBatched::Experimental;

namespace Test {

  template<typename VectorTagType,int VectorLength>
  void impl_test_batched_vector_view() {
    /// random data initialization
    typedef Vector<VectorTagType,VectorLength> vector_type;
    
    typedef typename vector_type::value_type value_type;    
    const int vector_length = vector_type::vector_length;
    
    typedef Kokkos::Details::ArithTraits<value_type> ats;
    typedef typename ats::mag_type mag_type;

    const mag_type eps = 1.0e3 * ats::epsilon();

    {
      Kokkos::View<vector_type*> a("a", 10);
      SimdViewAccess<Kokkos::View<vector_type*>, PackDim<0> > aa(a);
      
      for (int i=0;i<a.extent(0);++i) 
        a(i) = value_type(i);

      for (int i=0;i<aa.extent(0);++i) 
        EXPECT_NEAR_KK( aa(i), a(i/vector_length)[i%vector_length], eps*aa(i) );
    }
    {
      Kokkos::View<vector_type**> a("a", 10, 10);
      {
        SimdViewAccess<Kokkos::View<vector_type**>, PackDim<0> > aa(a);      
        for (int i0=0;i0<a.extent(0);++i0) 
          for (int i1=0;i1<a.extent(1);++i1) 
            a(i0,i1) = value_type(i0*10+i1);
        
        for (int i0=0;i0<aa.extent(0);++i0) 
          for (int i1=0;i1<aa.extent(1);++i1) 
            EXPECT_NEAR_KK( aa(i0,i1), a(i0/vector_length,i1)[i0%vector_length], eps*aa(i0,i1) );
      }
      {
        SimdViewAccess<Kokkos::View<vector_type**>, PackDim<1> > aa(a);      
        for (int i0=0;i0<a.extent(0);++i0) 
          for (int i1=0;i1<a.extent(1);++i1) 
            a(i0,i1) = value_type(i0*10+i1);
        
        for (int i0=0;i0<aa.extent(0);++i0) 
          for (int i1=0;i1<aa.extent(1);++i1) 
            EXPECT_NEAR_KK( aa(i0,i1), a(i0,i1/vector_length)[i0%vector_length], eps*aa(i0,i1) );
      }
    }
    {
      Kokkos::View<vector_type***> a("a", 10, 10, 10);
      {
        SimdViewAccess<Kokkos::View<vector_type***>, PackDim<0> > aa(a);      
        for (int i0=0;i0<a.extent(0);++i0) 
          for (int i1=0;i1<a.extent(1);++i1) 
            for (int i2=0;i2<a.extent(2);++i2) 
              a(i0,i1,i2) = value_type(i0*100+i1*10+i2);
        
        for (int i0=0;i0<aa.extent(0);++i0) 
          for (int i1=0;i1<aa.extent(1);++i1) 
            for (int i2=0;i2<aa.extent(2);++i2) 
              EXPECT_NEAR_KK( aa(i0,i1,i2), a(i0/vector_length,i1,i2)[i0%vector_length], eps*aa(i0,i1,i2) );
      }
      {
        SimdViewAccess<Kokkos::View<vector_type***>, PackDim<1> > aa(a);      
        for (int i0=0;i0<a.extent(0);++i0) 
          for (int i1=0;i1<a.extent(1);++i1) 
            for (int i2=0;i2<a.extent(2);++i2) 
              a(i0,i1,i2) = value_type(i0*100+i1*10+i2);
        
        for (int i0=0;i0<aa.extent(0);++i0) 
          for (int i1=0;i1<aa.extent(1);++i1) 
            for (int i2=0;i2<aa.extent(2);++i2) 
              EXPECT_NEAR_KK( aa(i0,i1,i2), a(i0,i1/vector_length,i2)[i1%vector_length], eps*aa(i0,i1,i2) );
      }
      {
        SimdViewAccess<Kokkos::View<vector_type***>, PackDim<2> > aa(a);      
        for (int i0=0;i0<a.extent(0);++i0) 
          for (int i1=0;i1<a.extent(1);++i1) 
            for (int i2=0;i2<a.extent(2);++i2) 
              a(i0,i1,i2) = value_type(i0*100+i1*10+i2);
        
        for (int i0=0;i0<aa.extent(0);++i0) 
          for (int i1=0;i1<aa.extent(1);++i1) 
            for (int i2=0;i2<aa.extent(2);++i2) 
              EXPECT_NEAR_KK( aa(i0,i1,i2), a(i0,i1,i2/vector_length)[i2%vector_length], eps*aa(i0,i1,i2) );
      }
    }
    {
      Kokkos::View<vector_type****> a("a", 10, 10, 10, 10);
      {
        SimdViewAccess<Kokkos::View<vector_type****>, PackDim<0> > aa(a);      
        for (int i0=0;i0<a.extent(0);++i0) 
          for (int i1=0;i1<a.extent(1);++i1) 
            for (int i2=0;i2<a.extent(2);++i2) 
              for (int i3=0;i3<a.extent(3);++i3) 
                a(i0,i1,i2,i3) = value_type(i0*1000+i1*100+i2*10+i3);
        
        for (int i0=0;i0<aa.extent(0);++i0) 
          for (int i1=0;i1<aa.extent(1);++i1) 
            for (int i2=0;i2<aa.extent(2);++i2) 
              for (int i3=0;i3<aa.extent(3);++i3) 
                EXPECT_NEAR_KK( aa(i0,i1,i2,i3), a(i0/vector_length,i1,i2,i3)[i0%vector_length], eps*aa(i0,i1,i2,i3) );
      }
      {
        SimdViewAccess<Kokkos::View<vector_type****>, PackDim<1> > aa(a);      
        for (int i0=0;i0<a.extent(0);++i0) 
          for (int i1=0;i1<a.extent(1);++i1) 
            for (int i2=0;i2<a.extent(2);++i2) 
              for (int i3=0;i3<a.extent(3);++i3) 
                a(i0,i1,i2,i3) = value_type(i0*1000+i1*100+i2*10+i3);
        
        for (int i0=0;i0<aa.extent(0);++i0) 
          for (int i1=0;i1<aa.extent(1);++i1) 
            for (int i2=0;i2<aa.extent(2);++i2) 
              for (int i3=0;i3<aa.extent(3);++i3) 
                EXPECT_NEAR_KK( aa(i0,i1,i2,i3), a(i0,i1/vector_length,i2,i3)[i1%vector_length], eps*aa(i0,i1,i2,i3) );
      }
      {
        SimdViewAccess<Kokkos::View<vector_type****>, PackDim<2> > aa(a);      
        for (int i0=0;i0<a.extent(0);++i0) 
          for (int i1=0;i1<a.extent(1);++i1) 
            for (int i2=0;i2<a.extent(2);++i2) 
              for (int i3=0;i3<a.extent(3);++i3) 
                a(i0,i1,i2,i3) = value_type(i0*1000+i1*100+i2*10+i3);
        
        for (int i0=0;i0<aa.extent(0);++i0) 
          for (int i1=0;i1<aa.extent(1);++i1) 
            for (int i2=0;i2<aa.extent(2);++i2) 
              for (int i3=0;i3<aa.extent(3);++i3) 
                EXPECT_NEAR_KK( aa(i0,i1,i2,i3), a(i0,i1,i2/vector_length,i3)[i2%vector_length], eps*aa(i0,i1,i2,i3) );
      }
      {
        SimdViewAccess<Kokkos::View<vector_type****>, PackDim<3> > aa(a);      
        for (int i0=0;i0<a.extent(0);++i0) 
          for (int i1=0;i1<a.extent(1);++i1) 
            for (int i2=0;i2<a.extent(2);++i2) 
              for (int i3=0;i3<a.extent(3);++i3) 
                a(i0,i1,i2,i3) = value_type(i0*1000+i1*100+i2*10+i3);
        
        for (int i0=0;i0<aa.extent(0);++i0) 
          for (int i1=0;i1<aa.extent(1);++i1) 
            for (int i2=0;i2<aa.extent(2);++i2) 
              for (int i3=0;i3<aa.extent(3);++i3) 
                EXPECT_NEAR_KK( aa(i0,i1,i2,i3), a(i0,i1,i2,i3/vector_length)[i3%vector_length], eps*aa(i0,i1,i2,i3) );
      }
    }
  }
}

template<typename DeviceType,typename VectorTagType,int VectorLength>
int test_batched_vector_view() {
  static_assert(Kokkos::Impl::SpaceAccessibility<DeviceType,Kokkos::HostSpace >::accessible,
                "vector datatype is only tested on host space");
  Test::impl_test_batched_vector_view<VectorTagType,VectorLength>();
  
  return 0;
}


///
/// SIMD
///

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F( TestCategory, batched_vector_view_simd_float8 ) {
  test_batched_vector_view<TestExecSpace,SIMD<float>,8>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F( TestCategory, batched_vector_view_simd_double4 ) {
  test_batched_vector_view<TestExecSpace,SIMD<double>,4>();
}
#endif

// #if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
// TEST_F( TestCategory, batched_vector_view_simd_dcomplex2 ) {
//   test_batched_vector_view<TestExecSpace,SIMD<Kokkos::complex<double> >,2>();
// }
// #endif
