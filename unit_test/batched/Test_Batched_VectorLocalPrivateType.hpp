/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

#include "KokkosBatched_Vector.hpp"

#include "KokkosKernels_TestUtils.hpp"

using namespace KokkosBatched::Experimental;

namespace Test {

  struct TestScalarViewTag_Var1 {};
  struct TestScalarViewTag_Var2 {};

  struct TestVectorViewTag_Var1 {};
  struct TestVectorViewTag_Var2 {};

  template<typename DeviceType, 
           typename ScalarType, int VectorLength>
  struct LocalTypeExample {
    typedef ScalarType scalar_type;
    typedef Vector<SIMD<ScalarType>,VectorLength> vector_type;

    typedef Kokkos::View<scalar_type***,DeviceType> scalar_view_type;
    typedef Kokkos::View<vector_type***,DeviceType> vector_view_type;

    scalar_view_type _As, _Bs, _Cs;
    vector_view_type _Av, _Bv, _Cv;

    KOKKOS_INLINE_FUNCTION
    LocalTypeExample() :
      _As(), _Bs(), _Cs(),
      _Av(), _Bv(), _Cv() {}
    
    KOKKOS_INLINE_FUNCTION
    void 
    setScalarView(const scalar_view_type As, 
                  const scalar_view_type Bs,
                  const scalar_view_type Cs) {
      _As = As; _Bs = Bs; _Cs = Cs;
    }

    KOKKOS_INLINE_FUNCTION
    void
    setVectorView(const vector_view_type Av,
                  const vector_view_type Bv,
                  const vector_view_type Cv) {
      _Av = Av; _Bv = Bv; _Cv = Cv;
    }

    // range policy with scalar for reference result
    KOKKOS_INLINE_FUNCTION
    void operator()(const int &k) const {
      auto aa = Kokkos::subview(_As, k, Kokkos::ALL(), Kokkos::ALL());
      auto bb = Kokkos::subview(_Bs, k, Kokkos::ALL(), Kokkos::ALL());
      auto cc = Kokkos::subview(_Cs, k, Kokkos::ALL(), Kokkos::ALL());

      const int m = cc.dimension_0();
      const int n = cc.dimension_1();
      const int q = aa.dimension_1();

      for (int i=0;i<m;++i)
        for (int j=0;j<n;++j) {
          scalar_type cval(0);
          for (int p=0;p<q;++p)
            cval += aa(i,p)*bb(p,j);
          cc(i,j) += cval;
        }
    }
    
    // team policy with scalar (nested parallel)
    // - with a proper view casting, this works with scalar view and vector view
    // - since it needs to use vector parallelism explicitly, this approach cannot 
    //   be used for both host and gpu.
    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const TestScalarViewTag_Var1 &, const MemberType &member) const {
      const int k = member.league_rank();

      auto aa = Kokkos::subview(_As, k, Kokkos::ALL(), Kokkos::ALL());
      auto bb = Kokkos::subview(_Bs, k, Kokkos::ALL(), Kokkos::ALL());
      auto cc = Kokkos::subview(_Cs, k, Kokkos::ALL(), Kokkos::ALL());

      const int m = cc.dimension_0();
      const int n = cc.dimension_1();
      const int q = aa.dimension_1();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n),
                          [&](const int j) {
                            Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, m),
                                                 [&](const int i) {
                                                   scalar_type cval(0);
                                                   for (int p=0;p<q;++p)
                                                     cval += aa(i,p)*bb(p,j);
                                                   cc(i,j) += cval;
                                                 });
                           });
    }

    // team policy with vector loop first
    // - with layout left, this gives coalecsed access to the batched matrices.
    // - a single function body can be used for both scalar and vector

    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const TestScalarViewTag_Var2 &, const MemberType &member) const {
      const int kbeg = member.league_rank()*VectorLength;

      // when implicit vectorization is used, the thread vector range should be replaced with a 
      // dummy loop (iteration 1); kbeg should also be recomputed
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, VectorLength),
                           [&](const int kk) {
                             const int k = kbeg + kk;

                             auto aa = Kokkos::subview(_As, k, Kokkos::ALL(), Kokkos::ALL());
                             auto bb = Kokkos::subview(_Bs, k, Kokkos::ALL(), Kokkos::ALL());
                             auto cc = Kokkos::subview(_Cs, k, Kokkos::ALL(), Kokkos::ALL());
                             
                             const int m = cc.dimension_0();
                             const int n = cc.dimension_1();
                             const int q = aa.dimension_1();

                             Kokkos::parallel_for(Kokkos::TeamThreadRange(member, m*n),
                                                  [&](const int ij) {
                                                    const int i = ij%m;
                                                    const int j = ij/m;

                                                    scalar_type cval(0);
                                                    for (int p=0;p<q;++p) 
                                                      cval += aa(i,p)*bb(p,j);
                                                    cc(i,j) += cval;
                                                  });
                           });
    }

    // team policy with simd type (host version)
    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const TestVectorViewTag_Var1 &, const MemberType &member) const {
      const int k = member.league_rank();

      auto aa = Kokkos::subview(_Av, k, Kokkos::ALL(), Kokkos::ALL());
      auto bb = Kokkos::subview(_Bv, k, Kokkos::ALL(), Kokkos::ALL());
      auto cc = Kokkos::subview(_Cv, k, Kokkos::ALL(), Kokkos::ALL());
      
      const int m = cc.dimension_0();
      const int n = cc.dimension_1();
      const int q = aa.dimension_1();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, m*n),
                           [&](const int ij) {
                             const int i = ij%m;
                             const int j = ij/m;
                             
                             vector_type cval(0);
                             for (int p=0;p<q;++p) 
                               cval += aa(i,p)*bb(p,j);
                             cc(i,j) += cval;
                           });
    }

    // team policy with simd type (cuda version)
    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const TestVectorViewTag_Var2 &, const MemberType &member) const {
      const int k = member.league_rank();

      auto aa = Kokkos::subview(_Av, k, Kokkos::ALL(), Kokkos::ALL());
      auto bb = Kokkos::subview(_Bv, k, Kokkos::ALL(), Kokkos::ALL());
      auto cc = Kokkos::subview(_Cv, k, Kokkos::ALL(), Kokkos::ALL());
      
      const int m = cc.dimension_0();
      const int n = cc.dimension_1();
      const int q = aa.dimension_1();

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, VectorLength),
                           [&](const int kk) {
                             Kokkos::parallel_for(Kokkos::TeamThreadRange(member, m*n),
                                                  [&](const int ij) {
                                                    const int i = ij%m;
                                                    const int j = ij/m;
                                                    
                                                    scalar_type cval(0);
                                                    for (int p=0;p<q;++p) 
                                                      cval += aa(i,p)[kk]*bb(p,j)[kk];
                                                    cc(i,j)[kk] += cval;
                                                  });
                           });
    }

  };
  
  template<typename DeviceType, 
           typename ScalarType, int VectorLength>
  void impl_test_batched_vector_local_type() {
    
    const int Nv = 1024;
    const int Ns = 1024*VectorLength;
    const int B = 3;

    enum : int {  vector_length = VectorLength };

    typedef DeviceType device_type;
    typedef Kokkos::DefaultHostExecutionSpace host_type;

    typedef ScalarType scalar_type;
    typedef Vector<SIMD<scalar_type>,vector_length> vector_type;

    typedef Kokkos::View<scalar_type***,device_type> scalar_view_type_device; 
    typedef Kokkos::View<scalar_type***,typename scalar_view_type_device::array_layout,host_type> scalar_view_type_host; 

    typedef Kokkos::View<vector_type***,device_type> vector_view_type_device; 

    scalar_view_type_host   Aref("Aref", Ns, B, B), Bref("Bref", Ns, B, B), Cref("Cref", Ns, B, B);
    scalar_view_type_device As  ("As",   Ns, B, B), Bs  ("Bs",   Ns, B, B), Cs  ("Cs",   Ns, B, B);
    vector_view_type_device Av  ("Av",   Nv, B, B), Bv  ("Bv",   Nv, B, B), Cv  ("Cs",   Nv, B, B);

    {
      // view initialization for testing
      Random<ScalarType> random;
      for (int k=0;k<Ns;++k) 
        for (int i=0;i<B;++i)
          for (int j=0;j<B;++j) {
            Aref(k, i,j) = random.value();
            Bref(k, i,j) = random.value();
            Cref(k, i,j) = random.value();
          }
      
      Kokkos::deep_copy(As, Aref);
      Kokkos::deep_copy(Bs, Bref);
      Kokkos::deep_copy(Cs, Cref);
      
      auto Ah = Kokkos::create_mirror_view(Av);
      auto Bh = Kokkos::create_mirror_view(Bv);
      auto Ch = Kokkos::create_mirror_view(Cv);

      {
        SimdViewAccess<decltype(Ah),PackDim<0> > aa(Ah), bb(Bh), cc(Ch);
        for (int k=0;k<Ns;++k) 
          for (int i=0;i<B;++i)
            for (int j=0;j<B;++j) {
              aa(k,i,j) = Aref(k,i,j);
              bb(k,i,j) = Bref(k,i,j);
              cc(k,i,j) = Cref(k,i,j);
            }
      }
      Kokkos::deep_copy(Av, Ah);
      Kokkos::deep_copy(Bv, Bh);
      Kokkos::deep_copy(Cv, Ch);
    }

    {
      // compute reference on host with range policy
      LocalTypeExample<host_type,scalar_type,vector_length> functor;
      functor.setScalarView(Aref, Bref, Cref);
      Kokkos::parallel_for(Kokkos::RangePolicy<host_type>(0, Ns), functor);
    }

    ///
    /// Case 1 : team policy on device with usual scalar views
    ///
    {
      // back up
      scalar_view_type_device As2("As2",   Ns, B, B), Bs2  ("Bs2",   Ns, B, B), Cs2  ("Cs2",   Ns, B, B);      
      
      Kokkos::deep_copy(As2, As);
      Kokkos::deep_copy(Bs2, Bs);
      Kokkos::deep_copy(Cs2, Cs);

      // compute scalar team policy on device
      const int team_size = 5; 
      LocalTypeExample<device_type,scalar_type,vector_length> functor;
      functor.setScalarView(As, Bs, Cs);
      Kokkos::parallel_for(Kokkos::TeamPolicy<device_type,TestScalarViewTag_Var1>(Ns, team_size, vector_length),
                           functor);

      // check scalar team policy on device
      auto Ch = Kokkos::create_mirror_view(Cs);
      Kokkos::deep_copy(Ch, Cs);
      {
        typedef Kokkos::Details::ArithTraits<scalar_type> ats;
        const typename ats::mag_type eps = 1.0e3 * ats::epsilon();
        for (int k=0;k<Ns;++k) 
          for (int i=0;i<B;++i) 
            for (int j=0;j<B;++j) 
              EXPECT_NEAR_KK( Ch(k,i,j), Cref(k,i,j), eps );
      }

      // restore initial values for the next test
      Kokkos::deep_copy(As, As2);
      Kokkos::deep_copy(Bs, Bs2);
      Kokkos::deep_copy(Cs, Cs2);      
    }

    ///
    /// Case 2 : team policy on device with usual scalar views
    ///
    {
      // back up
      scalar_view_type_device As2("As2",   Ns, B, B), Bs2  ("Bs2",   Ns, B, B), Cs2  ("Cs2",   Ns, B, B);      
      
      Kokkos::deep_copy(As2, As);
      Kokkos::deep_copy(Bs2, Bs);
      Kokkos::deep_copy(Cs2, Cs);

      // compute scalar team policy on device
      const int team_size = 5; 
      LocalTypeExample<device_type,scalar_type,vector_length> functor;
      functor.setScalarView(As, Bs, Cs);
      Kokkos::parallel_for(Kokkos::TeamPolicy<device_type,TestScalarViewTag_Var2>(Nv, team_size, vector_length),
                           functor);

      // check scalar team policy on device
      auto Ch = Kokkos::create_mirror_view(Cs);
      Kokkos::deep_copy(Ch, Cs);
      {
        typedef Kokkos::Details::ArithTraits<scalar_type> ats;
        const typename ats::mag_type eps = 1.0e3 * ats::epsilon();
        for (int k=0;k<Ns;++k) 
          for (int i=0;i<B;++i) 
            for (int j=0;j<B;++j) 
              EXPECT_NEAR_KK( Ch(k,i,j), Cref(k,i,j), eps );
      }

      // restore initial values for the next test
      Kokkos::deep_copy(As, As2);
      Kokkos::deep_copy(Bs, Bs2);
      Kokkos::deep_copy(Cs, Cs2);      
    }

    ///
    /// Case 3 : team policy on device with vector views
    ///
    {
      // back up
      vector_view_type_device Av2("Av2",   Nv, B, B), Bv2  ("Bv2",   Nv, B, B), Cv2  ("Cv2",   Nv, B, B);      
      
      Kokkos::deep_copy(Av2, Av);
      Kokkos::deep_copy(Bv2, Bv);
      Kokkos::deep_copy(Cv2, Cv);

      // compute scalar team policy on device
      const int team_size = 5; 
      LocalTypeExample<device_type,scalar_type,vector_length> functor;
      functor.setVectorView(Av, Bv, Cv);
      Kokkos::parallel_for(Kokkos::TeamPolicy<device_type,TestVectorViewTag_Var1>(Nv, team_size, vector_length),
                           functor);

      // check scalar team policy on device
      auto Ch = Kokkos::create_mirror_view(Cv);
      Kokkos::deep_copy(Ch, Cv);
      {
        typedef Kokkos::Details::ArithTraits<scalar_type> ats;
        const typename ats::mag_type eps = 1.0e3 * ats::epsilon();
        SimdViewAccess<decltype(Ch),PackDim<0> > cc(Ch);        
        for (int k=0;k<Ns;++k) 
          for (int i=0;i<B;++i) 
            for (int j=0;j<B;++j) 
              EXPECT_NEAR_KK( cc(k,i,j), Cref(k,i,j), eps );
      }

      // restore initial values for the next test
      Kokkos::deep_copy(Av, Av2);
      Kokkos::deep_copy(Bv, Bv2);
      Kokkos::deep_copy(Cv, Cv2);      
    }

    ///
    /// Case 4 : team policy on device with vector views
    ///
    {
      // back up
      vector_view_type_device Av2("Av2",   Nv, B, B), Bv2  ("Bv2",   Nv, B, B), Cv2  ("Cv2",   Nv, B, B);      
      
      Kokkos::deep_copy(Av2, Av);
      Kokkos::deep_copy(Bv2, Bv);
      Kokkos::deep_copy(Cv2, Cv);

      // compute scalar team policy on device
      const int team_size = 5; 
      LocalTypeExample<device_type,scalar_type,vector_length> functor;
      functor.setVectorView(Av, Bv, Cv);
      Kokkos::parallel_for(Kokkos::TeamPolicy<device_type,TestVectorViewTag_Var2>(Nv, team_size, vector_length),
                           functor);

      // check scalar team policy on device
      auto Ch = Kokkos::create_mirror_view(Cv);
      Kokkos::deep_copy(Ch, Cv);
      {
        typedef Kokkos::Details::ArithTraits<scalar_type> ats;
        const typename ats::mag_type eps = 1.0e3 * ats::epsilon();
        SimdViewAccess<decltype(Ch),PackDim<0> > cc(Ch);        
        for (int k=0;k<Ns;++k) 
          for (int i=0;i<B;++i) 
            for (int j=0;j<B;++j) 
              EXPECT_NEAR_KK( cc(k,i,j), Cref(k,i,j), eps );
      }

      // restore initial values for the next test
      Kokkos::deep_copy(Av, Av2);
      Kokkos::deep_copy(Bv, Bv2);
      Kokkos::deep_copy(Cv, Cv2);      
    }
  }
}

template<typename DeviceType,typename ScalarType, int VectorLength>
int test_batched_vector_local_type() {
  Test::impl_test_batched_vector_local_type<DeviceType,ScalarType,VectorLength>();
  
  return 0;
}



#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F( TestCategory, batched_vector_local_privae_type_simd_float ) {
  test_batched_vector_local_type<TestExecSpace,float,8>();
  test_batched_vector_local_type<TestExecSpace,float,8>();
  test_batched_vector_local_type<TestExecSpace,float,9>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F( TestCategory, batched_vector_local_private_type_simd_double ) {
  test_batched_vector_local_type<TestExecSpace,double,4>();
  test_batched_vector_local_type<TestExecSpace,double,4>();
  test_batched_vector_local_type<TestExecSpace,double,13>();
}
#endif

// #if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
// TEST_F( TestCategory, batched_vector_view_simd_scomplex4 ) {
//   test_batched_vector_view<TestExecSpace,SIMD<Kokkos::complex<float> >,4>();
// }
// #endif

// #if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
// TEST_F( TestCategory, batched_vector_view_simd_dcomplex2 ) {
//   test_batched_vector_view<TestExecSpace,SIMD<Kokkos::complex<double> >,2>();
// }
// #endif
