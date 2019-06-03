#include<gtest/gtest.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas1_copy.hpp>
#include<KokkosKernels_TestUtils.hpp>

namespace Test {
  template<class ViewTypeA, class ViewTypeB, class ViewTypeC, class ViewTypeD, class Device>
  void impl_test_copy(int N) {

    typedef typename ViewTypeA::value_type Scalar;

    typedef Kokkos::View<Scalar*[12],
                         typename std::conditional<
                            std::is_same<typename ViewTypeA::array_layout,Kokkos::LayoutStride>::value,
                            Kokkos::LayoutRight, Kokkos::LayoutLeft>::type,
                         Device> BaseTypeA;
    typedef Kokkos::View<Scalar*[12],
                         typename std::conditional<
                            std::is_same<typename ViewTypeB::array_layout,Kokkos::LayoutStride>::value,
                            Kokkos::LayoutRight, Kokkos::LayoutLeft>::type,
                         Device> BaseTypeB;
    typedef Kokkos::View<Scalar*[12],
                         typename std::conditional<
                            std::is_same<typename ViewTypeC::array_layout,Kokkos::LayoutStride>::value,
                            Kokkos::LayoutRight, Kokkos::LayoutLeft>::type,
                         Device> BaseTypeC;
    typedef Kokkos::View<Scalar*[12],
                         typename std::conditional<
                            std::is_same<typename ViewTypeD::array_layout,Kokkos::LayoutStride>::value,
                            Kokkos::LayoutRight, Kokkos::LayoutLeft>::type,
                         Device> BaseTypeD;

    BaseTypeA b_A("b_A",N);
    BaseTypeB b_B("b_B",N);
    BaseTypeC b_C("b_C",N);
    BaseTypeD b_D("b_D",N);
    
    ViewTypeA s_A = Kokkos::subview(b_A,Kokkos::ALL(),0);
    ViewTypeB s_B = Kokkos::subview(b_B,Kokkos::ALL(),1);
    ViewTypeC s_C = Kokkos::subview(b_C,Kokkos::ALL(),2);
    ViewTypeD s_D = Kokkos::subview(b_D,Kokkos::ALL(),3);

    typename BaseTypeA::HostMirror h_b_A = Kokkos::create_mirror_view(b_A);
    typename BaseTypeB::HostMirror h_b_B = Kokkos::create_mirror_view(b_B);
    typename BaseTypeC::HostMirror h_b_C = Kokkos::create_mirror_view(b_C);
    typename BaseTypeD::HostMirror h_b_D = Kokkos::create_mirror_view(b_D);

    typename ViewTypeA::HostMirror h_s_A = Kokkos::subview(h_b_A,Kokkos::ALL(),0);
    typename ViewTypeB::HostMirror h_s_B = Kokkos::subview(h_b_B,Kokkos::ALL(),1);
    typename ViewTypeC::HostMirror h_s_C = Kokkos::subview(h_b_C,Kokkos::ALL(),2);
    typename ViewTypeD::HostMirror h_s_D = Kokkos::subview(h_b_D,Kokkos::ALL(),3);

    Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);

    Kokkos::fill_random(b_A,rand_pool,Scalar(1));
    Kokkos::fill_random(b_B,rand_pool,Scalar(1));
    Kokkos::fill_random(b_C,rand_pool,Scalar(1));
    Kokkos::fill_random(b_D,rand_pool,Scalar(1));
	
    Kokkos::fence();

    //Src B: LayoutLeft --> Dst A:
    KokkosBlas::copy(s_B, s_A);

    Kokkos::deep_copy(h_b_A, b_A);
    Kokkos::deep_copy(h_b_B, b_B);

    {
      bool test_flag = true;
      for(int i=0;i<N;i++)
      if(h_s_A(i)!=h_s_B(i)) { test_flag=false; break;}
      ASSERT_EQ( test_flag, true );
    }

    //Src C: LayoutRight --> Dst A:
    KokkosBlas::copy(s_C, s_A);
	
    Kokkos::deep_copy(h_b_A, b_A);
    Kokkos::deep_copy(h_b_C, b_C);

    {
      bool test_flag = true;
      for(int i=0;i<N;i++)
      if(h_s_A(i)!=h_s_C(i)) { test_flag=false; break;}
      ASSERT_EQ( test_flag, true );
    }

    //Src D: LayoutStride --> Dst A:
    KokkosBlas::copy(s_D, s_A);
	
    Kokkos::deep_copy(h_b_A, b_A);
    Kokkos::deep_copy(h_b_D, b_D);

    {
      bool test_flag = true;
      for(int i=0;i<N;i++)
      if(h_s_A(i)!=h_s_D(i)) { test_flag=false; break;}
      ASSERT_EQ( test_flag, true );
    }
  }//impl_test_copy

  template<class ViewTypeA, class ViewTypeB, class ViewTypeC, class ViewTypeD, class Device>
  void impl_test_copy_mv(int N, int K) {

    typedef typename ViewTypeA::value_type Scalar;

    typedef multivector_layout_adapter<ViewTypeA> vfA_type;
    typedef multivector_layout_adapter<ViewTypeB> vfB_type;
    typedef multivector_layout_adapter<ViewTypeC> vfC_type;
    typedef multivector_layout_adapter<ViewTypeD> vfD_type;

    typename vfA_type::BaseType b_A("A",N,K);
    typename vfB_type::BaseType b_B("B",N,K);
    typename vfC_type::BaseType b_C("C",N,K);
    typename vfD_type::BaseType b_D("D",N,K);

    ViewTypeA A = vfA_type::view(b_A);
    ViewTypeB B = vfB_type::view(b_B);
    ViewTypeC C = vfC_type::view(b_C);
    ViewTypeD D = vfD_type::view(b_D);

    typedef multivector_layout_adapter<typename ViewTypeA::HostMirror> h_vfA_type;
    typedef multivector_layout_adapter<typename ViewTypeB::HostMirror> h_vfB_type;
    typedef multivector_layout_adapter<typename ViewTypeC::HostMirror> h_vfC_type;
    typedef multivector_layout_adapter<typename ViewTypeD::HostMirror> h_vfD_type;

    typename h_vfA_type::BaseType h_b_A = Kokkos::create_mirror_view(b_A);
    typename h_vfB_type::BaseType h_b_B = Kokkos::create_mirror_view(b_B);
    typename h_vfC_type::BaseType h_b_C = Kokkos::create_mirror_view(b_C);
    typename h_vfD_type::BaseType h_b_D = Kokkos::create_mirror_view(b_D);

    typename ViewTypeA::HostMirror h_A = h_vfA_type::view(h_b_A);
    typename ViewTypeB::HostMirror h_B = h_vfB_type::view(h_b_B);
    typename ViewTypeC::HostMirror h_C = h_vfC_type::view(h_b_C);
    typename ViewTypeD::HostMirror h_D = h_vfD_type::view(h_b_D);
	
    Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);

    Kokkos::fill_random(b_A,rand_pool,Scalar(1));
    Kokkos::fill_random(b_B,rand_pool,Scalar(1));
    Kokkos::fill_random(b_C,rand_pool,Scalar(1));
    Kokkos::fill_random(b_D,rand_pool,Scalar(1));

    Kokkos::fence();

    //Src B: LayoutLeft --> Dst A:
    KokkosBlas::copy(B, A);

    Kokkos::deep_copy(h_b_A, b_A);
    Kokkos::deep_copy(h_b_B, b_B);

    {
      bool test_flag = true;
      for(int i=0;i<N;i++)
        for(int j=0;j<K;j++)
          if(h_A(i,j)!=h_B(i,j)) { test_flag=false; break;}
      ASSERT_EQ( test_flag, true );
    }

    //Src C: LayoutRight --> Dst A:
    KokkosBlas::copy(C, A);

    Kokkos::deep_copy(h_b_A, b_A);
    Kokkos::deep_copy(h_b_C, b_C);

    {
      bool test_flag = true;
      for(int i=0;i<N;i++)
        for(int j=0;j<K;j++)
          if(h_A(i,j)!=h_C(i,j)) { test_flag=false; break;}
      ASSERT_EQ( test_flag, true );
    }

    //Src D: LayoutStride --> Dst A:
    KokkosBlas::copy(D, A);

    Kokkos::deep_copy(h_b_A, b_A);
    Kokkos::deep_copy(h_b_D, b_D);

    {
      bool test_flag = true;
      for(int i=0;i<N;i++)
        for(int j=0;j<K;j++)
          if(h_A(i,j)!=h_D(i,j)) { test_flag=false; break;}
      ASSERT_EQ( test_flag, true );
    }
  }//impl_test_copy_mv
}//namespace Test



template<class Scalar, class Device>
int test_copy() {
  typedef Kokkos::View<Scalar*, Kokkos::LayoutLeft,   Device> view_type_b_ll;
  typedef Kokkos::View<Scalar*, Kokkos::LayoutRight,  Device> view_type_c_lr;
  typedef Kokkos::View<Scalar*, Kokkos::LayoutStride, Device> view_type_d_ls;
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  //printf("test_copy -- LayoutLeft\n");
  typedef Kokkos::View<Scalar*, Kokkos::LayoutLeft, Device> view_type_a_ll;
  Test::impl_test_copy<view_type_a_ll, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(0);
  Test::impl_test_copy<view_type_a_ll, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(61);
  Test::impl_test_copy<view_type_a_ll, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(1024);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  //printf("test_copy -- LayoutRight\n");
  typedef Kokkos::View<Scalar*, Kokkos::LayoutRight, Device> view_type_a_lr;
  Test::impl_test_copy<view_type_a_lr, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(0);
  Test::impl_test_copy<view_type_a_lr, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(61);
  Test::impl_test_copy<view_type_a_lr, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(1024);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTSTRIDE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  //printf("test_copy -- LayoutStride\n");
  typedef Kokkos::View<Scalar*, Kokkos::LayoutStride, Device> view_type_a_ls;
  Test::impl_test_copy<view_type_a_ls, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(0);
  Test::impl_test_copy<view_type_a_ls, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(61);
  Test::impl_test_copy<view_type_a_ls, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(1024);
#endif

  return 1;
}

template<class Scalar, class Device>
int test_copy_mv() {
  typedef Kokkos::View<Scalar**, Kokkos::LayoutLeft,   Device> view_type_b_ll;
  typedef Kokkos::View<Scalar**, Kokkos::LayoutRight,  Device> view_type_c_lr;
  typedef Kokkos::View<Scalar**, Kokkos::LayoutStride, Device> view_type_d_ls;
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  //printf("test_copy_mv -- LayoutLeft\n");
  typedef Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device> view_type_a_ll;
  Test::impl_test_copy_mv<view_type_a_ll, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(0,5);
  Test::impl_test_copy_mv<view_type_a_ll, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(61,5);
  Test::impl_test_copy_mv<view_type_a_ll, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(1024,5);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  //printf("test_copy_mv -- LayoutRight\n");
  typedef Kokkos::View<Scalar**, Kokkos::LayoutRight, Device> view_type_a_lr;
  Test::impl_test_copy_mv<view_type_a_lr, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(0,5);
  Test::impl_test_copy_mv<view_type_a_lr, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(61,5);
  Test::impl_test_copy_mv<view_type_a_lr, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(1024,5);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTSTRIDE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  //printf("test_copy_mv -- LayoutStride\n");
  typedef Kokkos::View<Scalar**, Kokkos::LayoutStride, Device> view_type_a_ls;
  Test::impl_test_copy_mv<view_type_a_ls, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(0,5);
  Test::impl_test_copy_mv<view_type_a_ls, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(61,5);
  Test::impl_test_copy_mv<view_type_a_ls, view_type_b_ll, view_type_c_lr, view_type_d_ls, Device>(1024,5);
#endif

  return 1;
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, copy_float ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::copy_float"); 
    test_copy<float,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
TEST_F( TestCategory, copy_mv_float ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::copy_mv_float"); 
    test_copy_mv<float,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, copy_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::copy_double"); 
    test_copy<double,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
TEST_F( TestCategory, copy_mv_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::copy_mv_double"); 
    test_copy_mv<double,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, copy_complex_float ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::copy_complex_float"); 
    test_copy<Kokkos::complex<float>,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
TEST_F( TestCategory, copy_mv_complex_float ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::copy_mv_complex_float"); 
    test_copy_mv<Kokkos::complex<float>,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, copy_complex_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::copy_complex_double"); 
    test_copy<Kokkos::complex<double>,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
TEST_F( TestCategory, copy_mv_complex_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::copy_mv_complex_double"); 
    test_copy_mv<Kokkos::complex<double>,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
#endif

//#if defined(KOKKOSKERNELS_INST_INT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
//TEST_F( TestCategory, copy_int ) {
//  Kokkos::Profiling::pushRegion("KokkosBlas::Test::copy_int"); 
//    test_copy<int,TestExecSpace> ();
//  Kokkos::Profiling::popRegion();
//}
//TEST_F( TestCategory, copy_mv_int ) {
//  Kokkos::Profiling::pushRegion("KokkosBlas::Test::copy_mv_int"); 
//    test_copy_mv<int,TestExecSpace> ();
//  Kokkos::Profiling::popRegion();
//}
//#endif
