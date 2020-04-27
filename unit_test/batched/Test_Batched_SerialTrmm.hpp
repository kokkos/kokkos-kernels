#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

#include "KokkosBatched_Trmm_Decl.hpp"
#include "KokkosBatched_Trmm_Serial_Impl.hpp"

#include "KokkosKernels_TestUtils.hpp"

using namespace KokkosBatched;

namespace Test {

  template<class ViewTypeA, class ViewTypeB, class ViewTypeC, class ExecutionSpace>
  struct VanillaGEMM {
    bool A_t, B_t, A_c, B_c;
    int N,K;
    ViewTypeA A;
    ViewTypeB B;
    ViewTypeC C;

    typedef typename ViewTypeA::value_type ScalarA;
    typedef typename ViewTypeB::value_type ScalarB;
    typedef typename ViewTypeC::value_type ScalarC;
    typedef Kokkos::Details::ArithTraits<ScalarC> APT;
    typedef typename APT::mag_type mag_type;
    ScalarA alpha;
    ScalarC beta;

    KOKKOS_INLINE_FUNCTION
    void operator() (const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type& team) const {
// GNU COMPILER BUG WORKAROUND
#if defined(KOKKOS_COMPILER_GNU) && !defined(__CUDA_ARCH__)
      int i = team.league_rank();
#else
      const int i = team.league_rank();
#endif
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,N), [&] (const int& j) {
        ScalarC C_ij = 0.0;

        // GNU 5.3, 5.4 and 6.1 (and maybe more) crash with another nested lambda here

#if defined(KOKKOS_COMPILER_GNU) && !defined(KOKKOS_COMPILER_NVCC)
        for(int k=0; k<K; k++) {
          ScalarA A_ik = A_t?(A_c?APT::conj(A(k,i)):A(k,i)):A(i,k);
          ScalarB B_kj = B_t?(B_c?APT::conj(B(j,k)):B(j,k)):B(k,j);
          C_ij += A_ik*B_kj;
        }
#else
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,K), [&] (const int& k, ScalarC& lsum) {
           ScalarA A_ik = A_t?(A_c?APT::conj(A(k,i)):A(k,i)):A(i,k);
           ScalarB B_kj = B_t?(B_c?APT::conj(B(j,k)):B(j,k)):B(k,j);
           lsum += A_ik*B_kj;
        },C_ij);
#endif

        C(i,j) = beta*C(i,j) + alpha*C_ij;
      });
    }
  };

  template<typename S, typename U, typename T, typename D>
  struct ParamTag {
    typedef S side;
    typedef U uplo;
    typedef T trans;
    typedef D diag;
  };

  template<typename DeviceType,
           typename ViewType,
           typename ScalarType,
           typename ParamTagType,
           typename AlgoTagType>
  struct Functor_TestBatchedSerialTrmm {
    ViewType _a, _b;
    
    ScalarType _alpha;

    KOKKOS_INLINE_FUNCTION
    Functor_TestBatchedSerialTrmm(const ScalarType alpha, 
            const ViewType &a,
            const ViewType &b) 
      : _a(a), _b(b), _alpha(alpha) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const ParamTagType &, const int k) const {
      auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
      auto bb = Kokkos::subview(_b, k, Kokkos::ALL(), Kokkos::ALL());

      SerialTrmm<typename ParamTagType::side,
        typename ParamTagType::uplo,
        typename ParamTagType::trans,
        typename ParamTagType::diag,
        AlgoTagType>::
        invoke(_alpha, aa, bb);
    }

    inline
    void run() {
      typedef typename ViewType::value_type value_type;
      std::string name_region("KokkosBatched::Test::SerialTrmm");
      std::string name_value_type = ( std::is_same<value_type,float>::value ? "::Float" : 
                                      std::is_same<value_type,double>::value ? "::Double" :
                                      std::is_same<value_type,Kokkos::complex<float> >::value ? "::ComplexFloat" :
                                      std::is_same<value_type,Kokkos::complex<double> >::value ? "::ComplexDouble" : "::UnknownValueType" );                               
      std::string name = name_region + name_value_type;
      Kokkos::Profiling::pushRegion( name.c_str() );
      Kokkos::RangePolicy<DeviceType,ParamTagType> policy(0, _b.extent(0));
      Kokkos::parallel_for(name.c_str(), policy, *this);
      Kokkos::Profiling::popRegion();
    }
  };

  template<typename DeviceType,
           typename ViewType,
           typename ScalarType,
           typename ParamTagType,
           typename AlgoTagType>
  void impl_test_batched_trmm(const int N, const int nRows, const int nCols, const char *trans) {
    typedef typename ViewType::value_type value_type;
    typedef Kokkos::Details::ArithTraits<value_type> ats;

    ScalarType alpha(1.0);
    ScalarType beta(1.0);

    const bool is_side_right = std::is_same<typename ParamTagType::side,Side::Right>::value;
    const int K = is_side_right ? nCols : nRows;
    ViewType
      A("A", N, K, K),
      B_actual("B_actual", N, nRows, nCols), 
      B_expected("B_expected", N, nRows, nCols);

    Kokkos::Random_XorShift64_Pool<typename DeviceType::execution_space> random(13718);
    Kokkos::fill_random(A, random, value_type(1.0));
    Kokkos::fill_random(B_actual, random, value_type(1.0));

    Kokkos::deep_copy(B_expected, B_actual);
    Kokkos::fence();

    if (!is_side_right){
      // B_expected = alpha * op(A) * B + beta * C = 1 * op(A) * B + 0 * C
      struct VanillaGEMM<ViewTypeB,ViewTypeA,ViewTypeB,execution_space> vgemm;
      vgemm.A_t = (trans[0]!='N') && (trans[0]!='n'); vgemm.B_t = false;
      vgemm.A_c = (trans[0]=='C') || (trans[0]=='c'); vgemm.B_c = false;
      vgemm.N = N;    vgemm.K = K;
      vgemm.alpha = alpha;
      vgemm.beta = beta;
      for (int i = 0; i < N; i++) {
        vgemm.A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL()); 
        vgemm.B = Kokkos::subview(B_actual, i, Kokkos::ALL(), Kokkos::ALL());;
        vgemm.C = Kokkos::subview(B_expected, i, Kokkos::ALL(), Kokkos::ALL());;
        Kokkos::parallel_for("KokkosBlas::Test::VanillaGEMM", Kokkos::TeamPolicy<execution_space>(nRows,Kokkos::AUTO,16), vgemm);
      }
    }
    else {
      // B_expected = alpha * B * op(A) + beta * C = 1 * B * op(A) + 0 * C
      struct VanillaGEMM<ViewTypeB,ViewTypeA,ViewTypeB,execution_space> vgemm;
      vgemm.A_t = false; vgemm.B_t = (trans[0]!='N') && (trans[0]!='n');
      vgemm.A_c = false; vgemm.B_c = (trans[0]=='C') || (trans[0]=='c');
      vgemm.N = N;     vgemm.K = K;
      vgemm.alpha = alpha;
      vgemm.beta = beta;
      for (int i = 0; i < N; i++) {
        vgemm.A = Kokkos::subview(B_actual, i, Kokkos::ALL(), Kokkos::ALL()); 
        vgemm.B = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());;
        vgemm.C = Kokkos::subview(B_expected, i, Kokkos::ALL(), Kokkos::ALL());;
        Kokkos::parallel_for("KokkosBlas::Test::VanillaGEMM", Kokkos::TeamPolicy<execution_space>(nRows,Kokkos::AUTO,16), vgemm);
      }
    }

    Functor_TestBatchedSerialTrmm<DeviceType,ViewType,ScalarType,
      ParamTagType,Algo::Trmm::Unblocked>(alpha, A, B_actual).run();

    Kokkos::fence();
    Kokkos::deep_copy(host_B_expected, B_expected);

    Kokkos::fence();

    /// for comparison send it to host
    typename ViewType::HostMirror B_actual_host = Kokkos::create_mirror_view(B_actual);
    typename ViewType::HostMirror B_explected_host = Kokkos::create_mirror_view(B_expected);

    Kokkos::deep_copy(B_actual_host, B);
    Kokkos::deep_copy(B_explected_host, B_expected);

    /// check b0 = b1 ; this eps is about 10^-14
    typedef typename ats::mag_type mag_type;
    const mag_type eps = 1.0e3 * ats::epsilon();
    bool fail_flag = false;

    for (int k=0;k<N;++k) {
      for (int i=0;i<nRows;++i) {
        for (int j=0;j<nCols;++j) {
          if (ats::abs(B_actual_host(k,i,j)-B_explected_host(k,i,j)) > eps) {
            fail_flag = true;
          }
        }
      }
    }

    if (fail_flag)
      ASSERT_EQ( fail_flag, false );
  }
}


template<typename DeviceType,
         typename ValueType,
         typename ScalarType,
         typename ParamTagType,
         typename AlgoTagType,
         typename ParamTransType>
int test_batched_trmm() {
  char *trans = std::is_same<typename ParamTagType::side,Trans::NoTranspose>::value ? "N" :
                std::is_same<typename ParamTagType::side,Trans::Transpose>::value; ? "T" :
                std::is_same<typename ParamTagType::side,Trans::ConjTranspose>::value; ? "C" : "E"
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    typedef Kokkos::View<ValueType***,Kokkos::LayoutLeft,DeviceType> ViewType;
    Test::impl_test_batched_trmm<DeviceType,ViewType,ScalarType,ParamTagType,AlgoTagType>(     0, 10, 4, trans);
    for (int i=0;i<10;++i) {
      //printf("Testing: LayoutLeft,  Blksize %d\n", i);  
      Test::impl_test_batched_trmm<DeviceType,ViewType,ScalarType,ParamTagType,AlgoTagType>(1024,  i, 4, trans);
      Test::impl_test_batched_trmm<DeviceType,ViewType,ScalarType,ParamTagType,AlgoTagType>(1024,  i, 1, trans);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    typedef Kokkos::View<ValueType***,Kokkos::LayoutRight,DeviceType> ViewType;
    Test::impl_test_batched_trmm<DeviceType,ViewType,ScalarType,ParamTagType,AlgoTagType>(     0, 10, 4, trans);
    for (int i=0;i<10;++i) {
      //printf("Testing: LayoutRight, Blksize %d\n", i);  
      Test::impl_test_batched_trmm<DeviceType,ViewType,ScalarType,ParamTagType,AlgoTagType>(1024,  i, 4, trans);
      Test::impl_test_batched_trmm<DeviceType,ViewType,ScalarType,ParamTagType,AlgoTagType>(1024,  i, 1), trans;
    }
  }
#endif

  return 0;
}

