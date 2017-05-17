/// \author Kyungjoo Kim (kyukim@sandia.gov)


#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"

#include "KokkosKernels_Vector.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {
      namespace ArithmaticTest {
    
        enum : int  { TEST_ADD = 0,
                      TEST_SUBTRACT = 1,
                      TEST_MULT = 2,
                      TEST_DIV = 3,
                      TEST_UNARY_MINUS = 4,
                      TEST_MAX = 5 };
    
        template<typename ViewType, int TestID>
        struct Functor {
          ViewType _a, _b, _c;

          KOKKOS_INLINE_FUNCTION
          Functor(const ViewType &a, 
                  const ViewType &b, 
                  const ViewType &c) 
            : _a(a), _b(b), _c(c) {}
      
          inline
          const char* label() const {
            switch (TestID) {
            case TEST_ADD: return "add"; break;
            case TEST_SUBTRACT: return "subtract";  break;
            case TEST_MULT: return "multiply";  break;
            case TEST_DIV: return "divide";  break;
            case TEST_UNARY_MINUS: return "unary minus"; break;
            }
            return "nothing";
          }

          KOKKOS_INLINE_FUNCTION
          void operator()(const int i) const {
            switch (TestID) {
            case TEST_ADD: _c(i) = _a(i) + _b(i); break;
            case TEST_SUBTRACT: _c(i) = _a(i) - _b(i); break;
            case TEST_MULT: _c(i) = _a(i) * _b(i); break;
            case TEST_DIV: _c(i) = _a(i) / _b(i); break;
            case TEST_UNARY_MINUS: _c(i) = -_c(i); break;
            }
          }

          inline
          int run() {
            Kokkos::RangePolicy<DeviceSpaceType> policy(0, _c.dimension_0());
            Kokkos::parallel_for(policy, *this);
          }
      
        };

        template<typename VectorTagType, int TestID>
        void VectorArithmatic() {
          enum : int {
            N = 32768,
          };
            
          const int iter_begin = -3, iter_end = 10;
          Kokkos::Impl::Timer timer;
          double t = 0;

          ///
          /// random data initialization
          ///
          typedef Vector<VectorTagType> vector_type;
          constexpr int vector_length = vector_type::vector_length;
          typedef typename vector_type::value_type scalar_type;

          Kokkos::View<scalar_type*,HostSpaceType> 
            a_host("a_host", N), 
            b_host("b_host", N), 
            c_host("c_host", N);


          Kokkos::View<vector_type*,HostSpaceType> 
            avec_host("avec_host", N/vector_length), 
            bvec_host("bvec_host", N/vector_length), 
            cvec_host("cvec_host", N/vector_length);
      
          Random<scalar_type> random;
          for (int k=0;k<N;++k) {
            a_host(k) = random.value();
            b_host(k) = random.value();
            c_host(k) = random.value();

            const int i = k/vector_length, j = k%vector_length;
            avec_host(i)[j] = a_host(k);
            bvec_host(i)[j] = b_host(k);
            cvec_host(i)[j] = c_host(k);
          }

          ///
          /// test for reference
          ///

          {
            auto aref = Kokkos::create_mirror_view(typename DeviceSpaceType::memory_space(), a_host);
            auto bref = Kokkos::create_mirror_view(typename DeviceSpaceType::memory_space(), b_host);
            auto cref = Kokkos::create_mirror_view(typename DeviceSpaceType::memory_space(), c_host);
        
            Kokkos::deep_copy(aref, a_host);
            Kokkos::deep_copy(bref, b_host);
            Kokkos::deep_copy(cref, c_host);
        
            {
              t = 0;
              Functor<decltype(cref),TestID> test(aref,bref,cref);
              for (int iter=iter_begin;iter<iter_end;++iter) {
                DeviceSpaceType::fence();
                timer.reset();
                test.run();
                DeviceSpaceType::fence();
                t += (iter >= 0)*timer.seconds();
              }
              printf("Reference,     Test %12s, Time %e\n", test.label(), (t/iter_end));
            }
            Kokkos::deep_copy(c_host, cref);

            auto avec = Kokkos::create_mirror_view(typename DeviceSpaceType::memory_space(), avec_host);
            auto bvec = Kokkos::create_mirror_view(typename DeviceSpaceType::memory_space(), bvec_host);
            auto cvec = Kokkos::create_mirror_view(typename DeviceSpaceType::memory_space(), cvec_host);
        
            Kokkos::deep_copy(avec, avec_host);
            Kokkos::deep_copy(bvec, bvec_host);
            Kokkos::deep_copy(cvec, cvec_host);
        
            {
              t = 0;
              Functor<decltype(cvec),TestID> test(avec,bvec,cvec);
              for (int iter=iter_begin;iter<iter_end;++iter) {
                DeviceSpaceType::fence();
                timer.reset();
                test.run();
                DeviceSpaceType::fence();
                t += (iter >= 0)*timer.seconds();
              }
              printf("%9s,%3d, Test %12s, Time %e\n", vector_type::label(), vector_length, test.label(), (t/iter_end));
            }
        
            Kokkos::deep_copy(cvec_host, cvec);
          }
      
          ///
          /// check cref = cvec
          ///

          double diff = 0, sum = 0;
          for (int k=0;k<N;++k) {
            const int i = k/vector_length, j = k%vector_length;
            sum  += abs(c_host(k));
            diff += abs(c_host(k) - cvec_host(i)[j]);
          }
          EXPECT_TRUE((diff/sum) < std::numeric_limits<double>::epsilon());
        }
      }
    }
  }
}

using namespace KokkosKernels::Batched::Experimental;

///
/// double vector length 4
///

TEST( VectorArithmatic, double_SIMD4 ) {
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>, 4>,ArithmaticTest::TEST_ADD>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>, 4>,ArithmaticTest::TEST_SUBTRACT>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>, 4>,ArithmaticTest::TEST_MULT>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>, 4>,ArithmaticTest::TEST_DIV>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>, 4>,ArithmaticTest::TEST_UNARY_MINUS>();
}

TEST( VectorArithmatic, dcomplex_SIMD2 ) {
  typedef Kokkos::complex<double> dcomplex;

  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 2>,ArithmaticTest::TEST_ADD>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 2>,ArithmaticTest::TEST_SUBTRACT>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 2>,ArithmaticTest::TEST_MULT>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 2>,ArithmaticTest::TEST_DIV>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 2>,ArithmaticTest::TEST_UNARY_MINUS>();
}

#if defined(__AVX__) || defined(__AVX2__)
TEST( VectorArithmatic, double_AVX256 ) {
  ArithmaticTest::VectorArithmatic<VectorTag<AVX<double>, 4>,ArithmaticTest::TEST_ADD>();
  ArithmaticTest::VectorArithmatic<VectorTag<AVX<double>, 4>,ArithmaticTest::TEST_SUBTRACT>();
  ArithmaticTest::VectorArithmatic<VectorTag<AVX<double>, 4>,ArithmaticTest::TEST_MULT>();
  ArithmaticTest::VectorArithmatic<VectorTag<AVX<double>, 4>,ArithmaticTest::TEST_DIV>();
  ArithmaticTest::VectorArithmatic<VectorTag<AVX<double>, 4>,ArithmaticTest::TEST_UNARY_MINUS>();
}

TEST( VectorArithmatic, dcomplex_AVX256 ) {
  typedef Kokkos::complex<double> dcomplex;

  ArithmaticTest::VectorArithmatic<VectorTag<AVX<dcomplex>, 2>,ArithmaticTest::TEST_ADD>();
  ArithmaticTest::VectorArithmatic<VectorTag<AVX<dcomplex>, 2>,ArithmaticTest::TEST_SUBTRACT>();
  
#if defined (__FMA__)
  ArithmaticTest::VectorArithmatic<VectorTag<AVX<dcomplex>, 2>,ArithmaticTest::TEST_MULT>();
  //ArithmaticTest::VectorArithmatic<VectorTag<AVX<dcomplex>, 2>,ArithmaticTest::TEST_DIV>();
#endif 

  ArithmaticTest::VectorArithmatic<VectorTag<AVX<dcomplex>, 2>,ArithmaticTest::TEST_UNARY_MINUS>();
}
#endif

///
/// double vector length 8
///

TEST( VectorArithmatic, double_SIMD8 ) {
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>, 8>,ArithmaticTest::TEST_ADD>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>, 8>,ArithmaticTest::TEST_SUBTRACT>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>, 8>,ArithmaticTest::TEST_MULT>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>, 8>,ArithmaticTest::TEST_DIV>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>, 8>,ArithmaticTest::TEST_UNARY_MINUS>();
}

TEST( VectorArithmatic, dcomplex_SIMD4 ) {
  typedef Kokkos::complex<double> dcomplex;

  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 4>,ArithmaticTest::TEST_ADD>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 4>,ArithmaticTest::TEST_SUBTRACT>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 4>,ArithmaticTest::TEST_MULT>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 4>,ArithmaticTest::TEST_DIV>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 4>,ArithmaticTest::TEST_UNARY_MINUS>();
}

#if defined(__AVX512F__)
TEST( VectorArithmatic, AVX8 ) {
  ArithmaticTest::VectorArithmatic<VectorTag<AVX<double>, 8>,ArithmaticTest::TEST_ADD>();
  ArithmaticTest::VectorArithmatic<VectorTag<AVX<double>, 8>,ArithmaticTest::TEST_SUBTRACT>();

#if defined (__FMA__)
  ArithmaticTest::VectorArithmatic<VectorTag<AVX<double>, 8>,ArithmaticTest::TEST_MULT>();
  //ArithmaticTest::VectorArithmatic<VectorTag<AVX<double>, 8>,ArithmaticTest::TEST_DIV>();
#endif

  ArithmaticTest::VectorArithmatic<VectorTag<AVX<double>, 8>,ArithmaticTest::TEST_UNARY_MINUS>();
}
#endif

///
/// double vector length 64
///

TEST( VectorArithmatic, double_SIMD64 ) {
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>,64>,ArithmaticTest::TEST_ADD>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>,64>,ArithmaticTest::TEST_SUBTRACT>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>,64>,ArithmaticTest::TEST_MULT>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>,64>,ArithmaticTest::TEST_DIV>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<double>,64>,ArithmaticTest::TEST_UNARY_MINUS>();
}

TEST( VectorArithmatic, dcomplex_SIMD32 ) {
  typedef std::complex<double> dcomplex;

  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 32>,ArithmaticTest::TEST_ADD>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 32>,ArithmaticTest::TEST_SUBTRACT>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 32>,ArithmaticTest::TEST_MULT>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 32>,ArithmaticTest::TEST_DIV>();
  ArithmaticTest::VectorArithmatic<VectorTag<SIMD<dcomplex>, 32>,ArithmaticTest::TEST_UNARY_MINUS>();
}
