/// \author Kyungjoo Kim (kyukim@sandia.gov)


#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"

#include "KokkosKernels_Set_Decl.hpp"
#include "KokkosKernels_Set_Serial_Impl.hpp"

#include "KokkosKernels_Scale_Decl.hpp"
#include "KokkosKernels_Scale_Serial_Impl.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {
      namespace MatUtilTest {
        
        enum : int  { TEST_SET = 0,
                      TEST_SCALE = 1,
                      TEST_MAX = 2 };

        struct KokkosKernelVersion {};
        struct NaiveVersion {};
        
        template<typename ScalarType, typename ViewType, typename AlgoTagType, int TestID>
        struct Functor {
          
          ScalarType _alpha;
          ViewType _a;

          KOKKOS_INLINE_FUNCTION
          Functor(const ScalarType alpha, 
                  const ViewType &a) 
            : _alpha(alpha), _a(a) {}
      
          inline
          const char* label() const {
            switch (TestID) {
            case TEST_SET: return "set"; break;
            case TEST_SCALE: return "scale";  break;
            }
            return "nothing";
          }

          KOKKOS_INLINE_FUNCTION
          void operator()(const KokkosKernelVersion &, const int i) const {
            auto A = Kokkos::subview(_a, i, Kokkos::ALL(), Kokkos::ALL());
            switch (TestID) {
            case TEST_SET: Serial::Set::invoke(_alpha, A); break;
            case TEST_SCALE: Serial::Scale::invoke(_alpha, A); break;
            }
          }

          KOKKOS_INLINE_FUNCTION
          void operator()(const NaiveVersion &, const int i) const {
            auto A = Kokkos::subview(_a, i, Kokkos::ALL(), Kokkos::ALL());
            const int m = A.dimension_0(), n = A.dimension_1();
            switch (TestID) {
            case TEST_SET: {
              for (int i=0;i<m;++i) 
                for (int j=0;j<n;++j)
                  A(i,j)  = _alpha;
              break;
            }
            case TEST_SCALE: {
              for (int i=0;i<m;++i) 
                for (int j=0;j<n;++j)
                  A(i,j) *= _alpha;
              break;
            }
            }
          }

          inline
          int run() {
            Kokkos::RangePolicy<DeviceSpaceType,AlgoTagType> policy(0, _a.dimension_0());
            Kokkos::parallel_for(policy, *this);
          }      
        };

        template<typename ValueType, int TestID>
        void SetScale() {
          enum : int {
            N = 32768,
            BlkSize = 3
          };
          
          const int iter_begin = -3, iter_end = 10;
          Kokkos::Impl::Timer timer;
          double t = 0;

          ///
          /// random data initialization
          ///
          typedef ValueType value_type;
          Kokkos::View<value_type***,HostSpaceType> 
            a_host("a_host", N, BlkSize, BlkSize),
            b_host("b_host", N, BlkSize, BlkSize);
      
          Random<value_type> random;
          for (int k=0;k<N;++k) 
            for (int i=0;i<BlkSize;++i) 
              for (int j=0;j<BlkSize;++j) {
                a_host(k, i, j) = random.value();
                b_host(k, i, j) = a_host(k, i, j);
              }

          typedef value_type scalar_type;
          const scalar_type alpha = random.value();

          ///
          /// test for reference
          ///

          {
            auto a = Kokkos::create_mirror_view(typename DeviceSpaceType::memory_space(), a_host);
            auto b = Kokkos::create_mirror_view(typename DeviceSpaceType::memory_space(), b_host);
            
            Kokkos::deep_copy(a, a_host);
            Kokkos::deep_copy(b, b_host);
            
            {
              t = 0;
              Functor<scalar_type,decltype(a),NaiveVersion,TestID> test(alpha, a);
              for (int iter=iter_begin;iter<iter_end;++iter) {
                DeviceSpaceType::fence();
                timer.reset();
                test.run();
                DeviceSpaceType::fence();
                t += (iter >= 0)*timer.seconds();
              }
              printf("Reference,     Test %12s, Time %e\n", test.label(), (t/iter_end));
            }
            Kokkos::deep_copy(a_host, a);

            {
              t = 0;
              Functor<scalar_type,decltype(b),KokkosKernelVersion,TestID> test(alpha, b);
              for (int iter=iter_begin;iter<iter_end;++iter) {
                DeviceSpaceType::fence();
                timer.reset();
                test.run();
                DeviceSpaceType::fence();
                t += (iter >= 0)*timer.seconds();
              }
              printf("KK,            Test %12s, Time %e\n", test.label(), (t/iter_end));
            }
            Kokkos::deep_copy(b_host, b);
          }
      
          ///
          /// check a = b
          ///

          double sum = 0;
          for (int k=0;k<N;++k) 
            for (int i=0;i<BlkSize;++i) 
              for (int j=0;j<BlkSize;++j) {
                const auto diff = abs(a_host(k,i,j) - b_host(k,i,j));
                sum += diff;
              }
          EXPECT_TRUE(sum < std::numeric_limits<double>::epsilon());
        }
      }
      }
  }
  }
  
using namespace KokkosKernels::Batched::Experimental;

///
/// double vector length 4
///

TEST( MatUtil, double ) {
  MatUtilTest::SetScale<double,MatUtilTest::TEST_SET>();
  MatUtilTest::SetScale<double,MatUtilTest::TEST_SCALE>();
}

