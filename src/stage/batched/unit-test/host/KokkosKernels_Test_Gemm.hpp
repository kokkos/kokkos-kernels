/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include <iomanip>

#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"

#include "KokkosKernels_Vector.hpp"

#include "KokkosKernels_Gemm_Decl.hpp"
#include "KokkosKernels_Gemm_Serial_Impl.hpp"
//#include "KokkosKernels_Gemm_Team_Impl.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {
      namespace GemmTest {

#undef FLOP_MUL
#undef FLOP_ADD
#define FLOP_MUL 1.0
#define FLOP_ADD 1.0

        double FlopCount(int mm, int nn, int kk) {
          double m = (double)mm;    double n = (double)nn;    double k = (double)kk;
          return (FLOP_MUL*(m*n*k) +
                  FLOP_ADD*(m*n*k));
        }

        template<typename TA, typename TB>
        struct ParamTag { 
          typedef TA TransA;
          typedef TB TransB;
        };
 
        template<typename ViewType, typename ParamTagType, typename AlgoTagType>
        struct Functor {
          ViewType _a, _b, _c;

          KOKKOS_INLINE_FUNCTION
          Functor(const ViewType &a,
                  const ViewType &b,
                  const ViewType &c)
            : _a(a), _b(b), _c(c) {}

          KOKKOS_INLINE_FUNCTION
          void operator()(const ParamTagType &, const int k) const {
            auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
            auto bb = Kokkos::subview(_b, k, Kokkos::ALL(), Kokkos::ALL());
            auto cc = Kokkos::subview(_c, k, Kokkos::ALL(), Kokkos::ALL());
            
            Serial::Gemm<typename ParamTagType::TransA,typename ParamTagType::TransB,AlgoTagType>::
              invoke(1.0, aa, bb, 1.0, cc);
          }

          inline
          void run() {
            //Kokkos::RangePolicy<DeviceSpaceType,ParamTag<Trans::NoTranspose,Trans::NoTranspose> > policy(0, _c.dimension_0());
            Kokkos::RangePolicy<DeviceSpaceType,ParamTagType> policy(0, _c.dimension_0());
            Kokkos::parallel_for(policy, *this);            
          }
        };
    
        template<int BlkSize, typename VectorTagType, typename ParamTagType, typename AlgoTagType>
        void Gemm(const int N) {
          typedef ParamTagType param_tag_type;
          typedef typename VectorTagType::value_type value_type;
          constexpr int VectorLength = VectorTagType::length;
          
          const double flop = N*VectorLength*FlopCount(BlkSize,BlkSize,BlkSize);
          const double tmax = 1.0e15;

          const int iter_begin = -3, iter_end = 10;
          Kokkos::Impl::Timer timer;

          typedef typename DeviceSpaceType::array_layout array_layout;
          Kokkos::View<value_type***,array_layout,HostSpaceType> 
            amat("amat", N*VectorLength, BlkSize, BlkSize),
            bmat("bmat", N*VectorLength, BlkSize, BlkSize),
            cref("cref", N*VectorLength, BlkSize, BlkSize);

          typedef Vector<VectorTagType> VectorType;
          Kokkos::View<VectorType***,array_layout,HostSpaceType> 
            amat_simd("amat_simd", N, BlkSize, BlkSize),
            bmat_simd("bmat_simd", N, BlkSize, BlkSize);

          {
            Random<value_type> random;
            for (int k=0;k<N*VectorLength;++k) 
              for (int i=0;i<BlkSize;++i)
                for (int j=0;j<BlkSize;++j) {
                  amat(k, i, j) = random.value();
                  bmat(k, i, j) = random.value();

                  const int k0 = k/VectorLength, k1 = k%VectorLength;
                  amat_simd(k0, i, j)[k1] = amat(k, i, j);
                  bmat_simd(k0, i, j)[k1] = bmat(k, i, j);
                }
          }

          ///
          /// Reference version
          ///
          {
            typedef Kokkos::View<value_type***,DeviceSpaceType> view_type;
            view_type
              a("a", N*VectorLength, BlkSize, BlkSize),
              b("b", N*VectorLength, BlkSize, BlkSize),
              c("c", N*VectorLength, BlkSize, BlkSize);
            
            Functor<view_type,param_tag_type,Algo::Gemm::Unblocked> test(a, b, c);            
            {
              double tavg = 0, tmin = tmax;              
              for (int iter=iter_begin;iter<iter_end;++iter) {
                // initialize matrices
                Kokkos::deep_copy(a, amat);
                Kokkos::deep_copy(b, bmat);
                Kokkos::deep_copy(c, 0);

                DeviceSpaceType::fence();
                timer.reset();
                test.run();                
                DeviceSpaceType::fence();
                const double t = timer.seconds();
                tmin = std::min(tmin, t);
                tavg += (iter >= 0)*t;
              }
              tavg /= iter_end;              
              Kokkos::deep_copy(cref, c);
              printf("Reference, BlkSize = %3d, time = %e, avg flop/s = %e, max flop/s = %e\n",
                     BlkSize, tmin, (flop/tavg), (flop/tmin));
            }
          }

          ///
          /// Serial SIMD with appropriate data layout
          ///
          {
            typedef Kokkos::View<VectorType***,DeviceSpaceType> view_type;
            view_type
              a("a", N, BlkSize, BlkSize),
              b("b", N, BlkSize, BlkSize),
              c("c", N, BlkSize, BlkSize);

            Functor<view_type,param_tag_type,AlgoTagType> test(a, b, c);            
            {         
              double tavg = 0, tmin = tmax;
              for (int iter=iter_begin;iter<iter_end;++iter) {
                // initialize matrices
                Kokkos::deep_copy(a, amat_simd);
                Kokkos::deep_copy(b, bmat_simd);
                Kokkos::deep_copy(c, 0);

                DeviceSpaceType::fence();
                timer.reset();
                test.run();
                DeviceSpaceType::fence();
                const double t = timer.seconds();
                tmin = std::min(tmin, t);
                tavg += (iter >= 0)*t;
              }
              tavg /= iter_end;

              auto c_host = Kokkos::create_mirror_view(typename HostSpaceType::memory_space(), c);
              Kokkos::deep_copy(c_host, c);
              
              double diff = 0;
              for (int k=0;k<cref.dimension_0();++k)
                for (int i=0;i<cref.dimension_1();++i)
                  for (int j=0;j<cref.dimension_2();++j)
                    diff += abs(cref(k,i,j) - c_host(k/VectorLength,i,j)[k%VectorLength]);

              printf("KK,        BlkSize = %3d, time = %e, avg flop/s = %e, max flop/s = %e, diff = %e\n",
                     BlkSize, tmin, (flop/tavg), (flop/tmin), diff);
            }
          }
        }
      }
    }
  }
}

using namespace KokkosKernels::Batched::Experimental;

///
/// double SIMD 4
///

TEST( GemmBlocked_NT_NT, double_SIMD4 ) {
  enum : int { N = 512 }; // 512*4 = 2048
  GemmTest::Gemm< 3, VectorTag<SIMD<double>,4>,GemmTest::ParamTag<Trans::NoTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm< 5, VectorTag<SIMD<double>,4>,GemmTest::ParamTag<Trans::NoTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm<10, VectorTag<SIMD<double>,4>,GemmTest::ParamTag<Trans::NoTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
}

TEST( GemmBlocked_T_NT, double_SIMD4 ) {
  enum : int { N = 512 }; // 512*4 = 2048
  GemmTest::Gemm< 3, VectorTag<SIMD<double>,4>,GemmTest::ParamTag<Trans::Transpose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm< 5, VectorTag<SIMD<double>,4>,GemmTest::ParamTag<Trans::Transpose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm<10, VectorTag<SIMD<double>,4>,GemmTest::ParamTag<Trans::Transpose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
}

TEST( GemmBlocked_NT_T, double_SIMD4 ) {
  enum : int { N = 512 }; // 512*4 = 2048
  GemmTest::Gemm< 3, VectorTag<SIMD<double>,4>,GemmTest::ParamTag<Trans::NoTranspose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm< 5, VectorTag<SIMD<double>,4>,GemmTest::ParamTag<Trans::NoTranspose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm<10, VectorTag<SIMD<double>,4>,GemmTest::ParamTag<Trans::NoTranspose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
}

TEST( GemmBlocked_T_T, double_SIMD4 ) {
  enum : int { N = 512 }; // 512*4 = 2048
  GemmTest::Gemm< 3, VectorTag<SIMD<double>,4>,GemmTest::ParamTag<Trans::Transpose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm< 5, VectorTag<SIMD<double>,4>,GemmTest::ParamTag<Trans::Transpose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm<10, VectorTag<SIMD<double>,4>,GemmTest::ParamTag<Trans::Transpose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
}

///
/// Kokkos::complex<double> SIMD 2
///

TEST( GemmBlocked_NT_NT, dcomplex_SIMD2 ) {
  enum : int { N = 1024 }; // 1024*2 = 2048
  typedef Kokkos::complex<double> dcomplex;

  GemmTest::Gemm< 3, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm< 5, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm<10, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
}

TEST( GemmBlocked_T_NT, dcomplex_SIMD2 ) {
  enum : int { N = 1024 }; // 1024*2 = 2048
  typedef Kokkos::complex<double> dcomplex;

  GemmTest::Gemm< 3, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::Transpose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm< 5, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::Transpose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm<10, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::Transpose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
}

TEST( GemmBlocked_NT_T, dcomplex_SIMD2 ) {
  enum : int { N = 1024 }; // 1024*2 = 2048
  typedef Kokkos::complex<double> dcomplex;

  GemmTest::Gemm< 3, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm< 5, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm<10, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
}

// TEST( GemmBlocked_CT_NT, dcomplex_SIMD2 ) {
//   enum : int { N = 1024 }; // 1024*2 = 2048
//  typedef Kokkos::complex<double> dcomplex;
//   GemmTest::Gemm< 3, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::ConjTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm< 5, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::ConjTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm<10, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::ConjTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
// }

// TEST( GemmBlocked_NT_CT, dcomplex_SIMD2 ) {
//   enum : int { N = 1024 }; // 1024*2 = 2048
//  typedef Kokkos::complex<double> dcomplex;
//   GemmTest::Gemm< 3, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::ConjTranspose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm< 5, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::ConjTranspose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm<10, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::ConjTranspose>,Algo::Gemm::Blocked>(N);
// }

TEST( GemmBlocked_T_T, dcomplex_SIMD2 ) {
  enum : int { N = 1024 }; // 1024*2 = 2048
  typedef Kokkos::complex<double> dcomplex;

  GemmTest::Gemm< 3, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::Transpose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm< 5, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::Transpose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm<10, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::Transpose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
}

// TEST( GemmBlocked_CT_CT, dcomplex_SIMD2 ) {
//   enum : int { N = 1024 }; // 1024*2 = 2048
//  typedef Kokkos::complex<double> dcomplex;
//   GemmTest::Gemm< 3, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::ConjTranspose,Trans::ConjTranspose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm< 5, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::ConjTranspose,Trans::ConjTranspose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm<10, VectorTag<SIMD<dcomplex>,2>,GemmTest::ParamTag<Trans::ConjTranspose,Trans::ConjTranspose>,Algo::Gemm::Blocked>(N);
// }

///
/// double AVX 256
///

#if defined(__AVX__) || defined(__AVX2__)

TEST( GemmBlocked_NT_NT, double_AVX256 ) {
  enum : int { N = 512 }; // 512*4 = 2048
  GemmTest::Gemm< 3, VectorTag<AVX<double>,4>,GemmTest::ParamTag<Trans::NoTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm< 5, VectorTag<AVX<double>,4>,GemmTest::ParamTag<Trans::NoTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm<10, VectorTag<AVX<double>,4>,GemmTest::ParamTag<Trans::NoTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
}

TEST( GemmBlocked_T_NT, double_AVX256 ) {
  enum : int { N = 512 }; // 512*4 = 2048
  GemmTest::Gemm< 3, VectorTag<AVX<double>,4>,GemmTest::ParamTag<Trans::Transpose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm< 5, VectorTag<AVX<double>,4>,GemmTest::ParamTag<Trans::Transpose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm<10, VectorTag<AVX<double>,4>,GemmTest::ParamTag<Trans::Transpose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
}

TEST( GemmBlocked_NT_T, double_AVX256 ) {
  enum : int { N = 512 }; // 512*4 = 2048
  GemmTest::Gemm< 3, VectorTag<AVX<double>,4>,GemmTest::ParamTag<Trans::NoTranspose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm< 5, VectorTag<AVX<double>,4>,GemmTest::ParamTag<Trans::NoTranspose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm<10, VectorTag<AVX<double>,4>,GemmTest::ParamTag<Trans::NoTranspose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
}

TEST( GemmBlocked_T_T, double_AVX256 ) {
  enum : int { N = 512 }; // 512*4 = 2048
  GemmTest::Gemm< 3, VectorTag<AVX<double>,4>,GemmTest::ParamTag<Trans::Transpose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm< 5, VectorTag<AVX<double>,4>,GemmTest::ParamTag<Trans::Transpose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
  GemmTest::Gemm<10, VectorTag<AVX<double>,4>,GemmTest::ParamTag<Trans::Transpose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
}

///
/// dcomplex AVX 256
///

// TEST( GemmBlocked_NT_NT, dcomplex_AVX256 ) {
//   enum : int { N = 1024 }; // 1024*2 = 2048
//   typedef Kokkos::complex<double> dcomplex;

//   GemmTest::Gemm< 3, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm< 5, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm<10, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
// }

// TEST( GemmBlocked_T_NT, dcomplex_AVX256 ) {
//   enum : int { N = 1024 }; // 1024*2 = 2048
//   typedef Kokkos::complex<double> dcomplex;

//   GemmTest::Gemm< 3, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::Transpose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm< 5, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::Transpose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm<10, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::Transpose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
// }

// TEST( GemmBlocked_NT_T, dcomplex_AVX256 ) {
//   enum : int { N = 1024 }; // 1024*2 = 2048
//   typedef Kokkos::complex<double> dcomplex;

//   GemmTest::Gemm< 3, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm< 5, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm<10, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
// }

// // TEST( GemmBlocked_CT_NT, dcomplex_AVX256 ) {
// //   enum : int { N = 1024 }; // 1024*2 = 2048
// //  typedef Kokkos::complex<double> dcomplex;
// //   GemmTest::Gemm< 3, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::ConjTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
// //   GemmTest::Gemm< 5, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::ConjTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
// //   GemmTest::Gemm<10, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::ConjTranspose,Trans::NoTranspose>,Algo::Gemm::Blocked>(N);
// // }

// // TEST( GemmBlocked_NT_CT, dcomplex_AVX256 ) {
// //   enum : int { N = 1024 }; // 1024*2 = 2048
// //  typedef Kokkos::complex<double> dcomplex;
// //   GemmTest::Gemm< 3, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::ConjTranspose>,Algo::Gemm::Blocked>(N);
// //   GemmTest::Gemm< 5, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::ConjTranspose>,Algo::Gemm::Blocked>(N);
// //   GemmTest::Gemm<10, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::NoTranspose,Trans::ConjTranspose>,Algo::Gemm::Blocked>(N);
// // }

// TEST( GemmBlocked_T_T, dcomplex_AVX256 ) {
//   enum : int { N = 1024 }; // 1024*2 = 2048
//   typedef Kokkos::complex<double> dcomplex;

//   GemmTest::Gemm< 3, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::Transpose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm< 5, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::Transpose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
//   GemmTest::Gemm<10, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::Transpose,Trans::Transpose>,Algo::Gemm::Blocked>(N);
// }

// // TEST( GemmBlocked_T_T, dcomplex_AVX256 ) {
// //   enum : int { N = 1024 }; // 1024*2 = 2048
// //  typedef Kokkos::complex<double> dcomplex;
// //   GemmTest::Gemm< 3, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::ConjTranspose,Trans::ConjTranspose>,Algo::Gemm::Blocked>(N);
// //   GemmTest::Gemm< 5, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::ConjTranspose,Trans::ConjTranspose>,Algo::Gemm::Blocked>(N);
// //   GemmTest::Gemm<10, VectorTag<AVX<dcomplex>,2>,GemmTest::ParamTag<Trans::ConjTranspose,Trans::ConjTranspose>,Algo::Gemm::Blocked>(N);
// // }

#endif
