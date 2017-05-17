/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include <iomanip>

#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"

#include "KokkosKernels_Vector.hpp"

#include "KokkosKernels_Trsm_Decl.hpp"
#include "KokkosKernels_Trsm_Serial_Impl.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {
      namespace TrsmTest {

        template<typename ValueType>
        double FlopCountLower(int mm, int nn) {
          double m = (double)mm;    double n = (double)nn;
          double FLOP_MUL = 1.0;
          double FLOP_ADD = 1.0;
          if (std::is_same<ValueType,std::complex<double> >::value ||
              std::is_same<ValueType,Kokkos::complex<double> >::value) {
            FLOP_MUL = 6.0;
            FLOP_ADD = 2.0;
          }
          return (FLOP_MUL*(0.5*m*n*(n+1.0)) +
                  FLOP_ADD*(0.5*m*n*(n-1.0)));
        }
        template<typename ValueType>
        double FlopCountUpper(int mm, int nn) {
          double m = (double)mm;    double n = (double)nn;
          double FLOP_MUL = 1.0;
          double FLOP_ADD = 1.0;
          if (std::is_same<ValueType,std::complex<double> >::value ||
              std::is_same<ValueType,Kokkos::complex<double> >::value) {
            FLOP_MUL = 6.0;
            FLOP_ADD = 2.0;
          }
          return (FLOP_MUL*(0.5*m*n*(n+1.0)) +
                  FLOP_ADD*(0.5*m*n*(n-1.0)));
        }

        template<typename S, typename U, typename T, typename D>
        struct ParamTag {
          typedef S side;
          typedef U uplo;
          typedef T trans;
          typedef D diag;
        };

        template<typename ViewType, typename ParamTagType, typename AlgoTagType>
        struct Functor {
          ViewType _a, _b;

          KOKKOS_INLINE_FUNCTION
          Functor(const ViewType &a,
                  const ViewType &b) 
            : _a(a), _b(b) {}

          KOKKOS_INLINE_FUNCTION
          void operator()(const ParamTagType &, const int k) const {
            auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
            auto bb = Kokkos::subview(_b, k, Kokkos::ALL(), Kokkos::ALL());

            Serial::Trsm<typename ParamTagType::side,
                         typename ParamTagType::uplo,
                         typename ParamTagType::trans,
                         typename ParamTagType::diag,
                         AlgoTagType>::
              invoke(1.0, aa, bb);
          }

          inline
          void run() {
            Kokkos::RangePolicy<DeviceSpaceType,ParamTagType> policy(0, _b.dimension_0());
            Kokkos::parallel_for(policy, *this);
          }
        };

        template<int BlkSize, int NumCols, typename VectorTagType, typename ParamTagType, typename AlgoTagType>
        void Trsm(const int N) {
          typedef ParamTagType param_tag_type;
          typedef typename VectorTagType::value_type ValueType;
          constexpr int VectorLength = VectorTagType::length;

          // when m == n, lower upper does not matter (unit and nonunit)
          double flop = 0;
          if (std::is_same<typename ParamTagType::uplo,Uplo::Lower>::value)
            flop = FlopCountLower<ValueType>(BlkSize,NumCols);
          else
            flop = FlopCountUpper<ValueType>(BlkSize,NumCols);
          flop *= (N*VectorLength);

          const double tmax = 1.0e15;

          const int iter_begin = -3, iter_end = 10;
          Kokkos::Impl::Timer timer;

          typedef typename DeviceSpaceType::array_layout array_layout;
          Kokkos::View<ValueType***,array_layout,HostSpaceType>
            amat("amat", N*VectorLength, BlkSize, BlkSize),
            bmat("bmat", N*VectorLength, BlkSize, NumCols),
            bref("bref", N*VectorLength, BlkSize, NumCols);

          typedef Vector<VectorTagType> VectorType;
          Kokkos::View<VectorType***,array_layout,HostSpaceType>
            amat_simd("amat_simd", N, BlkSize, BlkSize),
            bmat_simd("bmat_simd", N, BlkSize, NumCols); 
      
          Random<ValueType> random;
          for (int k=0;k<N*VectorLength;++k) {
            const int k0 = k/VectorLength, k1 = k%VectorLength;
            for (int i=0;i<BlkSize;++i)
              for (int j=0;j<BlkSize;++j) {
                amat(k, i, j) = random.value() + 4.0*(i==j);
                amat_simd(k0, i, j)[k1] = amat(k, i, j);
              }
            for (int i=0;i<BlkSize;++i)
              for (int j=0;j<NumCols;++j) {
                bmat(k, i, j) = random.value(); 
                bmat_simd(k0, i, j)[k1] = bmat(k, i, j);
              }
          }
      
          ///
          /// Reference version
          ///
          {
            typedef Kokkos::View<ValueType***,array_layout,DeviceSpaceType> view_type;
            view_type
              a("a", N*VectorLength, BlkSize, BlkSize),
              b("b", N*VectorLength, BlkSize, NumCols);

            Functor<view_type,param_tag_type,Algo::Trsm::Unblocked> test(a, b);
            {
              double tavg = 0, tmin = tmax;
              for (int iter=iter_begin;iter<iter_end;++iter) {
                // initialize matrices
                Kokkos::deep_copy(a, amat);
                Kokkos::deep_copy(b, bmat);

                DeviceSpaceType::fence();
                timer.reset();
                test.run();
                DeviceSpaceType::fence();
                const double t = timer.seconds();
                tmin = std::min(tmin, t);
                tavg += (iter >= 0)*t;
              }
              tavg /= iter_end;
              Kokkos::deep_copy(bref, b);

              printf("Reference, BlkSize = %3d, NumCols = %d, time = %e, avg flop/s = %e, max flop/s = %e\n",
                     BlkSize, NumCols, tmin, (flop/tavg), (flop/tmin));
            }
          }

          ///
          /// SIMD with appropriate data layout
          ///
          {
            typedef Kokkos::View<VectorType***,DeviceSpaceType> view_type;
            view_type
              a("a", N, BlkSize, BlkSize),
              b("b", N, BlkSize, NumCols);

            Functor<view_type,param_tag_type,AlgoTagType> test(a, b);
            {
              double tavg = 0, tmin = tmax;
              for (int iter=iter_begin;iter<iter_end;++iter) {
                // initialize matrices
                Kokkos::deep_copy(a, amat_simd);
                Kokkos::deep_copy(b, bmat_simd);

                DeviceSpaceType::fence();
                timer.reset();
                test.run();
                DeviceSpaceType::fence();
                const double t = timer.seconds();
                tmin = std::min(tmin, t);
                tavg += (iter >= 0)*t;
              }
              tavg /= iter_end;

              double diff = 0;
              for (int k=0;k<bref.dimension_0();++k)
                for (int i=0;i<bref.dimension_1();++i)
                  for (int j=0;j<bref.dimension_2();++j)
                    diff += std::abs(bref(k,i,j) - b(k/VectorLength,i,j)[k%VectorLength]);

              printf("KK,        BlkSize = %3d, NumCols = %3d, time = %e, avg flop/s = %e, max flop/s = %e, diff = %e\n",
                     BlkSize, NumCols, tmin, (flop/tavg), (flop/tmin), diff);
            }
          }
          std::cout << "\n\n";
        }
      }
    }
  }
}

using namespace KokkosKernels::Batched::Experimental;

///
/// double SIMD 4
/// 

/// Left, Lower

TEST( TrsmBlocked_L_L_NT_U, double_SIMD4 ) {
  enum : int { N = 512 }; // 512*4 = 2048

  typedef VectorTag<SIMD<double>,4> vector_tag_type;
  typedef TrsmTest::ParamTag<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;

  TrsmTest::Trsm< 3, 3, vector_tag_type,param_tag_type,algo_tag_type>(N);
  TrsmTest::Trsm< 5, 5, vector_tag_type,param_tag_type,algo_tag_type>(N);
  TrsmTest::Trsm<10,10, vector_tag_type,param_tag_type,algo_tag_type>(N);
}

TEST( TrsmBlocked_L_L_NT_NU, double_SIMD4 ) {
  enum : int { N = 512 }; // 512*4 = 2048

  typedef VectorTag<SIMD<double>,4> vector_tag_type;
  typedef TrsmTest::ParamTag<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;

  TrsmTest::Trsm< 3, 3, vector_tag_type,param_tag_type,algo_tag_type>(N);
  TrsmTest::Trsm< 5, 5, vector_tag_type,param_tag_type,algo_tag_type>(N);
  TrsmTest::Trsm<10,10, vector_tag_type,param_tag_type,algo_tag_type>(N);
}

/// Left, Upper

TEST( TrsmBlocked_L_U_NT_U, double_SIMD4 ) {
  enum : int { N = 512 }; // 512*4 = 2048

  typedef VectorTag<SIMD<double>,4> vector_tag_type;
  typedef TrsmTest::ParamTag<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;

  TrsmTest::Trsm< 3, 3, vector_tag_type,param_tag_type,algo_tag_type>(N);
  TrsmTest::Trsm< 5, 5, vector_tag_type,param_tag_type,algo_tag_type>(N);
  TrsmTest::Trsm<10,10, vector_tag_type,param_tag_type,algo_tag_type>(N);
}

TEST( TrsmBlocked_L_U_NT_NU, double_SIMD4 ) {
  enum : int { N = 512 }; // 512*4 = 2048

  typedef VectorTag<SIMD<double>,4> vector_tag_type;
  typedef TrsmTest::ParamTag<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;

  TrsmTest::Trsm< 3, 3, vector_tag_type,param_tag_type,algo_tag_type>(N);
  TrsmTest::Trsm< 5, 5, vector_tag_type,param_tag_type,algo_tag_type>(N);
  TrsmTest::Trsm<10,10, vector_tag_type,param_tag_type,algo_tag_type>(N);
}

/// Right, Upper

TEST( TrsmBlocked_R_U_NT_U, double_SIMD4 ) {
  enum : int { N = 512 }; // 512*4 = 2048

  typedef VectorTag<SIMD<double>,4> vector_tag_type;
  typedef TrsmTest::ParamTag<Side::Right,Uplo::Upper,Trans::NoTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;

  TrsmTest::Trsm< 3, 3, vector_tag_type,param_tag_type,algo_tag_type>(N);
  TrsmTest::Trsm< 5, 5, vector_tag_type,param_tag_type,algo_tag_type>(N);
  TrsmTest::Trsm<10,10, vector_tag_type,param_tag_type,algo_tag_type>(N);
}

TEST( TrsmBlocked_R_U_NT_NU, double_SIMD4 ) {
  enum : int { N = 512 }; // 512*4 = 2048

  typedef VectorTag<SIMD<double>,4> vector_tag_type;
  typedef TrsmTest::ParamTag<Side::Right,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;

  TrsmTest::Trsm< 3, 3, vector_tag_type,param_tag_type,algo_tag_type>(N);
  TrsmTest::Trsm< 5, 5, vector_tag_type,param_tag_type,algo_tag_type>(N);
  TrsmTest::Trsm<10,10, vector_tag_type,param_tag_type,algo_tag_type>(N);
}

