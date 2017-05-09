#ifndef __KOKKOSKERNELS_GEMV_SERIAL_INTERNAL_HPP__
#define __KOKKOSKERNELS_GEMV_SERIAL_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"
//#include "KokkosKernels_InnerMultipleDotProduct_Serial_Impl.hpp"
#include "KokkosKernels_InnerGemmFixC_Serial_Impl.hpp"

namespace KokkosKernels {

  ///
  /// Serial Internal Impl
  /// ====================
  namespace Serial {

    template<typename ArgAlgo>
    struct GemvInternal {
      template<typename ScalarType,
               typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const int m, const int n, 
             const ScalarType alpha,
             const ValueType *__restrict__ A, const int as0, const int as1,
             const ValueType *__restrict__ x, const int xs0, 
             const ScalarType beta,
             /**/  ValueType *__restrict__ y, const int ys0) {
        //static_assert("KokkosKernels::GemvInternal:: Not yet implemented");
        return 0;
      }
    };

    template<>
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    int
    GemvInternal<Algo::Gemv::Unblocked>::
    invoke(const int m, const int n, 
           const ScalarType alpha,
           const ValueType *__restrict__ A, const int as0, const int as1,
           const ValueType *__restrict__ x, const int xs0,
           const ScalarType beta,
           /**/  ValueType *__restrict__ y, const int ys0) {
      // y = beta y + alpha A x
      // y (m), A(m x n), B(n)

      typedef ValueType value_type;

      if      (beta == 0) Util::set  (m, 1, y, ys0, 1, value_type(0)   );
      else if (beta != 1) Util::scale(m, 1, y, ys0, 1, value_type(beta));
      
      if (alpha != 0) {
        if (m <= 0 || n <= 0) return 0;
        
        for (int i=0;i<m;++i) {
          value_type t(0);
          const value_type *__restrict__ tA = (A + i*as0);
          for (int j=0;j<n;++j)
            t += tA[j*as1]*x[j*xs0];
          y[i*ys0] += alpha*t;
        }
      }
      return 0;
    }

    template<>
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    int
    GemvInternal<Algo::Gemv::Blocked>::
    invoke(const int m, const int n, 
           const ScalarType alpha,
           const ValueType *__restrict__ A, const int as0, const int as1,
           const ValueType *__restrict__ x, const int xs0,
           const ScalarType beta,
           /**/  ValueType *__restrict__ y, const int ys0) {
      // y = beta y + alpha A x
      // y (m), A(m x n), B(n)

      typedef ValueType value_type;

      if      (beta == 0) Util::set  (m, 1, y, ys0, 1, value_type(0)   );
      else if (beta != 1) Util::scale(m, 1, y, ys0, 1, value_type(beta));
      
      if (alpha != 0) {
        if (m <= 0 || n <= 0) return 0;
        
        enum : int {
          mb = Algo::Gemv::Blocked::mb };

        InnerGemmFixC<0,1> gemm(as0, as1, xs0, 1, ys0, 1); 

        const int mm = (m/mb)*mb, mp = (m%mb);

        for (int i=0;i<mm;i+=mb) 
          gemm.serial_invoke(alpha, A+i*as0,  x, mb, n, y+i*ys0 );

        if (mp) 
          gemm.serial_invoke(alpha, A+mm*as0, x, mp, n, y+mm*ys0);
      }
      return 0;
    }
  }
} // end namespace KokkosKernels

#endif
