#ifndef __KOKKOSKERNELS_GEMV_SERIAL_INTERNAL_HPP__
#define __KOKKOSKERNELS_GEMV_SERIAL_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"

#include "KokkosKernels_Set_Internal.hpp"
#include "KokkosKernels_Scale_Internal.hpp"

#include "KokkosKernels_InnerMultipleDotProduct_Serial_Impl.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {

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
                 /**/  ValueType *__restrict__ y, const int ys0);
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

          if      (beta == 0) Serial::SetInternal  ::invoke(m, value_type(0),    y, ys0);
          else if (beta != 1) Serial::ScaleInternal::invoke(m, value_type(beta), y, ys0);
      
          if (alpha != 0) {
            if (m <= 0 || n <= 0) return 0;
        
            for (int i=0;i<m;++i) {
              value_type t(0);
              const value_type *__restrict__ tA = (A + i*as0);

              KOKKOSKERNELS_LOOP_UNROLL
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
          enum : int {
            mbAlgo = Algo::Gemv::Blocked::mb<Kokkos::Impl::ActiveExecutionMemorySpace>()
          };

          if      (beta == 0) Serial::SetInternal  ::invoke(m, value_type(0),    y, ys0);
          else if (beta != 1) Serial::ScaleInternal::invoke(m, value_type(beta), y, ys0);
      
          if (alpha != 0) {
            if (m <= 0 || n <= 0) return 0;
        
            InnerMultipleDotProduct<mbAlgo> inner(as0, as1, xs0, ys0);
            const int mb = mbAlgo;
            for (int i=0;i<m;i+=mb) 
              inner.serial_invoke(alpha, A+i*as0, x, (i+mb) > m ? (m-i) : mb, n, y+i*ys0 );
          }
          return 0;
        }
      }
    }
  }
} // end namespace KokkosKernels

#endif
