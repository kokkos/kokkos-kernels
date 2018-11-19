#ifndef __KOKKOSBATCHED_SHIFTED_TRSV_SERIAL_INTERNAL_HPP__
#define __KOKKOSBATCHED_SHIFTED_TRSV_SERIAL_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

#include "KokkosBatched_Set_Internal.hpp"
#include "KokkosBatched_Scale_Internal.hpp"

#include "KokkosBatched_InnerTrsm_Serial_Impl.hpp"
#include "KokkosBatched_Gemv_Serial_Internal.hpp"


namespace KokkosBatched {
  namespace Experimental {

    ///
    /// Serial Internal Impl
    /// ====================

    ///
    /// Lower
    ///

    struct SerialShiftedTrsvInternalLower {
      template<typename ScalarType,
               typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const int m,
             const ScalarType lambda,
             const ValueType *__restrict__ A, const int as0, const int as1,
             /**/  ValueType *__restrict__ b, const int bs0) {
        for (int p=0;p<m;++p) {
          const int iend = m-p-1;
          
          const ValueType
            *__restrict__ a21   = iend ? A+(p+1)*as0+p*as1 : NULL;
          
          ValueType
            *__restrict__ beta1 =        b+p*bs0,
            *__restrict__ b2    = iend ? beta1+bs0 : NULL;
          
          // with __restrict__ a compiler assumes that the pointer is not accessed by others
          // op(/=) uses this pointer and changes the associated values, which brings a compiler problem
          *beta1 = *beta1 / (A[p*as0+p*as1]-lambda);

          for (int i=0;i<iend;++i)
            b2[i*bs0] -= a21[i*as0] * (*beta1);
        }
        return 0;
      }
    };

    ///
    /// Upper
    ///
    
    struct SerialShiftedTrsvInternalUpper {
      template<typename ScalarType,
               typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const int m,
             const ScalarType lambda,
             const ValueType *__restrict__ A, const int as0, const int as1,
             /**/  ValueType *__restrict__ b, const int bs0) {
        ValueType *__restrict__ b0 = b;
        for (int p=(m-1);p>=0;--p) {
          const int iend = p;
          
          const ValueType *__restrict__ a01   = A+p*as1;
          /**/  ValueType *__restrict__ beta1 = b+p*bs0;
          
          // with __restrict__ a compiler assumes that the pointer is not accessed by others
          // op(/=) uses this pointer and changes the associated values, which brings a compiler problem
          *beta1 = *beta1 / (A[p*as0+p*as1] - lambda);
          
          for (int i=0;i<iend;++i)
            b0[i*bs0] -= a01[i*as0] * (*beta1);
        }
        return 0;
      }
    };
    
  }
}

#endif
