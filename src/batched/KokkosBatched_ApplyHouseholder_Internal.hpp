#ifndef __KOKKOSBATCHED_HOUSEHOLDER_INTERNAL_HPP__
#define __KOKKOSBATCHED_HOUSEHOLDER_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"


namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Serial Internal Impl
    /// ==================== 
    ///
    /// this impl follows the flame interface of householder transformation
    ///
    struct SerialApplyLeftHouseholderInternal {
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const int m, // m = NumRows(A2) = numRows(u2)
             const int n, // n = NumCols(a1t)
             const ValueType   tau,
             /* */ ValueType * u2,  const int u2s,
             /* */ ValueType * a1t, const int a1ts,
             /* */ ValueType * A2,  const int as0, const int as1,
             /* */ ValueType * w1t) {
        // apply a single householder transform H from the left to a row vector a1t 
        // and a matrix A2
        typedef ValueType value_type;
        typedef typename Kokkos::Details::ArithTraits<ValueType>::mag_type mag_type;
        
        // compute the followings:
        // a1t -=    inv(tau)(a1t + u2'A2)
        // A2  -= u2 inv(tau)(a1t + u2'A2)

        // w1t = a1t
        for (int i=0;i<n;++i) 
          w1t[i] = a1t[i*a1ts];
        // SerialCopyInternal::invoke(n, w1t, 1, a1t, a1ts);

        // w1t += u2'A2 = A2^T conj(u2)
        for (int i=0;i<n;++i) { 
          value_type tmp(0);
          for (int j=0;j<m;++j) 
            tmp += A2[i*as1+j*as0]*Kokkos::Details::ArithTraits<value_type>::conj(u2[j*u2s]);
          w1t[i] += tmp;
        }
        // gemv is not implemented for conjugate transpose; if use this, no support for complex
        // SerialGemvInternal<Algo::Gemv::Unblocked>::                                                                     
        //   invoke(n, m, 
        //          1, 
        //          A2, as1, as0,
        //          u2, u2s,
        //          1, 
        //          w1t, 1);
        
        // w1t /= tau
        const value_type inv_tau = one/tau;
        for (int i=0;i<n;++i) 
          w1t[i] *= inv_tau;
        // SerialScaleInternal::invoke(n, one/tau, w1t, 1);

        // a1t -= w1t (axpy)
        for (int i=0;i<n;++i) 
          a1t[i*a1ts] -= w1t[i];

        // A2 -= u2 w1t (ger)
        for (int j=0;j<n;++j)
          for (int i=0;i<m;++i)
            A2[i*as0+j*as1] -= u2[i*u2s]*w1t[j];

        return 0;
      }
    };


    struct SerialApplyRightHouseholderInternal {
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const int m, // m = NumRows(a1) 
             const int n, // n = NumRows(u2) = NumCols(A2)
             const ValueType   tau,
             /* */ ValueType * u2,  const int u2s,
             /* */ ValueType * a1, const int a1s,
             /* */ ValueType * A2,  const int as0, const int as1,
             /* */ ValueType * w1) {
        // apply a single householder transform H from the left to a row vector a1t 
        // and a matrix A2
        typedef ValueType value_type;
        typedef typename Kokkos::Details::ArithTraits<ValueType>::mag_type mag_type;
        
        // compute the followings:
        // a1 -= inv(tau)(a1 + A2 u2)
        // A2 -= inv(tau)(a1 + A2 u2) u2'

        // w1 = a1
        for (int i=0;i<m;++i) 
          w1[i] = a1[i*a1s];
        // SerialCopyInternal::invoke(m, w1, 1, a1, a1s);

        // w1t += A2 u2
        for (int i=0;i<m;++i) { 
          value_type tmp(0);
          ValueType * A2_at_i = A2 + i*as0;
          for (int j=0;j<n;++j) 
            tmp += A2_at_i[j*as1]*u2[j*u2s];
          w1[i] += tmp;
        }
        // SerialGemvInternal<Algo::Gemv::Unblocked>::                                                                     
        //   invoke(m, n, 
        //          1, 
        //          A2, as0, as1,
        //          u2, u2s,
        //          1, 
        //          w1, 1);
        
        // w1t /= tau
        const value_type inv_tau = one/tau;
        for (int i=0;i<m;++i) 
          w1[i] *= inv_tau;
        // SerialScaleInternal::invoke(m, one/tau, w1, 1);

        // a1t -= w1 (axpy)
        for (int i=0;i<m;++i) 
          a1[i*a1s] -= w1[i];

        // A2 -= w1 * u2' (ger with conjugate)
        for (int j=0;j<n;++j)
          for (int i=0;i<m;++i)
            A2[i*as0+j*as1] -= w1[i]*Kokkos::Details::ArithTraits<ValueType>::conj(u2[j*u2s]);
        
        return 0;
      }
    };

  }//  end namespace Experimental
} // end namespace KokkosBatched


#endif
