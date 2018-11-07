#ifndef __KOKKOSBATCHED_APPLY_GIVENS_SERIAL_INTERNAL_HPP__
#define __KOKKOSBATCHED_APPLY_GIVENS_SERIAL_INTERNAL_HPP__


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
    template<int N>
    struct SerialApplyLeftGivensSizeSpecificInternal {
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const Kokkos::pair<ValueType,ValueType> &G,
             /* */ ValueType * a1t, const int &a1ts,
             /* */ ValueType * a2t, const int &a2ts) {
        typedef ValueType value_type;
        const value_type gamma = G.first;
        const value_type sigma = G.second;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int j=0;j<N;++j) {
          const value_type alpha1 = a1t[j*a1ts];
          const value_type alpha2 = a2t[j*a2ts];
          a1t[j*a1ts] = gamma*alpha1 - sigma*alpha2;
          a2t[j*a1ts] = sigma*alpha1 + gamma*alpha2;
        }
        return 0;
      }
    };


    struct SerialApplyLeftGivensInternal {
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const Kokkos::pair<ValueType,ValueType> G,
             const int n, 
             /* */ ValueType * a1t, const int a1ts,
             /* */ ValueType * a2t, const int a2ts) {
        typedef ValueType value_type;
        /// G = [ gamma -sigma;
        ///       sigma  gamma ];
        /// A := G A
        /// where gamma is G.first and sigma is G.second 
        
        const value_type gamma = G.first;
        const value_type sigma = G.second;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int j=0;j<n;++j) {
          const value_type alpha1 = a1t[j*a1ts];
          const value_type alpha2 = a2t[j*a2ts];
          a1t[j*a1ts] = gamma*alpha1 - sigma*alpha2;
          a2t[j*a1ts] = sigma*alpha1 + gamma*alpha2;
        }
        return 0;
      }
    };

    template<int M>
    struct SerialApplyRightGivensSizeSpecificInternal {
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const Kokkos::pair<ValueType,ValueType> &G,
             /* */ ValueType * a1, const int &a1s,
             /* */ ValueType * a2, const int &a2s) {
        typedef ValueType value_type;
        const value_type gamma = G.first;
        const value_type sigma = G.second;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int i=0;i<M;++i) {
          const value_type alpha1 = a1[i*a1s];
          const value_type alpha2 = a2[i*a2s];
          a1[i*a1s] = gamma*alpha1 - sigma*alpha2;
          a2[i*a1s] = sigma*alpha1 + gamma*alpha2;
        }
        return 0;
      }
    };

    struct SerialApplyRightGivensInternal {
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const Kokkos::pair<ValueType,ValueType> G,
             const int m, 
             /* */ ValueType * a1, const int a1s,
             /* */ ValueType * a2, const int a2s) {
        typedef ValueType value_type;
        /// G = [ gamma -sigma;
        ///       sigma  gamma ];
        /// A := A G'
        /// where gamma is G.first and sigma is G.second 
          
        const value_type gamma = G.first;
        const value_type sigma = G.second;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif        
        for (int i=0;i<m;++i) {
          const value_type alpha1 = a1[i*a1s];
          const value_type alpha2 = a2[i*a2s];
          a1[i*a1s] = gamma*alpha1 - sigma*alpha2;
          a2[i*a1s] = sigma*alpha1 + gamma*alpha2;
        }
        return 0;
      }
    };


    template<int M, int N>
    struct SerialApplyLeftRightGivensSizeSpecificInternal {
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const Kokkos::pair<ValueType,ValueType> &G12,
             /* */ ValueType *__restrict__ A, const int &as0, const int &as1) {
        typedef ValueType value_type;

        const value_type gamma12 = G12.first;
        const value_type sigma12 = G12.second;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int j=0;j<N;++j) {
          const value_type alpha1 = A[0*as0+j*as1];
          const value_type alpha2 = A[1*as0+j*as1];
          A[0*as0+j*as1] = ( gamma12*alpha1 - sigma12*alpha2 );
          A[1*as0+j*as1] = ( sigma12*alpha1 + gamma12*alpha2 );
        }

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int i=0;i<M;++i) {
          const value_type alpha1 = A[i*as0+0*as1];
          const value_type alpha2 = A[i*as0+1*as1];
          A[i*as0+0*as1] = ( gamma12*alpha1 - sigma12*alpha2 );
          A[i*as0+1*as1] = ( sigma12*alpha1 + gamma12*alpha2 );
        }
        return 0;
      }

      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const Kokkos::pair<ValueType,ValueType> &G12,
             const Kokkos::pair<ValueType,ValueType> &G13,
             /* */ ValueType *__restrict__ A, const int &as0, const int &as1) {
        typedef ValueType value_type;

        const value_type gamma12 = G12.first;
        const value_type sigma12 = G12.second;
        const value_type gamma13 = G13.first;
        const value_type sigma13 = G13.second;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int j=0;j<N;++j) {
          const value_type alpha2 = A[1*as0+j*as1];
          const value_type alpha3 = A[2*as0+j*as1];
          {
            const value_type alpha1 = A[0*as0+j*as1];
            A[0*as0+j*as1] = ( gamma12*alpha1 - sigma12*alpha2 );
            A[1*as0+j*as1] = ( sigma12*alpha1 + gamma12*alpha2 ); 
          }
          {
            const value_type alpha1 = A[0*as0+j*as1];
            A[0*as0+j*as1] = ( gamma13*alpha1 - sigma13*alpha3 );
            A[2*as0+j*as1] = ( sigma13*alpha1 + gamma13*alpha3 );
          }
        }

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int i=0;i<M;++i) {
          const value_type alpha2 = A[i*as0+1*as1];
          const value_type alpha3 = A[i*as0+2*as1];
          {
            const value_type alpha1 = A[i*as0+0*as1];
            A[i*as0+0*as1] = ( gamma12*alpha1 - sigma12*alpha2 );
            A[i*as0+1*as1] = ( sigma12*alpha1 + gamma12*alpha2 );
          }
          {
            const value_type alpha1 = A[i*as0+0*as1];
            A[i*as0+0*as1] = ( gamma13*alpha1 - sigma13*alpha3 );
            A[i*as0+2*as1] = ( sigma13*alpha1 + gamma13*alpha3 );
          }
        }
        return 0;
      }
    };

    
  }//  end namespace Experimental
} // end namespace KokkosBatched


#endif
