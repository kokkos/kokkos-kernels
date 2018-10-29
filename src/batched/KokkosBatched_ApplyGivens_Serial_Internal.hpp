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

        for (int j=0;j<n;++j) {
          const value_type alpha1 = a1t[j*a1ts];
          const value_type alpha2 = a2t[j*a2ts];
          a1t[j*a1ts] = gamma*alpha1 - sigma*alpha2;
          a2t[j*a1ts] = sigma*alpha1 + gamma*alpha2;
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

        for (int i=0;i<m;++i) {
          const value_type alpha1 = a1[i*a1s];
          const value_type alpha2 = a2[i*a2s];
          a1[i*a1s] = gamma*alpha1 - sigma*alpha2;
          a2[i*a1s] = sigma*alpha1 + gamma*alpha2;
        }
        return 0;
      }
    };

  }//  end namespace Experimental
} // end namespace KokkosBatched


#endif
