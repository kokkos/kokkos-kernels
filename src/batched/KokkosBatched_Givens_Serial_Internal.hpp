#ifndef __KOKKOSBATCHED_GIVENS_SERIAL_INTERNAL_HPP__
#define __KOKKOSBATCHED_GIVENS_SERIAL_INTERNAL_HPP__


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
    struct SerialGivensInternal {
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const ValueType   chi1,
             const Valuetype   chi2, 
             /* */ ValueType * gamma, 
             /* */ ValueType * sigma,
             /* */ ValueType * chi1_new) {
        typedef ValueType value_type;        
        const value_type zero(0);

        value_type cs, sn, r;
        if        (chi2 == zero) {
          r  = chi1;
          cs = one;
          sn = zero;
        } else if (chi1 == zero) {
          r  = chi2;
          cs = zero;
          sn = one;
        } else {
          // here we do not care overflow caused by the division although it is probable....
          r = Kokkos::Details::ArithTraits<value_type>::sqrt(chi1*chi1 + chi2*chi2);
          cs = chi1/r;
          sn = chi2/r;

          if (Kokkos::Details::ArithTraits<value_type>::abs(chi1) > 
              Kokkos::Details::ArithTraits<value_type>::abs(chi2) && cs < zero) {
            cs = -cs;
            sn = -sn;
            r  = -r;
          }

        }

        *gamma = cs;
        *sigma = sn;
        *chi1_new = r;

        return 0;
      }
    };

  }//  end namespace Experimental
} // end namespace KokkosBatched


#endif
