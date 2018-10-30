#ifndef __KOKKOSBATCHED_EIGENVALUE_SERIAL_INTERNAL_HPP__
#define __KOKKOSBATCHED_EIGENVALUE_SERIAL_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_WilkinsonShift_Serial_Internal.hpp"
#include "KokkosBatched_HessenbergQR_WithShift_Serial_Internal.hpp"

namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Serial Internal Impl
    /// ==================== 
    ///
    /// this impl follows the flame interface of householder transformation
    ///
    struct SerialEigenvalueInternal {
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const int m,
             /* */ ValueType * H, const int hs0, const int hs1,
             const int max_iteration = 1000) {
        typedef ValueType value_type;
        typedef typename Kokkos::Details::ArithTraits<value_type>::mag_type mag_type;

        const value_type zero(0);
        const mag_type tol = 1000*Kokkos::Details::ArithTraits<value_type>::epsilon();

        /// Given a strict Hessenberg matrix H (m x m), 
        /// it computes all eigenvalues via Hessenberg QR algorithm with a shift
        switch (m) {
        case 0:
        case 1: {
          // do nothing
          break;
        }
        default: {
          // when m > 2; first test use implicit hessenberg qr step
          const int hs = hs0+hs1;
          int iter(0);
          bool converge = false; 
          while (!converge && iter < max_iteration) {
            /// 0. find a subsystem to compute eigenvalues
            int cnt = 1;
            
            // - find mbeg (first nonzero subdiag value)
            for (;cnt<m;++cnt) {
              const value_type val = *(H+cnt*hs-hs1);
              if (Kokkos::Details::ArithTraits<value_type>::abs(val) > tol) break;
            }
            const int mbeg = cnt-1;
            
            // - find mend (first zero subdiag value)
            for (;cnt<m;++cnt) {
              // find the first zero subdiag
              const value_type val = *(H+cnt*hs-hs1);
              if (Kokkos::Details::ArithTraits<value_type>::abs(val) < tol) break;              
            }
            const int mend = cnt;
            
            // if there exist non-converged eigen values
            //printf("iter %d mbeg %d mend %d\n", iter, mbeg, mend);
            if (mbeg < mend && mbeg < (m-1)) {              
              /// 1. find shift
              value_type shift;
              {
                /// case 0. No shift (testing only)
                //shift = zero;
                
                /// case 1. Rayleigh quotient shift (all eigenvalues are real; testing only)
                shift = *(H+(mend-1)*hs);
                
                /// case 2. Wilkinson shift (francis algorithm)
                // value_type lambda1, lambda2;
                // bool is_complex;
                // SerialWilkinsonShiftInternal::invoke(last_2x2[0*hs0+0*hs1], last_2x2[0*hs0+1*hs1], 
                //                                      last_2x2[1*hs0+0*hs1], last_2x2[1*hs0+1*hs1],
                //                                      &lambda1, &lambda2,
                //                                      &is_complex);
                
                // const auto target = last_2x2[1*hs0+1*hs1];
                // shift = ( Kokkos::Details::ArithTraits<value_type>::abs(target - lambda1) > 
                //           Kokkos::Details::ArithTraits<value_type>::abs(target - lambda2) ? lambda2 : lambda1 );
              }
              
              /// 2. QR sweep
              SerialHessenbergQR_WithShiftInternal::invoke(mend-mbeg, 
                                                           H+hs*mbeg, hs0, hs1,
                                                           shift);
            } else {
              /// 3. all eigenvalues are converged
              converge = true;
            }
            ++iter;
          }
          printf("H in eigenvalues converges in %d = \n", iter);
          for (int i=0;i<m;++i) {
            for (int j=0;j<m;++j) 
              printf(" %e ", H[i*hs0+j*hs1]);
            printf("\n");
          }
          if (!converge) 
            Kokkos::abort("Error: eigenvalues are not converged and reached the maximum number of iterations");
        }
        }
        return 0;
      }
    };

  }//  end namespace Experimental
} // end namespace KokkosBatched


#endif
