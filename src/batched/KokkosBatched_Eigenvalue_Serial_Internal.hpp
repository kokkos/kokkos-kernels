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
        case 2: { 
          // do later
          break;
        }
        default: {
          // when m > 2; first test use implicit hessenberg qr step
          for (int m_H=m;m_H>1;--m_H) {
            const int last_idx = m_H-1;
            value_type *last_diag = H + last_idx*(hs0+hs1);
            value_type *left_from_last_diag = last_diag - hs1;
            value_type *last_2x2 = left_from_last_diag - hs0;
            int iter = 0; 
            bool converge = Kokkos::Details::ArithTraits<value_type>::abs(*left_from_last_diag) < tol;
            printf(" m_H = %d tol = %e\n", m_H, tol);
            while (!converge && iter < max_iteration) {
              value_type shift;
              {
                /// 0. No shift (for testing only)
                shift = zero;
                
                /// 1. Rayleigh quotient shift (all eigenvalues are real; used for testing)
                // shift = *last_diag;
                
                /// 2. Wilkinson shift (francis algorithm)
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

              /// QR sweep
              SerialHessenbergQR_WithShiftInternal::invoke(m_H, 
                                                           H, hs0, hs1, 
                                                           shift);

              converge = Kokkos::Details::ArithTraits<value_type>::abs(*left_from_last_diag) < tol;
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
        }
        return 0;
      }
    };

  }//  end namespace Experimental
} // end namespace KokkosBatched


#endif
