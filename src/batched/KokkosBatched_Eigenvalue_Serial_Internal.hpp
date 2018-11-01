#ifndef __KOKKOSBATCHED_EIGENVALUE_SERIAL_INTERNAL_HPP__
#define __KOKKOSBATCHED_EIGENVALUE_SERIAL_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_WilkinsonShift_Serial_Internal.hpp"
#include "KokkosBatched_HessenbergQR_WithShift_Serial_Internal.hpp"
#include "KokkosBatched_Francis_Serial_Internal.hpp"

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
             /* */ Kokkos::complex<ValueType> * e, const int es,
             const int max_iteration = 1000) {
        typedef ValueType value_type;
        typedef typename Kokkos::Details::ArithTraits<value_type>::mag_type mag_type;
        const mag_type tol = 10000*Kokkos::Details::ArithTraits<value_type>::epsilon();
        const value_type zero(0), nan(Kokkos::Details::ArithTraits<value_type>::nan());


        // printf("Hessenberg\n");
        // for (int i=0;i<m;++i) {
        //   for (int j=0;j<m;++j) 
        //     printf(" %e ", H[i*hs0+j*hs1]);
        //   printf("\n");
        // }

        /// Given a strict Hessenberg matrix H (m x m), 
        /// it computes all eigenvalues via Hessenberg QR algorithm with a shift
        switch (m) {
        case 0: {
          // do nothing
          break;
        }
        case 1: {
          // record the matrix value
          e[0] = Kokkos::complex<value_type>(H[0],zero);
          break;
        }
        case 2: {
          // compute eigenvalues from the characteristic determinant equation
          bool is_complex;
          SerialWilkinsonShiftInternal::invoke(H[0*hs0+0*hs1], H[0*hs0+1*hs1], 
                                               H[1*hs0+0*hs1], H[1*hs0+1*hs1],
                                               e, e+es,
                                               &is_complex);
          break;
        }
        default: {
          /// 0. clean eigenvalue array with Nan
          for (int i=0;i<m;++i) 
            e[i*es] = Kokkos::complex<value_type>(nan,zero);
          
          // when m > 2, use the francis method (or implicit hessenberg qr method for testing)
          const int hs = hs0+hs1;
          int iter(0);
          bool converge = false; 
          while (!converge && iter < max_iteration) {
            /// 1. find a subsystem to compute eigenvalues which are not converged yet
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
            if (mbeg < (mend-1)) { // && mbeg < (m-1)) {
#             if 1
              {
                /// Rayleigh quotient shift (all eigenvalues are real; testing only)
                const value_type shift = *(H+(mend-1)*hs); 

                /// QR sweep
                SerialHessenbergQR_WithShiftInternal::invoke(mend-mbeg, 
                                                             H+hs*mbeg, hs0, hs1,
                                                             shift);
                value_type *sub2x2 = H+(mend-2)*hs;
                if (Kokkos::Details::ArithTraits<value_type>::abs(*(sub2x2+hs0)) < tol) {
                  e[(mend-1)*es] = sub2x2[1*hs0+1*hs1];
                  printf(" --- single found at %d,%d, value = %e %e\n", mbeg, mend, e[(mend-1)*es].real(), e[(mend-1)*es].imag());
                } 
              }
#             endif

#             if 0
              {
                /// find a complex eigen pair
                Kokkos::complex<value_type> lambda1, lambda2;
                bool is_complex;
                value_type *sub2x2 = H+(mend-2)*hs;
                SerialWilkinsonShiftInternal::invoke(sub2x2[0*hs0+0*hs1], sub2x2[0*hs0+1*hs1], 
                                                     sub2x2[1*hs0+0*hs1], sub2x2[1*hs0+1*hs1],
                                                     &lambda1, &lambda2,
                                                     &is_complex);

                if ((mend-mbeg) == 2) {
                  /// short cut: eigenvalues are from wilkinson shift
                  sub2x2[1*hs0+0*hs1] = zero;
                  e[mbeg+0*es] = lambda1;
                  e[mbeg+1*es] = lambda2;

                  printf(" --- 2x2    found at %d, value = %e %e\n", (mbeg+1), e[(mbeg+1)*es].real(), e[(mbeg+1)*es].imag());
                  printf(" --- 2x2    found at %d, value = %e %e\n", (mbeg),   e[(mbeg  )*es].real(), e[(mbeg  )*es].imag());
                } else {
                  /// Francis step
                  SerialFrancisInternal::invoke(mend-mbeg,
                                                H+hs*mbeg, hs0, hs1,
                                                lambda1, lambda2,
                                                is_complex);
                  if        (Kokkos::Details::ArithTraits<value_type>::abs(*(sub2x2+hs0)) < tol) {
                    e[(mend-1)*es] = sub2x2[1*hs0+1*hs1];
                    printf(" --- single found at %d, value = %e %e\n", (mend-1), e[(mend-1)*es].real(), e[(mend-1)*es].imag());
                  } else if (Kokkos::Details::ArithTraits<value_type>::abs(*(sub2x2-hs1)) < tol) {
                    sub2x2[1*hs0+0*hs1] = zero;
                    e[(mend-1)*es] = lambda1;
                    e[(mend-2)*es] = lambda2;

                    printf(" --- double found at %d, value = %e %e\n", (mend-1), e[(mend-1)*es].real(), e[(mend-1)*es].imag());
                    printf(" --- double found at %d, value = %e %e\n", (mend-2), e[(mend-2)*es].real(), e[(mend-2)*es].imag());
                  }
                }

                // printf("H at iter %d\n", iter);
                // for (int i=0;i<m;++i) {
                //   for (int j=0;j<m;++j) 
                //     printf(" %e ", H[i*hs0+j*hs1]);
                //   printf("\n");
                // }
              }
#             endif
                                             
            } else {
              /// all eigenvalues are converged
              converge = true;
            }
            ++iter;
          }
          /// record missing real eigenvalues from the diagonals
          printf("iteration number %d\n", iter);
          for (int i=0;i<m;++i) {
            if (Kokkos::Details::ArithTraits<value_type>::isNan(e[i*es].real())) {
              printf(" not convering at %d\n", i );
              e[i*es] = Kokkos::complex<value_type>(H[i*hs],zero);
            }
          }
          if (!converge) {
            printf("Error: eigenvalues are not converged and reached the maximum number of iterations\n");
            //Kokkos::abort("Error: eigenvalues are not converged and reached the maximum number of iterations");
          }
          break;
        }          
        }
        return 0;
      }
    };

  }//  end namespace Experimental
} // end namespace KokkosBatched


#endif
