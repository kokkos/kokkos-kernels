#ifndef __KOKKOSBATCHED_RIGHT_EIGENVECTOR_FROM_SCHUR_SERIAL_INTERNAL_HPP__
#define __KOKKOSBATCHED_RIGHT_EIGENVECTOR_FROM_SCHUR_SERIAL_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_ShiftedTrsv_Serial_Internal.hpp"

namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Serial Internal Impl
    /// ==================== 
    ///
    /// this impl follows the flame interface of householder transformation
    ///
    struct SerialRightEigenvectorFromSchurInternal {
      /// Given a quasi upper triangular matrix S (m x m), this computes all right 
      /// eigenvectors.
      /// 
      /// Parameters:
      ///   [in]m 
      ///     A dimension of the square matrix S.
      ///   [in]S, [in]ss0, [in]ss1 
      ///     A quasi upper triangular part of Schur decomposition which is computed 
      ///       A = U^H S U
      ///   [out]V, [in]vs0, [out]vs1 
      ///     A set of right eigen vectors.
      ///   [in]w
      ///     contiguous workspace that can hold complex array (m)
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const int m,
             /* */ ValueType * S, const int ss0, const int ss1,
             /* */ ValueType * V, const int vs0, const int vs1,
             /* */ ValueType * w) {
        typedef ValueType value_type;
        typedef Kokkos::Details::ArithTraits<value_type> ats;
        typedef typename ats::mag_type mag_type;
        typedef Kokkos::complex<value_type> complex_type;

        const value_type zero(0), one(1);
        SerialSetInternal::invoke(m, m, zero, V, vs0, vs1);

        value_type *b = w; // consider complex case

        /// partitions used for loop iteration
        Partition2x2<value_type> S_part2x2(ss0, ss1);
        Partition3x3<value_type> S_part3x3(ss0, ss1);
        
        Partition2x1<value_type> V_part2x1(vs0);
        Partition3x1<value_type> V_part3x1(vs0);
        
        /// initial partition of S where ATL has a zero dimension
        S_part2x2.partWithATL(S, m, m, 0, 0);
        V_part2x1.partWithAT(V, m, 0);

        const mag_type tol = ats::epsilon();
        int m_stl = 0;
        for (;m_stl<(m-1);) {
          const value_type subdiag = ats::abs(*(S_part2x2.ABR+ss0));

          /// part 2x2 into 3x3
          const bool subdiag_is_zero = subdiag < tol;
          const int mA11 = subdiag_is_zero ? 1 : 2;
          S_part3x3.partWithABR(S_part2x2, mA11, mA11);
          V_part3x1.partWithAB(V_part2x1, mA11);

          const int m_stl_plus_mA11 = m_stl+mA11;
          if (subdiag_is_zero) {

            /// real eigenvalue 
            const value_type lambda = *S_part3x3.A11;

            /// initialize a right hand side
            b[m_stl] = one;
            printf("rhs = \n");
            for (int j=0;j<(m-m_stl_plus_mA11);++j) {
              b[j+m_stl_plus_mA11] = -S_part3x3.A12[j*ss1];
              printf(" %e \n",  b[j+m_stl_plus_mA11]);
            }
            /// perform shifted trsv (transposed)
            SerialShiftedTrsvInternalLower::invoke(m-m_stl_plus_mA11, lambda,
                                                   S_part3x3.A22, ss1, ss0,
                                                   w+m_stl_plus_mA11, 1);

            printf("sol = \n");
            for (int j=0;j<(m-m_stl_plus_mA11);++j) {
              printf(" %e \n",  b[j+m_stl_plus_mA11]);
            }

            /// copy back to V (row wise copy)
            for (int j=0;j<m_stl;++j) V_part3x1.A1[j*vs1] = zero;
            for (int j=m_stl;j<m;++j) V_part3x1.A1[j*vs1] = b[j];              
            
            printf("V, m_stl = %d, mA11 = %d\n", m_stl, mA11);
            for (int i=0;i<m;++i) {
              for (int j=0;j<m;++j) 
                printf(" %e ", V[i*vs0+j*vs1]);
              printf("\n");
            }

          } else {
            /// complex eigen pair  
            const value_type 
              alpha11 = S_part3x3.A11[0],
              alpha12 = S_part3x3.A11[ss1],
              alpha21 = S_part3x3.A11[ss0],
              beta = ats::sqrt(-alpha12*alpha21);

            const complex_type lambda(alpha11, beta);
            complex_type * wc = (complex_type*)(w);
            
            /// initialize a right hand side
            wc[m_stl  ] = complex_type(beta,  zero);
            wc[m_stl+1] = complex_type(zero, -alpha21);
            const value_type * S_A12_a = S_part3x3.A12;
            const value_type * S_A12_b = S_part3x3.A12 + ss0;
            for (int j=0;j<(m-m_stl_plus_mA11);++j)
              wc[j+m_stl_plus_mA11] = complex_type(-S_A12_a[j*ss1]*beta, S_A12_b[j*ss1]*alpha21);
            
            /// perform shifted trsv
            SerialShiftedTrsvInternalLower::invoke(m-m_stl_plus_mA11, lambda,
                                                   S_part3x3.A22, ss1, ss0,
                                                   wc+m_stl_plus_mA11, 1);
            
            /// copy back to V
            value_type * V_A1_r = V_part3x1.A1;
            value_type * V_A1_i = V_part3x1.A1 + vs0;
            for (int j=0;j<m_stl;++j) { 
              V_A1_r[j*vs1] = zero;
              V_A1_i[j*vs1] = zero;
            }
            for (int j=m_stl;j<m;++j) { 
              V_A1_r[j*vs1] = wc[j].real();              
              V_A1_i[j*vs1] = wc[j].imag();
            }              
            printf("V, m_stl = %d, mA11 = %d\n", m_stl, mA11);
            for (int i=0;i<m;++i) {
              for (int j=0;j<m;++j) 
                printf(" %e ", V[i*vs0+j*vs1]);
              printf("\n");
            }

            /// ---------------------------------------------------
          }
          S_part2x2.mergeToATL(S_part3x3);
          V_part2x1.mergeToAT(V_part3x1);
          m_stl += mA11;
        }
        
        /// case: m_stl = m-1
        if (m_stl < m) {
          value_type * VV = V+m_stl*vs0;
          for (int j=0;j<m_stl;++j) VV[j*vs1] = zero;
          VV[m_stl*vs1] = one;
        }

        return 0;
      }
    };

  }/// end namespace Experimental
} /// end namespace KokkosBatched


#endif
