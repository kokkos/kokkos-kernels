#ifndef __KOKKOSBATCHED_RIGHT_EIGENVECTOR_FROM_SCHUR_SERIAL_INTERNAL_HPP__
#define __KOKKOSBATCHED_RIGHT_EIGENVECTOR_FROM_SCHUR_SERIAL_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Normalize_Internal.hpp"
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
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const int m,
             /* */ ValueType * S, const int ss0, const int ss1,
             /* */ ValueType * V, const int vs0, const int vs1) {
        typedef ValueType value_type;
        typedef Kokkos::Details::ArithTraits<value_type> ats;
        typedef typename ats::mag_type mag_type;
        const value_type zero(0), one(1);
        
        /// partitions used for loop iteration
        Partition2x2<value_type> S_part2x2(ss0, ss1);
        Partition3x3<value_type> S_part3x3(ss0, ss1);
        
        Partition1x2<value_type> V_part1x2(vs1);
        Partition1x3<value_type> V_part1x3(vs1);
        
        /// initial partition of S where ABR has a zero dimension
        S_part2x2.partWithABR(S, m, m, 0, 0);
        V_part1x2.partWithAR(V, m, 0);

        const mag_type tol = ats::epsilon();
        int m_stl = m;
        for (;m_stl>0;) {
          const value_type subdiag = ats::abs(*(S_part2x2.ABR-ss0-2*ss1));
          if (subdiag < tol) {
            /// part 2x2 into 3x3
            S_part3x3.partWithATL(S_part2x2, 1, 1);
            V_part1x3.partWithAL(V_part1x2, 1);
            /// ---------------------------------------------------
            /// real eigenvalue 
            const value_type lambda = *S_part3x3.A11;
            
            /// initialize a right eigen vector
            for (int i=0;i<(m_stl-1);++i) V_part1x3.A1[i*vs0] = -S_part3x3.A01[i*ss0];
            V_part1x3.A1[(m_stl-1)*vs0] = one;
            for (int i=m_stl;i<m;++i) 
              V_part1x3.A1[i*vs0] = zero;
            
            /// perform shifted trsv
            SerialShiftedTrsvInternalUpper::invoke(m_stl-1, lambda,
                                                   S_part3x3.A00, ss0, ss1,
                                                   V_part1x3.A1,  vs0);
            
            /// normalize the vector
            SerialNormalizeInternal::invoke(m_stl, V_part1x3.A1, vs0);
            /// ---------------------------------------------------
            S_part2x2.mergeToABR(S_part3x3);
            V_part1x2.mergeToAR(V_part1x3);
            --m_stl;
          } else {
            // complex eigen pair 
            m_stl -= 2;
          }
        }
        
        /// case: m_stl = 0
        V[0] = one;
        for (int i=1;i<m;++i) V[i*vs0] = zero;

        return 0;
      }
    };

  }/// end namespace Experimental
} /// end namespace KokkosBatched


#endif
