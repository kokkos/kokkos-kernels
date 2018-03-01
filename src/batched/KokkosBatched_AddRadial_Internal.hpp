#ifndef __KOKKOSBATCHED_ADD_RADIAL_INTERNAL_HPP__
#define __KOKKOSBATCHED_ADD_RADIAL_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"


namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Serial Internal Impl
    /// ==================== 
    struct SerialAddRadialInternal {
      template<typename ScalarType,
               typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const int m, 
             const ScalarType tiny, 
             /* */ ValueType *__restrict__ A, const int as) {
        typedef Kokkos::Details::ArithTraits<ScalarType> ats;
        const auto       abs_tiny =  ats::abs(tiny);
        const auto minus_abs_tiny = -abs_tiny;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int i=0;i<m;++i) {
          A[i*as] +=  ValueType(minus_abs_tiny)*ValueType(A[i*as] <  0);
          A[i*as] +=  ValueType(      abs_tiny)*ValueType(A[i*as] >= 0);
        }
        
        return 0;
      }
    };

    ///
    /// Team Internal Impl
    /// ==================
    struct TeamAddRadialInternal {
      template<typename MemberType,
               typename ScalarType,
               typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType &member,
             const int m, 
             const ScalarType tiny, 
             /* */ ValueType *__restrict__ A, const int as) {
        typedef Kokkos::Details::ArithTraits<ScalarType> ats;
        const auto       abs_tiny =  ats::abs(tiny);
        const auto minus_abs_tiny = -abs_tiny;
        
        Kokkos::parallel_for
          (Kokkos::TeamThreadRange(member,0,m),
           [&](const int &i) {
            A[i*as] +=  ValueType(minus_abs_tiny)*ValueType(A[i*as] <  0);
            A[i*as] +=  ValueType(      abs_tiny)*ValueType(A[i*as] >= 0);
          });
        //member.team_barrier();
        return 0;
      }
      
    };

  }//  end namespace Experimental
} // end namespace KokkosBatched


#endif
