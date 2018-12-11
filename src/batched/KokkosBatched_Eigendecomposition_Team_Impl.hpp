#ifndef __KOKKOSBATCHED_EIGENDECOMPOSITION_TEAM_IMPL_HPP__
#define __KOKKOSBATCHED_EIGENDECOMPOSITION_TEAM_IMPL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Eigendecomposition_Team_Internal.hpp"

namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Team Impl
    /// =========

    template<typename MemberType>
    struct TeamEigendecomposition {
      template<typename AViewType,
               typename EViewType,
               typename UViewType,
               typename WViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType &member, 
             const AViewType &A,
             const EViewType &er, const EViewType &ei,
             const UViewType &UL, const UViewType &UR,
             const WViewType &W) {
        /// view checking
        const int m = A.extent(0);
        assert(m == A.extent(1)  && "Eigendecomposition: A is not square");
        assert(m == er.extent(0) && "Eigendecomposition: Length of er does not match to A's dimension");
        assert(m == ei.extent(0) && "Eigendecomposition: Length of ei does not match to A's dimension");
        assert(m == UL.extent(0) && "Eigendecomposition: Length of UL does not match to A's dimension");
        assert(m == UL.extent(1) && "Eigendecomposition: Width of UL does not match to A's dimension");
        assert(m == UR.extent(0) && "Eigendecomposition: Length of UR does not match to A's dimension");
        assert(m == UR.extent(1) && "Eigendecomposition: Width of UR does not match to A's dimension");
        assert(W.extent(0) >= (2*m*m+5*m) && "Eigendecomposition: workspace size is too small");
        assert(W.stride(0) == 1  && "Eigendecomposition: Provided workspace is not contiguous");
    
        /// static assert A,er,ei,UL,UR,W has the same value_type
        /// static assert all views have the same memory space
        return TeamEigendecompositionInternal
          ::invoke(member,
                   A.extent(0),
                   A.data(), A.stride(0), A.stride(1),
                   er.data(), er.stride(0),
                   ei.data(), ei.stride(0),
                   UL.data(), UL.stride(0), UL.stride(1),
                   UR.data(), UR.stride(0), UR.stride(1),
                   W.data(), W.extent(0));
      }
    };


  }/// end namespace Experimental
} /// end namespace KokkosBatched


#endif
