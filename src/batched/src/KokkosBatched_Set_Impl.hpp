#ifndef __KOKKOSKERNELS_SET_IMPL_HPP__
#define __KOKKOSKERNELS_SET_IMPL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"
#include "KokkosKernels_Set_Internal.hpp"


namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Serial Impl
    /// ===========
      
    namespace Serial {
        
      template<typename ScalarType,
	       typename AViewType>
      KOKKOS_INLINE_FUNCTION
      int
      Set::
      invoke(const ScalarType alpha,
	     const AViewType &A) {
	return SetInternal::
	  invoke(A.dimension_0(), A.dimension_1(),
		 alpha, 
		 A.data(), A.stride_0(), A.stride_1());
      }
    } // end namespace Serial

      ///
      /// Team Impl
      /// =========
      
    namespace Team {
        
      template<typename MemberType>
      template<typename ScalarType,
	       typename AViewType>
      KOKKOS_INLINE_FUNCTION
      int
      Set<MemberType>::
      invoke(const MemberType &member, 
	     const ScalarType alpha,
	     const AViewType &A) {
	return SetInternal::
	  invoke(member, 
		 A.dimension_0(), A.dimension_1(),
		 alpha, 
		 A.data(), A.stride_0(), A.stride_1());
      }
    } // end namespace Team
 
  } // end namespace Experimental
} //end namespace KokkosBatched


#endif
