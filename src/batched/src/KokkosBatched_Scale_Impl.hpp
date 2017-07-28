#ifndef __KOKKOSKERNELS_SCALE_IMPL_HPP__
#define __KOKKOSKERNELS_SCALE_IMPL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"
#include "KokkosKernels_Scale_Internal.hpp"


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
      Scale::
      invoke(const ScalarType alpha,
	     const AViewType &A) {
	return ScaleInternal::
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
      Scale<MemberType>::
      invoke(const MemberType &member, 
	     const ScalarType alpha,
	     const AViewType &A) {
	return ScaleInternal::
	  invoke(member, 
		 A.dimension_0(), A.dimension_1(),
		 alpha, 
		 A.data(), A.stride_0(), A.stride_1());
      }
    } // end namespace Team

  }
}


#endif
