#ifndef __KOKKOSBATCHED_SCALE_DECL_HPP__
#define __KOKKOSBATCHED_SCALE_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)


namespace KokkosBatched {
  namespace Experimental {

    ///
    /// Serial Scale
    ///

    namespace Serial {
      struct Scale {
	template<typename ScalarType,
		 typename AViewType>
	KOKKOS_INLINE_FUNCTION
	static int
	invoke(const ScalarType alpha,
	       const AViewType &A);
      };
    }

    ///
    /// Team Scale
    ///

    namespace Team {
      template<typename MemberType>
      struct Scale {
	template<typename ScalarType,
		 typename AViewType>
	KOKKOS_INLINE_FUNCTION
	static int
	invoke(const MemberType &member, 
	       const ScalarType alpha,
	       const AViewType &A);
      };
    }

  }
}


#endif
