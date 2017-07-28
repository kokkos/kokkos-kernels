#ifndef __KOKKOSKERNELS_SET_DECL_HPP__
#define __KOKKOSKERNELS_SET_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)


namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Serial Set
    ///

    namespace Serial {
      struct Set {
	template<typename ScalarType,
		 typename AViewType>
	KOKKOS_INLINE_FUNCTION
	static int
	invoke(const ScalarType alpha,
	       const AViewType &A);
      };
    }

    ///
    /// Team Set
    ///

    namespace Team {
      template<typename MemberType>
      struct Set {
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
