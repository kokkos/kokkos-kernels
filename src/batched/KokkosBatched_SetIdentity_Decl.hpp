#ifndef __KOKKOSBATCHED_SET_IDENTITY_DECL_HPP__
#define __KOKKOSBATCHED_SET_IDENTITY_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)


namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Serial SetIdentity
    ///

    struct SerialSetIdentity {
      template<typename AViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const AViewType &A);
    };

    ///
    /// Team Set
    ///

    template<typename MemberType>
    struct TeamSetIdentity {
      template<typename AViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType &member,
             const AViewType &A);
    };

  }
}


#endif
