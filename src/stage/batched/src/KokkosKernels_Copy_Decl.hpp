#ifndef __KOKKOSKERNELS_COPY_DECL_HPP__
#define __KOKKOSKERNELS_COPY_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {
      ///
      /// Serial Copy
      ///

      namespace Serial {
        template<typename ArgTrans>
        struct Copy {
          template<typename AViewType,
                   typename BViewType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const AViewType &A,
                 /* */ BViewType &B);
        };
      }

      ///
      /// Team Set
      ///

      namespace Team {
        template<typename MemberType, typename ArgTrans>
        struct Copy {
          template<typename AViewType,
                   typename BViewType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const MemberType &member,
                 const AViewType &A,
                 const BViewType &B);
        };
      }

    }
  }
}

#endif
