#ifndef __KOKKOSBATCHED_COPY_DECL_HPP__
#define __KOKKOSBATCHED_COPY_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Serial Copy
    ///

    template<typename ArgTrans>
    struct SerialCopy {
      template<typename AViewType,
               typename BViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const AViewType &A,
             /* */ BViewType &B);
    };

    ///
    /// Team Copy
    ///

    template<typename MemberType, typename ArgTrans>
    struct TeamCopy {
      template<typename AViewType,
               typename BViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType &member,
             const AViewType &A,
             const BViewType &B);
    };


    ///
    /// Selective Interface
    ///
    template<typename MemberType,
             typename ArgTrans,
             typename ArgMode>
    struct Copy {
      template<typename AViewType,
               typename BViewType>
      KOKKOS_FORCEINLINE_FUNCTION
      static int
      invoke(const MemberType &member,
             const AViewType &A,
             const BViewType &B) {
        int r_val = 0;
        if (std::is_same<ArgMode,Mode::Serial>::value) {
          r_val = SerialCopy<ArgTrans>::invoke(A, B);
        } else if (std::is_same<ArgMode,Mode::Team>::value) {
          r_val = TeamCopy<MemberType,ArgTrans>::invoke(member, A, B);
        } 
        return r_val;
      }
    };      


  }
}


#endif
