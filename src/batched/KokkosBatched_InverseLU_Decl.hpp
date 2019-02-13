#ifndef __KOKKOSBATCHED_INVERSELU_DECL_HPP__
#define __KOKKOSBATCHED_INVERSELU_DECL_HPP__


/// \author Vinh Dang (vqdang@sandia.gov)

#include "KokkosBatched_Vector.hpp"
#include "KokkosBatched_Copy_Decl.hpp"
#include "KokkosBatched_Copy_Impl.hpp"
#include "KokkosBatched_SetIdentity_Decl.hpp"
#include "KokkosBatched_SetIdentity_Impl.hpp"
#include "KokkosBatched_SolveLU_Decl.hpp"

namespace KokkosBatched {
  namespace Experimental {
      
    template<typename ArgAlgo>
    struct SerialInverseLU {
      template<typename AViewType,
               typename WViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const AViewType &A,
             const WViewType &W) {
        SerialCopy<Trans::NoTranspose>::invoke(A, W);
        SerialSetIdentity::invoke(A);
        SerialSolveLU<Trans::NoTranspose,ArgAlgo>::invoke(W, A);        
      }
    };       

    template<typename MemberType,
             typename ArgAlgo>
    struct TeamInverseLU {
      template<typename AViewType,
               typename WViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType &member, 
             const AViewType &A,
             const WViewType &W) {
        TeamCopy<MemberType,Trans::NoTranspose>::invoke(member, A, W);
        TeamSetIdentity<MemberType>::invoke(member, A);
        TeamSolveLU<MemberType,Trans::NoTranspose,ArgAlgo>::invoke(member, W, A);        
      }
    };       
      
  }
}

#endif
