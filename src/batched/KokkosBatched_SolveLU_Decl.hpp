#ifndef __KOKKOSBATCHED_SOLVELU_DECL_HPP__
#define __KOKKOSBATCHED_SOLVELU_DECL_HPP__


/// \author Vinh Dang (vqdang@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Serial_Impl.hpp"
#include "KokkosBatched_Trsm_Team_Impl.hpp"

namespace KokkosBatched {
      
  template<typename ArgTrans,
           typename ArgAlgo>
  struct SerialSolveLU {
    // no piv version
    template<typename AViewType,
             typename BViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const AViewType &A,
           const BViewType &B) {
      const double one(1.0);
      if (std::is_same<ArgTrans,Trans::NoTranspose>::value) {
        //First, compute Y (= U*X) by solving the system L*Y = B for Y
        SerialTrsm<Side::Left,Uplo::Lower,ArgTrans,Diag::Unit,ArgAlgo>::invoke(one, A, B);
        //Second, compute X by solving the system U*X = Y for X
        SerialTrsm<Side::Left,Uplo::Upper,ArgTrans,Diag::NonUnit,ArgAlgo>::invoke(one, A, B);
      } else if (std::is_same<ArgTrans,Trans::Transpose>::value || 
                 std::is_same<ArgTrans,Trans::ConjTranspose>::value) {
        //First, compute Y (= L'*X) by solving the system U'*Y = B for Y
        SerialTrsm<Side::Left,Uplo::Upper,ArgTrans,Diag::NonUnit,ArgAlgo>::invoke(one, A, B); 
        //Second, compute X by solving the system L'*X = Y for X
        SerialTrsm<Side::Left,Uplo::Lower,ArgTrans,Diag::Unit,ArgAlgo>::invoke(one, A, B);
      }
      return 0;
    }
  };       

  template<typename MemberType,
           typename ArgTrans,
           typename ArgAlgo>
  struct TeamSolveLU {
    // no piv version
    template<typename AViewType,
             typename BViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const AViewType &A,
           const BViewType &B) {
      int r_val[2] = {};
      /// the scalar value cannot be derived from the view either A or B. 
      /// this is because A and B can have non trivial data type in it. 
      const double one(1.0);
      if (std::is_same<ArgTrans,Trans::NoTranspose>::value) {
        //First, compute Y (= U*X) by solving the system L*Y = B for Y
        TeamTrsm<MemberType,Side::Left,Uplo::Lower,ArgTrans,Diag::Unit,ArgAlgo>::invoke(member, one, A, B);
        //Second, compute X by solving the system U*X = Y for X
        TeamTrsm<MemberType,Side::Left,Uplo::Upper,ArgTrans,Diag::NonUnit,ArgAlgo>::invoke(member, one, A, B);
      } else if (std::is_same<ArgTrans,Trans::Transpose>::value || 
                 std::is_same<ArgTrans,Trans::ConjTranspose>::value) {
        //First, compute Y (= L'*X) by solving the system U'*Y = B for Y
        TeamTrsm<MemberType,Side::Left,Uplo::Upper,ArgTrans,Diag::NonUnit,ArgAlgo>::invoke(member, one, A, B); 
        //Second, compute X by solving the system L'*X = Y for X
        TeamTrsm<MemberType,Side::Left,Uplo::Lower,ArgTrans,Diag::Unit,ArgAlgo>::invoke(member, one, A, B);
      }
      return 0;
    }
  };       
      

  ///
  /// Selective Interface
  ///
  template<typename MemberType,
           typename ArgTrans,
           typename ArgMode, typename ArgAlgo>
  struct SolveLU {
    // no piv version
    template<typename AViewType,
             typename BViewType>
    KOKKOS_FORCEINLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const AViewType &A,
           const BViewType &B) {
      int r_val = 0;
      if (std::is_same<ArgMode,Mode::Serial>::value) {
        r_val = SerialSolveLU<ArgTrans,ArgAlgo>::invoke(A, B);
      } else if (std::is_same<ArgMode,Mode::Team>::value) {
        r_val = TeamSolveLU<MemberType,ArgTrans,ArgAlgo>::invoke(member, A, B);
      } 
      return r_val;
    }
  };           
    
}


#endif
