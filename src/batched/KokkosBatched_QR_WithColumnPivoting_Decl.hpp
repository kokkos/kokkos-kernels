#ifndef __KOKKOSBATCHED_QR_WITH_COLUMNPIVOTING_DECL_HPP__
#define __KOKKOSBATCHED_QR_WITH_COLUMNPIVOTING_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

  ///
  /// Serial QR
  ///

  template<typename ArgAlgo>
  struct SerialQR_WithColumnPivoting {
    template<typename AViewType,
             typename tViewType,
	     typename pViewType,
             typename wViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const AViewType &A,
           const tViewType &t,
	   const pViewType &p,
           const wViewType &w);
  };

  ///
  /// Team QR
  ///

  template<typename MemberType,
           typename ArgAlgo>
  struct TeamQR_WithColumnPivoting {
    template<typename AViewType,
             typename tViewType,
	     typename pViewType,
             typename wViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member,
           const AViewType &A,
           const tViewType &t,
	   const pViewType &p,
           const wViewType &w) {
      /// not implemented
      return -1;
    }
  };

  ///
  /// TeamVector QR
  ///

  template<typename MemberType,
           typename ArgAlgo>
  struct TeamVectorQR_WithColumnPivoting {
    template<typename AViewType,
             typename tViewType,
	     typename pViewType,
             typename wViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const AViewType &A,
           const tViewType &t,
	   const pViewType &p,
           const wViewType &w);
  };

  ///
  /// Selective Interface
  ///
  template<typename MemberType,
           typename ArgMode,
           typename ArgAlgo>
  struct QR_WithColumnPivoting {
    template<typename AViewType,
             typename tViewType,
	     typename pViewType,
             typename wViewType>
    KOKKOS_FORCEINLINE_FUNCTION
    static int
    invoke(const MemberType &member,
           const AViewType &A,
           const tViewType &t,
	   const pViewType &p,
           const wViewType &w) {
      int r_val = 0;
      if (std::is_same<ArgMode,Mode::Serial>::value) {
        r_val = SerialQR_WithColumnPivoting<ArgAlgo>::invoke(A, t, p, w);
      } else if (std::is_same<ArgMode,Mode::Team>::value) {
        r_val = TeamQR_WithColumnPivoting<MemberType,ArgAlgo>::invoke(member, A, t, p, w);
      } else if (std::is_same<ArgMode,Mode::TeamVector>::value) {
        r_val = TeamVectorQR_WithColumnPivoting<MemberType,ArgAlgo>::invoke(member, A, t, p, w);
      } 
      return r_val;
    }
  };      

}

//#include "KokkosBatched_QR_WithColumnPivoting_Serial_Impl.hpp"
#include "KokkosBatched_QR_WithColumnPivoting_TeamVector_Impl.hpp"

#endif
