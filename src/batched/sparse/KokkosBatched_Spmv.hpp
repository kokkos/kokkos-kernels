#ifndef __KOKKOSBATCHED_SPMV_DECL_HPP__
#define __KOKKOSBATCHED_SPMV_DECL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

namespace KokkosBatched {

  ///
  /// Serial SPMV
  ///
  ///
  /// y <- alpha * A * x + beta * y
  ///
  ///

  template<typename ArgTrans>
  struct SerialSpmv {
    template<typename DViewType,
             typename IntView,
             typename xViewType,
             typename yViewType,
             typename alphaViewType,
             typename betaViewType,
             int dobeta>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const alphaViewType &alpha,
           const DViewType &D,
           const IntView &r,
           const IntView &c,
           const xViewType &x,
           const betaViewType &beta,
           const yViewType &y);
  };

  ///
  /// Team SPMV
  ///

  template<typename MemberType,
           typename ArgTrans>
  struct TeamSpmv {
    template<typename DViewType,
             typename IntView,
             typename xViewType,
             typename yViewType,
             typename alphaViewType,
             typename betaViewType,
             int dobeta>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const alphaViewType &alpha,
           const DViewType &D,
           const IntView &r,
           const IntView &c,
           const xViewType &x,
           const betaViewType &beta,
           const yViewType &y);
  };

  ///
  /// TeamVector SPMV
  ///

  template<typename MemberType,
           typename ArgTrans>
  struct TeamVectorSpmv {
    template<typename DViewType,
             typename IntView,
             typename xViewType,
             typename yViewType,
             typename alphaViewType,
             typename betaViewType,
             int dobeta>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const alphaViewType &alpha,
           const DViewType &D,
           const IntView &r,
           const IntView &c,
           const xViewType &x,
           const betaViewType &beta,
           const yViewType &y);
  };

  ///
  /// Selective Interface
  ///
  template<typename MemberType,
           typename ArgTrans,
           typename ArgMode>
  struct Spmv {
    template<typename DViewType,
             typename IntView,
             typename xViewType,
             typename yViewType,
             typename alphaViewType,
             typename betaViewType,
             int dobeta>
    KOKKOS_FORCEINLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const alphaViewType &alpha,
           const DViewType &D,
           const IntView &r,
           const IntView &c,
           const xViewType &x,
           const betaViewType &beta,
           const yViewType &y) {
      int r_val = 0;
      if (std::is_same<ArgMode,Mode::Serial>::value) {
        r_val = SerialSpmv<ArgTrans>::template invoke<DViewType, IntView, xViewType, yViewType, alphaViewType, betaViewType, dobeta>(alpha, D, r, c, x, beta, y);
      } else if (std::is_same<ArgMode,Mode::Team>::value) {
        r_val = TeamSpmv<MemberType,ArgTrans>::template invoke<DViewType, IntView, xViewType, yViewType, alphaViewType, betaViewType, dobeta>(member, alpha, D, r, c, x, beta, y);
      } else if (std::is_same<ArgMode,Mode::TeamVector>::value) {
        r_val = TeamVectorSpmv<MemberType,ArgTrans>::template invoke<DViewType, IntView, xViewType, yViewType, alphaViewType, betaViewType, dobeta>(member, alpha, D, r, c, x, beta, y);
      } 
      return r_val;
    }      
  };

}

#include "KokkosBatched_Spmv_Serial_Impl.hpp"
#include "KokkosBatched_Spmv_Team_Impl.hpp"
#include "KokkosBatched_Spmv_TeamVector_Impl.hpp"
#endif
