#ifndef __KOKKOSBATCHED_AXPY_DECL_HPP__
#define __KOKKOSBATCHED_AXPY_DECL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

namespace KokkosBatched {

  ///
  /// Serial AXPY
  ///
  ///
  /// y <- alpha * x + y
  ///
  ///

  struct SerialAxpy {
    template<typename ViewType,
             typename alphaViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const alphaViewType &alpha,
           const ViewType &X,
           const ViewType &Y);
  };

  ///
  /// Team AXPY
  ///

  template<typename MemberType>
  struct TeamAxpy {
    template<typename ViewType,
             typename alphaViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const alphaViewType &alpha,
           const ViewType &X,
           const ViewType &Y);
  };

  ///
  /// TeamVector AXPY
  ///

  template<typename MemberType>
  struct TeamVectorAxpy {
    template<typename ViewType,
             typename alphaViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const alphaViewType &alpha,
           const ViewType &X,
           const ViewType &Y);
  };

}

#include "KokkosBatched_Axpy_Impl.hpp"

#endif
