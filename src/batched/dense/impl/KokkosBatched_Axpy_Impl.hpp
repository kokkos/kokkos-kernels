#ifndef __KOKKOSBATCHED_AXPY_IMPL_HPP__
#define __KOKKOSBATCHED_AXPY_IMPL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

  ///
  /// Serial Internal Impl
  /// ==================== 
  struct SerialAxpyInternal {
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, 
           const ScalarType alpha, 
           /* */ ValueType *__restrict__ X, const int xs0,
           /* */ ValueType *__restrict__ Y, const int ys0) {

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (int i=0;i<m;++i)
        Y[i*ys0] += alpha*X[i*xs0];
        
      return 0;
    }

    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, 
           const ScalarType *__restrict__ alpha, const int alphas0,
           /* */ ValueType *__restrict__ X, const int xs0,
           /* */ ValueType *__restrict__ Y, const int ys0) {

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (int i=0;i<m;++i)
        Y[i*ys0] += alpha[i*alphas0]*X[i*xs0];
        
      return 0;
    }
      
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, const int n, 
           const ScalarType *__restrict__ alpha, const int alphas0,
           /* */ ValueType *__restrict__ X, const int xs0, const int xs1,
           /* */ ValueType *__restrict__ Y, const int ys0, const int ys1) {

      if (xs0 > xs1)
        for (int i=0;i<m;++i)
          invoke(n, alpha[i*alphas0], X+i*xs0, xs1, Y+i*ys0, ys1);
      else
        for (int j=0;j<n;++j)
          invoke(m, alpha, alphas0, X+j*xs1, xs0, Y+j*ys1, ys0);
        
      return 0;
    }
  };

  ///
  /// Team Internal Impl
  /// ==================== 
  struct TeamAxpyInternal {
    template<typename MemberType,
             typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, 
           const ScalarType alpha, 
           /* */ ValueType *__restrict__ X, const int xs0,
           /* */ ValueType *__restrict__ Y, const int ys0) {

      Kokkos::parallel_for
        (Kokkos::TeamThreadRange(member,m),
         [&](const int &i) {
          Y[i*ys0] += alpha*X[i*xs0];
        });
      //member.team_barrier();
      return 0;
    }

    template<typename MemberType,
             typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, 
           const ScalarType *__restrict__ alpha, const int alphas0, 
           /* */ ValueType *__restrict__ X, const int xs0,
           /* */ ValueType *__restrict__ Y, const int ys0) {

      Kokkos::parallel_for
        (Kokkos::TeamThreadRange(member,m),
         [&](const int &i) {
          Y[i*ys0] += alpha[i*alphas0]*X[i*xs0];
        });
      //member.team_barrier();
      return 0;
    }
      
    template<typename MemberType,
             typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, const int n, 
           const ScalarType *__restrict__ alpha, const int alphas0, 
           /* */ ValueType *__restrict__ X, const int xs0, const int xs1,
           /* */ ValueType *__restrict__ Y, const int ys0, const int ys1) {
      if (m > n) {
        Kokkos::parallel_for
          (Kokkos::TeamThreadRange(member,m),
           [&](const int &i) {
            SerialAxpyInternal::invoke(n, alpha[i*alphas0], X+i*xs0, xs1, Y+i*ys0, ys1);
          });
      } else {
        Kokkos::parallel_for
          (Kokkos::TeamThreadRange(member,n),
           [&](const int &j) {
            SerialAxpyInternal::invoke(m, alpha, alphas0, X+j*xs1, xs0, Y+j*ys1, ys0);
          });
      }
      //member.team_barrier();
      return 0;
    }
  };

  ///
  /// TeamVector Internal Impl
  /// ======================== 
  struct TeamVectorAxpyInternal {
    template<typename MemberType,
             typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, 
           const ScalarType alpha, 
           /* */ ValueType *__restrict__ X, const int xs0,
           /* */ ValueType *__restrict__ Y, const int ys0) {

      Kokkos::parallel_for
        (Kokkos::TeamVectorRange(member,m),
         [&](const int &i) {
          Y[i*ys0] += alpha*X[i*xs0];
        });
      //member.team_barrier();
      return 0;
    }

    template<typename MemberType,
             typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, 
           const ScalarType *__restrict__ alpha, const int alphas0, 
           /* */ ValueType *__restrict__ X, const int xs0,
           /* */ ValueType *__restrict__ Y, const int ys0) {

      Kokkos::parallel_for
        (Kokkos::TeamVectorRange(member,m),
         [&](const int &i) {
          Y[i*ys0] += alpha[i*alphas0]*X[i*xs0];
        });
      //member.team_barrier();
      return 0;
    }

    template<typename layout>
    KOKKOS_INLINE_FUNCTION
    static void
    getIndices(const int iTemp,
               const int n_rows,
               const int n_matrices,
               int &iRow,
               int &iMatrix) {
      if (std::is_same<layout, Kokkos::LayoutLeft>::value) {
        iRow    = iTemp / n_matrices;
        iMatrix = iTemp % n_matrices;
      }
      else {
        iRow    = iTemp % n_rows;
        iMatrix = iTemp / n_rows;
      }
    }

    template<typename MemberType,
             typename ScalarType,
             typename ValueType,
             typename layout>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, const int n, 
           const ScalarType *__restrict__ alpha, const int alphas0, 
           /* */ ValueType *__restrict__ X, const int xs0, const int xs1,
           /* */ ValueType *__restrict__ Y, const int ys0, const int ys1) {
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, 0, m * n),
          [&](const int& iTemp) {
            int i, j;
            getIndices<layout>(iTemp, n, m, j, i);
            Y[i*ys0+j*ys1] += alpha[i*alphas0] * X[i*xs0+j*xs1];
          });
      //member.team_barrier();
      return 0;
    }
  };

  ///
  /// Serial Impl
  /// ===========
  template<typename ViewType,
           typename alphaViewType>
  KOKKOS_INLINE_FUNCTION
  int
  SerialAxpy::
  invoke(const alphaViewType &alpha,
         const ViewType &X,
         const ViewType &Y) {
    return SerialAxpyInternal::template
      invoke<typename alphaViewType::non_const_value_type,
             typename ViewType::non_const_value_type>
             (X.extent(0), X.extent(1),
              alpha.data(), alpha.stride_0(),
              X.data(), X.stride_0(), X.stride_1(),
              Y.data(), Y.stride_0(), Y.stride_1());
  }

  ///
  /// Team Impl
  /// =========
    
  template<typename MemberType>
  template<typename ViewType,
           typename alphaViewType>
  KOKKOS_INLINE_FUNCTION
  int
  TeamAxpy<MemberType>::
  invoke(const MemberType &member, 
         const alphaViewType &alpha,
         const ViewType &X,
         const ViewType &Y) {
    return TeamAxpyInternal::template
      invoke<MemberType,
             typename alphaViewType::non_const_value_type,
             typename ViewType::non_const_value_type>
             (member, 
              X.extent(0), X.extent(1),
              alpha.data(), alpha.stride_0(),
              X.data(), X.stride_0(), X.stride_1(),
              Y.data(), Y.stride_0(), Y.stride_1());
  }

  ///
  /// TeamVector Impl
  /// ===============
    
  template<typename MemberType>
  template<typename ViewType,
           typename alphaViewType>
  KOKKOS_INLINE_FUNCTION
  int
  TeamVectorAxpy<MemberType>::
  invoke(const MemberType &member, 
         const alphaViewType &alpha,
         const ViewType &X,
         const ViewType &Y) {
    return TeamVectorAxpyInternal::
      invoke<MemberType,
             typename alphaViewType::non_const_value_type,
             typename ViewType::non_const_value_type,
             typename ViewType::array_layout>
             (member, 
             X.extent(0), X.extent(1),
             alpha.data(), alpha.stride_0(),
             X.data(), X.stride_0(), X.stride_1(),
             Y.data(), Y.stride_0(), Y.stride_1());
  }

}


#endif
