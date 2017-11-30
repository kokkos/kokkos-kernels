#ifndef __KOKKOSBATCHED_VECTOR_SIMD_RELATION_HPP__
#define __KOKKOSBATCHED_VECTOR_SIMD_RELATION_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Kokkos_Complex.hpp"

namespace KokkosBatched {
  namespace Experimental {

    // vector, vector

#define KOKKOSBATCHED_RELATION_OPERATOR_VV(op)                          \
    template<typename T1, typename T2, int l>                           \
    KOKKOS_INLINE_FUNCTION                                              \
    const Vector<SIMD<bool>,l> op (const Vector<SIMD<T1>,l> &a, const Vector<SIMD<T2>,l> &b) { \
      static_assert(std::is_convertible<T1,T2>::value, "value types must be convertible"); \
      Vector<SIMD<bool>,l> r_val;                                       \
      for (int i=0;i<l;++i)                                             \
        r_val[i] = op(a[i], b[i]);                                 \
      return r_val;                                                     \
    }                                                                   

    KOKKOSBATCHED_RELATION_OPERATOR_VV(operator<)
    KOKKOSBATCHED_RELATION_OPERATOR_VV(operator>)
    KOKKOSBATCHED_RELATION_OPERATOR_VV(operator<=)
    KOKKOSBATCHED_RELATION_OPERATOR_VV(operator>=)
    KOKKOSBATCHED_RELATION_OPERATOR_VV(operator==)
    KOKKOSBATCHED_RELATION_OPERATOR_VV(operator!=)

    // vector, scalar

#define KOKKOSBATCHED_RELATION_OPERATOR_VS(op)                          \
    template<typename T1, typename T2, int l>                           \
    KOKKOS_INLINE_FUNCTION                                              \
    const Vector<SIMD<bool>,l> op (const Vector<SIMD<T1>,l> &a, const T2 b) { \
      static_assert(std::is_convertible<T1,T2>::value, "value types must be convertible"); \
      Vector<SIMD<bool>,l> r_val;                                       \
      for (int i=0;i<l;++i)                                             \
        r_val[i] = op(a[i], b);                                         \
      return r_val;                                                     \
    }                                                                   

    KOKKOSBATCHED_RELATION_OPERATOR_VS(operator<)
    KOKKOSBATCHED_RELATION_OPERATOR_VS(operator>)
    KOKKOSBATCHED_RELATION_OPERATOR_VS(operator<=)
    KOKKOSBATCHED_RELATION_OPERATOR_VS(operator>=)
    KOKKOSBATCHED_RELATION_OPERATOR_VS(operator==)
    KOKKOSBATCHED_RELATION_OPERATOR_VS(operator!=)

    // scalar, vector

#define KOKKOSBATCHED_RELATION_OPERATOR_SV(op)                          \
    template<typename T1, typename T2, int l>                           \
    KOKKOS_INLINE_FUNCTION                                              \
    const Vector<SIMD<bool>,l> op (const T2 a, const Vector<SIMD<T1>,l> &b) { \
      static_assert(std::is_convertible<T1,T2>::value, "value types must be convertible"); \
      Vector<SIMD<bool>,l> r_val;                                       \
      for (int i=0;i<l;++i)                                             \
        r_val[i] = op(a, b[i]);                                         \
      return r_val;                                                     \
    }                                                                   

    KOKKOSBATCHED_RELATION_OPERATOR_SV(operator<)
    KOKKOSBATCHED_RELATION_OPERATOR_SV(operator>)
    KOKKOSBATCHED_RELATION_OPERATOR_SV(operator<=)
    KOKKOSBATCHED_RELATION_OPERATOR_SV(operator>=)
    KOKKOSBATCHED_RELATION_OPERATOR_SV(operator==)
    KOKKOSBATCHED_RELATION_OPERATOR_SV(operator!=)

    
  }
}

#endif
