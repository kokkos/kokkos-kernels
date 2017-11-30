#ifndef __KOKKOSBATCHED_VECTOR_SIMD_LOGICAL_HPP__
#define __KOKKOSBATCHED_VECTOR_SIMD_LOGICAL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Kokkos_Complex.hpp"

namespace KokkosBatched {
  namespace Experimental {

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<bool>,l> operator!(const Vector<SIMD<T>,l> &a) {
      Vector<SIMD<bool>,l> r_val;
      for (int i=0;i<l;++i)
        r_val[i] = !a[i];
      return r_val;
    }

    template<typename T0, typename T1, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<bool>,l> operator||(const Vector<SIMD<T0>,l> &a, const Vector<SIMD<T1>,l> &b) {
      Vector<SIMD<bool>,l> r_val;
      for (int i=0;i<l;++i)
        r_val[i] = a[i] || b[i];
      return r_val;
    }

    template<typename T0, typename T1, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<bool>,l> operator&&(const Vector<SIMD<T0>,l> &a, const Vector<SIMD<T1>,l> &b) {
      Vector<SIMD<bool>,l> r_val;
      for (int i=0;i<l;++i)
        r_val[i] = a[i] && b[i];
      return r_val;
    }

    template<typename T0, typename T1, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<bool>,l> operator||(const Vector<SIMD<T0>,l> &a, const T1 &b) {
      Vector<SIMD<bool>,l> r_val;
      for (int i=0;i<l;++i)
        r_val[i] = a[i] || b;
      return r_val;
    }

    template<typename T0, typename T1, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<bool>,l> operator&&(const Vector<SIMD<T0>,l> &a, const T1 &b) {
      Vector<SIMD<bool>,l> r_val;
      for (int i=0;i<l;++i)
        r_val[i] = a[i] && b;
      return r_val;
    }

    template<typename T0, typename T1, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<bool>,l> operator||(const T0 &a, const Vector<SIMD<T1>,l> &b) {
      Vector<SIMD<bool>,l> r_val;
      for (int i=0;i<l;++i)
        r_val[i] = a || b[i];
      return r_val;
    }

    template<typename T0, typename T1, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<bool>,l> operator&&(const T0 &a, const Vector<SIMD<T1>,l> &b) {
      Vector<SIMD<bool>,l> r_val;
      for (int i=0;i<l;++i)
        r_val[i] = a && b[i];
      return r_val;
    }

  }
}

#endif
