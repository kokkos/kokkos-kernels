#ifndef __KOKKOSBATCHED_VECTOR_SIMD_MATH_HPP__
#define __KOKKOSBATCHED_VECTOR_SIMD_MATH_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Kokkos_Complex.hpp"

namespace KokkosBatched {
  namespace Experimental {

    /// simd 

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    sqrt(const Vector<SIMD<T>,l> &a) {
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i) 
        r_val[i] = std::sqrt(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    cbrt(const Vector<SIMD<T>,l> &a) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::cbrt(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    log(const Vector<SIMD<T>,l> &a) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::log(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    log10(const Vector<SIMD<T>,l> &a) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::log10(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    exp(const Vector<SIMD<T>,l> &a) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::exp(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    pow(const Vector<SIMD<T>,l> &a, const Vector<SIMD<T>,l> &b) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::pow(a[i], b[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    pow(const T &a, const Vector<SIMD<T>,l> &b) {
      return pow(Vector<SIMD<T>,l>(a), b);
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    pow(const Vector<SIMD<T>,l> &a, const T &b) {
      return pow(a, Vector<SIMD<T>,l>(b));
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    sin(const Vector<SIMD<T>,l> &a) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::sin(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    cos(const Vector<SIMD<T>,l> &a) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::cos(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    tan(const Vector<SIMD<T>,l> &a) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::tan(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    sinh(const Vector<SIMD<T>,l> &a) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::sinh(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    cosh(const Vector<SIMD<T>,l> &a) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::cosh(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    tanh(const Vector<SIMD<T>,l> &a) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::tanh(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    asin(const Vector<SIMD<T>,l> &a) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::asin(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    acos(const Vector<SIMD<T>,l> &a) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::acos(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    atan(const Vector<SIMD<T>,l> &a) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::atan(a[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    atan2(const Vector<SIMD<T>,l> &a, const Vector<SIMD<T>,l> &b) {
      typedef Kokkos::Details::ArithTraits<T> ats;
      Vector<SIMD<T>,l> r_val;
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = std::atan2(a[i], b[i]);

      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    atan2(const T &a, const Vector<SIMD<T>,l> &b) {
      return atan2(Vector<SIMD<T>,l>(a), b);
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    atan2(const Vector<SIMD<T>,l> &a, const T &b) {
      return atan2(a, Vector<SIMD<T>,l>(b));
    }

  }
}

#endif
