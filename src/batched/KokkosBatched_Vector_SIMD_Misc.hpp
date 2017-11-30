#ifndef __KOKKOSBATCHED_VECTOR_SIMD_MISC_HPP__
#define __KOKKOSBATCHED_VECTOR_SIMD_MISC_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Kokkos_Complex.hpp"

namespace KokkosBatched {
  namespace Experimental {

    // scalar, scalar

    template<typename T>
    KOKKOS_INLINE_FUNCTION
    static
    T
    conditional_assign(const bool cond,
                       const T &if_true_val,
                       const T &if_false_val) {
      r_val = cond ? if_true_val : if_false_val;
    }

    template<typename T0, typename T1, typename T2>
    KOKKOS_INLINE_FUNCTION
    static
    void
    conditional_assign(/* */ T0 &r_val,
                       const bool cond,
                       const T1 &if_true_val,
                       const T2 &if_false_val) {
      static_assert(std::is_convertible<T0,T1>::value, "r_val type must be convertible to if_true_value type");
      static_assert(std::is_convertible<T0,T2>::value, "r_val type must be convertible to if_true_value type");
      r_val = cond ? if_true_val : if_false_val;
    }

    // vector, scalar

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    conditional_assign(const Vector<SIMD<bool>,l> &cond,
                       const Vector<SIMD<T>,l> &if_true_val,
                       const T &if_false_val) {
      Vector<SIMD<T>,l> r_val;
      for (int i=0;i<l;++i)
        r_val[i] = cond[i] ? if_true_val[i] : if_false_val;
      return r_val;
    }

    template<typename T0, typename T1, typename T2, int l>
    KOKKOS_INLINE_FUNCTION

    static
    void
    conditional_assign(/* */ Vector<SIMD<T0>,l> &r_val,
                       const Vector<SIMD<bool>,l> &cond,
                       const Vector<SIMD<T1>,l> &if_true_val,
                       const T2 &if_false_val) {
      static_assert(std::is_convertible<T0,T1>::value, "r_val type must be convertible to if_true_value type");
      static_assert(std::is_convertible<T0,T2>::value, "r_val type must be convertible to if_true_value type");
      for (int i=0;i<l;++i)
        r_val[i] = cond[i] ? if_true_val[i] : if_false_val;
    }

    // scalar, vector

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    conditional_assign(const Vector<SIMD<bool>,l> &cond,
                       const T &if_true_val,
                       const Vector<SIMD<T>,l> &if_false_val) {
      Vector<SIMD<T>,l> r_val;
      for (int i=0;i<l;++i)
        r_val[i] = cond[i] ? if_true_val : if_false_val[i];
      return r_val;
    }

    template<typename T0, typename T1, typename T2, int l>
    KOKKOS_INLINE_FUNCTION
    static
    void
    conditional_assign(/* */ Vector<SIMD<T0>,l> &r_val,
                       const Vector<SIMD<bool>,l> &cond,
                       const T1 &if_true_val,
                       const Vector<SIMD<T2>,l> &if_false_val){
      static_assert(std::is_convertible<T0,T1>::value, "r_val type must be convertible to if_true_value type");
      static_assert(std::is_convertible<T0,T2>::value, "r_val type must be convertible to if_true_value type");
      for (int i=0;i<l;++i)
        r_val[i] = cond[i] ? if_true_val : if_false_val[i];
    }

    // vector, vector

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    conditional_assign(const Vector<SIMD<bool>,l> &cond,
                       const Vector<SIMD<T>,l> &if_true_val,
                       const Vector<SIMD<T>,l> &if_false_val) {
      Vector<SIMD<T>,l> r_val;
      for (int i=0;i<l;++i)
        r_val[i] = cond[i] ? if_true_val[i] : if_false_val[i];
      return r_val;
    }

    template<typename T0, typename T1, typename T2, int l>
    KOKKOS_INLINE_FUNCTION
    static
    void
    conditional_assign(/* */ Vector<SIMD<T0>,l> &r_val,
                       const Vector<SIMD<bool>,l> &cond,
                       const Vector<SIMD<T1>,l> &if_true_val,
                       const Vector<SIMD<T2>,l> &if_false_val){
      static_assert(std::is_convertible<T0,T1>::value, "r_val type must be convertible to if_true_value type");
      static_assert(std::is_convertible<T0,T2>::value, "r_val type must be convertible to if_true_value type");
      for (int i=0;i<l;++i)
        r_val[i] = cond[i] ? if_true_val[i] : if_false_val[i];
    }
    
    template<typename T, int l, typename BinaryOp>
    KOKKOS_INLINE_FUNCTION
    static
    T
    reduce(const Vector<SIMD<T>,l> &val, const BinaryOp &func) {
      T r_val = val[0];
      for (int i=1;i<l;++i)
        r_val = func(r_val, val[i]);
    }
    
  }
}

#endif
