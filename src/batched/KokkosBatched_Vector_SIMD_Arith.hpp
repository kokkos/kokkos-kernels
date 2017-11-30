#ifndef __KOKKOSBATCHED_VECTOR_SIMD_ARITH_HPP__
#define __KOKKOSBATCHED_VECTOR_SIMD_ARITH_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Kokkos_Complex.hpp"

namespace KokkosBatched {
  namespace Experimental {

    /// simd, simd

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l>
    operator + (const Vector<SIMD<T>,l> &a,  const Vector<SIMD<T>,l> &b) {
      Vector<SIMD<T>,l> r_val;
#if                                                     \
  defined (KOKKOS_ENABLE_CUDA) &&                         \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
      // Kokkos::parallel_for
      //   (Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,typename VectorTag<SIMD<T,SpT>,l>::member_type>(l),
      //    [&](const int &i) {
      //     r_val[i] = a[i] + b[i];
      //   });
#else
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = a[i] + b[i];
#endif
      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l> &
    operator += (Vector<SIMD<T>,l> &a, const Vector<SIMD<T>,l> &b) {
      a = a + b;
      return a;
    }

    /// simd, real

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l>
    operator + (const Vector<SIMD<T>,l> &a, const T b) {
      return a + Vector<SIMD<T>,l>(b);
    }
    
    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l> 
    operator + (const T a, const Vector<SIMD<T>,l> &b) {
      return Vector<SIMD<T>,l>(a) + b;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l> & 
    operator += (Vector<SIMD<T>,l> &a, const T b) {
      a = a + b;
      return a;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l>
    operator ++ (Vector<SIMD<T>,l> &a, int) {
      Vector<SIMD<T>,l> a0 = a;
      a = a + typename Kokkos::Details::ArithTraits<T>::mag_type(1);
      return a0;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l> & 
    operator ++ (Vector<SIMD<T>,l> &a) {
      a = a + typename Kokkos::Details::ArithTraits<T>::mag_type(1);
      return a;
    }

    /// simd complex, real

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l>
    operator + (const Vector<SIMD<Kokkos::complex<T> >,l> &a, const T b) {
      return a + Vector<SIMD<Kokkos::complex<T> >,l>(b);
    }
    
    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator + (const T a, const Vector<SIMD<Kokkos::complex<T> >,l> &b) {
      return Vector<SIMD<Kokkos::complex<T> >,l>(a) + b;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> & 
    operator += (Vector<SIMD<Kokkos::complex<T> >,l> &a, const T b) {
      a = a + b;
      return a;
    }

    /// simd complex, complex 

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l>
    operator + (const Vector<SIMD<Kokkos::complex<T> >,l> &a, const Kokkos::complex<T> b) {
      return a + Vector<SIMD<Kokkos::complex<T> >,l>(b);
    }
    
    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator + (const Kokkos::complex<T> a, const Vector<SIMD<Kokkos::complex<T> >,l> &b) {
      return Vector<SIMD<Kokkos::complex<T> >,l>(a) + b;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> & 
    operator += (Vector<SIMD<Kokkos::complex<T> >,l> &a, const Kokkos::complex<T> b) {
      a = a + b;
      return a;
    }

    /// ---------------------------------------------------------------------------------------------------

    /// simd, simd

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l> 
    operator - (const Vector<SIMD<T>,l> &a, const Vector<SIMD<T>,l> &b) {
      Vector<SIMD<T>,l> r_val;
#if                                                     \
  defined (KOKKOS_ENABLE_CUDA) &&                         \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
      // Kokkos::parallel_for
      //   (Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,typename VectorTag<SIMD<T,SpT>,l>::member_type>(l),
      //    [&](const int &i) {
      //     r_val[i] = a[i] - b[i];
      //   });
#else
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = a[i] - b[i];
#endif
      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l> 
    operator - (const Vector<SIMD<T>,l> &a) {
#if                                                     \
  defined (KOKKOS_ENABLE_CUDA) &&                         \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
      // Kokkos::parallel_for
      //   (Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,typename VectorTag<SIMD<T,SpT>,l>::member_type>(l),
      //    [&](const int &i) {
      //     a[i] = -a[i];
      //   });
#else
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        a[i] = -a[i];
#endif
      return a;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l> &
    operator -= (Vector<SIMD<T>,l> &a, const Vector<SIMD<T>,l> &b) {
      a = a - b;
      return a;
    }

    /// simd, real

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l> 
    operator - (const Vector<SIMD<T>,l> &a, const T b) {
      return a - Vector<SIMD<T>,l>(b);
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l> 
    operator - (const T a, const Vector<SIMD<T>,l> &b) {
      return Vector<SIMD<T>,l>(a) - b;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l> &
    operator -= (Vector<SIMD<T>,l> &a, const T b) {
      a = a - b;
      return a;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l> 
    operator -- (Vector<SIMD<T>,l> &a, int) {
      Vector<SIMD<T>,l> a0 = a;
      a = a - typename Kokkos::Details::ArithTraits<T>::mag_type(1);
      return a0;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l> & 
    operator -- (Vector<SIMD<T>,l> &a) {
      a = a - typename Kokkos::Details::ArithTraits<T>::mag_type(1);
      return a;
    }

    /// simd complex, real
    
    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator - (const Vector<SIMD<Kokkos::complex<T> >,l> &a, const T b) {
      return a - Vector<SIMD<Kokkos::complex<T> >,l>(b);
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator - (const T a, const Vector<SIMD<Kokkos::complex<T> >,l> &b) {
      return Vector<SIMD<Kokkos::complex<T> >,l>(a) - b;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<Kokkos::complex<T> >,l> &
    operator -= (Vector<SIMD<Kokkos::complex<T> >,l> &a, const T b) {
      a = a - b;
      return a;
    }

    /// simd complex, complex
    
    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator - (const Vector<SIMD<Kokkos::complex<T> >,l> &a, const Kokkos::complex<T> b) {
      return a - Vector<SIMD<Kokkos::complex<T> >,l>(b);
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator - (const Kokkos::complex<T> a, const Vector<SIMD<Kokkos::complex<T> >,l> &b) {
      return Vector<SIMD<Kokkos::complex<T> >,l>(a) - b;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<Kokkos::complex<T> >,l> &
    operator -= (Vector<SIMD<Kokkos::complex<T> >,l> &a, const Kokkos::complex<T> b) {
      a = a - b;
      return a;
    }

    /// ---------------------------------------------------------------------------------------------------    

    /// simd, simd

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l> 
    operator * (const Vector<SIMD<T>,l> &a, const Vector<SIMD<T>,l> &b) {
      Vector<SIMD<T>,l> r_val;
#if                                                     \
  defined (KOKKOS_ENABLE_CUDA) &&                         \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
      // Kokkos::parallel_for
      //   (Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,typename VectorTag<SIMD<T,SpT>,l>::member_type>(l),
      //    [&](const int &i) {
      //     r_val[i] = a[i] * b[i];
      //   });
#else
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = a[i] * b[i];
#endif
      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l> &
    operator *= (Vector<SIMD<T>,l> &a, const Vector<SIMD<T>,l> &b) {
      a = a * b;
      return a;
    }


    /// simd, real

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l> 
    operator * (const Vector<SIMD<T>,l> &a, const T b) {
      return a * Vector<SIMD<T>,l>(b);
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l> 
    operator * (const T a, const Vector<SIMD<T>,l> &b) {
      return Vector<SIMD<T>,l>(a) * b;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l> &
    operator *= (Vector<SIMD<T>,l> &a, const T b) {
      a = a * b;
      return a;
    }

    /// simd complex, real

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator * (const Vector<SIMD<Kokkos::complex<T> >,l> &a, const T b) {
      return a * Vector<SIMD<Kokkos::complex<T> >,l>(b);
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator * (const T a, const Vector<SIMD<Kokkos::complex<T> >,l> &b) {
      return Vector<SIMD<Kokkos::complex<T> >,l>(a) * b;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<Kokkos::complex<T> >,l> &
    operator *= (Vector<SIMD<Kokkos::complex<T> >,l> &a, const T b) {
      a = a * b;
      return a;
    }

    /// simd complex, complex

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator * (const Vector<SIMD<Kokkos::complex<T> >,l> &a, const Kokkos::complex<T> b) {
      return a * Vector<SIMD<Kokkos::complex<T> >,l>(b);
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator * (const Kokkos::complex<T> a, const Vector<SIMD<Kokkos::complex<T> >,l> &b) {
      return Vector<SIMD<Kokkos::complex<T> >,l>(a) * b;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<Kokkos::complex<T> >,l> &
    operator *= (Vector<SIMD<Kokkos::complex<T> >,l> &a, const Kokkos::complex<T> b) {
      a = a * b;
      return a;
    }

    /// ---------------------------------------------------------------------------------------------------    

    /// simd, simd

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l> 
    operator / (const Vector<SIMD<T>,l> &a, const Vector<SIMD<T>,l> &b) {
      Vector<SIMD<T>,l> r_val;
#if                                                     \
  defined (KOKKOS_ENABLE_CUDA) &&                         \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
      // Kokkos::parallel_for
      //   (Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,typename VectorTag<SIMD<T,SpT>,l>::member_type>(l),
      //    [&](const int &i) {
      //     r_val[i] = a[i] / b[i];
      //   });
#else
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
      for (int i=0;i<l;++i)
        r_val[i] = a[i] / b[i];
#endif
      return r_val;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l> &
    operator /= (Vector<SIMD<T>,l> &a, const Vector<SIMD<T>,l> &b) {
      a = a / b;
      return a;
    }

    /// simd, real

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l> 
    operator / (const Vector<SIMD<T>,l> &a, const T b) {
      return a / Vector<SIMD<T>,l>(b);
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<T>,l> 
    operator / (const T a, const Vector<SIMD<T>,l> &b) {
      return Vector<SIMD<T>,l>(a) / b;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<T>,l> &
    operator /= (Vector<SIMD<T>,l> &a, const T b) {
      a = a / b;
      return a;
    }

    /// simd complex, real

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator / (const Vector<SIMD<Kokkos::complex<T> >,l> &a, const T b) {
      return a / Vector<SIMD<Kokkos::complex<T> >,l>(b);
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator / (const T a, const Vector<SIMD<Kokkos::complex<T> >,l> &b) {
      return Vector<SIMD<Kokkos::complex<T> >,l>(a) / b;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<Kokkos::complex<T> >,l> &
    operator /= (Vector<SIMD<Kokkos::complex<T> >,l> &a, const T b) {
      a = a / b;
      return a;
    }

    /// simd complex, complex

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator / (const Vector<SIMD<Kokkos::complex<T> >,l> &a, const Kokkos::complex<T> b) {
      return a / Vector<SIMD<Kokkos::complex<T> >,l>(b);
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static 
    Vector<SIMD<Kokkos::complex<T> >,l> 
    operator / (const Kokkos::complex<T> a, const Vector<SIMD<Kokkos::complex<T> >,l> &b) {
      return Vector<SIMD<Kokkos::complex<T> >,l>(a) / b;
    }

    template<typename T, int l>
    KOKKOS_INLINE_FUNCTION
    static
    Vector<SIMD<Kokkos::complex<T> >,l> &
    operator /= (Vector<SIMD<Kokkos::complex<T> >,l> &a, const Kokkos::complex<T> b) {
      a = a / b;
      return a;
    }

  }
}

#endif
