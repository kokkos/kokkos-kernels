#ifndef __KOKKOSBATCHED_VECTOR_SIMD_HPP__
#define __KOKKOSBATCHED_VECTOR_SIMD_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Kokkos_Complex.hpp"

namespace KokkosBatched {
  namespace Experimental {

    template<typename T, int l>
    class Vector<SIMD<T>,l> {
    public:
      using type = Vector<SIMD<T>,l>;
      using value_type = T;
      using mag_type = typename Kokkos::Details::ArithTraits<T>::mag_type;

      enum : int { vector_length = l };
      typedef value_type data_type[vector_length];

      KOKKOS_INLINE_FUNCTION
      static const char* label() { return "SIMD"; }

    private:
      mutable data_type _data;

    public:
      KOKKOS_INLINE_FUNCTION Vector() {
#if                                                     \
  defined (KOKKOS_ENABLE_CUDA) &&                         \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
        // Kokkos::parallel_for
        //   (Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,member_type>(vector_length),
        //    [&](const int &i) {
        //     _data[i] = 0;
        //   });
#else
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
        for (int i=0;i<vector_length;++i)
          _data[i] = 0;
#endif
      }
      template<typename ArgValueType>
      KOKKOS_INLINE_FUNCTION Vector(const ArgValueType val) {
        static_assert(std::is_convertible<T,ArgValueType>::value, "input type is not convertible");
#if                                                     \
  defined (KOKKOS_ENABLE_CUDA) &&                         \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
        // Kokkos::parallel_for
        //   (Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,member_type>(vector_length),
        //    [&](const int &i) {
        //     _data[i] = val;
        //   });
#else
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
        for (int i=0;i<vector_length;++i)
          _data[i] = val;
#endif
      }
      template<typename ArgValueType>
      KOKKOS_INLINE_FUNCTION Vector(const Vector<SIMD<ArgValueType>,l> &b) {
        static_assert(std::is_convertible<T,ArgValueType>::value, "input type is not convertible");
#if                                                     \
  defined (KOKKOS_ENABLE_CUDA) &&                         \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
        // Kokkos::parallel_for
        //   (Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,member_type>(vector_length),
        //    [&](const int &i) {
        //     _data[i] = b._data[i];
        //   });
#else
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
        for (int i=0;i<vector_length;++i)
          _data[i] = b._data[i];
#endif
      }

      KOKKOS_INLINE_FUNCTION
      type& loadAligned(value_type const *p) {
#if                                                     \
  defined (KOKKOS_ENABLE_CUDA) &&                         \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
        // Kokkos::parallel_for
        //   (Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,member_type>(vector_length),
        //    [&](const int &i) {
        //     _data[i] = p[i];
        //   });
#else
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
        for (int i=0;i<vector_length;++i)
          _data[i] = p[i];
#endif
        return *this;
      }

      KOKKOS_INLINE_FUNCTION
      type& loadUnaligned(value_type const *p) {
        return loadAligned(p);
      }
      
      KOKKOS_INLINE_FUNCTION
      void storeAligned(value_type *p) const {
#if                                                     \
  defined (KOKKOS_ENABLE_CUDA) &&                         \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
        // Kokkos::parallel_for
        //   (Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,member_type>(vector_length),
        //    [&](const int &i) {
        //     p[i] = _data[i];
        //   });
#else
#if defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
#if defined( KOKKOS_ENABLE_PRAGMA_VECTOR )
#pragma vector always
#endif
        for (int i=0;i<vector_length;++i)
          p[i] = _data[i];
#endif
      }

      KOKKOS_INLINE_FUNCTION
      void storeUnaligned(value_type *p) const {
        storeAligned(p);
      }

      KOKKOS_INLINE_FUNCTION
      value_type& operator[](const int i) const {
        return _data[i];
      }

    };

  }
}

#include "KokkosBatched_Vector_SIMD_Arith.hpp"
#include "KokkosBatched_Vector_SIMD_Logical.hpp"
#include "KokkosBatched_Vector_SIMD_Relation.hpp"
#include "KokkosBatched_Vector_SIMD_Math.hpp"
#include "KokkosBatched_Vector_SIMD_Misc.hpp"

#endif
