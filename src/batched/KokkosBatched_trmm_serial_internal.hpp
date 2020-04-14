/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef __KOKKOSBATCHED_TRMM_SERIAL_INTERNAL_HPP__
#define __KOKKOSBATCHED_TRMM_SERIAL_INTERNAL_HPP__

#include "KokkosBatched_Util.hpp"

#include "KokkosBatched_Set_Internal.hpp"
#include "KokkosBatched_Scale_Internal.hpp"

namespace KokkosBatched {

  template<typename AlgoType>
  struct SerialTrmmInternalLeftLower {
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int 
    invoke(const bool use_unit_diag,
           const bool do_conj,
           const int am, const int an, 
           const int bm, const int bn, 
           const ScalarType alpha,
           const ValueType *__restrict__ A, const int as0, const int as1,
           /**/  ValueType *__restrict__ B, const int bs0, const int bs1);
  };

  template<typename AlgoType>
  struct SerialTrmmInternalLeftUpper {
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int 
    invoke(const bool use_unit_diag,
           const bool do_conj,
           const int am, const int an, 
           const int bm, const int bn, 
           const ScalarType alpha,
           const ValueType *__restrict__ A, const int as0, const int as1,
           /**/  ValueType *__restrict__ B, const int bs0, const int bs1);
  };
  
  #if 0
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        printf("%*.13lf ", 20, h_A(m,n));
      }
      printf("\n");
    }
    printf("=============================================\n");
    int as0 = A.stride(0);
    int as1 = A.stride(1);
    auto *a_ptr = h_A.data();
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        printf("%*.13lf ", 20, a_ptr[m*as0 + n*as1]);
      }
      printf("\n");
    }
  #endif

  // ech-note: use_unit_diag intentionally ignored for now. Compiler can optimize
  // it out. Assuming that branching logic (especially on GPU) to handle use_unit_diag
  // will use more cycles than simply doing 1.0*B[idx] for the copy if use_unit_diag.
  template<>
  template<typename ScalarType,
           typename ValueType>
  KOKKOS_INLINE_FUNCTION
  int
  SerialTrmmInternalLeftLower<Algo::Trmm::Unblocked>::
  invoke(const bool use_unit_diag,
         const bool do_conj,
         const int am, const int an,
         const int bm, const int bn,
         const ScalarType alpha,
         const ValueType *__restrict__ A, const int as0, const int as1,
         /**/  ValueType *__restrict__ B, const int bs0, const int bs1) {

    const ScalarType one(1.0), zero(0.0);
    typedef Kokkos::Details::ArithTraits<ValueType> AT;
    int left_m = am;
    int right_n = bn;
    //echo-TODO: See about coniditionally setting conjOp at compile time.
    //auto dotOp = dotUpperLeft;
    //if (do_conj) {
    //  dotOp = dotUpperLeftConj;
    //}
    
    auto dotLowerLeftConj = [&](const ValueType *__restrict__ A, const int as0, const int as1, const int left_row, ValueType *__restrict__ B, const int bs0, const int bs1, const int right_col) {
      auto B_elems = left_row;
      auto A_elems = B_elems * as0;
      ScalarType sum = 0;
      #if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
      #pragma unroll
      #endif
      for (int i = 0; i <= B_elems; i++) {
        //printf("%lf * %lf\n", AT::conj(A[left_row*as0 + i*as1]), B[i*bs0 + bs1*right_col]);
        sum += AT::conj(A[left_row*as0 + i*as1]) * B[i*bs0 + bs1*right_col];
      }
      //printf("--sum=%lf\n", sum);
      return sum;
    };

    auto dotLowerLeft = [&](const ValueType *__restrict__ A, const int as0, const int as1, const int left_row, ValueType *__restrict__ B, const int bs0, const int bs1, const int right_col) {
      auto B_elems = left_row;
      auto A_elems = B_elems * as0;
      ScalarType sum = 0;
      #if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
      #pragma unroll
      #endif
      for (int i = 0; i <= B_elems; i++) {
        //printf("%lf * %lf\n", A[left_row*as0 + i*as1], B[i*bs0 + bs1*right_col]);
        sum += A[left_row*as0 + i*as1] * B[i*bs0 + bs1*right_col];
      }
      //printf("--sum=%lf\n", sum);
      return sum;
    };

    if (bm <= 0 || bn <= 0 || am <= 0 || an <= 0)
      return 0;

    if (alpha == zero)
      SerialSetInternal::invoke(bm, bn, zero,  B, bs0, bs1);
    else {
      if (alpha != one)
        SerialScaleInternal::invoke(bm, bn, alpha, B, bs0, bs1);

      #if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
      #pragma unroll
      #endif
      for (int m = left_m-1; m >= 0; m--) {
        #if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
        #pragma unroll
        #endif
        for (int n = 0; n < right_n; n++) {
          if (do_conj) {
            B[m*bs0 + n*bs1] = dotLowerLeftConj(A, as0, as1, m, B, bs0, bs1, n);
          } else {
            B[m*bs0 + n*bs1] = dotLowerLeft(A, as0, as1, m, B, bs0, bs1, n);
          }
        }
      }
    }
    return 0;
  }

  template<>
  template<typename ScalarType,
           typename ValueType>
  KOKKOS_INLINE_FUNCTION
  int
  SerialTrmmInternalLeftUpper<Algo::Trmm::Unblocked>::
  invoke(const bool use_unit_diag,
         const bool do_conj,
         const int am, const int an,
         const int bm, const int bn,
         const ScalarType alpha,
         const ValueType *__restrict__ A, const int as0, const int as1,
         /**/  ValueType *__restrict__ B, const int bs0, const int bs1) {

    const ScalarType one(1.0), zero(0.0);
    typedef Kokkos::Details::ArithTraits<ValueType> AT;
    int left_m = am;
    int right_n = bn;
    //echo-TODO: See about coniditionally setting conjOp at compile time.
    //auto dotOp = dotUpperLeft;
    //if (do_conj) {
    //  dotOp = dotUpperLeftConj;
    //}
    
    auto dotUpperLeft = [&](const ValueType *__restrict__ A, const int as0, const int as1, const int an, const int left_row, ValueType *__restrict__ B, const int bs0, const int bs1, const int right_col) {
      auto B_elems = an - left_row - 1;
      auto A_elems = B_elems * as0;
      ScalarType sum = 0;
      #if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
      #pragma unroll
      #endif
      for (int i = 0; i <= B_elems; i++) {
        //printf("%lf * %lf\n", A[left_row*as0 + (left_row+i)*as1], B[(left_row+i)*bs0 + bs1*right_col]);
        sum += A[left_row*as0 + (left_row+i)*as1] * B[(left_row+i)*bs0 + bs1*right_col];
      }
      //printf("--sum=%lf\n", sum);
      return sum;
    };

    auto dotUpperLeftConj = [&](const ValueType *__restrict__ A, const int as0, const int as1, const int an, const int left_row, ValueType *__restrict__ B, const int bs0, const int bs1, const int right_col) {
      auto B_elems = an - left_row - 1;
      auto A_elems = B_elems * as0;
      ScalarType sum = 0;
      #if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
      #pragma unroll
      #endif
      for (int i = 0; i <= B_elems; i++) {
        //printf("%lf * %lf\n", A[left_row*as0 + (left_row+i)*as1], B[(left_row+i)*bs0 + bs1*right_col]);
        sum += AT::conj(A[left_row*as0 + (left_row+i)*as1]) * B[(left_row+i)*bs0 + bs1*right_col];
      }
      //printf("--sum=%lf\n", sum);
      return sum;
    };

    if (bm <= 0 || bn <= 0 || am <= 0 || an <= 0)
      return 0;

    if (alpha == zero)
      SerialSetInternal::invoke(bm, bn, zero,  B, bs0, bs1);
    else {
      if (alpha != one)
        SerialScaleInternal::invoke(bm, bn, alpha, B, bs0, bs1);
      
      #if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
      #pragma unroll
      #endif
      for (int m = 0; m < left_m; ++m) {
        #if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
        #pragma unroll
        #endif
        for (int n = 0; n < right_n; ++n) {
          if (do_conj) {
            B[m*bs0 + n*bs1] = dotUpperLeftConj(A, as0, as1, an, m, B, bs0, bs1, n);
          } else {
            B[m*bs0 + n*bs1] = dotUpperLeft(A, as0, as1, an, m, B, bs0, bs1, n);
          }
        }
      }
    }
    return 0;
  }
} // namespace KokkosBatched
#endif // __KOKKOSBATCHED_TRMM_SERIAL_INTERNAL_HPP__
