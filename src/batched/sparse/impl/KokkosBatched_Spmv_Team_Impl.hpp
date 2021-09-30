//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.4
//       Copyright (2021) National Technology & Engineering
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
#ifndef __KOKKOSBATCHED_SPMV_TEAM_IMPL_HPP__
#define __KOKKOSBATCHED_SPMV_TEAM_IMPL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

  ///
  /// Team Internal Impl
  /// ==================== 
  struct TeamSpmvInternal {
    template <typename OrdinalType,
              typename layout>
    KOKKOS_INLINE_FUNCTION
    static void getIndices(const OrdinalType iTemp,
                    const OrdinalType n_rows,
                    const OrdinalType n_matrices,
                    OrdinalType &iRow,
                    OrdinalType &iMatrix);

    template <typename MemberType,
              typename ScalarType,
              typename ValueType,
              typename OrdinalType,
              typename layout,
              int dobeta>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member,
           const OrdinalType numMatrices, const OrdinalType numRows, 
           const ScalarType* KOKKOS_RESTRICT alpha, const OrdinalType alphas0,
           const ValueType* KOKKOS_RESTRICT values, const OrdinalType valuess0, const OrdinalType valuess1,
           const OrdinalType* KOKKOS_RESTRICT row_ptr, const OrdinalType row_ptrs0,
           const OrdinalType* KOKKOS_RESTRICT colIndices, const OrdinalType colIndicess0,
           const ValueType* KOKKOS_RESTRICT X, const OrdinalType xs0, const OrdinalType xs1, 
           const ScalarType* KOKKOS_RESTRICT beta, const OrdinalType betas0,
           /**/  ValueType* KOKKOS_RESTRICT Y, const OrdinalType ys0, const OrdinalType ys1);
  };


  template <typename OrdinalType,
            typename layout>
  KOKKOS_INLINE_FUNCTION
  void
  TeamSpmvInternal:: 
  getIndices(const OrdinalType iTemp,
             const OrdinalType numRows,
             const OrdinalType numMatrices,
             OrdinalType &iRow,
             OrdinalType &iMatrix) {
    if (std::is_same<layout, Kokkos::LayoutLeft>::value) {
      iRow    = iTemp / numMatrices;
      iMatrix = iTemp % numMatrices;
    }
    else {
      iRow    = iTemp % numRows;
      iMatrix = iTemp / numRows;
    }
  }

  template <typename MemberType,
            typename ScalarType,
            typename ValueType,
            typename OrdinalType,
            typename layout,
            int dobeta>
  KOKKOS_INLINE_FUNCTION
  int
  TeamSpmvInternal::
  invoke(const MemberType &member,
         const OrdinalType numMatrices, const OrdinalType numRows, 
         const ScalarType* KOKKOS_RESTRICT alpha, const OrdinalType alphas0,
         const ValueType* KOKKOS_RESTRICT values, const OrdinalType valuess0, const OrdinalType valuess1,
         const OrdinalType* KOKKOS_RESTRICT row_ptr, const OrdinalType row_ptrs0,
         const OrdinalType* KOKKOS_RESTRICT colIndices, const OrdinalType colIndicess0,
         const ValueType* KOKKOS_RESTRICT X, const OrdinalType xs0, const OrdinalType xs1,
         const ScalarType* KOKKOS_RESTRICT beta, const OrdinalType betas0,
         /**/  ValueType* KOKKOS_RESTRICT Y, const OrdinalType ys0, const OrdinalType ys1) {


    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, 0, numMatrices * numRows),
        [&](const OrdinalType& iTemp) {
          OrdinalType iRow, iMatrix;
          getIndices<OrdinalType,layout>(iTemp, numRows, numMatrices, iRow, iMatrix);

          const OrdinalType rowLength =
              row_ptr[(iRow+1)*row_ptrs0] - row_ptr[iRow*row_ptrs0];
          ValueType sum = 0;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
          for (OrdinalType iEntry = 0; iEntry < rowLength; ++iEntry) {
            sum += values[iMatrix*valuess0+(row_ptr[iRow*row_ptrs0]+iEntry)*valuess1]
                    * X[iMatrix*xs0+colIndices[(row_ptr[iRow*row_ptrs0]+iEntry)*colIndicess0]*xs1];
          }

          sum *= alpha[iMatrix*alphas0];

          if (dobeta == 0) {
            Y[iMatrix*ys0+iRow*ys1] = sum;
          } else {
            Y[iMatrix*ys0+iRow*ys1] = 
                beta[iMatrix*betas0] * Y[iMatrix*ys0+iRow*ys1] + sum;
          }
      });
      
    return 0;  
  }

  template<typename MemberType>
  struct TeamSpmv<MemberType,Trans::NoTranspose> {
          
    template<typename ValuesViewType,
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
           const ValuesViewType &values,
           const IntView &row_ptr,
           const IntView &colIndices,
           const xViewType &X,
           const betaViewType &beta,
           const yViewType &Y) {
      return TeamSpmvInternal::template
        invoke<MemberType,
               typename alphaViewType::non_const_value_type, 
               typename ValuesViewType::non_const_value_type, 
               typename IntView::non_const_value_type, 
               typename ValuesViewType::array_layout, 
               dobeta>
               (member, 
                X.extent(0), X.extent(1),
                alpha.data(), alpha.stride_0(),
                values.data(), values.stride_0(), values.stride_1(),
                row_ptr.data(), row_ptr.stride_0(),
                colIndices.data(), colIndices.stride_0(),
                X.data(), X.stride_0(), X.stride_1(),
                beta.data(), beta.stride_0(),
                Y.data(), Y.stride_0(), Y.stride_1());
    }
  };

}

#endif
