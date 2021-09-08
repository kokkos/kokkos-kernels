#ifndef __KOKKOSBATCHED_SPMV_TEAMVECTOR_IMPL_HPP__
#define __KOKKOSBATCHED_SPMV_TEAMVECTOR_IMPL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

  ///
  /// TeamVector Internal Impl
  /// ==================== 
  struct TeamVectorSpmvInternal {
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
           const OrdinalType m, const OrdinalType nrows, 
           const ScalarType *__restrict__ alpha, const OrdinalType alphas0,
           const ValueType *__restrict__ D, const OrdinalType ds0, const OrdinalType ds1,
           const OrdinalType *__restrict__ r, const OrdinalType rs0,
           const OrdinalType *__restrict__ c, const OrdinalType cs0,
           const ValueType *__restrict__ X, const OrdinalType xs0, const OrdinalType xs1, 
           const ScalarType *__restrict__ beta, const OrdinalType betas0,
           /**/  ValueType *__restrict__ Y, const OrdinalType ys0, const OrdinalType ys1);
  };

  template <typename OrdinalType,
            typename layout>
  KOKKOS_INLINE_FUNCTION
  void
  TeamVectorSpmvInternal:: 
  getIndices(const OrdinalType iTemp,
             const OrdinalType n_rows,
             const OrdinalType n_matrices,
             OrdinalType &iRow,
             OrdinalType &iMatrix) {
    if (std::is_same<layout, Kokkos::LayoutLeft>::value) {
      iRow    = iTemp / n_matrices;
      iMatrix = iTemp % n_matrices;
    }
    else {
      iRow    = iTemp % n_rows;
      iMatrix = iTemp / n_rows;
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
  TeamVectorSpmvInternal::
  invoke(const MemberType &member,
         const OrdinalType m, const OrdinalType nrows, 
         const ScalarType *__restrict__ alpha, const OrdinalType alphas0,
         const ValueType *__restrict__ D, const OrdinalType ds0, const OrdinalType ds1,
         const OrdinalType *__restrict__ r, const OrdinalType rs0,
         const OrdinalType *__restrict__ c, const OrdinalType cs0,
         const ValueType *__restrict__ X, const OrdinalType xs0, const OrdinalType xs1,
         const ScalarType *__restrict__ beta, const OrdinalType betas0,
         /**/  ValueType *__restrict__ Y, const OrdinalType ys0, const OrdinalType ys1) {


    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, 0, m * nrows),
        [&](const OrdinalType& iTemp) {
          OrdinalType iRow, iMatrix;
          getIndices<OrdinalType,layout>(iTemp, nrows, m, iRow, iMatrix);

          const OrdinalType row_length =
              r[(iRow+1)*rs0] - r[iRow*rs0];
          ValueType sum = 0;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
          for (OrdinalType iEntry = 0; iEntry < row_length; ++iEntry) {
            sum += D[iMatrix*ds0+(r[iRow*rs0]+iEntry)*ds1]
                    * X[iMatrix*xs0+c[(r[iRow*rs0]+iEntry)*cs0]*xs1];
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
  struct TeamVectorSpmv<MemberType,Trans::NoTranspose> {
          
    template<typename DViewType,
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
           const DViewType &D,
           const IntView &r,
           const IntView &c,
           const xViewType &X,
           const betaViewType &beta,
           const yViewType &Y) {
      return TeamVectorSpmvInternal::template
        invoke<MemberType, 
               typename alphaViewType::non_const_value_type, 
               typename DViewType::non_const_value_type, 
               typename IntView::non_const_value_type, 
               typename DViewType::array_layout, 
               dobeta>
               (member, 
                X.extent(0), X.extent(1),
                alpha.data(), alpha.stride_0(),
                D.data(), D.stride_0(), D.stride_1(),
                r.data(), r.stride_0(),
                c.data(), c.stride_0(),
                X.data(), X.stride_0(), X.stride_1(),
                beta.data(), beta.stride_0(),
                Y.data(), Y.stride_0(), Y.stride_1());
    }
  };

}

#endif
