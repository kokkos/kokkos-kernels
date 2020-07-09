#ifndef __KOKKOSBATCHED_APPLY_PIVOT_INTERNAL_HPP__
#define __KOKKOSBATCHED_APPLY_PIVOT_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

  ///
  /// TeamVector Internal Impl
  /// ========================
  struct TeamVectorApplyPivotVectorInternal {
    template<typename MemberType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member,
           const int piv,
           /* */ ValueType *__restrict__ A, const int as0) {
      Kokkos::single
	(Kokkos::PerTeam(member),
	 [&]() {
	   const int idx_p = piv*as0;
	   ValueType tmp = A[0];
	   A[0] = A[idx_p];
	   A[idx_p] = tmp;
	 });
      return 0;
    }

    template<typename MemberType,
	     typename IntType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member,
	   const int plen,
	   const IntType *__restrict__ piv, const int ps0,
           /* */ ValueType *__restrict__ A, const int as0) {
      Kokkos::single
	(Kokkos::PerTeam(member),
	 [&]() {
	   for (int i=0;i<plen;++i) {
	     const int idx_p = (i+piv[i*ps0])*as0;
	     ValueType tmp = A[i*as0];
	     A[i] = A[idx_p];
	     A[idx_p] = tmp;
	   }
	 });
      return 0;
    }
  };

  /// Pivot a row
  struct TeamVectorApplyPivotMatrixInternal {
    template<typename MemberType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member,
	   const int n,
           const int piv,
           /* */ ValueType *__restrict__ A, const int as0, const int as1) {
      Kokkos::parallel_for
	(Kokkos::TeamVectorRange(member, n),
	 [&](const int &j) {
	   const int idx_j = j*as1, idx_p = idx_j + piv*as0;
	   ValueType tmp = A[idx_j];
	   A[idx_j] = A[idx_p];
	   A[idx_p] = tmp;	   
	 });
      return 0;
    }

    template<typename MemberType,
	     typename IntType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member,
	   const int n, const int plen,
           const IntType *__restrict__ piv, const int ps0,
           /* */ ValueType *__restrict__ A, const int as0, const int as1) {
      Kokkos::parallel_for
	(Kokkos::TeamVectorRange(member, n),
	 [&](const int &j) {
	   for (int i=0;i<plen;++i) {
	     ValueType *__restrict__ A_at_i = A + i*as0;
	     const int idx_j = j*as1, idx_p = idx_j + piv[i*ps0]*as0;
	     ValueType tmp = A_at_i[idx_j];
	     A_at_i[idx_j] = A_at_i[idx_p];
	     A_at_i[idx_p] = tmp;	   
	   }
	 });
      return 0;
    }
};

}


#endif
