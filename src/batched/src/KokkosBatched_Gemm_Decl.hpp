#ifndef __KOKKOSKERNELS_GEMM_DECL_HPP__
#define __KOKKOSKERNELS_GEMM_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)


namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Serial Gemm 
    ///

    namespace Serial {
      template<typename ArgTransA,
               typename ArgTransB,
               typename ArgAlgo>
      struct Gemm {
        template<typename ScalarType,
                 typename AViewType,
                 typename BViewType,
                 typename CViewType>
        KOKKOS_INLINE_FUNCTION
        static int
        invoke(const ScalarType alpha,
               const AViewType &A,
               const BViewType &B,
               const ScalarType beta,
               const CViewType &C);
      };
    }

    ///
    /// Team Gemm
    ///

    namespace Team {
      template<typename MemberType,
               typename ArgTransA,
               typename ArgTransB,
               typename ArgAlgo>
      struct Gemm {
        template<typename ScalarType,
                 typename AViewType,
                 typename BViewType,
                 typename CViewType>
        KOKKOS_INLINE_FUNCTION
        static int
        invoke(const MemberType &member,
               const ScalarType alpha,
               const AViewType &A,
               const BViewType &B,
               const ScalarType beta,
               const CViewType &C);
      };
    }
  
    // specialized for different m and n
    // C(mxn) += alpha * A(mxk) B(kxn)

  }
}

#endif
