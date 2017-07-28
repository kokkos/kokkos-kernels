#ifndef __KOKKOSKERNELS_TRSM_DECL_HPP__
#define __KOKKOSKERNELS_TRSM_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {

      namespace Serial {        
        template<typename ArgSide,
                 typename ArgUplo,
                 typename ArgTrans,
                 typename ArgDiag,
                 typename ArgAlgo>
        struct Trsm {
          template<typename ScalarType,
                   typename AViewType,
                   typename BViewType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const ScalarType alpha,
                 const AViewType &A,
                 const BViewType &B);
        };
      }

      namespace Team {        
        template<typename MemberType,
                 typename ArgSide,
                 typename ArgUplo,
                 typename ArgTrans,
                 typename ArgDiag,
                 typename ArgAlgo>
        struct Trsm {
          template<typename ScalarType,
                   typename AViewType,
                   typename BViewType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const MemberType &member,
                 const ScalarType alpha,
                 const AViewType &A,
                 const BViewType &B);
        };
      }

    }
  }
}
#endif
