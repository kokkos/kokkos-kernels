#ifndef __KOKKOSKERNELS_TRSM_DECL_HPP__
#define __KOKKOSKERNELS_TRSM_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace KokkosKernels {

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
             const BViewType &B) {
        //static_assert(false, "KokkosKernels::Trsm::invoke:: Not yet implemented");
        return 0;
      }      
    };

  }

}

#endif
