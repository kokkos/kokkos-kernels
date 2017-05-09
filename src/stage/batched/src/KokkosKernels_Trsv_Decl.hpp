#ifndef __KOKKOSKERNELS_TRSV_DECL_HPP__
#define __KOKKOSKERNELS_TRSV_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace KokkosKernels {
  
  ///
  /// Serial Trsv
  ///

  namespace Serial {
    template<typename ArgUplo,
             typename ArgTrans,
             typename ArgDiag,
             typename ArgAlgo>
    struct Trsv {

      template<typename ScalarType,
               typename AViewType,
               typename bViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const ScalarType alpha,
             const AViewType &A,
             const bViewType &b) {
        //static_assert(false, "KokkosKernels::Trsv:: Not yet implemented");
        return 0;
      }
    };
  }

}

#endif
