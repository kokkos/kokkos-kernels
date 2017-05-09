#ifndef __KOKKOSKERNELS_GEMV_DECL_HPP__
#define __KOKKOSKERNELS_GEMV_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace KokkosKernels {

  ///
  /// Serial Gemv 
  ///

  namespace Serial {
    template<typename ArgTrans,
             typename ArgAlgo>
    struct Gemv {
      template<typename ScalarType,
               typename AViewType,
               typename xViewType,
               typename yViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const ScalarType alpha,
             const AViewType &A,
             const xViewType &x,
             const ScalarType beta,
             const yViewType &y) {
        //static_assert(false, "KokkosKernels::Gemv:: Not yet implemented");
        return 0;
      }
    };
  }
  
}

#endif
