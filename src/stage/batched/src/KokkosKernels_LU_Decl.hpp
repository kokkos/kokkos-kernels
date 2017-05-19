#ifndef __KOKKOSKERNELS_LU_DECL_HPP__
#define __KOKKOSKERNELS_LU_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace KokkosKernels {

  namespace Serial {

    template<typename ArgAlgo>
    struct LU {

      // no piv version
      template<typename AViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const AViewType &A) {
        //static_assert(false, "KokkosKernels::LU:: Not yet implemented");
        return 0;
      }

    };

  }
}

#endif
