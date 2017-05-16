#ifndef __KOKKOSKERNELS_SCALE_DECL_HPP__
#define __KOKKOSKERNELS_SCALE_DECL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {

      ///
      /// Serial Scale
      ///

      namespace Serial {
        struct Scale {
          template<typename ScalarType,
                   typename AViewType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const ScalarType alpha,
                 const AViewType &A);
        };
      }

    }
  }
}

#endif
