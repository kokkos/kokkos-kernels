#ifndef __KOKKOSKERNELS_LU_SERIAL_IMPL_HPP__
#define __KOKKOSKERNELS_LU_SERIAL_IMPL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"
#include "KokkosKernels_LU_Serial_Internal.hpp"

namespace KokkosKernels {

  namespace Serial {

    ///
    /// LU no piv
    ///

#if \
  defined(__KOKKOSKERNELS_INTEL_MKL__) &&                       \
  defined(__KOKKOSKERNELS_INTEL_MKL_BATCHED__) &&               \
  defined(__KOKKOSKERNELS_INTEL_MKL_COMPACT_BATCHED__)
    template<>
    template<typename AViewType>
    KOKKOS_INLINE_FUNCTION
    int
    LU<Algo::LU::CompactMKL>::
    invoke(const AViewType &A) {
      typedef typename AViewType::value_type vector_type;
      typedef typename vector_type::value_type value_type;

      const int
        m = A.dimension(0),
        n = A.dimension(1),
        vl = vector_type::vector_length;
      LAPACKE_dgetrf_compact(CblasRowMajor, 
                             m, n, 
                             (double*)A.data(), A.stride_0(), 
                             (MKL_INT)vl, (MKL_INT)1);
    }
#endif

    template<>
    template<typename AViewType>
    KOKKOS_INLINE_FUNCTION
    int
    LU<Algo::LU::Unblocked>::
    invoke(const AViewType &A) {
      return LU_Internal<Algo::LU::Unblocked>::invoke(A.dimension_0(), A.dimension_1(),
                                                      A.data(), A.stride_0(), A.stride_1());
    }
    
    template<>
    template<typename AViewType>
    KOKKOS_INLINE_FUNCTION
    int
    LU<Algo::LU::Blocked>::
    invoke(const AViewType &A) {
      return LU_Internal<Algo::LU::Blocked>::invoke(A.dimension_0(), A.dimension_1(),
                                                    A.data(), A.stride_0(), A.stride_1());
    }

  }
}

#endif
