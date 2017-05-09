#ifndef __KOKKOSKERNELS_TRSV_SERIAL_INTERNAL_HPP__
#define __KOKKOSKERNELS_TRSV_SERIAL_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"
#include "KokkosKernels_Trsm_Serial_Internal.hpp"  

namespace KokkosKernels {

  ///
  /// Serial Internal Impl
  /// ====================
  namespace Serial {

    ///
    /// Lower
    ///

    template<typename AlgoType>
    struct TrsvInternalLower {
      template<typename ScalarType,
               typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const bool use_unit_diag,
             const int m, 
             const ScalarType alpha,
             const ValueType *__restrict__ A, const int as0, const int as1,
             /**/  ValueType *__restrict__ b, const int bs0) {
        //static_assert("KokkosKernels::TrsvInternalLower:: Not yet implemented");
        return 0;
      }
    };

    template<>
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    int 
    TrsvInternalLower<Algo::Trsv::Unblocked>::
    invoke(const bool use_unit_diag,
           const int m, 
           const ScalarType alpha,
           const ValueType *__restrict__ A, const int as0, const int as1,
           /**/  ValueType *__restrict__ b, const int bs0) {
      return TrsmInternalLeftLower<Algo::Trsm::Unblocked>::
        invoke(use_unit_diag,
               m, 1,
               alpha,
               A, as0, as1,
               b, bs0, 1);
    }

    template<>
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    int 
    TrsvInternalLower<Algo::Trsv::Blocked>::
    invoke(const bool use_unit_diag,
           const int m, 
           const ScalarType alpha,
           const ValueType *__restrict__ A, const int as0, const int as1,
           /**/  ValueType *__restrict__ b, const int bs0) {
      return TrsmInternalLeftLower<Algo::Trsm::Blocked>::
        invoke(use_unit_diag,
               m, 1,
               alpha,
               A, as0, as1,
               b, bs0, 1);
    }

    ///
    /// Upper
    ///

    template<typename AlgoType>
    struct TrsvInternalUpper {
      template<typename ScalarType,
               typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const bool use_unit_diag,
             const int m, 
             const ScalarType alpha,
             const ValueType *__restrict__ A, const int as0, const int as1,
             /**/  ValueType *__restrict__ b, const int bs0) {
        //static_assert("KokkosKernels::TrsvInternalUpper:: Not yet implemented");
        return 0;
      }
    };

    template<>
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    int 
    TrsvInternalUpper<Algo::Trsv::Unblocked>::
    invoke(const bool use_unit_diag,
           const int m, 
           const ScalarType alpha,
           const ValueType *__restrict__ A, const int as0, const int as1,
           /**/  ValueType *__restrict__ b, const int bs0) {
      return TrsmInternalLeftUpper<Algo::Trsm::Unblocked>::
        invoke(use_unit_diag,
               m, 1,
               alpha,
               A, as0, as1,
               b, bs0, 1);
    }

    template<>
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    int 
    TrsvInternalUpper<Algo::Trsv::Blocked>::
    invoke(const bool use_unit_diag,
           const int m, 
           const ScalarType alpha,
           const ValueType *__restrict__ A, const int as0, const int as1,
           /**/  ValueType *__restrict__ b, const int bs0) {
      return TrsmInternalLeftUpper<Algo::Trsm::Blocked>::
        invoke(use_unit_diag,
               m, 1,
               alpha,
               A, as0, as1,
               b, bs0, 1);
    }
  }
}

#endif
