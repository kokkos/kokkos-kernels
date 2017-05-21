#ifndef __KOKKOSKERNELS_GEMV_TEAM_IMPL_HPP__
#define __KOKKOSKERNELS_GEMV_TEAM_IMPL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"
#include "KokkosKernels_Gemv_Team_Internal.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {

      ///
      /// Team Impl
      /// =========
      namespace Team {

        ///
        /// Implemented:
        /// NT, T
        ///
        /// Not yet implemented
        /// CT

        ///
        /// NT
        ///

        template<typename MemberType>
        struct Gemv<MemberType,Trans::NoTranspose,Algo::Gemv::Unblocked> {
          
          template<typename ScalarType,
                   typename AViewType,
                   typename xViewType,
                   typename yViewType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const MemberType &member,
                 const ScalarType alpha,
                 const AViewType &A,
                 const xViewType &x,
                 const ScalarType beta,
                 const yViewType &y) {
            return GemvInternal<Algo::Gemv::Unblocked>::
              invoke(member,
                     A.dimension_0(), A.dimension_1(),
                     alpha, 
                     A.data(), A.stride_0(), A.stride_1(),
                     x.data(), x.stride_0(),
                     beta,
                     y.data(), y.stride_0());
          }
        };
    
        template<typename MemberType>
        struct Gemv<MemberType,Trans::NoTranspose,Algo::Gemv::Blocked> {

          template<typename ScalarType,
                   typename AViewType,
                   typename xViewType,
                   typename yViewType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const MemberType &member,
                 const ScalarType alpha,
                 const AViewType &A,
                 const xViewType &x,
                 const ScalarType beta,
                 const yViewType &y) {
            return GemvInternal<Algo::Gemv::Blocked>::
              invoke(member,
                     A.dimension_0(), A.dimension_1(),
                     alpha, 
                     A.data(), A.stride_0(), A.stride_1(),
                     x.data(), x.stride_0(),
                     beta,
                     y.data(), y.stride_0());
          }
        };

        ///
        /// T
        ///

        template<typename MemberType>
        Gemv<MemberType,Trans::Transpose,Algo::Gemv::Unblocked> {

          template<typename ScalarType,
                   typename AViewType,
                   typename xViewType,
                   typename yViewType>
            KOKKOS_INLINE_FUNCTION
            static int
            invoke(const MemberType &member,
                   const ScalarType alpha,
                   const AViewType &A,
                   const xViewType &x,
                   const ScalarType beta,
                   const yViewType &y) {
            return GemvInternal<Algo::Gemv::Unblocked>::
              invoke(member,
                     A.dimension_1(), A.dimension_0(),
                     alpha, 
                     A.data(), A.stride_1(), A.stride_0(),
                     x.data(), x.stride_0(),
                     beta,
                     y.data(), y.stride_0());
          }
        };
        
        template<typename MemberType>
        Gemv<MemberType,Trans::Transpose,Algo::Gemv::Blocked> {

          template<typename ScalarType,
                   typename AViewType,
                   typename xViewType,
                   typename yViewType>
            KOKKOS_INLINE_FUNCTION
            static int
            invoke(const MemberType &member, 
                   const ScalarType alpha,
                   const AViewType &A,
                   const xViewType &x,
                   const ScalarType beta,
                   const yViewType &y) {
            return GemvInternal<Algo::Gemv::Blocked>::
              invoke(member,
                     A.dimension_1(), A.dimension_0(),
                     alpha, 
                     A.data(), A.stride_1(), A.stride_0(),
                     x.data(), x.stride_0(),
                     beta,
                     y.data(), y.stride_0());
          }
        };
      }
    }
  }
} // end namespace KokkosKernels
#endif
