#ifndef __KOKKOSKERNELS_COPY_IMPL_HPP__
#define __KOKKOSKERNELS_COPY_IMPL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"
#include "KokkosKernels_Copy_Internal.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {
      ///
      /// Serial Impl
      /// ===========
      
      namespace Serial {
        
        template<>
        template<typename AViewType,
                 typename BViewType>
        KOKKOS_INLINE_FUNCTION
        int
        Copy<Trans::NoTranspose>::
        invoke(const AViewType &A,
               /* */ BViewType &B) {
          return CopyInternal::
            invoke(min(A.dimension_0(), B.dimension_0()), 
                   min(A.dimension_1(), B.dimension_1()),
                   A.data(), A.stride_0(), A.stride_1(),
                   B.data(), B.stride_0(), B.stride_1());
        }
        
        template<>
        template<typename AViewType,
                 typename BViewType>
        KOKKOS_INLINE_FUNCTION
        int
        Copy<Trans::Transpose>::
        invoke(const AViewType &A,
               /* */ BViewType &B) {
          return CopyInternal::
            invoke(min(A.dimension_1(), B.dimension_0()), 
                   min(A.dimension_0(), B.dimension_1()),
                   A.data(), A.stride_1(), A.stride_0(),
                   B.data(), B.stride_0(), B.stride_1());
        }
        
      } // end namespace Serial
      
      ///
      /// Team Impl
      /// =========
      
      namespace Team {
        
        template<typename MemberType>
        struct Copy<MemberType,Trans::NoTranspose> {
          template<typename AViewType,
                   typename BViewType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const MemberType &member, 
                 const AViewType &A,
                 /* */ BViewType &B) {
            return CopyInternal::
              invoke(member,
                     min(A.dimension_0(), B.dimension_0()), 
                     min(A.dimension_1(), B.dimension_1()),
                     A.data(), A.stride_0(), A.stride_1(),
                     B.data(), B.stride_0(), B.stride_1());
          }
        };

        template<typename MemberType>
        struct Copy<MemberType,Trans::Transpose> {
          template<typename AViewType,
                   typename BViewType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const MemberType &member, 
                 const AViewType &A,
                 /* */ BViewType &B) {
            return CopyInternal::
              invoke(member,
                     min(A.dimension_1(), B.dimension_0()), 
                     min(A.dimension_0(), B.dimension_1()),
                     A.data(), A.stride_1(), A.stride_0(),
                     B.data(), B.stride_0(), B.stride_1());
          }
        };
        
      } // end namespace Team
      
    } // end namespace Experimental
  } //end namespace Batched
} // end namespace KokkosKernels

#endif
