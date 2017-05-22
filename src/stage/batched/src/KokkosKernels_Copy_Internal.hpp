#ifndef __KOKKOSKERNELS_COPY_INTERNAL_HPP__
#define __KOKKOSKERNELS_COPY_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {
      ///
      /// Serial Internal Impl
      /// ==================== 
      namespace Serial {
        struct CopyInternal {
          template<typename ValueType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const int m, const int n, 
                 const ValueType *__restrict__ A, const int as0, const int as1,
                 /* */ ValueType *__restrict__ B, const int bs0, const int bs1) {
            if (A == B) return 0;
            if (as0 > as1)
              for (int i=0;i<m;++i)
                for (int j=0;j<n;++j)
                  B[i*bs0+j*bs1] = A[i*as0+j*as1];
            else
              for (int j=0;j<n;++j)
                for (int i=0;i<m;++i)
                  B[i*bs0+j*bs1] = A[i*as0+j*as1];
            return 0;
          }
        };
      } // end namespace Serial

      ///
      /// Team Internal Impl
      /// ==================
      namespace Team {
        struct CopyInternal {
          template<typename MemberType,
                   typename ValueType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const MemberType &member,
                 const int m, const int n, 
                 const ValueType *__restrict__ A, const int as0, const int as1,
                 /* */ ValueType *__restrict__ B, const int bs0, const int bs1) {
            if (A == B) return 0;
            Kokkos::parallel_for
              (Kokkos::TeamThreadRange(member,0,m*n),
               [&](const int &ij) {
#if                             \
  defined (KOKKOS_HAVE_CUDA) &&                         \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
                const int i = ij%m, j = ij/m;
#else
                const int i = ij/n, j = ij%n;
#endif
                B[i*bs0+j*bs1] = A[i*as0+j*as1];
              });
            //member.team_barrier();
            return 0;
          }
        };
      } // end namespace Team

    }//  end namespace Experimental
  } // end namespace Batched
} // end namespace KokkosKernels

#endif
