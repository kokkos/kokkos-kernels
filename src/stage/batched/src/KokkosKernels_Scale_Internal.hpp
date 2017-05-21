#ifndef __KOKKOSKERNELS_SCALE_INTERNAL_HPP__
#define __KOKKOSKERNELS_SCALE_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {
      ///
      /// Serial Internal Impl
      /// ==================== 
      namespace Serial {
        struct ScaleInternal {
          template<typename ScalarType,
                   typename ValueType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const int m, const int n, 
                 const ScalarType alpha, 
                 /* */ ValueType *__restrict__ A, const int as0, const int as1) {
            const int mn = m*n;

            if ( (m == as0 && as1 == 1) ||
                 (n == as1 && as0 == 1) )
              for (int k=0;k<mn;++k)
                A[k] *= alpha;
            else
              if (as0 > as1)
                for (int i=0;i<m;++i)
                  for (int j=0;j<n;++j)
                    A[i*as0+j*as1] *= alpha;
              else
                for (int j=0;j<n;++j)
                  for (int i=0;i<m;++i)
                    A[i*as0+j*as1] *= alpha;
        
            return 0;
          }
        };
      } // end namespace Serial

      ///
      /// Team Internal Impl
      /// ==================== 
      namespace Team {
        struct ScaleInternal {
          template<typename MemberType,
                   typename ScalarType,
                   typename ValueType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const MemberType &member, 
                 const int m, const int n, 
                 const ScalarType alpha, 
                 /* */ ValueType *__restrict__ A, const int as0, const int as1) {
            if ( (m == as0 && as1 == 1) ||
                 (n == as1 && as0 == 1) )
              Kokkos::parallel_for
                (Kokkos::TeamThreadRange(member,0,m*n),
                 [&](const int &k) {
                  A[k] *= alpha;
                });
            else
              Kokkos::parallel_for
                (Kokkos::TeamThreadRange(member,0,m*n),
                 [&](const int &ij) {
#if \
  defined (KOKKOS_HAVE_CUDA) && \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
                  const int i = ij%m, j = ij/m;
#else
                  const int i = ij/n, j = ij%n;
#endif
                  A[i*as0+j*as1] *= alpha;
                });
            member.team_barrier();
            return 0;
          }
        };
      } // end namespace Team

    }
  }
} // end namespace KokkosKernels

#endif
