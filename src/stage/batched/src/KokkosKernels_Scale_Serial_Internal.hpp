#ifndef __KOKKOSKERNELS_SCALE_SERIAL_INTERNAL_HPP__
#define __KOKKOSKERNELS_SCALE_SERIAL_INTERNAL_HPP__


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
    }
  }
} // end namespace KokkosKernels

#endif
