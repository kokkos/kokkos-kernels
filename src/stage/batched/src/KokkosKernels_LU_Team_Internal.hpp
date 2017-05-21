#ifndef __KOKKOSKERNELS_LU_TEAM_INTERNAL_HPP__
#define __KOKKOSKERNELS_LU_TEAM_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"

#include "KokkosKernels_InnerLU_Serial_Impl.hpp"
#include "KokkosKernels_InnerTrsm_Serial_Impl.hpp"
#include "KokkosKernels_Gemm_Team_Internal.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {
  
      ///
      /// Team Internal Impl
      /// ==================

      namespace Team {

        template<typename AlgoType>
        struct LU_Internal {
          template<typename MemberType, typename ValueType>
          KOKKOS_INLINE_FUNCTION
          static int 
          invoke(const MemberType &member,
                 const int m, const int n,
                 ValueType *__restrict__ A, const int as0, const int as1);
        };

        template<>
        template<typename MemberType, typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int
        LU_Internal<Algo::LU::Unblocked>::
        invoke(const MemberType &member, 
               const int m, const int n,
               ValueType *__restrict__ A, const int as0, const int as1) {
          typedef ValueType value_type;
          const int k = (m < n ? m : n);
          if (k <= 0) return 0;

          for (int p=0;p<k;++p) {
            const int iend = m-p-1, jend = n-p-1;

            const value_type 
              // inv_alpha11 = 1.0/A(p,p),
              alpha11 = A[p*as0+p*as1],
              *__restrict__ a12t = A+(p  )*as0+(p+1)*as1;
            
            value_type
              *__restrict__ a21  = A+(p+1)*as0+(p  )*as1,
              *__restrict__ A22  = A+(p+1)*as0+(p+1)*as1;
            
            member.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,iend),[&](const int &i) {
                // a21[i*as0] *= inv_alpha11; 
                a21[i*as0] /= alpha11;
              });
            member.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,iend*jend),[&](const int &ij) {
#if                             \
  defined (KOKKOS_HAVE_CUDA) &&                         \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
                const int i = ij%iend, j = ij/iend;
#else
                const int i = ij/jend, j = ij%jend;
#endif
                A22[i*as0+j*as1] -= a21[i*as0] * a12t[j*as1];
              });
          }
          return 0;
        }
    
        template<>
        template<typename MemberType, typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int
        LU_Internal<Algo::LU::Blocked>::
        invoke(const MemberType &member, 
               const int m, const int n,
               ValueType *__restrict__ A, const int as0, const int as1) {
          typedef ValueType value_type;
          const int k = (m < n ? m : n);
          if (k <= 0) return 0;

          {
            enum : int {
              mb = Algo::LU::Blocked::mb<Kokkos::Impl::ActiveExecutionMemorySpace>()
            };

            InnerLU<mb> lu(as0, as1);
          
            InnerTrsmLeftLowerUnitDiag<mb>    trsm_llu(as0, as1, as0, as1);
            InnerTrsmLeftLowerNonUnitDiag<mb> trsm_run(as1, as0, as1, as0);

            auto lu_factorize = [&](const int ib,
                                    const int jb,
                                    value_type *__restrict__ AA) {
              const int kb = ib < jb ? ib : jb; 
              for (int p=0;p<kb;p+=mb) {
                const int pb = (p+mb) > kb ? (kb-p) : mb;

                // diagonal block
                value_type *__restrict__ Ap = AA+p*as0+p*as1;

                // lu on a block             
                member.team_barrier();
                lu.serial_invoke(pb, Ap);
                member.team_barrier();

                // dimension ABR
                const int 
                  m_abr = ib-p-mb, n_abr = jb-p-mb,
                  mp_abr = m_abr%mb, np_abr = n_abr%mb,
                  mq_abr = (m_abr/mb)+(mp_abr>0), nq_abr = (n_abr/mb)+(np_abr>0);
                
                // trsm update
                Kokkos::parallel_for
                  (Kokkos::TeamThreadRange(member,0,mq_abr+nq_abr),
                   [&](const int &ij) {
                    if (ij < nq_abr) {
                      const int j = ij*mb, qb = (j+mb) > n_abr ? np_abr : mb;
                      trsm_llu.serial_invoke(Ap, pb, qb, Ap+j*as1);
                    } else {
                      const int i = (ij-nq_abr)*mb , qb = (i+mb) > m_abr ? mp_abr : mb;
                      trsm_run.serial_invoke(Ap, pb, qb, Ap+i*as0);
                    }
                  });
                member.team_barrier();

                // gemm update
                GemmInternal<Algo::Gemm::Blocked>::
                  invoke(member, 
                         m_abr, n_abr, pb,
                         -1, 
                         Ap+mb*as0, as0, as1,
                         Ap+mb*as1, as0, as1,
                         1,
                         Ap+mb*as0+mb*as1, as0, as1);
              }
            };
            
            const bool is_small = true; //(m*n <= 64*64);
            if (is_small) {
              lu_factorize(m, n, A);
            } else {
              // // some cache blocking may need (not priority yet);
              // lu_factorize(m, n, A);
            }

          }
          return 0;
        }
      }
    }
  }
}
#endif
