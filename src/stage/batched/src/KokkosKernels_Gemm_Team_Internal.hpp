#ifndef __KOKKOSKERNELS_GEMM_TEAM_INTERNAL_HPP__
#define __KOKKOSKERNELS_GEMM_TEAM_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"

#include "KokkosKernels_Set_Serial_Internal.hpp"
#include "KokkosKernels_Scale_Serial_Internal.hpp"

//#include "KokkosKernels_InnerGemmFixA_Team_Impl.hpp"
//#include "KokkosKernels_InnerGemmFixB_Team_Impl.hpp"
#include "KokkosKernels_InnerGemmFixC_Team_Impl.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {
      ///
      /// Team Internal Impl
      /// ==================== 
      namespace Team {

        template<typename ArgAlgo>
        struct GemmInternal {
          template<typename MemberType,
                   typename ScalarType,
                   typename ValueType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const MemberType &member, 
                 const int m, const int n, const int k,
                 const ScalarType alpha, 
                 const ValueType *__restrict__ A, const int as0, const int as1,
                 const ValueType *__restrict__ B, const int bs0, const int bs1,
                 const ScalarType beta,
                 /**/  ValueType *__restrict__ C, const int cs0, const int cs1);
        };

        template<>
        template<typename MemberType,
                 typename ScalarType,
                 typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int
        GemmInternal<Algo::Gemm::Unblocked>::
        invoke(const MemberType &member, 
               const int m, const int n, const int k,
               const ScalarType alpha, 
               const ValueType *__restrict__ A, const int as0, const int as1,
               const ValueType *__restrict__ B, const int bs0, const int bs1,
               const ScalarType beta,
               /**/  ValueType *__restrict__ C, const int cs0, const int cs1) {
          // C = beta C + alpha A B
          // C (m x n), A(m x k), B(k x n)
      
          typedef ValueType value_type;
        
          const int team_rank = member.team_rank();
        
          // later change set and scale with team
          if (team_rank == 0) {
            if      (beta == 0) Serial::SetInternal  ::invoke(m, n, value_type(0),    C, cs0, cs1);
            else if (beta != 1) Serial::ScaleInternal::invoke(m, n, value_type(beta), C, cs0, cs1);
          }
          member.team_barrier();
        
          if (alpha != 0) {
            if (m <= 0 || n <= 0 || k <= 0) return 0;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,m*n),[&](const int &ij) {
                const int
                  i = ij/n,
                  j = ij%n;
            
                const value_type
                  *__restrict__ pA = A+i*as0,
                  *__restrict__ pB = B+j*bs1;
            
                value_type c = 0;
                for (int p=0;p<k;++p) 
                  c += pA[p*as1]*pB[p*bs0];
                C[i*cs0+j*cs1] += alpha*c;
              });
          }
          return 0;
        }
    
        template<>
        template<typename MemberType,
                 typename ScalarType,
                 typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int
        GemmInternal<Algo::Gemm::Blocked>::
        invoke(const MemberType &member, 
               const int m, const int n, const int k,
               const ScalarType alpha, 
               const ValueType *__restrict__ A, const int as0, const int as1,
               const ValueType *__restrict__ B, const int bs0, const int bs1,
               const ScalarType beta,
               /**/  ValueType *__restrict__ C, const int cs0, const int cs1) {
          // C = beta C + alpha A B
          // C (m x n), A(m x k), B(k x n)

          typedef ValueType value_type;
        
          const int team_rank = member.team_rank();

          if (team_rank == 0) {
            if      (beta == 0) Serial::SetInternal  ::invoke(m, n, value_type(0),    C, cs0, cs1);
            else if (beta != 1) Serial::ScaleInternal::invoke(m, n, value_type(beta), C, cs0, cs1);
          }
          member.team_barrier();

          if (alpha != 0) {
            if (m <= 0 || n <= 0 || k <= 0) return 0;

            enum : int {
              mb = Algo::Gemm::Blocked::mb<Kokkos::Impl::ActiveExecutionMemorySpace>(),
              nb = Algo::Gemm::Blocked::nb<Kokkos::Impl::ActiveExecutionMemorySpace>() };
        
            InnerGemmFixC<mb,nb> inner(as0, as1, bs0, bs1, cs0, cs1);
            InnerGemmFixC<0,0> remainder(as0, as1, bs0, bs1, cs0, cs1);

            auto gemm = [&](const int ib, 
                            const int jb,
                            const int pb,
                            const value_type *__restrict__ AA,
                            const value_type *__restrict__ BB,
                            /**/  value_type *__restrict__ CC) {
              if (ib <= 5 && ib == jb) {
                remainder.team_invoke(member, alpha, AA, BB, ib, jb, pb, CC);
              } else {
                const int
                mq = (ib/mb), mm = (mq*mb), mp = (ib%mb), 
                nq = (jb/nb), nn = (nq*nb), np = (jb%nb);         
            
                {
                  // square tiling
                  Kokkos::parallel_for
                    (Kokkos::TeamThreadRange(member, mq*nq ),
                     [&](const int &ij) {
                      const int i = ij%mq*mb, j = ij/mq*nb;
                      inner.serial_invoke(alpha, AA+i*as0, BB+j*bs1, k, CC+i*cs0+j*cs1);
                    });
                  if (mp)
                    Kokkos::parallel_for
                      (Kokkos::TeamThreadRange(member, nq ),
                       [&](const int &jj) {
                        const int j = jj*nb;
                        inner.serial_invoke(alpha, AA+mm*as0, BB+j*bs1, mp, nb, pb, CC+mm*cs0+j*cs1);            
                      });
                  
                  if (np)
                    Kokkos::parallel_for
                      (Kokkos::TeamThreadRange(member, mq ),
                       [&](const int &ii) {
                        const int i = ii*mb;
                        inner.serial_invoke(alpha, AA+i*as0, BB+nn*bs1, mb, np, pb, CC+i*cs0+nn*cs1);
                      });

                  if (mp && np) {
                    remainder.team_invoke(member, alpha, AA+mm*as0, BB+nn*bs1, mp, np, pb, CC+mm*cs0+nn*cs1);
                  }
                }
              }
            };          
        
            const bool is_small = (m*n*k <= 64*64*64);
            if (is_small) {
              gemm(m, n, k, A, B, C);
            } else {
              // cache blocking
              const int 
                nc = nb*10, kc = mb*4, mc = mb*4;
          
              for (int jj=0;jj<n;jj+=nc) {
                const int tj = n-jj, jb = (tj < nc ? tj : nc);
                for (int pp=0;pp<k;pp+=kc) {
                  const int tp = k-pp, pb = (tp < kc ? tp : kc);
                  //const int pb = k, pp = 0;
                  for (int ii=0;ii<m;ii+=mc) {
                    const int ti = m-ii, ib = (ti < mc ? ti : mc);
                
                    const value_type *__restrict__ AA = A+ii*as0+pp*as1;
                    const value_type *__restrict__ BB = B+pp*bs0+jj*bs1;
                    /**/  value_type *__restrict__ CC = C+ii*cs0+jj*cs1;
                
                    gemm(ib, jb, pb, AA, BB, CC);                  
                  } // for ii
                } // for pp
              } // for jj          
            }

          }
          return 0;
        };

      } // end namespace Team

    }
  }
}
#endif
