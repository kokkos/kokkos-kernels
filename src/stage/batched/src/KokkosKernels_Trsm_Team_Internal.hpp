#ifndef __KOKKOSKERNELS_TRSM_TEAM_INTERNAL_HPP__
#define __KOKKOSKERNELS_TRSM_TEAM_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"

#include "KokkosKernels_Set_Internal.hpp"
#include "KokkosKernels_Scale_Internal.hpp"

#include "KokkosKernels_InnerTrsm_Serial_Impl.hpp"
#include "KokkosKernels_Gemm_Team_Internal.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {
      ///
      /// Team Internal Impl
      /// ====================
      namespace Team {
        template<typename AlgoType>
        struct TrsmInternalLeftLower {
          template<typename MemberType,
                   typename ScalarType,
                   typename ValueType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const MemberType &member, 
                 const bool use_unit_diag,
                 const int m, const int n, 
                 const ScalarType alpha,
                 const ValueType *__restrict__ A, const int as0, const int as1,
                 /**/  ValueType *__restrict__ B, const int bs0, const int bs1);
        };

        template<>
        template<typename MemberType,
                 typename ScalarType,
                 typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int 
        TrsmInternalLeftLower<Algo::Trsm::Unblocked>::
        invoke(const MemberType &member, 
               const bool use_unit_diag,
               const int m, const int n,
               const ScalarType alpha,
               const ValueType *__restrict__ A, const int as0, const int as1,
               /**/  ValueType *__restrict__ B, const int bs0, const int bs1) {
          typedef ValueType value_type;

          // note that parallel range is different ( m*n vs m-1*n);        
          const bool team_barrier = true;
          if (alpha == 0)   Team::SetInternal::invoke(member, m, n, value_type(0), B, bs0, bs1, team_barrier);
          else {
            if (alpha != 1) Team::ScaleInternal::invoke(member, m, n, value_type(alpha), B, bs0, bs1, team_barrier);
            if (m <= 0 || n <= 0) return 0;

            member.team_barrier();
            for (int p=0;p<m;++p) {
              const int iend = m-p-1, jend = n;
          
              const value_type
                *__restrict__ a21 = iend ? A+(p+1)*as0+p*as1 : NULL;
            
              value_type
                *__restrict__ b1t =        B+p*bs0,
                *__restrict__ B2  = iend ? B+(p+1)*bs0 : NULL;

              if (!use_unit_diag) {
                const value_type alpha11 = A[p*as0+p*as1];
                Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,jend),[&](const int &j) {
                    b1t[j*bs1] /= alpha11;
                  });
                member.team_barrier();
              }
              Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,iend*jend),[&](const int &ij) {
#if \
  defined (KOKKOS_HAVE_CUDA) && \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
                  const int i = ij%iend, j = ij/iend;
#else
                  const int i = ij/jend, j = ij%jend;
#endif
                  B2[i*bs0+j*bs1] -= a21[i*as0] * b1t[j*bs1];
                });          
            }
          }      
          return 0;
        }

        template<>
        template<typename MemberType,
                 typename ScalarType,
                 typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int
        TrsmInternalLeftLower<Algo::Trsm::Blocked>::
        invoke(const MemberType &member, 
               const bool use_unit_diag,
               const int m, const int n,
               const ScalarType alpha,
               const ValueType *__restrict__ A, const int as0, const int as1,
               /**/  ValueType *__restrict__ B, const int bs0, const int bs1) {
          typedef ValueType value_type;

          // note that parallel range is different ( m*n vs m-1*n);        
          const bool team_barrier = true;
          if (alpha == 0)   Team::SetInternal::invoke(member, m, n, value_type(0), B, bs0, bs1, team_barrier);
          else {
            if (alpha != 1) Team::ScaleInternal::invoke(member, m, n, value_type(alpha), B, bs0, bs1, team_barrier);
            if (m <= 0 || n <= 0) return 0;

            {
              enum : int {
                mb = Algo::Trsm::Blocked::mb<Kokkos::Impl::ActiveExecutionMemorySpace>()
              };

              ///
              /// case host: team size is small and blocksize (mb,nb) is large

              ///
              /// case cuda: team size is large and blocksize (mb,nb) is small
              InnerTrsmLeftLowerUnitDiag<mb>    trsm_u(as0, as1, bs0, bs1);
              InnerTrsmLeftLowerNonUnitDiag<mb> trsm_n(as0, as1, bs0, bs1);
              
              auto trsm = [&](const int ib, 
                              const int jb,
                              const value_type *__restrict__ AA,
                              /**/  value_type *__restrict__ BB) {
                for (int p=0;p<ib;p+=mb) {
                  const int pb = (p+mb) > ib ? (ib-p) : mb; 
                  
                  // trsm update
                  const value_type *__restrict__ Ap = AA+p*as0+p*as1;
                  /**/  value_type *__restrict__ Bp = BB+p*bs0;
                  
                  const int np = jb%mb;
                  Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,(jb/mb)+(np>0)),[&](const int &jj) {
                      const int j = jj*mb, qb = (j+mb) > jb ? np : mb;
                      if (use_unit_diag) trsm_u.serial_invoke(Ap, pb, qb, Bp+j*bs1);
                      else               trsm_n.serial_invoke(Ap, pb, qb, Bp+j*bs1);
                    });
                  member.team_barrier();
                  
                  // gemm update
                  GemmInternal<Algo::Gemm::Blocked>
                  ::invoke(member,
                           ib-p-pb, jb, pb,
                           -1,
                           Ap+pb*as0, as0, as1,
                           Bp, bs0, bs1,
                           1,
                           Bp+pb*bs0, bs0, bs1);
                }
              };
              
              const bool is_small = true; //(m*n <= 64*64);
              if (is_small) {
                trsm(m, n, A, B);
              } else {
                // // some cache blocking may need (not priority yet);
                // trsm(m, n, A, B);
              }
            }        
          }
          return 0;
        }

        template<typename AlgoType>
        struct TrsmInternalLeftUpper {
          template<typename MemberType,
                   typename ScalarType,
                   typename ValueType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const MemberType &member, 
                 const bool use_unit_diag,
                 const int m, const int n, 
                 const ScalarType alpha,
                 const ValueType *__restrict__ A, const int as0, const int as1,
                 /**/  ValueType *__restrict__ B, const int bs0, const int bs1);
        };

        template<>
        template<typename MemberType,
                 typename ScalarType,
                 typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int 
        TrsmInternalLeftUpper<Algo::Trsm::Unblocked>::
        invoke(const MemberType &member, 
               const bool use_unit_diag,
               const int m, const int n,
               const ScalarType alpha,
               const ValueType *__restrict__ A, const int as0, const int as1,
               /**/  ValueType *__restrict__ B, const int bs0, const int bs1) {
          typedef ValueType value_type;
  
          // note that parallel range is different ( m*n vs m-1*n);        
          const bool team_barrier = true;
          if (alpha == 0)   Team::SetInternal::invoke(m, n, value_type(0), B, bs0, bs1, team_barrier);
          else {
            if (alpha != 1) Team::ScaleInternal::invoke(m, n, value_type(alpha), B, bs0, bs1, team_barrier);
            if (m <= 0 || n <= 0) return 0;
        
            value_type *__restrict__ B0 = B;
            for (int p=(m-1);p>=0;--p) {
              const int iend = p, jend = n;

              const value_type *__restrict__ a01 = A+p*as1;
              /**/  value_type *__restrict__ b1t = B+p*bs0;
            
              if (!use_unit_diag) {
                const value_type alpha11 = A[p*as0+p*as1];
                Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,jend),[&](const int &j) {
                    b1t[j*bs1] /= alpha11;
                  });
                member.team_barrier();
              }

              Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,iend*jend),[&](const int &ij) {
#if \
  defined (KOKKOS_HAVE_CUDA) && \
  defined (KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
                  const int i = ij%iend, j = ij/iend;
#else
                  const int i = ij/jend, j = ij%jend;
#endif
                  B0[i*bs0+j*bs1] -= a01[i*as0] * b1t[j*bs1];
                });          
            }
          }
          return 0;
        };

        template<>
        template<typename MemberType,
                 typename ScalarType,
                 typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int 
        TrsmInternalLeftUpper<Algo::Trsm::Blocked>::
        invoke(const MemberType &member,
               const bool use_unit_diag,
               const int m, const int n,
               const ScalarType alpha,
               const ValueType *__restrict__ A, const int as0, const int as1,
               /**/  ValueType *__restrict__ B, const int bs0, const int bs1) {
          typedef ValueType value_type;

          // note that parallel range is different ( m*n vs m-1*n);        
          const bool team_barrier = true;
          if (alpha == 0)   Team::SetInternal::invoke(m, n, value_type(0), B, bs0, bs1, team_barrier);
          else {
            if (alpha != 1) Team::ScaleInternal::invoke(m, n, value_type(alpha), B, bs0, bs1, team_barrier);
            if (m <= 0 || n <= 0) return 0;

            {
              enum : int {
                mb = Algo::Trsm::Blocked::mb<Kokkos::Impl::ActiveExecutionMemorySpace>()
              };

              InnerTrsmLeftUpperUnitDiag<mb>    trsm_u(as0, as1, bs0, bs1);
              InnerTrsmLeftUpperNonUnitDiag<mb> trsm_n(as0, as1, bs0, bs1);
          
              auto trsm = [&](const int ib, 
                              const int jb,
                              const value_type *__restrict__ AA,
                              /**/  value_type *__restrict__ BB) {
                for (int pp=0;pp<ib;pp+=mb) {
                  const int 
                  ptmp = ib - pp - mb, 
                  p = ptmp < 0 ? 0 : ptmp, 
                  pb = mb + (ptmp < 0)*ptmp;
              
                  // trsm update
                  const value_type *__restrict__ Ap = AA+p*as0+p*as1;
                  /**/  value_type *__restrict__ Bp = BB+p*bs0;

                  const int np = jb%mb;
                  Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,(jb/mb)+(np>0)),[&](const int &jj) {
                      const int j = jj*mb, qb = (j+mb) > jb ? np : mb;     
                      if (use_unit_diag) trsm_u.serial_invoke(Ap, pb, qb, Bp+j*bs1);
                      else               trsm_n.serial_invoke(Ap, pb, qb, Bp+j*bs1);
                    });
                  
                  // gemm update
                  GemmInternal<Algo::Gemm::Blocked>
                  ::invoke(member,
                           p, jb, pb,
                           -1,
                           Ap-p*as0, as0, as1,
                           Bp, bs0, bs1,
                           1,
                           BB, bs0, bs1);
                }
              };
          
              const bool is_small = true; //(m*n <= 64*64);
              if (is_small) {
                trsm(m, n, A, B);
              } else {
                // // some cache blocking may need (not priority yet);
                // trsm(m, n, A, B);
              }
            }        
          }
          return 0;      
        };

      }
    }
  }
}
#endif
