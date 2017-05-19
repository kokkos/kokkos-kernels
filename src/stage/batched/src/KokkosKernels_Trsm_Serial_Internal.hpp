#ifndef __KOKKOSKERNELS_TRSM_SERIAL_INTERNAL_HPP__
#define __KOKKOSKERNELS_TRSM_SERIAL_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosKernels_Util.hpp"

#include "KokkosKernels_Set_Internal.hpp"
#include "KokkosKernels_Scale_Internal.hpp"

#include "KokkosKernels_InnerGemmFixA_Serial_Impl.hpp"
#include "KokkosKernels_InnerTrsm_Serial_Impl.hpp"

namespace KokkosKernels {
  namespace Batched {
    namespace Experimental {
      ///
      /// Serial Internal Impl
      /// ====================
      namespace Serial {
        template<typename AlgoType>
        struct TrsmInternalLeftLower {
          template<typename ScalarType,
                   typename ValueType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const bool use_unit_diag,
                 const int m, const int n, 
                 const ScalarType alpha,
                 const ValueType *__restrict__ A, const int as0, const int as1,
                 /**/  ValueType *__restrict__ B, const int bs0, const int bs1);
        };

        template<>
        template<typename ScalarType,
                 typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int 
        TrsmInternalLeftLower<Algo::Trsm::Unblocked>::
        invoke(const bool use_unit_diag,
               const int m, const int n,
               const ScalarType alpha,
               const ValueType *__restrict__ A, const int as0, const int as1,
               /**/  ValueType *__restrict__ B, const int bs0, const int bs1) {
          typedef ValueType value_type;
        
          if (alpha == 0)   Serial::SetInternal::invoke(m, n, value_type(0), B, bs0, bs1);
          else {
            if (alpha != 1) Serial::ScaleInternal::invoke(m, n, value_type(alpha), B, bs0, bs1);
            if (m <= 0 || n <= 0) return 0;

            for (int p=0;p<m;++p) {
              const int
                iend = m-p-1,
                jend = n;
          
              const value_type
                *__restrict__ a21 = iend ? A+(p+1)*as0+p*as1 : NULL;
            
              value_type
                *__restrict__ b1t =        B+p*bs0,
                *__restrict__ B2  = iend ? B+(p+1)*bs0 : NULL;
          
              if (!use_unit_diag) {
                const value_type alpha11 = A[p*as0+p*as1];
                for (int j=0;j<jend;++j)
                  b1t[j*bs1] /= alpha11;
              }
          
              for (int i=0;i<iend;++i)
                for (int j=0;j<jend;++j)
                  B2[i*bs0+j*bs1] -= a21[i*as0] * b1t[j*bs1];
            }
          }      
          return 0;
        }

        template<>
        template<typename ScalarType,
                 typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int
        TrsmInternalLeftLower<Algo::Trsm::Blocked>::
        invoke(const bool use_unit_diag,
               const int m, const int n,
               const ScalarType alpha,
               const ValueType *__restrict__ A, const int as0, const int as1,
               /**/  ValueType *__restrict__ B, const int bs0, const int bs1) {
          typedef ValueType value_type;

          if (alpha == 0)   Serial::SetInternal::invoke(m, n, value_type(0), B, bs0, bs1);
          else {
            if (alpha != 1) Serial::ScaleInternal::invoke(m, n, value_type(alpha), B, bs0, bs1);
            if (m <= 0 || n <= 0) return 0;

            {
              enum : int {
                mb = Algo::Trsm::Blocked::mb };

              InnerTrsmLeftLowerUnitDiag<mb>    trsm_u(as0, as1, bs0, bs1);
              InnerTrsmLeftLowerNonUnitDiag<mb> trsm_n(as0, as1, bs0, bs1);

              InnerTrsmLeftLowerUnitDiag<0>    remainder_trsm_u(as0, as1, bs0, bs1);
              InnerTrsmLeftLowerNonUnitDiag<0> remainder_trsm_n(as0, as1, bs0, bs1);

              InnerGemmFixA<mb,mb> gemm(as0, as1, bs0, bs1, bs0, bs1);
          
              auto trsm = [&](const int ib, 
                              const int jb,
                              const value_type *__restrict__ AA,
                              /**/  value_type *__restrict__ BB) {
                const int 
                mm = (ib/mb)*mb, mp = (ib%mb);
            
                for (int p=0;p<mm;p+=mb) {
                  // trsm update
                  {
                    const value_type *__restrict__ Ap = AA+p*as0+p*as1;
                    /**/  value_type *__restrict__ Bp = BB+p*bs0;
                
                    if (use_unit_diag) trsm_u.serial_invoke(Ap, jb, Bp);
                    else               trsm_n.serial_invoke(Ap, jb, Bp);
                  }

                  // gemm update
                  {
                    for (int i=p+mb;i<mm;i+=mb)
                      gemm.serial_invoke(-1, AA+i*as0+p*as1, BB+p*bs0, jb, BB+i*bs0);
                
                    if (mp) 
                      gemm.serial_invoke(-1, AA+mm*as0+p*as1, BB+p*bs0, mp, jb, mb, BB+mm*bs0); 
                  }
                }
            
                if (mp) {
                  {
                    const value_type *__restrict__ Ap = AA+mm*as0+mm*as1;
                    /**/  value_type *__restrict__ Bp = BB+mm*bs0;
                
                    if (use_unit_diag) remainder_trsm_u.serial_invoke(Ap, mp, jb, Bp);
                    else               remainder_trsm_n.serial_invoke(Ap, mp, jb, Bp);            
                  }
                }
              };

              const bool is_small = (m*n <= 64*64);
              if (is_small) {
                trsm(m, n, A, B);
              } else {
                // some cache blocking may need (not priority yet);
                trsm(m, n, A, B);
              }
            }        
          }
          return 0;
        }

        template<typename AlgoType>
        struct TrsmInternalLeftUpper {
          template<typename ScalarType,
                   typename ValueType>
          KOKKOS_INLINE_FUNCTION
          static int
          invoke(const bool use_unit_diag,
                 const int m, const int n, 
                 const ScalarType alpha,
                 const ValueType *__restrict__ A, const int as0, const int as1,
                 /**/  ValueType *__restrict__ B, const int bs0, const int bs1) {
            //static_assert("KokkosKernels::TrsmInternalLeftUpper:: Not yet implemented");
            return 0;
          }
        };

        template<>
        template<typename ScalarType,
                 typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int 
        TrsmInternalLeftUpper<Algo::Trsm::Unblocked>::
        invoke(const bool use_unit_diag,
               const int m, const int n,
               const ScalarType alpha,
               const ValueType *__restrict__ A, const int as0, const int as1,
               /**/  ValueType *__restrict__ B, const int bs0, const int bs1) {
          typedef ValueType value_type;
  
          if (alpha == 0)   Serial::SetInternal::invoke(m, n, value_type(0), B, bs0, bs1);
          else {
            if (alpha != 1) Serial::ScaleInternal::invoke(m, n, value_type(alpha), B, bs0, bs1);
            if (m <= 0 || n <= 0) return 0;
        
            value_type *__restrict__ B0 = B;
            for (int p=(m-1);p>=0;--p) {
              const int iend = p, jend = n;

              const value_type *__restrict__ a01 = A+p*as1;
              /**/  value_type *__restrict__ b1t = B+p*bs0;
            
              if (!use_unit_diag) {
                const value_type alpha11 = A[p*as0+p*as1];
                for (int j=0;j<n;++j)
                  b1t[j*bs1] /= alpha11;
              }
              for (int i=0;i<iend;++i)
                for (int j=0;j<jend;++j)
                  B0[i*bs0+j*bs1] -= a01[i*as0] * b1t[j*bs1];
            }
          }
          return 0;
        };

        template<>
        template<typename ScalarType,
                 typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int 
        TrsmInternalLeftUpper<Algo::Trsm::Blocked>::
        invoke(const bool use_unit_diag,
               const int m, const int n,
               const ScalarType alpha,
               const ValueType *__restrict__ A, const int as0, const int as1,
               /**/  ValueType *__restrict__ B, const int bs0, const int bs1) {
          typedef ValueType value_type;

          if (alpha == 0)   Serial::SetInternal::invoke(m, n, value_type(0), B, bs0, bs1);
          else {
            if (alpha != 1) Serial::ScaleInternal::invoke(m, n, value_type(alpha), B, bs0, bs1);
            if (m <= 0 || n <= 0) return 0;

            {
              enum : int {
                mb = Algo::Trsm::Blocked::mb };

              InnerTrsmLeftUpperUnitDiag<mb>    trsm_u(as0, as1, bs0, bs1);
              InnerTrsmLeftUpperNonUnitDiag<mb> trsm_n(as0, as1, bs0, bs1);

              InnerTrsmLeftUpperUnitDiag<0> remainder_trsm_u(as0, as1, bs0, bs1);
              InnerTrsmLeftUpperNonUnitDiag<0> remainder_trsm_n(as0, as1, bs0, bs1);
          
              InnerGemmFixA<mb,mb> gemm(as0, as1, bs0, bs1, bs0, bs1);
          
              auto trsm = [&](const int ib, 
                              const int jb,
                              const value_type *__restrict__ AA,
                              /**/  value_type *__restrict__ BB) {
                const int mm = (m/mb)*mb, mp = (m%mb);
            
                for (int pp=0;pp<mm;pp+=mb) {
                  const int p = m - pp - mb;
              
                  // trsm update
                  {
                    const value_type *__restrict__ Ap = AA+p*as0+p*as1;
                    /**/  value_type *__restrict__ Bp = BB+p*bs0;
                
                    if (use_unit_diag) trsm_u.serial_invoke(Ap, jb, Bp);
                    else               trsm_n.serial_invoke(Ap, jb, Bp);
                  }

                  // gemm update
                  {
                    for (int i=(p-mb);i>=0;i-=mb)
                      gemm.serial_invoke(-1, AA+i*as0+p*as1, BB+p*bs0, jb, BB+i*bs0);
                
                    if (mp) 
                      gemm.serial_invoke(-1, AA+p*as1, BB+p*bs0, mp, jb, mb, BB);               
                  }
                }

                if (mp) {
                  if (use_unit_diag) remainder_trsm_u.serial_invoke(AA, mp, jb, BB);
                  else               remainder_trsm_n.serial_invoke(AA, mp, jb, BB);
                }
              };
          
              const bool is_small = (m*n <= 64*64);
              if (is_small) {
                trsm(m, n, A, B);
              } else {
                // some cache blocking may need (not priority yet);
                trsm(m, n, A, B);
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
