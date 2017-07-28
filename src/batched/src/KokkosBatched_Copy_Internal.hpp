#ifndef __KOKKOSBATCHED_COPY_INTERNAL_HPP__
#define __KOKKOSBATCHED_COPY_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"


namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Serial Internal Impl
    /// ==================== 
    namespace Serial {
      // do i need this ?
      template<int mb>
      struct CopyUnrolled {
        const int _as, _bs;
        KOKKOS_INLINE_FUNCTION
        CopyUnrolled(const int as, const int bs) : _as(as), _bs(bs) {}

        template<typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int invoke(const ValueType *__restrict__ A, 
                   /* */ ValueType *__restrict__ B);

        template<typename ValueType>
        KOKKOS_INLINE_FUNCTION
        int invoke(const int k,
                   const ValueType *__restrict__ A, 
                   /* */ ValueType *__restrict__ B);
      };

      template<>
      template<typename ValueType> 
      KOKKOS_INLINE_FUNCTION int
      CopyUnrolled<8>::invoke(const ValueType *__restrict__ A,
                              /* */ ValueType *__restrict__ B) {
        B[0*_bs] = A[0*_as]; B[1*_bs] = A[1*_as]; B[2*_bs] = A[2*_as]; B[3*_bs] = A[3*_as];
        B[4*_bs] = A[4*_as]; B[5*_bs] = A[5*_as]; B[6*_bs] = A[6*_as]; B[7*_bs] = A[7*_as];
        return 8;
      }

      template<>
      template<typename ValueType> 
      KOKKOS_INLINE_FUNCTION int
      CopyUnrolled<7>::invoke(const ValueType *__restrict__ A,
                              /* */ ValueType *__restrict__ B) {
        B[0*_bs] = A[0*_as]; B[1*_bs] = A[1*_as]; B[2*_bs] = A[2*_as]; B[3*_bs] = A[3*_as];
        B[4*_bs] = A[4*_as]; B[5*_bs] = A[5*_as]; B[6*_bs] = A[6*_as]; 
        return 8;
      }

      template<>
      template<typename ValueType> 
      KOKKOS_INLINE_FUNCTION int
      CopyUnrolled<6>::invoke(const ValueType *__restrict__ A,
                              /* */ ValueType *__restrict__ B) {
        B[0*_bs] = A[0*_as]; B[1*_bs] = A[1*_as]; B[2*_bs] = A[2*_as]; B[3*_bs] = A[3*_as];
        B[4*_bs] = A[4*_as]; B[5*_bs] = A[5*_as]; 
        return 8;
      }

      template<>
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION int
      CopyUnrolled<5>::invoke(const ValueType *__restrict__ A,
                              /* */ ValueType *__restrict__ B) {
        B[0*_bs] = A[0*_as]; B[1*_bs] = A[1*_as]; B[2*_bs] = A[2*_as]; B[3*_bs] = A[3*_as]; 
        B[4*_bs] = A[4*_as];
        return 5;
      }

      template<>
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION int
      CopyUnrolled<4>::invoke(const ValueType *__restrict__ A,
                              /* */ ValueType *__restrict__ B) {
        B[0*_bs] = A[0*_as]; B[1*_bs] = A[1*_as]; B[2*_bs] = A[2*_as]; B[3*_bs] = A[3*_as];
        return 4;
      }

      template<>
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION int
      CopyUnrolled<3>::invoke(const ValueType *__restrict__ A,
                              /* */ ValueType *__restrict__ B) {
        B[0*_bs] = A[0*_as]; B[1*_bs] = A[1*_as]; B[2*_bs] = A[2*_as];
        return 3;
      }

      template<>
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION int
      CopyUnrolled<2>::invoke(const ValueType *__restrict__ A,
                              /* */ ValueType *__restrict__ B) {
        B[0*_bs] = A[0*_as]; B[1*_bs] = A[1*_as]; 
        return 2;
      }

      template<>
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION int
      CopyUnrolled<1>::invoke(const ValueType *__restrict__ A,
                              /* */ ValueType *__restrict__ B) {
        B[0*_bs] = A[0*_as];
        return 1;
      }

      template<>
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION int
      CopyUnrolled<0>::invoke(const int k,
                              const ValueType *__restrict__ A,
                              /* */ ValueType *__restrict__ B) {
        for (int p=0;p<k;) {
          const int pas = p*_as, pbs = p*_bs;
          switch (k-p) {
          case 8: { CopyUnrolled<8> inner(_as, _bs); p += inner.invoke(A+pas, B+pbs); break; }
          case 7: { CopyUnrolled<7> inner(_as, _bs); p += inner.invoke(A+pas, B+pbs); break; }
          case 6: { CopyUnrolled<6> inner(_as, _bs); p += inner.invoke(A+pas, B+pbs); break; }
          case 5: { CopyUnrolled<5> inner(_as, _bs); p += inner.invoke(A+pas, B+pbs); break; }
          case 4: { CopyUnrolled<4> inner(_as, _bs); p += inner.invoke(A+pas, B+pbs); break; }
          case 3: { CopyUnrolled<3> inner(_as, _bs); p += inner.invoke(A+pas, B+pbs); break; }
          case 2: { CopyUnrolled<2> inner(_as, _bs); p += inner.invoke(A+pas, B+pbs); break; }
          case 1: { CopyUnrolled<1> inner(_as, _bs); p += inner.invoke(A+pas, B+pbs); break; }
          }
        }
        return 0;
      }
        
      struct CopyInternal {
        template<typename ValueType>
        KOKKOS_FORCEINLINE_FUNCTION
        static int
        invoke(const int m, 
               const ValueType *__restrict__ A, const int as0, 
               /* */ ValueType *__restrict__ B, const int bs0) {
            
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
          for (int i=0;i<m;++i) 
            B[i*bs0] = A[i*as0];

          // // manuall unroll: not sure if this works
          // Serial::CopyUnrolled<0> inner(as0, bs0);
          // for (int i=0;i<m;i+=8) 
          //   inner.invoke(i+8 > m ? m-i : 8, A+(i*as0), B+(i*bs0));

          return 0;
        }
        template<typename ValueType>
        KOKKOS_FORCEINLINE_FUNCTION
        static int
        invoke(const int m, const int n, 
               const ValueType *__restrict__ A, const int as0, const int as1,
               /* */ ValueType *__restrict__ B, const int bs0, const int bs1) {
          if (as1 < as0) 
            for (int i=0;i<m;++i) 
              invoke(n, A+i*as0, as1, B+i*bs0, bs1);
          else 
            for (int j=0;j<n;++j)               
              invoke(m, A+j*as1, as0, B+j*bs1, bs0);
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
        KOKKOS_FORCEINLINE_FUNCTION
        static int
        invoke(const MemberType &member,
               const int m, 
               const ValueType *__restrict__ A, const int as0, 
               /* */ ValueType *__restrict__ B, const int bs0) {
          Kokkos::parallel_for
            (Kokkos::TeamThreadRange(member,0,m),[&](const int &i) {
              B[i*bs0] = A[i*as0];
            });
          //member.team_barrier();
          return 0;
        }
        template<typename MemberType,
                 typename ValueType>
        KOKKOS_FORCEINLINE_FUNCTION
        static int
        invoke(const MemberType &member,
               const int m, const int n, 
               const ValueType *__restrict__ A, const int as0, const int as1,
               /* */ ValueType *__restrict__ B, const int bs0, const int bs1) {
          if (m > n) { 
            Kokkos::parallel_for
              (Kokkos::TeamThreadRange(member,0,m),[&](const int &i) {
                Serial::CopyInternal::invoke(n, A+i*as0, as1, B+i*bs0, bs1);
              });
          } else {
            Kokkos::parallel_for
              (Kokkos::TeamThreadRange(member,0,n),[&](const int &j) {
                Serial::CopyInternal::invoke(m, A+j*as1, as0, B+j*bs1, bs0);
              });
          }
          //member.team_barrier();
          return 0;
        }
      };
    } // end namespace Team

  }//  end namespace Experimental
} // end namespace KokkosBatched


#endif
