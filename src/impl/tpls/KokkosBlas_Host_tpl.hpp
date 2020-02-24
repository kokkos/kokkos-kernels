#ifndef KOKKOSBLAS_HOST_TPL_HPP_
#define KOKKOSBLAS_HOST_TPL_HPP_

/// \file  KokkosBlas_Host_tpl.hpp 
/// \brief BLAS wrapper
/// \author Kyungjoo Kim (kyukim@sandia.gov)


#include "KokkosKernels_config.h"
#include "Kokkos_ArithTraits.hpp"
#if defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK)
// TODO: include lapacke_config.h for lapack_int
#endif // KOKKOSKERNELS_ENABLE_TPL_LAPACK

#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) || defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK)

namespace KokkosBlas {
  namespace Impl {

    template<typename T>
    struct HostBlas {
      typedef Kokkos::ArithTraits<T> ats;
      typedef typename ats::mag_type mag_type;
  
#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS)
      static
      void scal(int n,
                const T alpha,
                /* */ T *x, int x_inc);

      static 
      int iamax(int n, 
                const T *x, int x_inc);
      
      static 
      mag_type nrm2(int n,
                    const T *x, int x_inc);

      static
      mag_type asum(int n, 
                    const T *x, int x_inc);
      
      static 
      T dot(int n,
            const T *x, int x_inc,
            const T *y, int y_inc);

      static
      void axpy(int n, 
                const T alpha,
                const T *x, int x_inc,
                /* */ T *y, int y_inc);

      static 
      void gemv(const char trans, 
                int m, int n, 
                const T alpha, 
                const T *a, int lda,
                const T *b, int ldb,
                const T beta,
                /* */ T *c, int ldc);

      static 
      void trsv(const char uplo, const char transa, const char diag, 
                int m, 
                const T *a, int lda,
                /* */ T *b, int ldb);

      static 
      void gemm(const char transa, const char transb, 
                int m, int n, int k,
                const T alpha, 
                const T *a, int lda,
                const T *b, int ldb,
                const T beta,
                /* */ T *c, int ldc);

      static 
      void herk(const char transa, const char transb, 
                int n, int k,
                const T alpha, 
                const T *a, int lda,
                const T beta,
                /* */ T *c, int ldc);

      static 
      void trmm(const char side, const char uplo, const char transa, const char diag,
                int m, int n, 
                const T alpha, 
                const T *a, int lda,
                /* */ T *b, int ldb);

      static 
      void trsm(const char side, const char uplo, const char transa, const char diag,
                int m, int n, 
                const T alpha, 
                const T *a, int lda,
                /* */ T *b, int ldb);

      static
      void gesv(int n, int rhs,
                T *a, int lda, int *ipiv,
                T *b, int ldb,
                int info);
#endif // KOKKOSKERNELS_ENABLE_TPL_BLAS

#if defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK)
      //static
      //lapack_int_t trtri(const int matrix_layout, const char uplo, const char diag,
      //           lapack_int_t n,
      //           const T *a, lapack_int_t lda);
      static
      int trtri(const int matrix_layout, const char uplo, const char diag,
                 int n,
                 const T *a, int lda);
#endif // KOKKOSKERNELS_ENABLE_TPL_LAPACK
    };
  }
}

#endif // KOKKOSKERNELS_ENABLE_TPL_BLAS || defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK)

#endif // KOKKOSBLAS_HOST_TPL_HPP_
