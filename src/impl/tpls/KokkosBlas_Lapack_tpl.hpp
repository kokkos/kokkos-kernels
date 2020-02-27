#ifndef KOKKOSBLASLAPACK_LAPACK_TPL_HPP_
#define KOKKOSBLASLAPACK_LAPACK_TPL_HPP_

#include<KokkosBlas_Host_tpl.hpp> // trtri prototype
#include<KokkosBlas_tpl_spec.hpp> // LAPACKE_

#if defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK)

namespace KokkosBlas {
  namespace Impl {
    //// TRTRI ////
    template<>
    int 
    HostBlas<double >::trtri(const int matrix_layout, const char uplo, const char diag,
                                         int n, // TODO: n,lda lapack_int
                                         const double *a, int lda) {
      return LAPACKE_dtrtri(matrix_layout, uplo, diag, n, const_cast <double*> (a), lda);
    }
    template<>
    int 
    HostBlas<float >::trtri(const int matrix_layout, const char uplo, const char diag,
                                         int n, // TODO: n,lda lapack_int
                                         const float *a, int lda) {
      return LAPACKE_strtri(matrix_layout, uplo, diag, n, const_cast <float*> (a), lda);
    }
    template<>
    int 
    HostBlas<std::complex<double> >::trtri(const int matrix_layout, const char uplo, const char diag,
                                         int n, // TODO: n,lda lapack_int
                                         const std::complex<double> *a, int lda) {
      return LAPACKE_ztrtri(matrix_layout, uplo, diag, n, reinterpret_cast<lapack_complex_double*> (const_cast <std::complex<double>*> (a)), lda);
    }
    template<>
    int 
    HostBlas<std::complex<float> >::trtri(const int matrix_layout, const char uplo, const char diag,
                                         int n, // TODO: n,lda lapack_int
                                         const std::complex<float> *a, int lda) {
      return LAPACKE_ctrtri(matrix_layout, uplo, diag, n, reinterpret_cast<lapack_complex_float*> (const_cast <std::complex<float>*> (a)), lda);
    }
  } // Impl
} // KokkosBlas
#endif // KOKKOSKERNELS_ENABLE_TPL_LAPACK
#endif // KOKKOSBLASLAPACK_LAPACK_TPL_HPP_
