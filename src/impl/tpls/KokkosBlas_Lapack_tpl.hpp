#ifndef KOKKOSBLASLAPACK_LAPACK_TPL_HPP_
#define KOKKOSBLASLAPACK_LAPACK_TPL_HPP_

#include<KokkosBlas_Host_tpl.hpp>

#if defined (KOKKOSKERNELS_ENABLE_TPL_LAPACK)
#include<KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
  namespace Impl {
    template<>
    int 
    HostBlas<double >::trtri(const int matrix_layout, const char uplo, const char diag,
                                         int n, // TODO: n,lda lapack_int
                                         const double *a, int lda) {
      return LAPACKE_dtrtri(matrix_layout, uplo, diag, n, const_cast <double*> (a), lda);
    }
    template<>
    int 
    HostBlas<std::complex<double> >::trtri(const int matrix_layout, const char uplo, const char diag,
                                         int n, // TODO: n,lda lapack_int
                                         const std::complex<double> *a, int lda) {
      return LAPACKE_ztrtri(matrix_layout, uplo, diag, n, reinterpret_cast<lapack_complex_double*> (const_cast <std::complex<double>*> (a)), lda);
    }
  } // Impl
} // KokkosBlas
#endif // KOKKOSKERNELS_ENABLE_TPL_LAPACK
#endif // KOKKOSBLASLAPACK_LAPACK_TPL_HPP_
