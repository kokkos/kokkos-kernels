#ifndef KOKKOSLAPACK_HOST_TPL_HPP_
#define KOKKOSLAPACK_HOST_TPL_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_ArithTraits.hpp"

#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) && \
    defined(KOKKOSKERNELS_ENABLE_TPL_LAPACKE)
#include <lapacke.h>
#include <cblas.h>

namespace KokkosBlas {
namespace Impl {

template <typename T>
struct HostLapack {
  typedef Kokkos::ArithTraits<T> ats;
  typedef typename ats::mag_type mag_type;

  static void potrf(bool matrix_layout, char uplo, int n, T* a, int lda);

  static void geqp3(bool matrix_layout, int m, int n, T* a, int lda, int* jpvt,
                    T* tau);

  static void geqrf(bool matrix_layout, int m, int n, T* a, int lda, T* tau,
                    T* work, int lwork);

  static void unmqr(bool matrix_layout, char side, char trans, int m, int n,
                    int k, const T* a, int lda, const T* tau,
                    /* */ T* c, int ldc,
                    /* */ T* work, int lwork);

  static void ormqr(bool matrix_layout, char side, char trans, int m, int n,
                    int k, const T* a, int lda, const T* tau,
                    /* */ T* c, int ldc,
                    /* */ T* work, int lwork);

};  // HostLapack

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // ENABLE BLAS/LAPACK

#endif  // KOKKOSLAPACK_HOST_TPL_HPP_
