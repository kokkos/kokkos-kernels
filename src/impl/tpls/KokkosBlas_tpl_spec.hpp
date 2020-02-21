#ifndef KOKKOSBLAS_TPL_SPEC_HPP_
#define KOKKOSBLAS_TPL_SPEC_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include "cuda_runtime.h"
#include "cublas_v2.h"

namespace KokkosBlas {
namespace Impl {

struct CudaBlasSingleton {
  cublasHandle_t handle;

  CudaBlasSingleton();

  static CudaBlasSingleton & singleton();
};

}
}
#endif // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include "magma_v2.h"
#include "magma_lapack.h"

namespace KokkosBlas {
namespace Impl {

struct MagmaSingleton {

  MagmaSingleton();

  static MagmaSingleton & singleton();
};

}
}
#endif // KOKKOSKERNELS_ENABLE_TPL_MAGMA

#endif // KOKKOSBLAS_TPL_SPEC_HPP_
