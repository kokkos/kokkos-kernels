#ifndef KOKKOSBLAS_CUDA_TPL_HPP_
#define KOKKOSBLAS_CUDA_TPL_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include<KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

CudaBlasSingleton::CudaBlasSingleton()
{
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
        Kokkos::abort("CUBLAS initialization failed\n");
    printf("cuBLAS context initialized.\n");
    Kokkos::push_finalize_hook ([&] () { 
        cublasDestroy(handle); 
        printf("cuBLAS context released.\n");
    });
}

CudaBlasSingleton & CudaBlasSingleton::singleton()
{ static CudaBlasSingleton s ; return s ; }

}
}
#endif

#endif