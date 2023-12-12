#ifndef __KOKKOSBLAS_MKL_TPL_HPP__
#define __KOKKOSBLAS_MKL_TPL_HPP__

#if defined(KOKKOSKERNELS_ENABLE_TPL_MKL)

#define KOKKOSKERNELS_IMPL_MKL_VERSION = 
  ((__INTEL_MKL__        * 100)
+ (__INTEL_MKL_MINOR__  * 10)
+ (__INTEL_MKL_UPDATE__ * 1));

#endif

#endif
