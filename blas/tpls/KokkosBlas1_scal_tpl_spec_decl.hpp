// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS1_SCAL_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS1_SCAL_TPL_SPEC_DECL_HPP_

namespace KokkosBlas {
namespace Impl {

namespace {
template <class RV, class AS, class XV>
inline void scal_print_specialization() {
#if defined(KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION)
  printf("KokkosBlas1::scal<> TPL Blas specialization for < %s , %s , %s >\n", typeid(RV).name(), typeid(AS).name(),
         typeid(XV).name());
#endif
}
}  // namespace
}  // namespace Impl
}  // namespace KokkosBlas

#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS)
#include "KokkosBlas_Host_tpl.hpp"

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_BLAS(EXEC_SPACE, SCALAR_TYPE, BASE_SCALAR_TYPE, LAYOUT, ETI_SPEC_AVAIL)        \
  template <>                                                                                                          \
  struct Scal<EXEC_SPACE, Kokkos::View<SCALAR_TYPE*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,    \
              SCALAR_TYPE,                                                                                             \
              Kokkos::View<const SCALAR_TYPE*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1, true, \
              ETI_SPEC_AVAIL> {                                                                                        \
    typedef Kokkos::View<SCALAR_TYPE*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > RV;               \
    typedef SCALAR_TYPE AS;                                                                                            \
    typedef Kokkos::View<const SCALAR_TYPE*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XV;         \
    typedef typename XV::size_type size_type;                                                                          \
                                                                                                                       \
    static void scal(const EXEC_SPACE& space, const RV& R, const AS& alpha, const XV& X) {                             \
      Kokkos::Profiling::pushRegion("KokkosBlas::scal[TPL_BLAS," #SCALAR_TYPE "]");                                    \
      const size_type numElems = X.extent(0);                                                                          \
      if ((numElems < static_cast<size_type>(INT_MAX)) && (R.data() == X.data())) {                                    \
        scal_print_specialization<RV, AS, XV>();                                                                       \
        int N                          = numElems;                                                                     \
        int one                        = 1;                                                                            \
        const BASE_SCALAR_TYPE alpha_b = static_cast<BASE_SCALAR_TYPE>(alpha);                                         \
        HostBlas<BASE_SCALAR_TYPE>::scal(N, alpha_b, reinterpret_cast<BASE_SCALAR_TYPE*>(R.data()), one);              \
      } else {                                                                                                         \
        Scal<EXEC_SPACE, RV, AS, XV, 1, false, ETI_SPEC_AVAIL>::scal(space, R, alpha, X);                              \
      }                                                                                                                \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

#define KOKKOSBLAS1_DSCAL_TPL_SPEC_DECL_BLAS(EXEC_SPACE, LAYOUT, ETI_SPEC_AVAIL) \
  KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_BLAS(EXEC_SPACE, double, double, LAYOUT, ETI_SPEC_AVAIL)

#define KOKKOSBLAS1_SSCAL_TPL_SPEC_DECL_BLAS(EXEC_SPACE, LAYOUT, ETI_SPEC_AVAIL) \
  KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_BLAS(EXEC_SPACE, float, float, LAYOUT, ETI_SPEC_AVAIL)

#define KOKKOSBLAS1_ZSCAL_TPL_SPEC_DECL_BLAS(EXEC_SPACE, LAYOUT, ETI_SPEC_AVAIL)                          \
  KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_BLAS(EXEC_SPACE, Kokkos::complex<double>, std::complex<double>, LAYOUT, \
                                       ETI_SPEC_AVAIL)

#define KOKKOSBLAS1_CSCAL_TPL_SPEC_DECL_BLAS(EXEC_SPACE, LAYOUT, ETI_SPEC_AVAIL) \
  KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_BLAS(EXEC_SPACE, Kokkos::complex<float>, std::complex<float>, LAYOUT, ETI_SPEC_AVAIL)

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS1_DSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, true)
KOKKOSBLAS1_DSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, false)

KOKKOSBLAS1_SSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, true)
KOKKOSBLAS1_SSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, false)

KOKKOSBLAS1_ZSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, true)
KOKKOSBLAS1_ZSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, false)

KOKKOSBLAS1_CSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, true)
KOKKOSBLAS1_CSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, false)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS1_DSCAL_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, true)
KOKKOSBLAS1_DSCAL_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, false)

KOKKOSBLAS1_SSCAL_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, true)
KOKKOSBLAS1_SSCAL_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, false)

KOKKOSBLAS1_ZSCAL_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, true)
KOKKOSBLAS1_ZSCAL_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, false)

KOKKOSBLAS1_CSCAL_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, true)
KOKKOSBLAS1_CSCAL_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, false)
#endif

#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS1_DSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, true)
KOKKOSBLAS1_DSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, false)

KOKKOSBLAS1_SSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, true)
KOKKOSBLAS1_SSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, false)

KOKKOSBLAS1_ZSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, true)
KOKKOSBLAS1_ZSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, false)

KOKKOSBLAS1_CSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, true)
KOKKOSBLAS1_CSCAL_TPL_SPEC_DECL_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, false)
#endif

}  // namespace Impl
}  // namespace KokkosBlas

#endif

// cuBLAS
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_CUBLAS(SCALAR_TYPE, CUDA_SCALAR_TYPE, CUBLAS_FN, LAYOUT, ETI_SPEC_AVAIL)       \
  template <>                                                                                                          \
  struct Scal<Kokkos::Cuda,                                                                                            \
              Kokkos::View<SCALAR_TYPE*, LAYOUT, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, SCALAR_TYPE, \
              Kokkos::View<const SCALAR_TYPE*, LAYOUT, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1,     \
              true, ETI_SPEC_AVAIL> {                                                                                  \
    typedef Kokkos::View<SCALAR_TYPE*, LAYOUT, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > RV;             \
    typedef SCALAR_TYPE AS;                                                                                            \
    typedef Kokkos::View<const SCALAR_TYPE*, LAYOUT, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XV;       \
    typedef typename XV::size_type size_type;                                                                          \
                                                                                                                       \
    static void scal(const Kokkos::Cuda& space, const RV& R, const AS& alpha, const XV& X) {                           \
      Kokkos::Profiling::pushRegion("KokkosBlas::scal[TPL_CUBLAS," #SCALAR_TYPE "]");                                  \
      const size_type numElems = X.extent(0);                                                                          \
      if ((numElems < static_cast<size_type>(INT_MAX)) && (R.data() == X.data())) {                                    \
        scal_print_specialization<RV, AS, XV>();                                                                       \
        const int N                            = static_cast<int>(numElems);                                           \
        constexpr int one                      = 1;                                                                    \
        KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                     \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                              \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(CUBLAS_FN(s.handle, N, reinterpret_cast<const CUDA_SCALAR_TYPE*>(&alpha),     \
                                                   reinterpret_cast<CUDA_SCALAR_TYPE*>(R.data()), one));               \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                             \
      } else {                                                                                                         \
        Scal<Kokkos::Cuda, RV, AS, XV, 1, false, ETI_SPEC_AVAIL>::scal(space, R, alpha, X);                            \
      }                                                                                                                \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

#define KOKKOSBLAS1_DSCAL_TPL_SPEC_DECL_CUBLAS(LAYOUT, ETI_SPEC_AVAIL) \
  KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_CUBLAS(double, double, cublasDscal, LAYOUT, ETI_SPEC_AVAIL)

#define KOKKOSBLAS1_SSCAL_TPL_SPEC_DECL_CUBLAS(LAYOUT, ETI_SPEC_AVAIL) \
  KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_CUBLAS(float, float, cublasSscal, LAYOUT, ETI_SPEC_AVAIL)

#define KOKKOSBLAS1_ZSCAL_TPL_SPEC_DECL_CUBLAS(LAYOUT, ETI_SPEC_AVAIL) \
  KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_CUBLAS(Kokkos::complex<double>, cuDoubleComplex, cublasZscal, LAYOUT, ETI_SPEC_AVAIL)

#define KOKKOSBLAS1_CSCAL_TPL_SPEC_DECL_CUBLAS(LAYOUT, ETI_SPEC_AVAIL) \
  KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_CUBLAS(Kokkos::complex<float>, cuComplex, cublasCscal, LAYOUT, ETI_SPEC_AVAIL)

KOKKOSBLAS1_DSCAL_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, true)
KOKKOSBLAS1_DSCAL_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, false)

KOKKOSBLAS1_SSCAL_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, true)
KOKKOSBLAS1_SSCAL_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, false)

KOKKOSBLAS1_ZSCAL_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, true)
KOKKOSBLAS1_ZSCAL_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, false)

KOKKOSBLAS1_CSCAL_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, true)
KOKKOSBLAS1_CSCAL_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif

// rocBLAS
#if defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_ROCBLAS(SCALAR_TYPE, ROCBLAS_SCALAR_TYPE, ROCBLAS_FN, LAYOUT, ETI_SPEC_AVAIL) \
  template <>                                                                                                         \
  struct Scal<Kokkos::HIP, Kokkos::View<SCALAR_TYPE*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
              SCALAR_TYPE,                                                                                            \
              Kokkos::View<const SCALAR_TYPE*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1,     \
              true, ETI_SPEC_AVAIL> {                                                                                 \
    typedef Kokkos::View<SCALAR_TYPE*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > RV;             \
    typedef SCALAR_TYPE AS;                                                                                           \
    typedef Kokkos::View<const SCALAR_TYPE*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XV;       \
    typedef typename XV::size_type size_type;                                                                         \
                                                                                                                      \
    static void scal(const Kokkos::HIP& space, const RV& R, const AS& alpha, const XV& X) {                           \
      Kokkos::Profiling::pushRegion("KokkosBlas::scal[TPL_ROCBLAS," #SCALAR_TYPE "]");                                \
      const size_type numElems = X.extent(0);                                                                         \
      if ((numElems < static_cast<size_type>(INT_MAX)) && (R.data() == X.data())) {                                   \
        scal_print_specialization<RV, AS, XV>();                                                                      \
        const int N                           = static_cast<int>(numElems);                                           \
        constexpr int one                     = 1;                                                                    \
        KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                      \
        KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, space.hip_stream()));                          \
        rocblas_pointer_mode pointer_mode;                                                                            \
        KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_get_pointer_mode(s.handle, &pointer_mode));                         \
        KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_pointer_mode(s.handle, rocblas_pointer_mode_host));             \
        KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(ROCBLAS_FN(s.handle, N,                                                     \
                                                     reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(&alpha),            \
                                                     reinterpret_cast<ROCBLAS_SCALAR_TYPE*>(R.data()), one));         \
        KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_pointer_mode(s.handle, pointer_mode));                          \
      } else {                                                                                                        \
        Scal<Kokkos::HIP, RV, AS, XV, 1, false, ETI_SPEC_AVAIL>::scal(space, R, alpha, X);                            \
      }                                                                                                               \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS1_DSCAL_TPL_SPEC_DECL_ROCBLAS(LAYOUT, ETI_SPEC_AVAIL) \
  KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_ROCBLAS(double, double, rocblas_dscal, LAYOUT, ETI_SPEC_AVAIL)

#define KOKKOSBLAS1_SSCAL_TPL_SPEC_DECL_ROCBLAS(LAYOUT, ETI_SPEC_AVAIL) \
  KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_ROCBLAS(float, float, rocblas_sscal, LAYOUT, ETI_SPEC_AVAIL)

#define KOKKOSBLAS1_ZSCAL_TPL_SPEC_DECL_ROCBLAS(LAYOUT, ETI_SPEC_AVAIL)                                           \
  KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_ROCBLAS(Kokkos::complex<double>, rocblas_double_complex, rocblas_zscal, LAYOUT, \
                                          ETI_SPEC_AVAIL)

#define KOKKOSBLAS1_CSCAL_TPL_SPEC_DECL_ROCBLAS(LAYOUT, ETI_SPEC_AVAIL)                                         \
  KOKKOSBLAS1_XSCAL_TPL_SPEC_DECL_ROCBLAS(Kokkos::complex<float>, rocblas_float_complex, rocblas_cscal, LAYOUT, \
                                          ETI_SPEC_AVAIL)

KOKKOSBLAS1_DSCAL_TPL_SPEC_DECL_ROCBLAS(Kokkos::LayoutLeft, true)
KOKKOSBLAS1_DSCAL_TPL_SPEC_DECL_ROCBLAS(Kokkos::LayoutLeft, false)

KOKKOSBLAS1_SSCAL_TPL_SPEC_DECL_ROCBLAS(Kokkos::LayoutLeft, true)
KOKKOSBLAS1_SSCAL_TPL_SPEC_DECL_ROCBLAS(Kokkos::LayoutLeft, false)

KOKKOSBLAS1_ZSCAL_TPL_SPEC_DECL_ROCBLAS(Kokkos::LayoutLeft, true)
KOKKOSBLAS1_ZSCAL_TPL_SPEC_DECL_ROCBLAS(Kokkos::LayoutLeft, false)

KOKKOSBLAS1_CSCAL_TPL_SPEC_DECL_ROCBLAS(Kokkos::LayoutLeft, true)
KOKKOSBLAS1_CSCAL_TPL_SPEC_DECL_ROCBLAS(Kokkos::LayoutLeft, false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif

#endif
