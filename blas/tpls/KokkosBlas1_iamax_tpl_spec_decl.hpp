// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS1_IAMAX_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS1_IAMAX_TPL_SPEC_DECL_HPP_

namespace KokkosBlas {
namespace Impl {
template <class RV, class XV>
inline void iamax_print_specialization() {
#if defined(KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION)
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)
  printf("KokkosBlas1::iamax<> TPL cuBLAS specialization for < %s , %s >\n", typeid(RV).name(), typeid(XV).name());
#elif defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)
  printf("KokkosBlas1::iamax<> TPL rocBLAS specialization for < %s , %s >\n", typeid(RV).name(), typeid(XV).name());
#else
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
  printf("KokkosBlas1::iamax<> TPL Blas specialization for < %s , %s >\n", typeid(RV).name(), typeid(XV).name());
#endif
#endif
#endif
}
}  // namespace Impl
}  // namespace KokkosBlas

// Generic Host side BLAS (could be MKL or whatever)
#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS)
#include "KokkosBlas_Host_tpl.hpp"

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_BLAS(SCALAR_TYPE, BASE_SCALAR_TYPE)                                            \
  template <typename ExecSpace, bool ETI_SPEC_AVAIL>                                                                    \
    requires(std::is_same_v<typename ExecSpace::memory_space, Kokkos::HostSpace>)                                       \
  struct Iamax<                                                                                                         \
      ExecSpace,                                                                                                        \
      Kokkos::View<unsigned long, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>,      \
      Kokkos::View<const SCALAR_TYPE*, Kokkos::LayoutLeft, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>, 1,      \
      true, ETI_SPEC_AVAIL> {                                                                                           \
    typedef Kokkos::View<unsigned long, Kokkos::LayoutLeft, Kokkos::HostSpace,                                          \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>                                                       \
        RV;                                                                                                             \
    typedef Kokkos::View<const SCALAR_TYPE*, Kokkos::LayoutLeft, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>    \
        XV;                                                                                                             \
    typedef typename XV::size_type size_type;                                                                           \
                                                                                                                        \
    static void iamax(const ExecSpace& space, RV& R, const XV& X) {                                                     \
      Kokkos::Profiling::pushRegion("KokkosBlas::iamax[TPL_BLAS," #SCALAR_TYPE "]");                                    \
      const size_type numElems = X.extent(0);                                                                           \
      if (numElems == 0) {                                                                                              \
        R() = 0;                                                                                                        \
        return;                                                                                                         \
      }                                                                                                                 \
      if (numElems < static_cast<size_type>(INT_MAX)) {                                                                 \
        iamax_print_specialization<RV, XV>();                                                                           \
        int N         = static_cast<int>(numElems);                                                                     \
        const int XST = X.stride(0);                                                                                    \
        const int LDX = (XST == 0) ? 1 : XST;                                                                           \
        int idx       = HostBlas<BASE_SCALAR_TYPE>::iamax(N, reinterpret_cast<const BASE_SCALAR_TYPE*>(X.data()), LDX); \
        R()           = static_cast<size_type>(idx);                                                                    \
      } else {                                                                                                          \
        Iamax<ExecSpace, RV, XV, 1, false, ETI_SPEC_AVAIL>::iamax(space, R, X);                                         \
      }                                                                                                                 \
      Kokkos::Profiling::popRegion();                                                                                   \
    }                                                                                                                   \
  };

KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_BLAS(double, double)
KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_BLAS(float, float)
KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_BLAS(Kokkos::complex<double>, std::complex<double>)
KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_BLAS(Kokkos::complex<float>, std::complex<float>)

}  // namespace Impl
}  // namespace KokkosBlas

#endif

// cuBLAS
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

template <class ExecutionSpace, class RViewType, class XViewType, bool ETI_SPEC_AVAIL>
void cublasIamaxWrapper(const ExecutionSpace& space, RViewType& R, const XViewType& X) {
  using scalar_type = typename XViewType::non_const_value_type;
  using size_type   = typename XViewType::size_type;
  using index_type  = typename RViewType::non_const_value_type;

  iamax_print_specialization<RViewType, XViewType>();
  const int N   = static_cast<int>(X.extent(0));
  const int XST = X.stride(0);
  const int LDX = (XST == 0) ? 1 : XST;

  KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();
  KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));

  cublasPointerMode_t prevPtrMode;
  KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasGetPointerMode(s.handle, &prevPtrMode));
  if constexpr (std::is_same_v<typename RViewType::execution_space, Kokkos::Cuda>) {
    if (prevPtrMode == CUBLAS_POINTER_MODE_HOST) {
      // With a return view using Kokkos::Cuda, we need the device point mode
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetPointerMode(s.handle, CUBLAS_POINTER_MODE_DEVICE));
    }
  } else if constexpr (std::is_same_v<typename RViewType::memory_space, Kokkos::HostSpace>) {
    if (prevPtrMode == CUBLAS_POINTER_MODE_DEVICE) {
      // With a return view using Kokkos::HostSpace, we need the host point mode
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetPointerMode(s.handle, CUBLAS_POINTER_MODE_HOST));
    }
  }

  if constexpr (std::is_same_v<index_type, int>) {
    if (X.extent(0) < static_cast<size_type>(INT_MAX)) {
      Kokkos::Profiling::pushRegion("KokkosBlas::iamax[TPL_CUBLAS," + KokkosKernels::ArithTraits<scalar_type>::name() +
                                    "]");
      if constexpr (std::is_same_v<scalar_type, float>) {
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasIsamax(s.handle, N, X.data(), LDX, R.data()));
      } else if constexpr (std::is_same_v<scalar_type, double>) {
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasIdamax(s.handle, N, X.data(), LDX, R.data()));
      } else if constexpr (std::is_same_v<scalar_type, Kokkos::complex<float>>) {
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(
            cublasIcamax(s.handle, N, reinterpret_cast<const cuComplex*>(X.data()), LDX, R.data()));
      } else if constexpr (std::is_same_v<scalar_type, Kokkos::complex<double>>) {
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(
            cublasIzamax(s.handle, N, reinterpret_cast<const cuDoubleComplex*>(X.data()), LDX, R.data()));
      }
      Kokkos::Profiling::popRegion();
    } else {
      Iamax<Kokkos::Cuda, RViewType, XViewType, 1, false, ETI_SPEC_AVAIL>::iamax(space, R, X);
    }
  } else if constexpr (std::is_same_v<index_type, std::int64_t>) {
    Kokkos::Profiling::pushRegion("KokkosBlas::iamax[TPL_CUBLAS_64," + KokkosKernels::ArithTraits<scalar_type>::name() +
                                  "]");
    if constexpr (std::is_same_v<scalar_type, float>) {
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasIsamax_64(s.handle, N, X.data(), LDX, R.data()));
    } else if constexpr (std::is_same_v<scalar_type, double>) {
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasIdamax_64(s.handle, N, X.data(), LDX, R.data()));
    } else if constexpr (std::is_same_v<scalar_type, Kokkos::complex<float>>) {
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(
          cublasIcamax_64(s.handle, N, reinterpret_cast<const cuComplex*>(X.data()), LDX, R.data()));
    } else if constexpr (std::is_same_v<scalar_type, Kokkos::complex<double>>) {
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(
          cublasIzamax_64(s.handle, N, reinterpret_cast<const cuDoubleComplex*>(X.data()), LDX, R.data()));
    }
    Kokkos::Profiling::popRegion();
  }

  if constexpr (std::is_same_v<typename RViewType::execution_space, Kokkos::Cuda>) {
    if (prevPtrMode == CUBLAS_POINTER_MODE_HOST) {
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetPointerMode(s.handle, CUBLAS_POINTER_MODE_HOST));
    }
  } else if constexpr (std::is_same_v<typename RViewType::memory_space, Kokkos::HostSpace>) {
    if (prevPtrMode == CUBLAS_POINTER_MODE_DEVICE) {
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetPointerMode(s.handle, CUBLAS_POINTER_MODE_DEVICE));
    }
  }
}

#define KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_CUBLAS_WRAPPER(SCALAR_TYPE, INDEX_TYPE, RET_DEVICE_TYPE)                      \
  template <bool ETI_SPEC_AVAIL>                                                                                       \
  struct Iamax<                                                                                                        \
      Kokkos::Cuda,                                                                                                    \
      Kokkos::View<INDEX_TYPE, Kokkos::LayoutLeft, RET_DEVICE_TYPE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>,          \
      Kokkos::View<const SCALAR_TYPE*, Kokkos::LayoutLeft, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged>>, 1,  \
      true, ETI_SPEC_AVAIL> {                                                                                          \
    using RV = Kokkos::View<INDEX_TYPE, Kokkos::LayoutLeft, RET_DEVICE_TYPE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>; \
    using XV =                                                                                                         \
        Kokkos::View<const SCALAR_TYPE*, Kokkos::LayoutLeft, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;   \
                                                                                                                       \
    static void iamax(const Kokkos::Cuda& space, RV& R, const XV& X) {                                                 \
      if (X.extent(0) == 0) {                                                                                          \
        Kokkos::deep_copy(R, 0);                                                                                       \
        return;                                                                                                        \
      }                                                                                                                \
      cublasIamaxWrapper<Kokkos::Cuda, RV, XV, ETI_SPEC_AVAIL>(space, R, X);                                           \
    }                                                                                                                  \
  };

#define KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_CUBLAS(SCALAR_TYPE, INDEX_TYPE)                      \
  KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_CUBLAS_WRAPPER(SCALAR_TYPE, INDEX_TYPE, Kokkos::HostSpace) \
  KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_CUBLAS_WRAPPER(SCALAR_TYPE, INDEX_TYPE, Kokkos::Cuda)

KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_CUBLAS(double, std::int64_t)
KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_CUBLAS(float, std::int64_t)
KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_CUBLAS(Kokkos::complex<double>, std::int64_t)
KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_CUBLAS(Kokkos::complex<float>, std::int64_t)

KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_CUBLAS(double, int)
KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_CUBLAS(float, int)
KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_CUBLAS(Kokkos::complex<double>, int)
KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_CUBLAS(Kokkos::complex<float>, int)

}  // namespace Impl
}  // namespace KokkosBlas

#endif

// rocBLAS
#if defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

using ROCBLAS_DEVICE_TYPE = Kokkos::HIP;

#define KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_ROCBLAS_WRAPPER(SCALAR_TYPE, ROCBLAS_SCALAR_TYPE, ROCBLAS_FN, INDEX_TYPE,     \
                                                         RET_DEVICE_TYPE, ROCBLAS_PTR_MODE_1, ROCBLAS_PTR_MODE_2)      \
  template <bool ETI_SPEC_AVAIL>                                                                                       \
  struct Iamax<                                                                                                        \
      Kokkos::HIP,                                                                                                     \
      Kokkos::View<INDEX_TYPE, Kokkos::LayoutLeft, RET_DEVICE_TYPE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>,          \
      Kokkos::View<const SCALAR_TYPE*, Kokkos::LayoutLeft, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged>>, 1,   \
      true, ETI_SPEC_AVAIL> {                                                                                          \
    using execution_space = Kokkos::HIP;                                                                               \
    typedef Kokkos::View<INDEX_TYPE, Kokkos::LayoutLeft, RET_DEVICE_TYPE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> RV; \
    typedef Kokkos::View<const SCALAR_TYPE*, Kokkos::LayoutLeft, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged>> \
        XV;                                                                                                            \
    typedef typename XV::size_type size_type;                                                                          \
                                                                                                                       \
    static void iamax(const execution_space& space, RV& R, const XV& X) {                                              \
      Kokkos::Profiling::pushRegion("KokkosBlas::iamax[TPL_ROCBLAS," #SCALAR_TYPE "]");                                \
      const size_type numElems = X.extent(0);                                                                          \
      if (numElems == 0) {                                                                                             \
        Kokkos::deep_copy(R, 0);                                                                                       \
        return;                                                                                                        \
      }                                                                                                                \
      if (numElems < static_cast<size_type>(INT_MAX)) {                                                                \
        iamax_print_specialization<RV, XV>();                                                                          \
        const int N                           = static_cast<int>(numElems);                                            \
        const int XST                         = X.stride(0);                                                           \
        const int LDX                         = (XST == 0) ? 1 : XST;                                                  \
        KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                       \
        KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, space.hip_stream()));                           \
        rocblas_pointer_mode prevPtrMode;                                                                              \
        KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_get_pointer_mode(s.handle, &prevPtrMode));                           \
        if (prevPtrMode == ROCBLAS_PTR_MODE_2) {                                                                       \
          KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_pointer_mode(s.handle, ROCBLAS_PTR_MODE_1));                   \
        }                                                                                                              \
        KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(ROCBLAS_FN(s.handle, N,                                                      \
                                                     reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(X.data()), LDX,      \
                                                     reinterpret_cast<int*>(R.data())));                               \
        KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, NULL));                                         \
        if (prevPtrMode == ROCBLAS_PTR_MODE_2) {                                                                       \
          KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_pointer_mode(s.handle, ROCBLAS_PTR_MODE_2));                   \
        }                                                                                                              \
      } else {                                                                                                         \
        Iamax<execution_space, RV, XV, 1, false, ETI_SPEC_AVAIL>::iamax(space, R, X);                                  \
      }                                                                                                                \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

#define KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_ROCBLAS(SCALAR_TYPE, ROCBLAS_SCALAR_TYPE, ROCBLAS_FN, INDEX_TYPE)   \
  KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_ROCBLAS_WRAPPER(SCALAR_TYPE, ROCBLAS_SCALAR_TYPE, ROCBLAS_FN, INDEX_TYPE, \
                                                   Kokkos::HostSpace, rocblas_pointer_mode_host,             \
                                                   rocblas_pointer_mode_device)                              \
  KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_ROCBLAS_WRAPPER(SCALAR_TYPE, ROCBLAS_SCALAR_TYPE, ROCBLAS_FN, INDEX_TYPE, \
                                                   ROCBLAS_DEVICE_TYPE, rocblas_pointer_mode_device,         \
                                                   rocblas_pointer_mode_host)

#define KOKKOSBLAS1_DIAMAX_TPL_SPEC_DECL_ROCBLAS(INDEX_TYPE) \
  KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_ROCBLAS(double, double, rocblas_idamax, INDEX_TYPE)

#define KOKKOSBLAS1_SIAMAX_TPL_SPEC_DECL_ROCBLAS(INDEX_TYPE) \
  KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_ROCBLAS(float, float, rocblas_isamax, INDEX_TYPE)

#define KOKKOSBLAS1_ZIAMAX_TPL_SPEC_DECL_ROCBLAS(INDEX_TYPE) \
  KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_ROCBLAS(Kokkos::complex<double>, rocblas_double_complex, rocblas_izamax, INDEX_TYPE)

#define KOKKOSBLAS1_CIAMAX_TPL_SPEC_DECL_ROCBLAS(INDEX_TYPE) \
  KOKKOSBLAS1_XIAMAX_TPL_SPEC_DECL_ROCBLAS(Kokkos::complex<float>, rocblas_float_complex, rocblas_icamax, INDEX_TYPE)

KOKKOSBLAS1_DIAMAX_TPL_SPEC_DECL_ROCBLAS(unsigned long)

KOKKOSBLAS1_SIAMAX_TPL_SPEC_DECL_ROCBLAS(unsigned long)

KOKKOSBLAS1_ZIAMAX_TPL_SPEC_DECL_ROCBLAS(unsigned long)

KOKKOSBLAS1_CIAMAX_TPL_SPEC_DECL_ROCBLAS(unsigned long)

KOKKOSBLAS1_DIAMAX_TPL_SPEC_DECL_ROCBLAS(unsigned int)

KOKKOSBLAS1_SIAMAX_TPL_SPEC_DECL_ROCBLAS(unsigned int)

KOKKOSBLAS1_ZIAMAX_TPL_SPEC_DECL_ROCBLAS(unsigned int)

KOKKOSBLAS1_CIAMAX_TPL_SPEC_DECL_ROCBLAS(unsigned int)

}  // namespace Impl
}  // namespace KokkosBlas

#endif

#endif
