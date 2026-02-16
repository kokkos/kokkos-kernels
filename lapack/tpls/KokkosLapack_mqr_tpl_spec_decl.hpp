// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSLAPACK_MQR_TPL_SPEC_DECL_HPP_
#define KOKKOSLAPACK_MQR_TPL_SPEC_DECL_HPP_

#include <iostream>
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosLapack {
namespace Impl {
template <class AViewType, class TauViewType, class InfoViewType>
inline void mqr_print_specialization() {
#ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
  printf("KokkosLapack::mqr<> TPL MAGMA specialization for < %s , %s, %s >\n", typeid(AViewType).name(),
         typeid(TauViewType).name(), typeid(InfoViewType).name());
#else
#ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK
  printf("KokkosLapack::mqr<> TPL Lapack specialization for < %s , %s, %s >\n", typeid(AViewType).name(),
         typeid(TauViewType).name(), typeid(InfoViewType).name());
#endif
#endif
#endif
}
}  // namespace Impl
}  // namespace KokkosLapack

// Generic Host side LAPACK (could be MKL or whatever)
#if defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK) || defined(KOKKOSKERNELS_ENABLE_TPL_ACCELERATE)
#include <KokkosLapack_Host_tpl.hpp>

namespace KokkosLapack {
namespace Impl {

template <class AViewType, class TauViewType, class CViewType, class InfoViewType>
void lapackMqrWrapper(const char side[], const char trans[], const AViewType& A, const TauViewType& Tau,
                      const CViewType& C, const InfoViewType& Info) {
  using memory_space = typename AViewType::memory_space;
  using Scalar       = typename AViewType::non_const_value_type;
  using ALayout_t    = typename AViewType::array_layout;
  static_assert(std::is_same_v<ALayout_t, Kokkos::LayoutLeft>,
                "KokkosLapack - mqr: A needs to have a Kokkos::LayoutLeft");
  const int m   = C.extent_int(0);
  const int n   = C.extent_int(1);
  const int k   = Tau.extent_int(0);
  const int lda = A.stride(1);
  const int ldc = C.stride(1);

  int lwork = -1;
  // work needs to be at least length 1 to store the returned value for lwork
  Kokkos::View<Scalar*, memory_space> work("work array", 1);

  if constexpr (KokkosKernels::ArithTraits<Scalar>::is_complex) {
    using MagType = typename KokkosKernels::ArithTraits<Scalar>::mag_type;

    HostLapack<std::complex<MagType>>::mqr(
        side[0], trans[0], m, n, k, reinterpret_cast<std::complex<MagType>*>(A.data()), lda,
        reinterpret_cast<std::complex<MagType>*>(Tau.data()), reinterpret_cast<std::complex<MagType>*>(C.data()), ldc,
        reinterpret_cast<std::complex<MagType>*>(work.data()), lwork, Info.data());

    if (Info[0] < 0) return;

    lwork = static_cast<int>(work(0).real());

    work = Kokkos::View<Scalar*, memory_space>("mqr work buffer", lwork);

    HostLapack<std::complex<MagType>>::mqr(
        side[0], trans[0], m, n, k, reinterpret_cast<std::complex<MagType>*>(A.data()), lda,
        reinterpret_cast<std::complex<MagType>*>(Tau.data()), reinterpret_cast<std::complex<MagType>*>(C.data()), ldc,
        reinterpret_cast<std::complex<MagType>*>(work.data()), lwork, Info.data());
  } else {
    HostLapack<Scalar>::mqr(side[0], trans[0], m, n, k, A.data(), lda, Tau.data(), C.data(), ldc, work.data(), lwork,
                            Info.data());

    if (Info[0] < 0) return;

    lwork = static_cast<int>(work(0));

    work = Kokkos::View<Scalar*, memory_space>("mqr work buffer", lwork);

    HostLapack<Scalar>::mqr(side[0], trans[0], m, n, k, A.data(), lda, Tau.data(), C.data(), ldc, work.data(), lwork,
                            Info.data());
  }
}

#define KOKKOSLAPACK_MQR_LAPACK(SCALAR, LAYOUT, EXECSPACE, MEM_SPACE)                                                  \
  template <>                                                                                                          \
  struct MQR<                                                                                                          \
      EXECSPACE,                                                                                                       \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>,   \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>,    \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>,   \
      Kokkos::View<int*, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>, true, \
      mqr_eti_spec_avail<EXECSPACE,                                                                                    \
                         Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>,                          \
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                        \
                         Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>,                           \
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                        \
                         Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>,                          \
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                        \
                         Kokkos::View<int*, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>,                              \
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::value> {                              \
    using AViewType =                                                                                                  \
        Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>; \
    using TauViewType =                                                                                                \
        Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;  \
    using CViewType =                                                                                                  \
        Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>; \
    using InfoViewType =                                                                                               \
        Kokkos::View<int*, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;     \
                                                                                                                       \
    static void mqr(const EXECSPACE& /* space */, const char side[], const char trans[], const AViewType& A,           \
                    const TauViewType& Tau, const CViewType& C, const InfoViewType& Info) {                            \
      Kokkos::Profiling::pushRegion("KokkosLapack::mqr[TPL_LAPACK," #SCALAR "]");                                      \
      mqr_print_specialization<AViewType, TauViewType, InfoViewType>();                                                \
      lapackMqrWrapper(side, trans, A, Tau, C, Info);                                                                  \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

#if defined(KOKKOS_ENABLE_SERIAL)
KOKKOSLAPACK_MQR_LAPACK(float, Kokkos::LayoutLeft, Kokkos::Serial, Kokkos::HostSpace)
KOKKOSLAPACK_MQR_LAPACK(double, Kokkos::LayoutLeft, Kokkos::Serial, Kokkos::HostSpace)
KOKKOSLAPACK_MQR_LAPACK(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::Serial, Kokkos::HostSpace)
KOKKOSLAPACK_MQR_LAPACK(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::Serial, Kokkos::HostSpace)
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
KOKKOSLAPACK_MQR_LAPACK(float, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::HostSpace)
KOKKOSLAPACK_MQR_LAPACK(double, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::HostSpace)
KOKKOSLAPACK_MQR_LAPACK(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::HostSpace)
KOKKOSLAPACK_MQR_LAPACK(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::HostSpace)
#endif

#if defined(KOKKOS_ENABLE_THREADS)
KOKKOSLAPACK_MQR_LAPACK(float, Kokkos::LayoutLeft, Kokkos::Threads, Kokkos::HostSpace)
KOKKOSLAPACK_MQR_LAPACK(double, Kokkos::LayoutLeft, Kokkos::Threads, Kokkos::HostSpace)
KOKKOSLAPACK_MQR_LAPACK(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::Threads, Kokkos::HostSpace)
KOKKOSLAPACK_MQR_LAPACK(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::Threads, Kokkos::HostSpace)
#endif

}  // namespace Impl
}  // namespace KokkosLapack
#endif  // KOKKOSKERNELS_ENABLE_TPL_LAPACK

// CUSOLVER
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSOLVER
#include "KokkosLapack_cusolver.hpp"

namespace KokkosLapack {
namespace Impl {

template <class ExecutionSpace, class AViewType, class TauViewType, class CViewType, class InfoViewType>
void cusolverMqrWrapper(const ExecutionSpace& space, const char side[], const char trans[], const AViewType& A,
                        const TauViewType& Tau, const CViewType& C, const InfoViewType& Info) {
  using memory_space = typename AViewType::memory_space;
  using Scalar       = typename AViewType::non_const_value_type;

  using ALayout_t = typename AViewType::array_layout;
  static_assert(std::is_same_v<ALayout_t, Kokkos::LayoutLeft>,
                "KokkosLapack - cusolver {or,un}mqr: A needs to have a Kokkos::LayoutLeft");
  const int m   = C.extent_int(0);
  const int n   = C.extent_int(1);
  const int k   = Tau.extent_int(0);
  const int lda = A.stride(1);
  const int ldc = C.stride(1);
  int lwork     = 0;

  const cublasSideMode_t cu_side   = KokkosBlas::Impl::side_mode_kk_to_cublas(side);
  const cublasOperation_t cu_trans = KokkosBlas::Impl::trans_mode_kk_to_cublas(trans);

  CudaLapackSingleton& s = CudaLapackSingleton::singleton();
  KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(cusolverDnSetStream(s.handle, space.cuda_stream()));
  if constexpr (std::is_same_v<Scalar, float>) {
    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(cusolverDnSormqr_bufferSize(s.handle, cu_side, cu_trans, m, n, k, A.data(),
                                                                     lda, Tau.data(), C.data(), ldc, &lwork));
    Kokkos::View<float*, memory_space> Workspace("cusolver sormar workspace", lwork);

    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(cusolverDnSormqr(s.handle, cu_side, cu_trans, m, n, k, A.data(), lda,
                                                          Tau.data(), C.data(), ldc, Workspace.data(), lwork,
                                                          Info.data()));
  }
  if constexpr (std::is_same_v<Scalar, double>) {
    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(cusolverDnDormqr_bufferSize(s.handle, cu_side, cu_trans, m, n, k, A.data(),
                                                                     lda, Tau.data(), C.data(), ldc, &lwork));
    Kokkos::View<double*, memory_space> Workspace("cusolver dormqr workspace", lwork);

    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(cusolverDnDormqr(s.handle, cu_side, cu_trans, m, n, k, A.data(), lda,
                                                          Tau.data(), C.data(), ldc, Workspace.data(), lwork,
                                                          Info.data()));
  }
  if constexpr (std::is_same_v<Scalar, Kokkos::complex<float>>) {
    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(cusolverDnCunmqr_bufferSize(
        s.handle, cu_side, cu_trans, m, n, k, reinterpret_cast<cuComplex*>(A.data()), lda,
        reinterpret_cast<cuComplex*>(Tau.data()), reinterpret_cast<cuComplex*>(C.data()), ldc, &lwork));
    Kokkos::View<cuComplex*, memory_space> Workspace("cusolver cunmqr workspace", lwork);

    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(
        cusolverDnCunmqr(s.handle, cu_side, cu_trans, m, n, k, reinterpret_cast<cuComplex*>(A.data()), lda,
                         reinterpret_cast<cuComplex*>(Tau.data()), reinterpret_cast<cuComplex*>(C.data()), ldc,
                         reinterpret_cast<cuComplex*>(Workspace.data()), lwork, Info.data()));
  }
  if constexpr (std::is_same_v<Scalar, Kokkos::complex<double>>) {
    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(cusolverDnZunmqr_bufferSize(
        s.handle, cu_side, cu_trans, m, n, k, reinterpret_cast<cuDoubleComplex*>(A.data()), lda,
        reinterpret_cast<cuDoubleComplex*>(Tau.data()), reinterpret_cast<cuDoubleComplex*>(C.data()), ldc, &lwork));
    Kokkos::View<cuDoubleComplex*, memory_space> Workspace("cusolver zunmqr workspace", lwork);

    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(
        cusolverDnZunmqr(s.handle, cu_side, cu_trans, m, n, k, reinterpret_cast<cuDoubleComplex*>(A.data()), lda,
                         reinterpret_cast<cuDoubleComplex*>(Tau.data()), reinterpret_cast<cuDoubleComplex*>(C.data()),
                         ldc, reinterpret_cast<cuDoubleComplex*>(Workspace.data()), lwork, Info.data()));
  }
  KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(cusolverDnSetStream(s.handle, NULL));
}

#define KOKKOSLAPACK_MQR_CUSOLVER(SCALAR, LAYOUT, MEM_SPACE)                                                           \
  template <>                                                                                                          \
  struct MQR<                                                                                                          \
      Kokkos::Cuda,                                                                                                    \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                                          \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                                           \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                                          \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                                           \
      Kokkos::View<int*, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>,    \
      true,                                                                                                            \
      mqr_eti_spec_avail<Kokkos::Cuda,                                                                                 \
                         Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                       \
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                        \
                         Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                        \
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                        \
                         Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                       \
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                        \
                         Kokkos::View<int*, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                           \
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::value> {                              \
    using AViewType   = Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                        \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                                         \
    using TauViewType = Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                         \
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                                         \
    using CViewType   = Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                        \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                                         \
    using InfoViewType =                                                                                               \
        Kokkos::View<int*, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;  \
                                                                                                                       \
    static void mqr(const Kokkos::Cuda& space, const char side[], const char trans[], const AViewType& A,              \
                    const TauViewType& Tau, const CViewType& C, const InfoViewType& Info) {                            \
      Kokkos::Profiling::pushRegion("KokkosLapack::mqr[TPL_CUSOLVER," #SCALAR "]");                                    \
      mqr_print_specialization<AViewType, TauViewType, InfoViewType>();                                                \
                                                                                                                       \
      cusolverMqrWrapper(space, side, trans, A, Tau, C, Info);                                                         \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

KOKKOSLAPACK_MQR_CUSOLVER(float, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSLAPACK_MQR_CUSOLVER(double, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSLAPACK_MQR_CUSOLVER(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSLAPACK_MQR_CUSOLVER(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::CudaSpace)

#if defined(KOKKOSKERNELS_INST_MEMSPACE_CUDAUVMSPACE)
KOKKOSLAPACK_MQR_CUSOLVER(float, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSLAPACK_MQR_CUSOLVER(double, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSLAPACK_MQR_CUSOLVER(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSLAPACK_MQR_CUSOLVER(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
#endif

}  // namespace Impl
}  // namespace KokkosLapack
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUSOLVER

// ROCSOLVER
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER
#include <KokkosBlas_tpl_spec.hpp>
#include <rocsolver/rocsolver.h>

namespace KokkosLapack {
namespace Impl {

template <class ExecutionSpace, class AViewType, class TauViewType, class CViewType, class InfoViewType>
void rocsolverMqrWrapper(const ExecutionSpace& space, const char side[], const char trans[], const AViewType& A,
                         const TauViewType& Tau, const CViewType& C, const InfoViewType& Info) {
  using Scalar = typename AViewType::non_const_value_type;

  using ALayout_t = typename AViewType::array_layout;
  static_assert(std::is_same_v<ALayout_t, Kokkos::LayoutLeft>,
                "KokkosLapack - rocsolver mqr: A needs to have a Kokkos::LayoutLeft");
  const rocblas_int m   = static_cast<rocblas_int>(C.extent(0));
  const rocblas_int n   = static_cast<rocblas_int>(C.extent(1));
  const rocblas_int k   = static_cast<rocblas_int>(Tau.extent(0));
  const rocblas_int lda = static_cast<rocblas_int>(A.stride(1));
  const rocblas_int ldc = static_cast<rocblas_int>(C.stride(1));

  rocblas_side roc_side       = KokkosBlas::Impl::side_mode_kk_to_rocblas(side);
  rocblas_operation roc_trans = KokkosBlas::Impl::trans_mode_kk_to_rocblas(trans);

  KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();
  KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, space.hip_stream()));
  if constexpr (std::is_same_v<Scalar, float>) {
    KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(
        rocsolver_sormqr(s.handle, roc_side, roc_trans, m, n, k, A.data(), lda, Tau.data(), C.data(), ldc));
  }
  if constexpr (std::is_same_v<Scalar, double>) {
    KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(
        rocsolver_dormqr(s.handle, roc_side, roc_trans, m, n, k, A.data(), lda, Tau.data(), C.data(), ldc));
  }
  if constexpr (std::is_same_v<Scalar, Kokkos::complex<float>>) {
    KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocsolver_cunmqr(
        s.handle, roc_side, roc_trans, m, n, k, reinterpret_cast<rocblas_float_complex*>(A.data()), lda,
        reinterpret_cast<rocblas_float_complex*>(Tau.data()), reinterpret_cast<rocblas_float_complex*>(C.data()), ldc));
  }
  if constexpr (std::is_same_v<Scalar, Kokkos::complex<double>>) {
    KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocsolver_zunmqr(s.handle, roc_side, roc_trans, m, n, k,
                                                       reinterpret_cast<rocblas_double_complex*>(A.data()), lda,
                                                       reinterpret_cast<rocblas_double_complex*>(Tau.data()),
                                                       reinterpret_cast<rocblas_double_complex*>(C.data()), ldc));
  }
  Kokkos::deep_copy(Info, 0);  // Success
  KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, NULL));
}

#define KOKKOSLAPACK_MQR_ROCSOLVER(SCALAR, LAYOUT, MEM_SPACE)                                                          \
  template <>                                                                                                          \
  struct MQR<                                                                                                          \
      Kokkos::HIP,                                                                                                     \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>,  \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
      Kokkos::View<int*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>,     \
      true,                                                                                                            \
      mqr_eti_spec_avail<Kokkos::HIP,                                                                                  \
                         Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                        \
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                        \
                         Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                         \
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                        \
                         Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                        \
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                        \
                         Kokkos::View<int*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                            \
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::value> {                              \
    using AViewType   = Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                         \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                                         \
    using TauViewType = Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                          \
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                                         \
    using CViewType   = Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                         \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                                         \
    using InfoViewType =                                                                                               \
        Kokkos::View<int*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;   \
                                                                                                                       \
    static void mqr(const Kokkos::HIP& space, const char side[], const char trans[], const AViewType& A,               \
                    const TauViewType& Tau, const CViewType& C, const InfoViewType& Info) {                            \
      Kokkos::Profiling::pushRegion("KokkosLapack::mqr[TPL_ROCSOLVER," #SCALAR "]");                                   \
      mqr_print_specialization<AViewType, TauViewType, InfoViewType>();                                                \
                                                                                                                       \
      rocsolverMqrWrapper(space, side, trans, A, Tau, C, Info);                                                        \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

KOKKOSLAPACK_MQR_ROCSOLVER(float, Kokkos::LayoutLeft, Kokkos::HIPSpace)
KOKKOSLAPACK_MQR_ROCSOLVER(double, Kokkos::LayoutLeft, Kokkos::HIPSpace)
KOKKOSLAPACK_MQR_ROCSOLVER(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::HIPSpace)
KOKKOSLAPACK_MQR_ROCSOLVER(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::HIPSpace)

}  // namespace Impl
}  // namespace KokkosLapack
#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER

#endif
