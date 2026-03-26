// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSLAPACK_POTRF_TPL_SPEC_DECL_HPP_
#define KOKKOSLAPACK_POTRF_TPL_SPEC_DECL_HPP_

#include <KokkosKernels_ArithTraits.hpp>

namespace KokkosLapack {
namespace Impl {
template <class AViewType>
inline void potrf_print_specialization() {
#ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSOLVER
  printf("KokkosLapack::potrf<> TPL cuSolver specialization for < %s >\n", typeid(AViewType).name());
#elif defined(KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER)
  printf("KokkosLapack::potrf<> TPL rocSolver specialization for < %s >\n", typeid(AViewType).name());
#elif defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK)
  printf("KokkosLapack::potrf<> TPL Lapack specialization for < %s >\n", typeid(AViewType).name());
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

template <class AViewType>
void lapackPotrfWrapper(const char uplo[], const int& n, AViewType& A, const int& lda) {
  using Scalar   = typename AViewType::non_const_value_type;
  using Layout_t = typename AViewType::array_layout;
  static_assert(std::is_same_v<Layout_t, Kokkos::LayoutLeft>,
                "KokkosLapack - potrf: A needs to have a Kokkos::LayoutLeft");

  int info = 0;
  if constexpr (KokkosKernels::ArithTraits<Scalar>::is_complex) {
    using MagType = typename KokkosKernels::ArithTraits<Scalar>::mag_type;
    HostLapack<std::complex<MagType>>::potrf(uplo[0], n, reinterpret_cast<std::complex<MagType>*>(A.data()), lda);
  } else {
    HostLapack<Scalar>::potrf(uplo[0], n, A.data(), lda);
  }

  if (info != 0) {
    Kokkos::abort("KokkosLapack::potrf: LAPACK potrf failed with nonzero info");
  }
}

#define KOKKOSLAPACK_POTRF_LAPACK(SCALAR, LAYOUT, EXECSPACE, MEM_SPACE)                                                \
  template <>                                                                                                          \
  struct Potrf<                                                                                                        \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>,   \
      true,                                                                                                            \
      potrf_eti_spec_avail<Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>,                        \
                                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::value> {                            \
    using AViewType =                                                                                                  \
        Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>; \
    static void potrf(const char uplo[], const int& n, AViewType& A, const int& lda) {                                 \
      Kokkos::Profiling::pushRegion("KokkosLapack::potrf[TPL_LAPACK," #SCALAR "]");                                    \
      potrf_print_specialization<AViewType>();                                                                         \
      lapackPotrfWrapper(uplo, n, A, lda);                                                                             \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

#if defined(KOKKOS_ENABLE_SERIAL)
KOKKOSLAPACK_POTRF_LAPACK(float, Kokkos::LayoutLeft, Kokkos::Serial, Kokkos::HostSpace)
KOKKOSLAPACK_POTRF_LAPACK(double, Kokkos::LayoutLeft, Kokkos::Serial, Kokkos::HostSpace)
KOKKOSLAPACK_POTRF_LAPACK(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::Serial, Kokkos::HostSpace)
KOKKOSLAPACK_POTRF_LAPACK(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::Serial, Kokkos::HostSpace)
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
KOKKOSLAPACK_POTRF_LAPACK(float, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::HostSpace)
KOKKOSLAPACK_POTRF_LAPACK(double, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::HostSpace)
KOKKOSLAPACK_POTRF_LAPACK(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::HostSpace)
KOKKOSLAPACK_POTRF_LAPACK(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::HostSpace)
#endif

#if defined(KOKKOS_ENABLE_THREADS)
KOKKOSLAPACK_POTRF_LAPACK(float, Kokkos::LayoutLeft, Kokkos::Threads, Kokkos::HostSpace)
KOKKOSLAPACK_POTRF_LAPACK(double, Kokkos::LayoutLeft, Kokkos::Threads, Kokkos::HostSpace)
KOKKOSLAPACK_POTRF_LAPACK(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::Threads, Kokkos::HostSpace)
KOKKOSLAPACK_POTRF_LAPACK(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::Threads, Kokkos::HostSpace)
#endif

}  // namespace Impl
}  // namespace KokkosLapack
#endif  // KOKKOSKERNELS_ENABLE_TPL_LAPACK || KOKKOSKERNELS_ENABLE_TPL_ACCELERATE

// CUSOLVER
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSOLVER
#include "KokkosLapack_cusolver.hpp"

namespace KokkosLapack {
namespace Impl {

template <class AViewType>
void cusolverPotrfWrapper(const char uplo[], const int& n, AViewType& A, const int& lda) {
  using memory_space = typename AViewType::memory_space;
  using Scalar       = typename AViewType::non_const_value_type;

  cublasFillMode_t uplo_t = (uplo[0] == 'U' || uplo[0] == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  int lwork = 0;
  Kokkos::View<int, memory_space> info("potrf info");
  CudaLapackSingleton& s = CudaLapackSingleton::singleton();

  if constexpr (std::is_same_v<Scalar, float>) {
    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(cusolverDnSpotrf_bufferSize(s.handle, uplo_t, n, A.data(), lda, &lwork));
    Kokkos::View<float*, memory_space> Workspace("potrf workspace", lwork);
    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(
        cusolverDnSpotrf(s.handle, uplo_t, n, A.data(), lda, Workspace.data(), lwork, info.data()));
  }
  if constexpr (std::is_same_v<Scalar, double>) {
    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(cusolverDnDpotrf_bufferSize(s.handle, uplo_t, n, A.data(), lda, &lwork));
    Kokkos::View<double*, memory_space> Workspace("potrf workspace", lwork);
    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(
        cusolverDnDpotrf(s.handle, uplo_t, n, A.data(), lda, Workspace.data(), lwork, info.data()));
  }
  if constexpr (std::is_same_v<Scalar, Kokkos::complex<float>>) {
    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(
        cusolverDnCpotrf_bufferSize(s.handle, uplo_t, n, reinterpret_cast<cuComplex*>(A.data()), lda, &lwork));
    Kokkos::View<cuComplex*, memory_space> Workspace("potrf workspace", lwork);
    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(cusolverDnCpotrf(s.handle, uplo_t, n, reinterpret_cast<cuComplex*>(A.data()),
                                                          lda, reinterpret_cast<cuComplex*>(Workspace.data()), lwork,
                                                          info.data()));
  }
  if constexpr (std::is_same_v<Scalar, Kokkos::complex<double>>) {
    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(
        cusolverDnZpotrf_bufferSize(s.handle, uplo_t, n, reinterpret_cast<cuDoubleComplex*>(A.data()), lda, &lwork));
    Kokkos::View<cuDoubleComplex*, memory_space> Workspace("potrf workspace", lwork);
    KOKKOSLAPACK_IMPL_CUSOLVER_SAFE_CALL(
        cusolverDnZpotrf(s.handle, uplo_t, n, reinterpret_cast<cuDoubleComplex*>(A.data()), lda,
                         reinterpret_cast<cuDoubleComplex*>(Workspace.data()), lwork, info.data()));
  }
}

#define KOKKOSLAPACK_POTRF_CUSOLVER(SCALAR, LAYOUT, MEM_SPACE)                                              \
  template <>                                                                                               \
  struct Potrf<Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                      \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                       \
               true,                                                                                        \
               potrf_eti_spec_avail<Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>, \
                                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::value> {        \
    using AViewType = Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,               \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                                \
    static void potrf(const char uplo[], const int& n, AViewType& A, const int& lda) {                      \
      Kokkos::Profiling::pushRegion("KokkosLapack::potrf[TPL_CUSOLVER," #SCALAR "]");                       \
      potrf_print_specialization<AViewType>();                                                              \
      cusolverPotrfWrapper(uplo, n, A, lda);                                                                \
      Kokkos::Profiling::popRegion();                                                                       \
    }                                                                                                       \
  };

KOKKOSLAPACK_POTRF_CUSOLVER(float, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSLAPACK_POTRF_CUSOLVER(double, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSLAPACK_POTRF_CUSOLVER(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSLAPACK_POTRF_CUSOLVER(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::CudaSpace)

#if defined(KOKKOSKERNELS_INST_MEMSPACE_CUDAUVMSPACE)
KOKKOSLAPACK_POTRF_CUSOLVER(float, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSLAPACK_POTRF_CUSOLVER(double, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSLAPACK_POTRF_CUSOLVER(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSLAPACK_POTRF_CUSOLVER(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
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

template <class AViewType>
void rocsolverPotrfWrapper(const char uplo[], const int& n, AViewType& A, const int& lda) {
  using Scalar = typename AViewType::non_const_value_type;

  rocblas_fill uplo_t = (uplo[0] == 'U' || uplo[0] == 'u') ? rocblas_fill_upper : rocblas_fill_lower;

  const rocblas_int n_   = static_cast<rocblas_int>(n);
  const rocblas_int lda_ = static_cast<rocblas_int>(lda);
  Kokkos::View<rocblas_int, typename AViewType::memory_space> info("potrf info");

  KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();
  if constexpr (std::is_same_v<Scalar, float>) {
    KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocsolver_spotrf(s.handle, uplo_t, n_, A.data(), lda_, info.data()));
  }
  if constexpr (std::is_same_v<Scalar, double>) {
    KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocsolver_dpotrf(s.handle, uplo_t, n_, A.data(), lda_, info.data()));
  }
  if constexpr (std::is_same_v<Scalar, Kokkos::complex<float>>) {
    KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(
        rocsolver_cpotrf(s.handle, uplo_t, n_, reinterpret_cast<rocblas_float_complex*>(A.data()), lda_, info.data()));
  }
  if constexpr (std::is_same_v<Scalar, Kokkos::complex<double>>) {
    KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(
        rocsolver_zpotrf(s.handle, uplo_t, n_, reinterpret_cast<rocblas_double_complex*>(A.data()), lda_, info.data()));
  }
}

#define KOKKOSLAPACK_POTRF_ROCSOLVER(SCALAR, LAYOUT, MEM_SPACE)                                                        \
  template <>                                                                                                          \
  struct Potrf<                                                                                                        \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
      true,                                                                                                            \
      potrf_eti_spec_avail<Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                      \
                                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::value> {                            \
    using AViewType = Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                           \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                                           \
    static void potrf(const char uplo[], const int& n, AViewType& A, const int& lda) {                                 \
      Kokkos::Profiling::pushRegion("KokkosLapack::potrf[TPL_ROCSOLVER," #SCALAR "]");                                 \
      potrf_print_specialization<AViewType>();                                                                         \
      rocsolverPotrfWrapper(uplo, n, A, lda);                                                                          \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

KOKKOSLAPACK_POTRF_ROCSOLVER(float, Kokkos::LayoutLeft, Kokkos::HIPSpace)
KOKKOSLAPACK_POTRF_ROCSOLVER(double, Kokkos::LayoutLeft, Kokkos::HIPSpace)
KOKKOSLAPACK_POTRF_ROCSOLVER(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::HIPSpace)
KOKKOSLAPACK_POTRF_ROCSOLVER(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::HIPSpace)

}  // namespace Impl
}  // namespace KokkosLapack
#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER

#endif  // KOKKOSLAPACK_POTRF_TPL_SPEC_DECL_HPP_
