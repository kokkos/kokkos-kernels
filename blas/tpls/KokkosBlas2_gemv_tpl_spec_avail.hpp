// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class ExecutionSpace, class AT, class XT, class YT>
struct gemv_tpl_spec_avail {
  enum : bool { value = false };
};

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS

#define KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(EXEC_SPACE, SCALAR, LAYOUTA, LAYOUTX, LAYOUTY)                    \
  template <>                                                                                                  \
  struct gemv_tpl_spec_avail<                                                                                  \
      EXEC_SPACE, Kokkos::View<const SCALAR**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR*, LAYOUTX, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<SCALAR*, LAYOUTY, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                 \
    enum : bool { value = true };                                                                              \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, double, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, float, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)

KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, double, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, float, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, double, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, float, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)

KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, double, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, float, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
#endif

#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, double, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, float, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)

KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, double, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, float, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
#endif

#endif

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS

#define KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUTA, LAYOUTX, LAYOUTY)                                  \
  template <>                                                                                                      \
  struct gemv_tpl_spec_avail<                                                                                      \
      Kokkos::Cuda, Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR*, LAYOUTX, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<SCALAR*, LAYOUTY, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                   \
    enum : bool { value = true };                                                                                  \
  };

// Note BMK: We use the same layout for A, X and Y because the GEMV
// interface will switch the layouts of X and Y to that of A.
// So this TPL version will match any layout combination, as long
// as none are LayoutStride.

KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                       Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                       Kokkos::LayoutLeft)

KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                       Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                       Kokkos::LayoutRight)

#endif

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS

#define KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ROCBLAS(SCALAR, LAYOUT)                                                 \
  template <>                                                                                                   \
  struct gemv_tpl_spec_avail<                                                                                   \
      Kokkos::HIP, Kokkos::View<const SCALAR**, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,               \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                  \
    enum : bool { value = true };                                                                               \
  };

KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft)

KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>, Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>, Kokkos::LayoutRight)

#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL

#if defined(KOKKOS_ENABLE_SYCL)

#define KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ONEMKL(SCALAR, LAYOUT)                                                    \
  template <>                                                                                                     \
  struct gemv_tpl_spec_avail<                                                                                     \
      Kokkos::SYCL, Kokkos::View<const SCALAR**, LAYOUT, Kokkos::SYCL, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR*, LAYOUT, Kokkos::SYCL, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::SYCL, Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                   \
    enum : bool { value = true };                                                                                 \
  };

KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ONEMKL(double, Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ONEMKL(float, Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ONEMKL(Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ONEMKL(Kokkos::complex<float>, Kokkos::LayoutLeft)

KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ONEMKL(double, Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ONEMKL(float, Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ONEMKL(Kokkos::complex<double>, Kokkos::LayoutRight)
KOKKOSBLAS2_GEMV_TPL_SPEC_AVAIL_ONEMKL(Kokkos::complex<float>, Kokkos::LayoutRight)

#endif

#endif

}  // namespace Impl
}  // namespace KokkosBlas

#endif
