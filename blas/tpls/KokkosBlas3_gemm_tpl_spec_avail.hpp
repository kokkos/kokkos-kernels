// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class execution_space, class AT, class XT, class YT>
struct gemm_tpl_spec_avail {
  enum : bool { value = false };
};

// Generic Host side BLAS (could be MKL or whatever)
#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS)

#define KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(EXEC_SPACE, SCALAR, LAYOUTA, LAYOUTB, LAYOUTC)                    \
  template <>                                                                                                  \
  struct gemm_tpl_spec_avail<                                                                                  \
      EXEC_SPACE, Kokkos::View<const SCALAR**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR**, LAYOUTB, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<SCALAR**, LAYOUTC, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                \
    enum : bool { value = true };                                                                              \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, double, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, float, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)

KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, double, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, float, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, double, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, float, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)

KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, double, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, float, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
#endif

#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, double, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, float, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft)

KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, double, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, float, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                     Kokkos::LayoutRight)
#endif

#endif

// cuBLAS
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)

#define KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUTA, LAYOUTB, LAYOUTC)                                  \
  template <>                                                                                                      \
  struct gemm_tpl_spec_avail<                                                                                      \
      Kokkos::Cuda, Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR**, LAYOUTB, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,               \
      Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                  \
    enum : bool { value = true };                                                                                  \
  };

KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                       Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                       Kokkos::LayoutLeft)

KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                       Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight,
                                       Kokkos::LayoutRight)

#endif

// rocBLAS
#if defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)

#define KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_ROCBLAS(SCALAR, LAYOUT)                                                 \
  template <>                                                                                                   \
  struct gemm_tpl_spec_avail<                                                                                   \
      Kokkos::HIP, Kokkos::View<const SCALAR**, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR**, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                 \
    enum : bool { value = true };                                                                               \
  };

KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft)

KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>, Kokkos::LayoutRight)
KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>, Kokkos::LayoutRight)

#endif
}  // namespace Impl
}  // namespace KokkosBlas

#endif
