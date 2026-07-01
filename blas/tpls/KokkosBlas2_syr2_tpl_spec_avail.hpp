// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class EXEC_SPACE, class XT, class YT, class AT>
struct syr2_tpl_spec_avail {
  enum : bool { value = false };
};

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS

#define KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(SCALAR, LAYOUT, EXEC_SPACE)                                     \
  template <>                                                                                                \
  struct syr2_tpl_spec_avail<                                                                                \
      EXEC_SPACE, Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<SCALAR**, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {               \
    enum : bool { value = true };                                                                            \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::Serial)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::Serial)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::Serial)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::Serial)

KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutRight, Kokkos::Serial)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutRight, Kokkos::Serial)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::Serial)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::Serial)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::OpenMP)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::OpenMP)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::OpenMP)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::OpenMP)

KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutRight, Kokkos::OpenMP)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutRight, Kokkos::OpenMP)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::OpenMP)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::OpenMP)
#endif

#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::Threads)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::Threads)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::Threads)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::Threads)

KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutRight, Kokkos::Threads)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutRight, Kokkos::Threads)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::Threads)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::Threads)
#endif

#endif

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS

#define KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUT, EXEC_SPACE)                                   \
  template <>                                                                                                \
  struct syr2_tpl_spec_avail<                                                                                \
      EXEC_SPACE, Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<SCALAR**, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {               \
    enum : bool { value = true };                                                                            \
  };

KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft, Kokkos::Cuda)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft, Kokkos::Cuda)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::Cuda)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::Cuda)

KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutRight, Kokkos::Cuda)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutRight, Kokkos::Cuda)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::Cuda)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::Cuda)

#endif

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS

#define KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_ROCBLAS(SCALAR, LAYOUT, EXEC_SPACE)                                  \
  template <>                                                                                                \
  struct syr2_tpl_spec_avail<                                                                                \
      EXEC_SPACE, Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<SCALAR**, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {               \
    enum : bool { value = true };                                                                            \
  };

KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutLeft, Kokkos::HIP)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutLeft, Kokkos::HIP)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::HIP)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::HIP)

KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutRight, Kokkos::HIP)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutRight, Kokkos::HIP)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::HIP)
KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::HIP)

#endif
}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSBLAS2_SYR2_TPL_SPEC_AVAIL_HPP_
