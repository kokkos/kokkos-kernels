// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class execution_space, class RV, class XMV, int Xrank = XMV::rank>
struct iamax_tpl_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

namespace KokkosBlas {
namespace Impl {

// Generic Host side BLAS (could be MKL or whatever)
#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS)
// double
#define KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_BLAS(EXEC_SPACE, INDEX_TYPE, SCALAR, LAYOUT)                            \
  template <>                                                                                                    \
  struct iamax_tpl_spec_avail<                                                                                   \
      EXEC_SPACE, Kokkos::View<INDEX_TYPE, LAYOUT, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> {            \
    enum : bool { value = true };                                                                                \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, unsigned long, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, unsigned long, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, unsigned long, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, unsigned long, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, unsigned long, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, unsigned long, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, unsigned long, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, unsigned long, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif

#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, unsigned long, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, unsigned long, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, unsigned long, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, unsigned long, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif

#endif

// cuBLAS
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)
// double
#define KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_CUBLAS(INDEX_TYPE, SCALAR, LAYOUT)                                        \
  template <>                                                                                                      \
  struct iamax_tpl_spec_avail<                                                                                     \
      Kokkos::Cuda, Kokkos::View<INDEX_TYPE, LAYOUT, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR*, LAYOUT, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> {            \
    enum : bool { value = true };                                                                                  \
  };                                                                                                               \
  template <>                                                                                                      \
  struct iamax_tpl_spec_avail<                                                                                     \
      Kokkos::Cuda, Kokkos::View<INDEX_TYPE, LAYOUT, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,      \
      Kokkos::View<const SCALAR*, LAYOUT, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> {            \
    enum : bool { value = true };                                                                                  \
  };

KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_CUBLAS(unsigned long, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_CUBLAS(unsigned int, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_CUBLAS(unsigned long, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_CUBLAS(unsigned int, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_CUBLAS(unsigned long, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_CUBLAS(unsigned int, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_CUBLAS(unsigned long, Kokkos::complex<float>, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_CUBLAS(unsigned int, Kokkos::complex<float>, Kokkos::LayoutLeft)

#endif

// rocBLAS
#if defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)

#define KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_ROCBLAS(INDEX_TYPE, SCALAR, LAYOUT)                                      \
  template <>                                                                                                     \
  struct iamax_tpl_spec_avail<                                                                                    \
      Kokkos::HIP, Kokkos::View<INDEX_TYPE, LAYOUT, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> {            \
    enum : bool { value = true };                                                                                 \
  };                                                                                                              \
  template <>                                                                                                     \
  struct iamax_tpl_spec_avail<                                                                                    \
      Kokkos::HIP, Kokkos::View<INDEX_TYPE, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,       \
      Kokkos::View<const SCALAR*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> {            \
    enum : bool { value = true };                                                                                 \
  };

KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_ROCBLAS(unsigned long, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_ROCBLAS(unsigned int, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_ROCBLAS(unsigned long, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_ROCBLAS(unsigned int, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_ROCBLAS(unsigned long, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_ROCBLAS(unsigned int, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_ROCBLAS(unsigned long, Kokkos::complex<float>, Kokkos::LayoutLeft)
KOKKOSBLAS1_IAMAX_TPL_SPEC_AVAIL_ROCBLAS(unsigned int, Kokkos::complex<float>, Kokkos::LayoutLeft)

#endif

}  // namespace Impl
}  // namespace KokkosBlas
#endif
