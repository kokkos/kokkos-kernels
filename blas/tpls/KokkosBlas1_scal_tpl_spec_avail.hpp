// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class execution_space, class RV, class AV, class XV, int Xrank = XV::rank>
struct scal_tpl_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

namespace KokkosBlas {
namespace Impl {

// Generic Host side BLAS (could be MKL or whatever)
#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS)
// double
#define KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_BLAS(EXEC_SPACE, SCALAR, LAYOUT)                                       \
  template <>                                                                                                  \
  struct scal_tpl_spec_avail<                                                                                  \
      EXEC_SPACE, Kokkos::View<SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, SCALAR, \
      Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> {          \
    enum : bool { value = true };                                                                              \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif
#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif

#endif

// cuBLAS
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)
// double
#define KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUT, EXECSPACE)                                    \
  template <>                                                                                                \
  struct scal_tpl_spec_avail<                                                                                \
      EXECSPACE, Kokkos::View<SCALAR*, LAYOUT, EXECSPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, SCALAR, \
      Kokkos::View<const SCALAR*, LAYOUT, EXECSPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> {         \
    enum : bool { value = true };                                                                            \
  };

KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft, Kokkos::Cuda)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft, Kokkos::Cuda)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::Cuda)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::Cuda)

#endif

// rocBLAS
#if defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)

#define KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_ROCBLAS(SCALAR, LAYOUT, EXECSPACE)                                   \
  template <>                                                                                                \
  struct scal_tpl_spec_avail<                                                                                \
      EXECSPACE, Kokkos::View<SCALAR*, LAYOUT, EXECSPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, SCALAR, \
      Kokkos::View<const SCALAR*, LAYOUT, EXECSPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> {         \
    enum : bool { value = true };                                                                            \
  };

KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutLeft, Kokkos::HIP)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutLeft, Kokkos::HIP)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::HIP)
KOKKOSBLAS1_SCAL_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::HIP)

#endif

}  // namespace Impl
}  // namespace KokkosBlas
#endif
