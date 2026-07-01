// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class ExecutionSpace, class VectorView, class MagnitudeView, class ScalarView>
struct rot_tpl_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

namespace KokkosBlas {
namespace Impl {

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#define KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_BLAS(SCALAR, LAYOUT, EXECSPACE)                                      \
  template <>                                                                                               \
  struct rot_tpl_spec_avail<                                                                                \
      EXECSPACE, Kokkos::View<SCALAR*, LAYOUT, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
      Kokkos::View<typename KokkosKernels::ArithTraits<SCALAR>::mag_type, LAYOUT, Kokkos::HostSpace,        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                                \
      Kokkos::View<SCALAR, LAYOUT, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {           \
    enum : bool { value = true };                                                                           \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::Serial)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::Serial)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::Serial)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::Serial)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::OpenMP)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::OpenMP)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::OpenMP)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::OpenMP)
#endif

#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::Threads)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::Threads)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::Threads)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::Threads)
#endif
#endif

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#define KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUT, EXECSPACE)                                           \
  template <>                                                                                                      \
  struct rot_tpl_spec_avail<EXECSPACE,                                                                             \
                            Kokkos::View<SCALAR*, LAYOUT, EXECSPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>,     \
                            Kokkos::View<typename KokkosKernels::ArithTraits<SCALAR>::mag_type, LAYOUT, EXECSPACE, \
                                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                 \
                            Kokkos::View<SCALAR, LAYOUT, EXECSPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {    \
    enum : bool { value = true };                                                                                  \
  };

KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft, Kokkos::Cuda)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft, Kokkos::Cuda)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::Cuda)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::Cuda)
#endif

// rocBLAS
/*
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#define KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_ROCBLAS(SCALAR, LAYOUT, EXECSPACE)                  \
  template <>                                                             \
  struct rot_tpl_spec_avail<                                              \
      EXECSPACE,                                                          \
      Kokkos::View<SCALAR*, LAYOUT, EXECSPACE,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,              \
      Kokkos::View<SCALAR, LAYOUT, EXECSPACE,   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {            \
    enum : bool { value = true };                                         \
  };

KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutLeft, Kokkos::HIP)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutLeft, Kokkos::HIP)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::HIP)
KOKKOSBLAS1_ROT_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::HIP)
#endif
*/

}  // namespace Impl
}  // namespace KokkosBlas
#endif
