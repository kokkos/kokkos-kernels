// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class execution_space, class AV, class XMV, class BV, class YMV, int rank = YMV::rank>
struct axpby_tpl_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

namespace KokkosBlas {
namespace Impl {

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS

#define KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_BLAS(EXEC_SPACE, SCALAR, LAYOUT)                                            \
  template <>                                                                                                        \
  struct axpby_tpl_spec_avail<                                                                                       \
      EXEC_SPACE, SCALAR, Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      SCALAR, Kokkos::View<SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> {              \
    enum : bool { value = true };                                                                                    \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif
#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif

#endif

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS

#define KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUT)                                            \
  template <>                                                                                              \
  struct axpby_tpl_spec_avail<                                                                             \
      Kokkos::Cuda, SCALAR,                                                                                \
      Kokkos::View<const SCALAR*, LAYOUT, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, SCALAR, \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> {          \
    enum : bool { value = true };                                                                          \
  };

KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft)
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft)
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_AXPBY_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft)

#endif
}  // namespace Impl
}  // namespace KokkosBlas
#endif
