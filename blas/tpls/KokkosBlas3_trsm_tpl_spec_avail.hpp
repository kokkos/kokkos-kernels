// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {

// Specialization struct which defines whether a specialization exists
template <class execution_space, class AVT, class BVT>
struct trsm_tpl_spec_avail {
  enum : bool { value = false };
};

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS

#define KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(EXEC_SPACE, SCALAR, LAYOUTA, LAYOUTB)                             \
  template <>                                                                                                  \
  struct trsm_tpl_spec_avail<                                                                                  \
      EXEC_SPACE, Kokkos::View<const SCALAR**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<SCALAR**, LAYOUTB, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                \
    enum : bool { value = true };                                                                              \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, double, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, float, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft)

KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, double, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, float, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, double, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, float, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft)

KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, double, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, float, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight)
#endif

#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, double, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, float, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft)

KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, double, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, float, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight)
#endif

#endif

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS

#define KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUTA, LAYOUTB)                                           \
  template <>                                                                                                      \
  struct trsm_tpl_spec_avail<                                                                                      \
      Kokkos::Cuda, Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                  \
    enum : bool { value = true };                                                                                  \
  };

KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft)

KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight)
KOKKOSBLAS3_TRSM_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight)

#endif
}  // namespace Impl
}  // namespace KokkosBlas

#endif
