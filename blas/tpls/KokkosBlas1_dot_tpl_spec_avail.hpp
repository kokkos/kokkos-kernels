// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class execution_space, class AV, class XMV, class YMV, int Xrank = XMV::rank, int Yrank = YMV::rank>
struct dot_tpl_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

namespace KokkosBlas {
namespace Impl {

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
// double
#define KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_BLAS(EXEC_SPACE, SCALAR, LAYOUT)                                      \
  template <>                                                                                                \
  struct dot_tpl_spec_avail<                                                                                 \
      EXEC_SPACE, Kokkos::View<SCALAR, LAYOUT, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1, 1> {     \
    enum : bool { value = true };                                                                            \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, float, Kokkos::LayoutLeft)

// TODO: we met difficuties in FindTPLMKL.cmake to set the BLAS library properly
// such that the test in CheckHostBlasReturnComplex.cmake could not be
// compiled and run to give a correct answer on KK_BLAS_RESULT_AS_POINTER_ARG.
// This resulted in segfault in dot() with MKL and complex.
// So we just temporarily disable it until FindTPLMKL.cmake is fixed.
#if !defined(KOKKOSKERNELS_ENABLE_TPL_MKL)
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, float, Kokkos::LayoutLeft)

// TODO: we met difficuties in FindTPLMKL.cmake to set the BLAS library properly
// such that the test in CheckHostBlasReturnComplex.cmake could not be
// compiled and run to give a correct answer on KK_BLAS_RESULT_AS_POINTER_ARG.
// This resulted in segfault in dot() with MKL and complex.
// So we just temporarily disable it until FindTPLMKL.cmake is fixed.
#if !defined(KOKKOSKERNELS_ENABLE_TPL_MKL)
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif
#endif

#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, float, Kokkos::LayoutLeft)

// TODO: we met difficuties in FindTPLMKL.cmake to set the BLAS library properly
// such that the test in CheckHostBlasReturnComplex.cmake could not be
// compiled and run to give a correct answer on KK_BLAS_RESULT_AS_POINTER_ARG.
// This resulted in segfault in dot() with MKL and complex.
// So we just temporarily disable it until FindTPLMKL.cmake is fixed.
#if !defined(KOKKOSKERNELS_ENABLE_TPL_MKL)
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif
#endif

#endif

#define KOKKOSBLAS1_DOT_TPL_SPEC(SCALAR, LAYOUT, EXECSPACE)                                                 \
  template <>                                                                                               \
  struct dot_tpl_spec_avail<                                                                                \
      EXECSPACE, Kokkos::View<SCALAR, LAYOUT, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<const SCALAR*, LAYOUT, EXECSPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<const SCALAR*, LAYOUT, EXECSPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1, 1> {     \
    enum : bool { value = true };                                                                           \
  };

#define KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL(LAYOUT, EXECSPACE)             \
  KOKKOSBLAS1_DOT_TPL_SPEC(float, LAYOUT, EXECSPACE)                  \
  KOKKOSBLAS1_DOT_TPL_SPEC(double, LAYOUT, EXECSPACE)                 \
  KOKKOSBLAS1_DOT_TPL_SPEC(Kokkos::complex<float>, LAYOUT, EXECSPACE) \
  KOKKOSBLAS1_DOT_TPL_SPEC(Kokkos::complex<double>, LAYOUT, EXECSPACE)

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
// Note BMK: CUBLAS dot is consistently slower than our native dot
// (measured 11.2, 11.8, 12.0 using perf test, and all are similar)
// If a future version improves performance, re-enable it here and
// in the tpl_spec_decl file.
#if 0
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL(Kokkos::LayoutLeft, Kokkos::Cuda)
#endif
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL(Kokkos::LayoutLeft, Kokkos::HIP)
#endif

#if defined(KOKKOSKERNELS_ENABLE_TPL_MKL) && defined(KOKKOS_ENABLE_SYCL)
KOKKOSBLAS1_DOT_TPL_SPEC_AVAIL(Kokkos::LayoutLeft, Kokkos::SYCL)
#endif
}  // namespace Impl
}  // namespace KokkosBlas
#endif
