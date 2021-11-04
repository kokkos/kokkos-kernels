#ifndef KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {

template <class AVT, class TVT, class CVT, class WVT>
struct unmqr_tpl_spec_avail {
  enum : bool { value = false };
};

template <class AVT, class TVT, class CVT>
struct unmqr_workspace_tpl_spec_avail {
  enum : bool { value = false };
};

// Hostspace LAPACKE(netlib) or MKL
// TODO: Check if these have the same syntax

#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) && \
    defined(KOKKOSKERNELS_ENABLE_TPL_LAPACKE)

#define KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_LAPACK(SCALAR, LAYOUTA, LAYOUTB,   \
                                               LAYOUTC, MEMSPACE)          \
  template <class ExecSpace>                                               \
  struct unmqr_tpl_spec_avail<                                             \
      Kokkos::View<const SCALAR**, LAYOUTA,                                \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<const SCALAR*, LAYOUTB,                                 \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {           \
    enum : bool { value = true };                                          \
  };

#define KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_LAPACK(                  \
    SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE)                           \
  template <class ExecSpace>                                               \
  struct unmqr_workspace_tpl_spec_avail<                                   \
      Kokkos::View<const SCALAR**, LAYOUTA,                                \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<const SCALAR*, LAYOUTB,                                 \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {           \
    enum : bool { value = true };                                          \
  };

#if defined(KOKKOSKERNELS_INST_DOUBLE) && defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_LAPACK(double, Kokkos::LayoutLeft,
                                                 Kokkos::LayoutLeft,
                                                 Kokkos::LayoutLeft,
                                                 Kokkos::HostSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_LAPACK(double, Kokkos::LayoutLeft,
                                       Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                       Kokkos::HostSpace)
#endif

#if defined(KOKKOSKERNELS_INST_FLOAT) && defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_LAPACK(float, Kokkos::LayoutLeft,
                                                 Kokkos::LayoutLeft,
                                                 Kokkos::LayoutLeft,
                                                 Kokkos::HostSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_LAPACK(float, Kokkos::LayoutLeft,
                                       Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                       Kokkos::HostSpace)
#endif

#if defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) && \
    defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_LAPACK(Kokkos::complex<double>,
                                                 Kokkos::LayoutLeft,
                                                 Kokkos::LayoutLeft,
                                                 Kokkos::LayoutLeft,
                                                 Kokkos::HostSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_LAPACK(Kokkos::complex<double>,
                                       Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                       Kokkos::LayoutLeft, Kokkos::HostSpace)
#endif

#if defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) && \
    defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_LAPACK(Kokkos::complex<float>,
                                                 Kokkos::LayoutLeft,
                                                 Kokkos::LayoutLeft,
                                                 Kokkos::LayoutLeft,
                                                 Kokkos::HostSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_LAPACK(Kokkos::complex<float>,
                                       Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                       Kokkos::LayoutLeft, Kokkos::HostSpace)
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) && \
    defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_LAPACK(double, Kokkos::LayoutRight,
                                                 Kokkos::LayoutRight,
                                                 Kokkos::LayoutRight,
                                                 Kokkos::HostSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_LAPACK(double, Kokkos::LayoutRight,
                                       Kokkos::LayoutRight, Kokkos::LayoutRight,
                                       Kokkos::HostSpace)
#endif

#if defined(KOKKOSKERNELS_INST_FLOAT) && defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_LAPACK(float, Kokkos::LayoutRight,
                                                 Kokkos::LayoutRight,
                                                 Kokkos::LayoutRight,
                                                 Kokkos::HostSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_LAPACK(float, Kokkos::LayoutRight,
                                       Kokkos::LayoutRight, Kokkos::LayoutRight,
                                       Kokkos::HostSpace)
#endif

#if defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) && \
    defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_LAPACK(Kokkos::complex<double>,
                                                 Kokkos::LayoutRight,
                                                 Kokkos::LayoutRight,
                                                 Kokkos::LayoutRight,
                                                 Kokkos::HostSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_LAPACK(Kokkos::complex<double>,
                                       Kokkos::LayoutRight, Kokkos::LayoutRight,
                                       Kokkos::LayoutRight, Kokkos::HostSpace)
#endif

#if defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) && \
    defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_LAPACK(Kokkos::complex<float>,
                                                 Kokkos::LayoutRight,
                                                 Kokkos::LayoutRight,
                                                 Kokkos::LayoutRight,
                                                 Kokkos::HostSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_LAPACK(Kokkos::complex<float>,
                                       Kokkos::LayoutRight, Kokkos::LayoutRight,
                                       Kokkos::LayoutRight, Kokkos::HostSpace)
#endif

#endif  // if BLAS && LAPACK

// CUSOLVER
//
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSOLVER)

#define KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_CUSOLVER(SCALAR, LAYOUTA, LAYOUTB, \
                                                 LAYOUTC, MEMSPACE)        \
  template <class ExecSpace>                                               \
  struct unmqr_tpl_spec_avail<                                             \
      Kokkos::View<const SCALAR**, LAYOUTA,                                \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<const SCALAR*, LAYOUTB,                                 \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {           \
    enum : bool { value = true };                                          \
  };

#define KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_CUSOLVER(                \
    SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE)                           \
  template <class ExecSpace>                                               \
  struct unmqr_workspace_tpl_spec_avail<                                   \
      Kokkos::View<const SCALAR**, LAYOUTA,                                \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<const SCALAR*, LAYOUTB,                                 \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {           \
    enum : bool { value = true };                                          \
  };

// CUDA Space
KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_CUSOLVER(double, Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::CudaSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_CUSOLVER(double, Kokkos::LayoutLeft,
                                         Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                         Kokkos::CudaSpace)

KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_CUSOLVER(float, Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::CudaSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_CUSOLVER(float, Kokkos::LayoutLeft,
                                         Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                         Kokkos::CudaSpace)

KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_CUSOLVER(Kokkos::complex<double>,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::CudaSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_CUSOLVER(Kokkos::complex<double>,
                                         Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                         Kokkos::LayoutLeft, Kokkos::CudaSpace)

KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_CUSOLVER(Kokkos::complex<float>,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::CudaSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_CUSOLVER(Kokkos::complex<float>,
                                         Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                         Kokkos::LayoutLeft, Kokkos::CudaSpace)

// CUDA UVM Space
KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_CUSOLVER(double, Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::CudaUVMSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_CUSOLVER(double, Kokkos::LayoutLeft,
                                         Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                         Kokkos::CudaUVMSpace)

KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_CUSOLVER(float, Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::CudaUVMSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_CUSOLVER(float, Kokkos::LayoutLeft,
                                         Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                         Kokkos::CudaUVMSpace)

KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_CUSOLVER(Kokkos::complex<double>,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::CudaUVMSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_CUSOLVER(Kokkos::complex<double>,
                                         Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                         Kokkos::LayoutLeft,
                                         Kokkos::CudaUVMSpace)

KOKKOSBLAS_UNMQR_WORKSPACE_TPL_SPEC_AVAIL_CUSOLVER(Kokkos::complex<float>,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::LayoutLeft,
                                                   Kokkos::CudaUVMSpace)
KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_CUSOLVER(Kokkos::complex<float>,
                                         Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                         Kokkos::LayoutLeft,
                                         Kokkos::CudaUVMSpace)

#endif  // if CUBLAS && CUSOLVER

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSBLAS_UNMQR_TPL_SPEC_AVAIL_HPP_
