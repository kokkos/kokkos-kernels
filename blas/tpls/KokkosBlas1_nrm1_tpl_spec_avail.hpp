// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class execution_space, class AV, class XMV, int Xrank = XMV::rank>
struct nrm1_tpl_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

namespace KokkosBlas {
namespace Impl {

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
// double
#define KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_BLAS(EXEC_SPACE, SCALAR, LAYOUT)                               \
  template <>                                                                                          \
  struct nrm1_tpl_spec_avail<                                                                          \
      EXEC_SPACE,                                                                                      \
      Kokkos::View<typename KokkosKernels::Details::InnerProductSpaceTraits<SCALAR>::mag_type, LAYOUT, \
                   Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                       \
      Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> {  \
    enum : bool { value = true };                                                                      \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif
#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif

#endif

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
// double
#define KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUT)                                          \
  template <>                                                                                           \
  struct nrm1_tpl_spec_avail<                                                                           \
      Kokkos::Cuda,                                                                                     \
      Kokkos::View<typename KokkosKernels::Details::InnerProductSpaceTraits<SCALAR>::mag_type, LAYOUT,  \
                   Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                        \
      Kokkos::View<const SCALAR*, LAYOUT, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> { \
    enum : bool { value = true };                                                                       \
  };

KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft)

#endif

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#define KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_ROCBLAS(SCALAR, LAYOUT)                                        \
  template <>                                                                                          \
  struct nrm1_tpl_spec_avail<                                                                          \
      Kokkos::HIP,                                                                                     \
      Kokkos::View<typename KokkosKernels::Details::InnerProductSpaceTraits<SCALAR>::mag_type, LAYOUT, \
                   Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                       \
      Kokkos::View<const SCALAR*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> { \
    enum : bool { value = true };                                                                      \
  };

KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft)

#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCBLAS

// oneMKL
#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL

#if defined(KOKKOS_ENABLE_SYCL)

#define KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_MKL_SYCL(SCALAR, LAYOUT)                                        \
  template <>                                                                                           \
  struct nrm1_tpl_spec_avail<                                                                           \
      Kokkos::SYCL,                                                                                     \
      Kokkos::View<typename KokkosKernels::Details::InnerProductSpaceTraits<SCALAR>::mag_type, LAYOUT,  \
                   Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                        \
      Kokkos::View<const SCALAR*, LAYOUT, Kokkos::SYCL, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> { \
    enum : bool { value = true };                                                                       \
  };

KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_MKL_SYCL(double, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_MKL_SYCL(float, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_MKL_SYCL(Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM1_TPL_SPEC_AVAIL_MKL_SYCL(Kokkos::complex<float>, Kokkos::LayoutLeft)

#endif  // KOKKOS_ENABLE_SYCL
#endif  // KOKKOSKERNELS_ENABLE_TPL_MKL

}  // namespace Impl
}  // namespace KokkosBlas
#endif
