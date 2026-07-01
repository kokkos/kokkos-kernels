// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class execution_space, class RV, class XMV, int Xrank = XMV::rank>
struct nrm2_tpl_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

namespace KokkosBlas {
namespace Impl {
// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
// double
#define KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_BLAS(EXEC_SPACE, SCALAR, LAYOUT)                               \
  template <>                                                                                          \
  struct nrm2_tpl_spec_avail<                                                                          \
      EXEC_SPACE,                                                                                      \
      Kokkos::View<typename KokkosKernels::Details::InnerProductSpaceTraits<SCALAR>::mag_type, LAYOUT, \
                   Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                       \
      Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> {  \
    enum : bool { value = true };                                                                      \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_BLAS(Kokkos::Serial, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_BLAS(Kokkos::OpenMP, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif
#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, double, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, float, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL_BLAS(Kokkos::Threads, Kokkos::complex<float>, Kokkos::LayoutLeft)
#endif

#endif

#define KOKKOSBLAS1_NRM2_TPL_SPEC(SCALAR, LAYOUT, EXEC_SPACE)                                         \
  template <>                                                                                         \
  struct nrm2_tpl_spec_avail<                                                                         \
      EXEC_SPACE,                                                                                     \
      Kokkos::View<typename KokkosKernels::ArithTraits<SCALAR>::mag_type, LAYOUT, Kokkos::HostSpace,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                         \
      Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, 1> { \
    enum : bool { value = true };                                                                     \
  };

#define KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL(LAYOUT, EXEC_SPACE)             \
  KOKKOSBLAS1_NRM2_TPL_SPEC(float, LAYOUT, EXEC_SPACE)                  \
  KOKKOSBLAS1_NRM2_TPL_SPEC(double, LAYOUT, EXEC_SPACE)                 \
  KOKKOSBLAS1_NRM2_TPL_SPEC(Kokkos::complex<float>, LAYOUT, EXEC_SPACE) \
  KOKKOSBLAS1_NRM2_TPL_SPEC(Kokkos::complex<double>, LAYOUT, EXEC_SPACE)

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL(Kokkos::LayoutLeft, Kokkos::Cuda)
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL(Kokkos::LayoutLeft, Kokkos::HIP)
#endif

#if defined(KOKKOSKERNELS_ENABLE_TPL_MKL) && defined(KOKKOS_ENABLE_SYCL)
KOKKOSBLAS1_NRM2_TPL_SPEC_AVAIL(Kokkos::LayoutLeft, Kokkos::SYCL)
#endif

}  // namespace Impl
}  // namespace KokkosBlas
#endif
