// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSPARSE_BSPGEMM_NUMERIC_TPL_SPEC_AVAIL_HPP_
#define KOKKOSPARSE_BSPGEMM_NUMERIC_TPL_SPEC_AVAIL_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
#include <mkl.h>
#endif

namespace KokkosSparse {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class KernelHandle, class a_size_view_t_, class a_lno_view_t, class a_scalar_view_t, class b_size_view_t_,
          class b_lno_view_t, class b_scalar_view_t, class c_size_view_t_, class c_lno_view_t, class c_scalar_view_t>
struct bspgemm_numeric_tpl_spec_avail {
  enum : bool { value = false };
};

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
#define BSPGEMM_NUMERIC_AVAIL_MKL(SCALAR, EXEC)                                                             \
  template <>                                                                                               \
  struct bspgemm_numeric_tpl_spec_avail<                                                                    \
      KokkosKernels::Experimental::KokkosKernelsHandle<const MKL_INT, const MKL_INT, const SCALAR, EXEC,    \
                                                       Kokkos::HostSpace, Kokkos::HostSpace>,               \
      Kokkos::View<const MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                               \
      Kokkos::View<const MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                               \
      Kokkos::View<const SCALAR *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                               \
      Kokkos::View<const MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                               \
      Kokkos::View<const MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                               \
      Kokkos::View<const SCALAR *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                               \
      Kokkos::View<const MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                               \
      Kokkos::View<MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                               \
      Kokkos::View<SCALAR *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                                            \
    enum : bool { value = true };                                                                           \
  };

#define BSPGEMM_NUMERIC_AVAIL_MKL_E(EXEC)                 \
  BSPGEMM_NUMERIC_AVAIL_MKL(float, EXEC)                  \
  BSPGEMM_NUMERIC_AVAIL_MKL(double, EXEC)                 \
  BSPGEMM_NUMERIC_AVAIL_MKL(Kokkos::complex<float>, EXEC) \
  BSPGEMM_NUMERIC_AVAIL_MKL(Kokkos::complex<double>, EXEC)

#ifdef KOKKOS_ENABLE_SERIAL
BSPGEMM_NUMERIC_AVAIL_MKL_E(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
BSPGEMM_NUMERIC_AVAIL_MKL_E(Kokkos::OpenMP)
#endif
#endif  // KOKKOSKERNELS_ENABLE_TPL_MKL

}  // namespace Impl
}  // namespace KokkosSparse

#endif
