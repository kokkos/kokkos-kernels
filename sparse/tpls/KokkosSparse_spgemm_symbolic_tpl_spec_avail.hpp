/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOSPARSE_SPGEMM_SYMBOLIC_TPL_SPEC_AVAIL_HPP_
#define KOKKOSPARSE_SPGEMM_SYMBOLIC_TPL_SPEC_AVAIL_HPP_

namespace KokkosSparse {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class KernelHandle, class a_size_view_t_, class a_lno_view_t,
          class b_size_view_t_, class b_lno_view_t, class c_size_view_t_>
struct spgemm_symbolic_tpl_spec_avail {
  enum : bool { value = false };
};

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
// NOTE: all versions of cuSPARSE 10.x and 11.x support exactly the same matrix
// types, so there is no ifdef'ing on versions needed in avail. Offset and
// Ordinal must both be 32-bit. Even though the "generic" API lets you specify
// offsets and ordinals independently as either 16, 32 or 64-bit, SpGEMM will
// just fail at runtime if you don't use 32 for both.

#define SPGEMM_SYMBOLIC_AVAIL_CUSPARSE(SCALAR, MEMSPACE)              \
  template <>                                                         \
  struct spgemm_symbolic_tpl_spec_avail<                              \
      KokkosKernels::Experimental::KokkosKernelsHandle<               \
          const int, const int, const SCALAR, Kokkos::Cuda, MEMSPACE, \
          MEMSPACE>,                                                  \
      Kokkos::View<const int*, default_layout,                        \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
      Kokkos::View<const int*, default_layout,                        \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
      Kokkos::View<const int*, default_layout,                        \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
      Kokkos::View<const int*, default_layout,                        \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
      Kokkos::View<int*, default_layout,                              \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {      \
    enum : bool { value = true };                                     \
  };

#define SPGEMM_SYMBOLIC_AVAIL_CUSPARSE_S(SCALAR)            \
  SPGEMM_SYMBOLIC_AVAIL_CUSPARSE(SCALAR, Kokkos::CudaSpace) \
  SPGEMM_SYMBOLIC_AVAIL_CUSPARSE(SCALAR, Kokkos::CudaUVMSpace)

SPGEMM_SYMBOLIC_AVAIL_CUSPARSE_S(float)
SPGEMM_SYMBOLIC_AVAIL_CUSPARSE_S(double)
SPGEMM_SYMBOLIC_AVAIL_CUSPARSE_S(Kokkos::complex<float>)
SPGEMM_SYMBOLIC_AVAIL_CUSPARSE_S(Kokkos::complex<double>)
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#define SPGEMM_SYMBOLIC_AVAIL_ROCSPARSE(SCALAR)                              \
  template <>                                                                \
  struct spgemm_symbolic_tpl_spec_avail<                                     \
      KokkosKernels::Experimental::KokkosKernelsHandle<                      \
          const int, const int, const SCALAR, Kokkos::HIP, Kokkos::HIPSpace, \
          Kokkos::HIPSpace>,                                                 \
      Kokkos::View<const int*, default_layout,                               \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<const int*, default_layout,                               \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<const int*, default_layout,                               \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<const int*, default_layout,                               \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<int*, default_layout,                                     \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {             \
    enum : bool { value = true };                                            \
  };

SPGEMM_SYMBOLIC_AVAIL_ROCSPARSE(float)
SPGEMM_SYMBOLIC_AVAIL_ROCSPARSE(double)
SPGEMM_SYMBOLIC_AVAIL_ROCSPARSE(Kokkos::complex<float>)
SPGEMM_SYMBOLIC_AVAIL_ROCSPARSE(Kokkos::complex<double>)
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
#define SPGEMM_SYMBOLIC_AVAIL_MKL(SCALAR, EXEC)               \
  template <>                                                                \
  struct spgemm_symbolic_tpl_spec_avail<                                     \
      KokkosKernels::Experimental::KokkosKernelsHandle<                       \
          const int, const int, const SCALAR, EXEC, Kokkos::HostSpace,  \
          Kokkos::HostSpace>,                                                  \
      Kokkos::View<const int *, default_layout,                               \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,             \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<const int *, default_layout,                               \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,             \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<const int *, default_layout,                               \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,             \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<const int *, default_layout,                               \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,             \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<int *, default_layout,                                     \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,             \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {  \
    enum : bool { value = true };                                            \
  };

#define SPGEMM_SYMBOLIC_AVAIL_MKL_E(EXEC) \
SPGEMM_SYMBOLIC_AVAIL_MKL(float, EXEC) \
SPGEMM_SYMBOLIC_AVAIL_MKL(double, EXEC) \
SPGEMM_SYMBOLIC_AVAIL_MKL(Kokkos::complex<float>, EXEC) \
SPGEMM_SYMBOLIC_AVAIL_MKL(Kokkos::complex<double>, EXEC)

#ifdef KOKKOS_ENABLE_SERIAL
SPGEMM_SYMBOLIC_AVAIL_MKL_E(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
SPGEMM_SYMBOLIC_AVAIL_MKL_E(Kokkos::OpenMP)
#endif
#endif

}  // namespace Impl
}  // namespace KokkosSparse

#endif
