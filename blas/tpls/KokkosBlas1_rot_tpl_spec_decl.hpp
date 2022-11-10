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

#ifndef KOKKOSBLAS1_ROT_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS1_ROT_TPL_SPEC_DECL_HPP_

namespace KokkosBlas {
namespace Impl {

namespace {
template <class ExecutionSpace, class VectorView, class ScalarView>
inline void rot_print_specialization() {
#ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
  printf("KokkosBlas::rot<> TPL Blas specialization for < %s, %s, %s >\n",
         typeid(VectorView).name(), typeid(ScalarView).name(),
         typeid(ExecutionSpace).name);
#endif
}
}  // namespace
}  // namespace Impl
}  // namespace KokkosBlas

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#include "KokkosBlas_Host_tpl.hpp"

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS1_DROT_TPL_SPEC_DECL_BLAS(LAYOUT, EXECSPACE, ETI_SPEC_AVAIL) \
  template <>                                                                  \
  struct Rot<EXECSPACE,                                                        \
             Kokkos::View<double*, LAYOUT,                                     \
                          Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,        \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>,            \
             Kokkos::View<double, LAYOUT,                                      \
                          Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,        \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>,            \
             true, ETI_SPEC_AVAIL> {                                           \
    using VectorView =                                                         \
        Kokkos::View<double*, LAYOUT,                                          \
                     Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,             \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    using ScalarView =                                                         \
        Kokkos::View<double, LAYOUT,                                           \
                     Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,             \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    static void rot(EXECSPACE const& /*space*/, VectorView const& X,           \
                    VectorView const& Y, ScalarView const& c,                  \
                    ScalarView const& s) {                                     \
      Kokkos::Profiling::pushRegion("KokkosBlas::rot[TPL_BLAS,double]");       \
      HostBlas<double>::rot(X.extent_int(0), X.data(), 1, Y.data(), 1,         \
                            c.data(), s.data());                               \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS1_SROT_TPL_SPEC_DECL_BLAS(LAYOUT, EXECSPACE, ETI_SPEC_AVAIL) \
  template <>                                                                  \
  struct Rot<EXECSPACE,                                                        \
             Kokkos::View<float*, LAYOUT,                                      \
                          Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,        \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>,            \
             Kokkos::View<float, LAYOUT,                                       \
                          Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,        \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>,            \
             true, ETI_SPEC_AVAIL> {                                           \
    using VectorView =                                                         \
        Kokkos::View<float*, LAYOUT,                                           \
                     Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,             \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    using ScalarView =                                                         \
        Kokkos::View<float, LAYOUT,                                            \
                     Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,             \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    static void rot(EXECSPACE const& /*space*/, VectorView const& X,           \
                    VectorView const& Y, ScalarView const& c,                  \
                    ScalarView const& s) {                                     \
      Kokkos::Profiling::pushRegion("KokkosBlas::rot[TPL_BLAS,float]");        \
      HostBlas<float>::rot(X.extent_int(0), X.data(), 1, Y.data(), 1,          \
                           c.data(), s.data());                                \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_BLAS(LAYOUT, EXECSPACE, ETI_SPEC_AVAIL) \
  template <class MEMSPACE>                                                    \
  struct Rot<Kokkos::complex<double>, EXECSPACE, MEMSPACE, true,               \
             ETI_SPEC_AVAIL> {                                                 \
    using VectorView =                                                         \
        Kokkos::View<Kokkos::complex<double>*, LAYOUT,                         \
                     Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,             \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    using ScalarView =                                                         \
        Kokkos::View<double, LAYOUT,                                           \
                     Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,             \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    static void rot(EXECSPACE const& /*space*/, VectorView const& X,           \
                    VectorView const& Y, ScalarView const& c,                  \
                    ScalarView const& s) {                                     \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::rot[TPL_BLAS,complex<double>]");                        \
      HostBlas<std::complex<double>>::rot(                                     \
          X.extent_int(0), reinterpret_cast<std::complex<double>*>(X.data()),  \
          1, reinterpret_cast<std::complex<double>*>(Y.data()), 1, c.data(),   \
          s.data());                                                           \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS1_CROT_TPL_SPEC_DECL_BLAS(LAYOUT, EXECSPACE, ETI_SPEC_AVAIL) \
  template <class MEMSPACE>                                                    \
  struct Rot<Kokkos::complex<float>, EXECSPACE, MEMSPACE, true,                \
             ETI_SPEC_AVAIL> {                                                 \
    using VectorView =                                                         \
        Kokkos::View<Kokkos::complex<float>*, LAYOUT,                          \
                     Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,             \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    using ScalarView =                                                         \
        Kokkos::View<float, LAYOUT,                                            \
                     Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,             \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    static void rot(EXECSPACE const& /*space*/, VectorView const& X,           \
                    VectorView const& Y, ScalarView const& c,                  \
                    ScalarView const& s) {                                     \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::rot[TPL_BLAS,complex<float>]");                         \
      HostBlas<std::complex<float>>::rot(                                      \
          X.extent_int(0), reinterpret_cast<std::complex<float>*>(X.data()),   \
          1, reinterpret_cast<std::complex<float>*>(Y.data()), 1, c.data(),    \
          s.data());                                                           \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS1_DROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::Serial, true)
KOKKOSBLAS1_DROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::Serial, false)

KOKKOSBLAS1_SROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::Serial, true)
KOKKOSBLAS1_SROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::Serial, false)

KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::Serial, true)
KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::Serial, false)

KOKKOSBLAS1_CROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::Serial, true)
KOKKOSBLAS1_CROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::Serial, false)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS1_DROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::OpenMP, true)
KOKKOSBLAS1_DROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::OpenMP, false)

KOKKOSBLAS1_SROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::OpenMP, true)
KOKKOSBLAS1_SROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::OpenMP, false)

KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::OpenMP, true)
KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::OpenMP, false)

KOKKOSBLAS1_CROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::OpenMP, true)
KOKKOSBLAS1_CROT_TPL_SPEC_DECL_BLAS(Kokkos::LayoutLeft, Kokkos::OpenMP, false)
#endif

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSKERNELS_ENABLE_TPL_BLAS

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS1_DROT_TPL_SPEC_DECL_CUBLAS(LAYOUT, EXECSPACE, MEMSPACE,     \
                                              ETI_SPEC_AVAIL)                  \
  template <>                                                                  \
  struct Rot<                                                                  \
      EXECSPACE,                                                               \
      Kokkos::View<double*, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      Kokkos::View<double, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      true, ETI_SPEC_AVAIL> {                                                  \
    using VectorView =                                                         \
        Kokkos::View<double*, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,     \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    using ScalarView =                                                         \
        Kokkos::View<double, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,      \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    static void rot(EXECSPACE const& space, VectorView const& X,               \
                    VectorView const& Y, ScalarView const& c,                  \
                    ScalarView const& s) {                                     \
      Kokkos::Profiling::pushRegion("KokkosBlas::rot[TPL_CUBLAS,double]");     \
      rot_print_specialization<EXECSPACE, VectorView, ScalarView>();           \
      KokkosBlas::Impl::CudaBlasSingleton& singleton =                         \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                    \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetStream(singleton.handle, space.cuda_stream()));             \
      cublasPointerMode_t pointer_mode;                                        \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasGetPointerMode(singleton.handle, &pointer_mode));              \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetPointerMode(singleton.handle, CUBLAS_POINTER_MODE_DEVICE)); \
      cublasDrot(singleton.handle, X.extent_int(0), X.data(), 1, Y.data(), 1,  \
                 c.data(), s.data());                                          \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetPointerMode(singleton.handle, pointer_mode));               \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS1_SROT_TPL_SPEC_DECL_CUBLAS(LAYOUT, EXECSPACE, MEMSPACE,     \
                                              ETI_SPEC_AVAIL)                  \
  template <>                                                                  \
  struct Rot<EXECSPACE,                                                        \
             Kokkos::View<float*, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>, \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>,            \
             Kokkos::View<float, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,  \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>,            \
             true, ETI_SPEC_AVAIL> {                                           \
    using VectorView =                                                         \
        Kokkos::View<float*, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,      \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    using ScalarView =                                                         \
        Kokkos::View<float, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,       \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    static void rot(EXECSPACE const& space, VectorView const& X,               \
                    VectorView const& Y, ScalarView const& c,                  \
                    ScalarView const& s) {                                     \
      Kokkos::Profiling::pushRegion("KokkosBlas::rot[TPL_CUBLAS,float]");      \
      rot_print_specialization<EXECSPACE, VectorView, ScalarView>();           \
      KokkosBlas::Impl::CudaBlasSingleton& singleton =                         \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                    \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetStream(singleton.handle, space.cuda_stream()));             \
      cublasPointerMode_t pointer_mode;                                        \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasGetPointerMode(singleton.handle, &pointer_mode));              \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetPointerMode(singleton.handle, CUBLAS_POINTER_MODE_DEVICE)); \
      cublasSrot(singleton.handle, X.extent_int(0), X.data(), 1, Y.data(), 1,  \
                 c.data(), s.data());                                          \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetPointerMode(singleton.handle, pointer_mode));               \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_CUBLAS(LAYOUT, EXECSPACE, MEMSPACE,     \
                                              ETI_SPEC_AVAIL)                  \
  template <>                                                                  \
  struct Rot<EXECSPACE,                                                        \
             Kokkos::View<Kokkos::complex<double>*, LAYOUT,                    \
                          Kokkos::Device<EXECSPACE, MEMSPACE>,                 \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>,            \
             Kokkos::View<double, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>, \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>,            \
             true, ETI_SPEC_AVAIL> {                                           \
    using VectorView = Kokkos::View<Kokkos::complex<double>*, LAYOUT,          \
                                    Kokkos::Device<EXECSPACE, MEMSPACE>,       \
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;  \
    using ScalarView =                                                         \
        Kokkos::View<double, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,      \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    static void rot(EXECSPACE const& space, VectorView const& X,               \
                    VectorView const& Y, ScalarView const& c,                  \
                    ScalarView const& s) {                                     \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::rot[TPL_CUBLAS,complex<double>]");                      \
      rot_print_specialization<EXECSPACE, VectorView, ScalarView>();           \
      KokkosBlas::Impl::CudaBlasSingleton& singleton =                         \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                    \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetStream(singleton.handle, space.cuda_stream()));             \
      cublasPointerMode_t pointer_mode;                                        \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasGetPointerMode(singleton.handle, &pointer_mode));              \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetPointerMode(singleton.handle, CUBLAS_POINTER_MODE_DEVICE)); \
      cublasZdrot(singleton.handle, X.extent_int(0),                           \
                  reinterpret_cast<cuDoubleComplex*>(X.data()), 1,             \
                  reinterpret_cast<cuDoubleComplex*>(Y.data()), 1, c.data(),   \
                  s.data());                                                   \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetPointerMode(singleton.handle, pointer_mode));               \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS1_CROT_TPL_SPEC_DECL_CUBLAS(LAYOUT, EXECSPACE, MEMSPACE,     \
                                              ETI_SPEC_AVAIL)                  \
  template <>                                                                  \
  struct Rot<EXECSPACE,                                                        \
             Kokkos::View<Kokkos::complex<float>*, LAYOUT,                     \
                          Kokkos::Device<EXECSPACE, MEMSPACE>,                 \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>,            \
             Kokkos::View<float, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,  \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>,            \
             true, ETI_SPEC_AVAIL> {                                           \
    using VectorView = Kokkos::View<Kokkos::complex<float>, LAYOUT,            \
                                    Kokkos::Device<EXECSPACE, MEMSPACE>,       \
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;  \
    using ScalarView =                                                         \
        Kokkos::View<float, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,       \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    static void rot(EXECSPACE const& space, VectorView const& X,               \
                    VectorView const& Y, ScalarView const& c,                  \
                    ScalarView const& s) {                                     \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::rot[TPL_CUBLAS,complex<float>]");                       \
      rot_print_specialization<EXECSPACE, VectorView, ScalarView>();           \
      KokkosBlas::Impl::CudaBlasSingleton& singleton =                         \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                    \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetStream(singleton.handle, space.cuda_stream()));             \
      cublasPointerMode_t pointer_mode;                                        \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasGetPointerMode(singleton.handle, &pointer_mode));              \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetPointerMode(singleton.handle, CUBLAS_POINTER_MODE_DEVICE)); \
      cublasCsrot(singleton.handle, X.extent_int(0),                           \
                  reinterpret_cast<cuComplex*>(X.data()), 1,                   \
                  reinterpret_cast<cuComplex*>(Y.data()), 1, c.data(),         \
                  s.data());                                                   \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetPointerMode(singleton.handle, pointer_mode));               \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

KOKKOSBLAS1_DROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaSpace, true)
KOKKOSBLAS1_DROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaSpace, false)
KOKKOSBLAS1_DROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaSpace, true)
KOKKOSBLAS1_DROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaSpace, false)
KOKKOSBLAS1_DROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, true)
KOKKOSBLAS1_DROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, false)
KOKKOSBLAS1_DROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, true)
KOKKOSBLAS1_DROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, false)

KOKKOSBLAS1_SROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaSpace, true)
KOKKOSBLAS1_SROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaSpace, false)
KOKKOSBLAS1_SROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaSpace, true)
KOKKOSBLAS1_SROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaSpace, false)
KOKKOSBLAS1_SROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, true)
KOKKOSBLAS1_SROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, false)
KOKKOSBLAS1_SROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, true)
KOKKOSBLAS1_SROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, false)

KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaSpace, true)
KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaSpace, false)
KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaSpace, true)
KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaSpace, false)
KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, true)
KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, false)
KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, true)
KOKKOSBLAS1_ZROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, false)

KOKKOSBLAS1_CROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaSpace, true)
KOKKOSBLAS1_CROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaSpace, false)
KOKKOSBLAS1_CROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaSpace, true)
KOKKOSBLAS1_CROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaSpace, false)
KOKKOSBLAS1_CROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, true)
KOKKOSBLAS1_CROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutLeft, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, false)
KOKKOSBLAS1_CROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, true)
KOKKOSBLAS1_CROT_TPL_SPEC_DECL_CUBLAS(Kokkos::LayoutRight, Kokkos::Cuda,
                                      Kokkos::CudaUVMSpace, false)
}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

#endif
