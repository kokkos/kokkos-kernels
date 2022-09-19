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

#ifndef KOKKOSBLAS1_ROTG_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS1_ROTG_TPL_SPEC_DECL_HPP_

namespace KokkosBlas {
namespace Impl {

namespace {
template <class Scalar, class ExecutionSpace>
inline void rotg_print_specialization() {
#ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
  printf("KokkosBlas1::rotg<> TPL Blas specialization for < %s, %s >\n",
         typeid(Scalar).name(), typeid(ExecutionSpace).name);
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

#define KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_BLAS(EXECSPACE, ETI_SPEC_AVAIL)   \
  template <class MEMSPACE>                                               \
  struct Rotg<double, EXECSPACE, MEMSPACE, true, ETI_SPEC_AVAIL> {        \
    static void rotg(double& a, double& b, double& c, double& s) {        \
      Kokkos::Profiling::pushRegion("KokkosBlas::rotg[TPL_BLAS,double]"); \
      HostBlas<double>::rotg(&a, &b, &c, &s);                             \
      Kokkos::Profiling::popRegion();                                     \
    }                                                                     \
  };

#define KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_BLAS(EXECSPACE, ETI_SPEC_AVAIL)  \
  template <class MEMSPACE>                                              \
  struct Rotg<float, EXECSPACE, MEMSPACE, true, ETI_SPEC_AVAIL> {        \
    static void rotg(float& a, float& b, float& c, float& s) {           \
      Kokkos::Profiling::pushRegion("KokkosBlas::rotg[TPL_BLAS,float]"); \
      HostBlas<float>::rotg(&a, &b, &c, &s);                             \
      Kokkos::Profiling::popRegion();                                    \
    }                                                                    \
  };

#define KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_BLAS(EXECSPACE, ETI_SPEC_AVAIL)      \
  template <class MEMSPACE>                                                  \
  struct Rotg<Kokkos::complex<double>, EXECSPACE, MEMSPACE, true,            \
              ETI_SPEC_AVAIL> {                                              \
    static void rotg(Kokkos::complex<double>& a, Kokkos::complex<double>& b, \
                     double& c, Kokkos::complex<double>& s) {                \
      Kokkos::Profiling::pushRegion(                                         \
          "KokkosBlas::rotg[TPL_BLAS,complex<double>]");                     \
      HostBlas<std::complex<double> >::rotg(                                 \
          reinterpret_cast<std::complex<double>*>(&a),                       \
          reinterpret_cast<std::complex<double>*>(&b), &c,                   \
          reinterpret_cast<std::complex<double>*>(&s));                      \
      Kokkos::Profiling::popRegion();                                        \
    }                                                                        \
  };

#define KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_BLAS(EXECSPACE, ETI_SPEC_AVAIL)    \
  template <class MEMSPACE>                                                \
  struct Rotg<Kokkos::complex<float>, EXECSPACE, MEMSPACE, true,           \
              ETI_SPEC_AVAIL> {                                            \
    static void rotg(Kokkos::complex<float>& a, Kokkos::complex<float>& b, \
                     float& c, Kokkos::complex<float>& s) {                \
      Kokkos::Profiling::pushRegion(                                       \
          "KokkosBlas::rotg[TPL_BLAS,complex<float>]");                    \
      HostBlas<std::complex<float> >::rotg(                                \
          reinterpret_cast<std::complex<float>*>(&a),                      \
          reinterpret_cast<std::complex<float>*>(&b), &c,                  \
          reinterpret_cast<std::complex<float>*>(&s));                     \
      Kokkos::Profiling::popRegion();                                      \
    }                                                                      \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_BLAS(Kokkos::Serial, true)
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_BLAS(Kokkos::Serial, false)

KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_BLAS(Kokkos::Serial, true)
KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_BLAS(Kokkos::Serial, false)

KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_BLAS(Kokkos::Serial, true)
KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_BLAS(Kokkos::Serial, false)

KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_BLAS(Kokkos::Serial, true)
KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_BLAS(Kokkos::Serial, false)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, true)
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, false)

KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, true)
KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, false)

KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, true)
KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, false)

KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, true)
KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_BLAS(Kokkos::OpenMP, false)
#endif

}  // namespace Impl
}  // namespace KokkosBlas

#endif

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_CUBLAS(EXECSPACE, ETI_SPEC_AVAIL)   \
  template <class MEMSPACE>                                                 \
  struct Rotg<double, EXECSPACE, MEMSPACE, true, ETI_SPEC_AVAIL> {          \
    static void rotg(double& a, double& b, double& c, double& s) {          \
      Kokkos::Profiling::pushRegion("KokkosBlas::nrm1[TPL_CUBLAS,double]"); \
      rotg_print_specialization<double, EXECSPACE>();                       \
      KokkosBlas::Impl::CudaBlasSingleton& singleton =                      \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                 \
      cublasDrotg(singleton.handle, &a, &b, &c, &s);                        \
      Kokkos::Profiling::popRegion();                                       \
    }                                                                       \
  };

#define KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_CUBLAS(EXECSPACE, ETI_SPEC_AVAIL)  \
  template <class MEMSPACE>                                                \
  struct Rotg<float, EXECSPACE, MEMSPACE, true, ETI_SPEC_AVAIL> {          \
    static void rotg(float& a, float& b, float& c, float& s) {             \
      Kokkos::Profiling::pushRegion("KokkosBlas::nrm1[TPL_CUBLAS,float]"); \
      rotg_print_specialization<float, EXECSPACE>();                       \
      KokkosBlas::Impl::CudaBlasSingleton& singleton =                     \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                \
      cublasSrotg(singleton.handle, &a, &b, &c, &s);                       \
      Kokkos::Profiling::popRegion();                                      \
    }                                                                      \
  };

#define KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_CUBLAS(EXECSPACE, ETI_SPEC_AVAIL)    \
  template <class MEMSPACE>                                                  \
  struct Rotg<Kokkos::complex<double>, EXECSPACE, MEMSPACE, true,            \
              ETI_SPEC_AVAIL> {                                              \
    static void rotg(Kokkos::complex<double>& a, Kokkos::complex<double>& b, \
                     double& c, Kokkos::complex<double>& s) {                \
      Kokkos::Profiling::pushRegion(                                         \
          "KokkosBlas::nrm1[TPL_CUBLAS,complex<double>]");                   \
      rotg_print_specialization<Kokkos::complex<double>, EXECSPACE>();       \
      KokkosBlas::Impl::CudaBlasSingleton& singleton =                       \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                  \
      cublasZrotg(singleton.handle, reinterpret_cast<cuDoubleComplex*>(&a),  \
                  reinterpret_cast<cuDoubleComplex*>(&b), &c,                \
                  reinterpret_cast<cuDoubleComplex*>(&s));                   \
      Kokkos::Profiling::popRegion();                                        \
    }                                                                        \
  };

#define KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_CUBLAS(EXECSPACE, ETI_SPEC_AVAIL)  \
  template <class MEMSPACE>                                                \
  struct Rotg<Kokkos::complex<float>, EXECSPACE, MEMSPACE, true,           \
              ETI_SPEC_AVAIL> {                                            \
    static void rotg(Kokkos::complex<float>& a, Kokkos::complex<float>& b, \
                     float& c, Kokkos::complex<float>& s) {                \
      Kokkos::Profiling::pushRegion(                                       \
          "KokkosBlas::nrm1[TPL_CUBLAS,complex<float>]");                  \
      rotg_print_specialization<Kokkos::complex<float>, EXECSPACE>();      \
      KokkosBlas::Impl::CudaBlasSingleton& singleton =                     \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                \
      cublasCrotg(singleton.handle, reinterpret_cast<cuComplex*>(&a),      \
                  reinterpret_cast<cuComplex*>(&b), &c,                    \
                  reinterpret_cast<cuComplex*>(&s));                       \
      Kokkos::Profiling::popRegion();                                      \
    }                                                                      \
  };

KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_CUBLAS(Kokkos::Cuda, true)
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_CUBLAS(Kokkos::Cuda, false)

KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_CUBLAS(Kokkos::Cuda, true)
KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_CUBLAS(Kokkos::Cuda, false)

KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_CUBLAS(Kokkos::Cuda, true)
KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_CUBLAS(Kokkos::Cuda, false)

KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_CUBLAS(Kokkos::Cuda, true)
KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_CUBLAS(Kokkos::Cuda, false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_ROCBLAS(EXECSPACE, ETI_SPEC_AVAIL)   \
  template <class MEMSPACE>                                                  \
  struct Rotg<double, EXECSPACE, MEMSPACE, true, ETI_SPEC_AVAIL> {           \
    static void rotg(double& a, double& b, double& c, double& s) {           \
      Kokkos::Profiling::pushRegion("KokkosBlas::nrm1[TPL_ROCBLAS,double]"); \
      rotg_print_specialization<double, EXECSPACE>();                        \
      KokkosBlas::Impl::RocBlasSingleton& singleton =                        \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                   \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                         \
          rocblas_drotg(singleton.handle, &a, &b, &c, &s));                  \
      Kokkos::Profiling::popRegion();                                        \
    }                                                                        \
  };

#define KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_ROCBLAS(EXECSPACE, ETI_SPEC_AVAIL)  \
  template <class MEMSPACE>                                                 \
  struct Rotg<float, EXECSPACE, MEMSPACE, true, ETI_SPEC_AVAIL> {           \
    static void rotg(float& a, float& b, float& c, float& s) {              \
      Kokkos::Profiling::pushRegion("KokkosBlas::nrm1[TPL_ROCBLAS,float]"); \
      rotg_print_specialization<float, EXECSPACE>();                        \
      KokkosBlas::Impl::RocBlasSingleton& singleton =                       \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                  \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                        \
          rocblas_srotg(singleton.handle, &a, &b, &c, &s));                 \
      Kokkos::Profiling::popRegion();                                       \
    }                                                                       \
  };

#define KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_ROCBLAS(EXECSPACE, ETI_SPEC_AVAIL)   \
  template <class MEMSPACE>                                                  \
  struct Rotg<Kokkos::complex<double>, EXECSPACE, MEMSPACE, true,            \
              ETI_SPEC_AVAIL> {                                              \
    static void rotg(Kokkos::complex<double>& a, Kokkos::complex<double>& b, \
                     double& c, Kokkos::complex<double>& s) {                \
      Kokkos::Profiling::pushRegion(                                         \
          "KokkosBlas::nrm1[TPL_ROCBLAS,complex<double>]");                  \
      rotg_print_specialization<Kokkos::complex<double>, EXECSPACE>();       \
      KokkosBlas::Impl::RocBlasSingleton& singleton =                        \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                   \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_zrotg(                           \
          singleton.handle, reinterpret_cast<rocblas_double_complex*>(&a),   \
          reinterpret_cast<rocblas_double_complex*>(&b), &c,                 \
          reinterpret_cast<rocblas_double_complex*>(&s)));                   \
      Kokkos::Profiling::popRegion();                                        \
    }                                                                        \
  };

#define KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_ROCBLAS(EXECSPACE, ETI_SPEC_AVAIL) \
  template <class MEMSPACE>                                                \
  struct Rotg<Kokkos::complex<float>, EXECSPACE, MEMSPACE, true,           \
              ETI_SPEC_AVAIL> {                                            \
    static void rotg(Kokkos::complex<float>& a, Kokkos::complex<float>& b, \
                     float& c, Kokkos::complex<float>& s) {                \
      Kokkos::Profiling::pushRegion(                                       \
          "KokkosBlas::nrm1[TPL_ROCBLAS,complex<float>]");                 \
      rotg_print_specialization<Kokkos::complex<float>, EXECSPACE>();      \
      KokkosBlas::Impl::RocBlasSingleton& singleton =                      \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                 \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_crotg(                         \
          singleton.handle, reinterpret_cast<rocblas_float_complex*>(&a),  \
          reinterpret_cast<rocblas_float_complex*>(&b), &c,                \
          reinterpret_cast<rocblas_float_complex*>(&s)));                  \
      Kokkos::Profiling::popRegion();                                      \
    }                                                                      \
  };

KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_ROCBLAS(Kokkos::HIP, true)
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_ROCBLAS(Kokkos::HIP, false)

KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_ROCBLAS(Kokkos::HIP, true)
KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_ROCBLAS(Kokkos::HIP, false)

KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_ROCBLAS(Kokkos::HIP, true)
KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_ROCBLAS(Kokkos::HIP, false)

KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_ROCBLAS(Kokkos::HIP, true)
KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_ROCBLAS(Kokkos::HIP, false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif

#endif
