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
template <class Scalar>
inline void rotg_print_specialization() {
#ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
  printf("KokkosBlas1::rotg<> TPL Blas specialization for < %s >\n",
         typeid(Scalar).name());
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

#define KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_BLAS(ETI_SPEC_AVAIL)              \
  template <>                                                             \
  struct Rotg<double, true, ETI_SPEC_AVAIL> {                             \
    static void rotg(double& a, double& b, double& c, double& s) {        \
      Kokkos::Profiling::pushRegion("KokkosBlas::rotg[TPL_BLAS,double]"); \
      HostBlas<double>::rotg(&a, &b, &c, &s);                             \
      Kokkos::Profiling::popRegion();                                     \
    }                                                                     \
  };

#define KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_BLAS(ETI_SPEC_AVAIL)             \
  template <>                                                            \
  struct Rotg<float, true, ETI_SPEC_AVAIL> {                             \
    static void rotg(float& a, float& b, float& c, float& s) {           \
      Kokkos::Profiling::pushRegion("KokkosBlas::rotg[TPL_BLAS,float]"); \
      HostBlas<float>::rotg(&a, &b, &c, &s);                             \
      Kokkos::Profiling::popRegion();                                    \
    }                                                                    \
  };

#define KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_BLAS(ETI_SPEC_AVAIL)                 \
  template <>                                                                \
  struct Rotg<Kokkos::complex<double>, true, ETI_SPEC_AVAIL> {               \
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

#define KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_BLAS(ETI_SPEC_AVAIL)               \
  template <>                                                              \
  struct Rotg<Kokkos::complex<float>, true, ETI_SPEC_AVAIL> {              \
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

KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_BLAS(true)
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_BLAS(false)

KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_BLAS(true)
KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_BLAS(false)

KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_BLAS(true)
KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_BLAS(false)

KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_BLAS(true)
KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_BLAS(false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_CUBLAS(ETI_SPEC_AVAIL)              \
  template <>                                                               \
  struct Rotg<double, true, ETI_SPEC_AVAIL> {                               \
    static void rotg(double& a, double& b, double& c, double& s) {          \
      Kokkos::Profiling::pushRegion("KokkosBlas::nrm1[TPL_CUBLAS,double]"); \
      rotg_print_specialization<double>();                                  \
      KokkosBlas::Impl::CudaBlasSingleton& singleton =                      \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                 \
      cublasDrotg(singleton.handle, &a, &b, &c, &s);                        \
      Kokkos::Profiling::popRegion();                                       \
    }                                                                       \
  };

#define KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_CUBLAS(ETI_SPEC_AVAIL)             \
  template <>                                                              \
  struct Rotg<float, true, ETI_SPEC_AVAIL> {                               \
    static void rotg(float& a, float& b, float& c, float& s) {             \
      Kokkos::Profiling::pushRegion("KokkosBlas::nrm1[TPL_CUBLAS,float]"); \
      rotg_print_specialization<float>();                                  \
      KokkosBlas::Impl::CudaBlasSingleton& singleton =                     \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                \
      cublasSrotg(singleton.handle, &a, &b, &c, &s);                       \
      Kokkos::Profiling::popRegion();                                      \
    }                                                                      \
  };

#define KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_CUBLAS(ETI_SPEC_AVAIL)               \
  template <>                                                                \
  struct Rotg<Kokkos::complex<double>, true, ETI_SPEC_AVAIL> {               \
    static void rotg(Kokkos::complex<double>& a, Kokkos::complex<double>& b, \
                     double& c, Kokkos::complex<double>& s) {                \
      Kokkos::Profiling::pushRegion(                                         \
          "KokkosBlas::nrm1[TPL_CUBLAS,complex<double>]");                   \
      rotg_print_specialization<Kokkos::complex<double> >();                 \
      KokkosBlas::Impl::CudaBlasSingleton& singleton =                       \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                  \
      cublasZrotg(singleton.handle, reinterpret_cast<cuDoubleComplex*>(&a),  \
                  reinterpret_cast<cuDoubleComplex*>(&b), &c,                \
                  reinterpret_cast<cuDoubleComplex*>(&s));                   \
      Kokkos::Profiling::popRegion();                                        \
    }                                                                        \
  };

#define KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_CUBLAS(ETI_SPEC_AVAIL)             \
  template <>                                                              \
  struct Rotg<Kokkos::complex<float>, true, ETI_SPEC_AVAIL> {              \
    static void rotg(Kokkos::complex<float>& a, Kokkos::complex<float>& b, \
                     float& c, Kokkos::complex<float>& s) {                \
      Kokkos::Profiling::pushRegion(                                       \
          "KokkosBlas::nrm1[TPL_CUBLAS,complex<float>]");                  \
      rotg_print_specialization<Kokkos::complex<float> >();                \
      KokkosBlas::Impl::CudaBlasSingleton& singleton =                     \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                \
      cublasCrotg(singleton.handle, reinterpret_cast<cuComplex*>(&a),      \
                  reinterpret_cast<cuComplex*>(&b), &c,                    \
                  reinterpret_cast<cuComplex*>(&s));                       \
      Kokkos::Profiling::popRegion();                                      \
    }                                                                      \
  };

KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_CUBLAS(true)
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_CUBLAS(false)

KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_CUBLAS(true)
KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_CUBLAS(false)

KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_CUBLAS(true)
KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_CUBLAS(false)

KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_CUBLAS(true)
KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_CUBLAS(false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_ROCBLAS(ETI_SPEC_AVAIL)              \
  template <>                                                                \
  struct Rotg<double, true, ETI_SPEC_AVAIL> {                                \
    static void rotg(double& a, double& b, double& c, double& s) {           \
      Kokkos::Profiling::pushRegion("KokkosBlas::nrm1[TPL_ROCBLAS,double]"); \
      rotg_print_specialization<double>();                                   \
      KokkosBlas::Impl::RocBlasSingleton& singleton =                        \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                   \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                         \
          rocblas_drotg(singleton.handle, &a, &b, &c, &s));                  \
      Kokkos::Profiling::popRegion();                                        \
    }                                                                        \
  };

#define KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_ROCBLAS(ETI_SPEC_AVAIL)             \
  template <>                                                               \
  struct Rotg<float, true, ETI_SPEC_AVAIL> {                                \
    static void rotg(float& a, float& b, float& c, float& s) {              \
      Kokkos::Profiling::pushRegion("KokkosBlas::nrm1[TPL_ROCBLAS,float]"); \
      rotg_print_specialization<float>();                                   \
      KokkosBlas::Impl::RocBlasSingleton& singleton =                       \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                  \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                        \
          rocblas_srotg(singleton.handle, &a, &b, &c, &s));                 \
      Kokkos::Profiling::popRegion();                                       \
    }                                                                       \
  };

#define KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_ROCBLAS(ETI_SPEC_AVAIL)              \
  template <>                                                                \
  struct Rotg<Kokkos::complex<double>, true, ETI_SPEC_AVAIL> {               \
    static void rotg(Kokkos::complex<double>& a, Kokkos::complex<double>& b, \
                     double& c, Kokkos::complex<double>& s) {                \
      Kokkos::Profiling::pushRegion(                                         \
          "KokkosBlas::nrm1[TPL_ROCBLAS,complex<double>]");                  \
      rotg_print_specialization<Kokkos::complex<double> >();                 \
      KokkosBlas::Impl::RocBlasSingleton& singleton =                        \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                   \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_zrotg(                           \
          singleton.handle, reinterpret_cast<rocblas_double_complex*>(&a),   \
          reinterpret_cast<rocblas_double_complex*>(&b), &c,                 \
          reinterpret_cast<rocblas_double_complex*>(&s)));                   \
      Kokkos::Profiling::popRegion();                                        \
    }                                                                        \
  };

#define KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_ROCBLAS(ETI_SPEC_AVAIL)            \
  template <>                                                              \
  struct Rotg<Kokkos::complex<float>, true, ETI_SPEC_AVAIL> {              \
    static void rotg(Kokkos::complex<float>& a, Kokkos::complex<float>& b, \
                     float& c, Kokkos::complex<float>& s) {                \
      Kokkos::Profiling::pushRegion(                                       \
          "KokkosBlas::nrm1[TPL_ROCBLAS,complex<float>]");                 \
      rotg_print_specialization<Kokkos::complex<float> >();                \
      KokkosBlas::Impl::RocBlasSingleton& singleton =                      \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                 \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_crotg(                         \
          singleton.handle, reinterpret_cast<rocblas_float_complex*>(&a),  \
          reinterpret_cast<rocblas_float_complex*>(&b), &c,                \
          reinterpret_cast<rocblas_float_complex*>(&s)));                  \
      Kokkos::Profiling::popRegion();                                      \
    }                                                                      \
  };

KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_ROCBLAS(true)
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_ROCBLAS(false)

KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_ROCBLAS(true)
KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_ROCBLAS(false)

KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_ROCBLAS(true)
KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_ROCBLAS(false)

KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_ROCBLAS(true)
KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_ROCBLAS(false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif

#endif
