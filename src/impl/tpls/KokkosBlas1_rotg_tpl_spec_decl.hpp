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
template <class Scalar>
inline void rotg_print_specialization() {
#ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
  printf("KokkosBlas1::rotg<> TPL cuBLAS specialization for %s\n",
         typeid(Scalar).name());
#else
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
  printf("KokkosBlas1::rotg<> TPL Blas specialization for %s\n",
         typeid(Scalar).name());
#endif
#endif
#endif
}
}  // namespace Impl
}  // namespace KokkosBlas

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#include "KokkosBlas_Host_tpl.hpp"

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_BLAS(ETI_SPEC_AVAIL, EXECSPACE)	\
  template <>								\
  struct Rotg<float, EXECSPACE, true, ETI_SPEC_AVAIL> {			\
  									\
    static void rotg(EXECSPACE /*space*/,				\
		     float& a, float& b, float& c, float& s) {		\
      Kokkos::Profiling::pushRegion("KokkosBlas::rotg[TPL_BLAS,double]"); \
      HostBlas<float>::rotg(a, b, c, s);				\
      Kokkos::Profiling::popRegion();					\
    }									\
  };

#define KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_BLAS(ETI_SPEC_AVAIL, EXECSPACE)	\
  template <>								\
  struct Rotg<double, EXECSPACE, true, ETI_SPEC_AVAIL> {		\
  									\
    static void rotg(EXECSPACE /*space*/,				\
		     double& a, double& b, double& c, double& s) {	\
      Kokkos::Profiling::pushRegion("KokkosBlas::rotg[TPL_BLAS,double]"); \
      HostBlas<double>::rotg(a, b, c, s);				\
      Kokkos::Profiling::popRegion();					\
    }									\
  };

#define KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_BLAS(ETI_SPEC_AVAIL, EXECSPACE)	\
  template <>								\
  struct Rotg<double, EXECSPACE, true, ETI_SPEC_AVAIL> {		\
  									\
    static void rotg(EXECSPACE /*space*/,				\
		     Kokkos::complex<float>& a,				\
		     Kokkos::complex<float>& b,				\
		     Kokkos::complex<float>& c,				\
		     Kokkos::complex<float>& s) {			\
      Kokkos::Profiling::pushRegion("KokkosBlas::rotg[TPL_BLAS,double]"); \
      HostBlas<std::complex<float>>::rotg(static_cast<std::complex<float>>(a), \
					  static_cast<std::complex<float>>(b), \
					  static_cast<std::complex<float>>(c), \
					  static_cast<std::complex<float>>(s));	\
      Kokkos::Profiling::popRegion();					\
    }									\
  };

#define KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_BLAS(ETI_SPEC_AVAIL, EXECSPACE)	\
  template <>						\
  struct Rotg<Kokkos::complex<double>, EXECSPACE, true, ETI_SPEC_AVAIL> { \
  									\
    static void rotg(EXECSPACE /*space*/,				\
		     Kokkos::complex<double>& a,			\
		     Kokkos::complex<double>& b,			\
		     Kokkos::complex<double>& c,			\
		     Kokkos::complex<double>& s) {			\
      Kokkos::Profiling::pushRegion("KokkosBlas::rotg[TPL_BLAS,double]"); \
      HostBlas<std::complex<double>>::rotg(static_cast<std::complex<double>>(a), \
					   static_cast<std::complex<double>>(b), \
					   static_cast<std::complex<double>>(c), \
					   static_cast<std::complex<double>>(s)); \
      Kokkos::Profiling::popRegion();					\
    }									\
  };

KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_BLAS(true, Kokkos::Serial)
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_BLAS(false, Kokkos::Serial)

KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_BLAS(true, Kokkos::Serial)
KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_BLAS(false, Kokkos::Serial)

KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_BLAS(true, Kokkos::Serial)
KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_BLAS(false, Kokkos::Serial)

KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_BLAS(true, Kokkos::OpenMP)
KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_BLAS(false, Kokkos::OpenMP)

KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_BLAS(true, Kokkos::OpenMP)
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_BLAS(false, Kokkos::OpenMP)

KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_BLAS(true, Kokkos::OpenMP)
KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_BLAS(false, Kokkos::OpenMP)

KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_BLAS(true, Kokkos::OpenMP)
KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_BLAS(false, Kokkos::OpenMP)

KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_BLAS(true, Kokkos::OpenMP)
KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_BLAS(false, Kokkos::OpenMP)

}  // namespace Impl
}  // namespace KokkosBlas

#endif

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_CUBLAS(ETI_SPEC_AVAIL, MEMSPACE) \
  template <>								\
  struct Rotg<double, Kokkos::Cuda, MEMSPACE, true, ETI_SPEC_AVAIL> {	\
  									\
    static void rotg(Kokkos::Cuda space,				\
		     double& a, double& b, double& c, double& s) {	\
      Kokkos::Profiling::pushRegion("KokkosBlas::rotg[TPL_CUBLAS,double]"); \
      rotg_print_specialization<double>();				\
      KokkosBlas::Impl::CudaBlasSingleton& singleton =			\
	KokkosBlas::Impl::CudaBlasSingleton::singleton();		\
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(					\
				   cublasSetStream(singleton.handle, space.cuda_stream())); \
      cublasDrotg(singleton.handle, &a, &b, &c, &s);			\
      Kokkos::Profiling::popRegion();					\
    }									\
  };

#define KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_CUBLAS(ETI_SPEC_AVAIL, MEMSPACE) \
  template <>								\
  struct Rotg<float, Kokkos::Cuda, MEMSPACE, true, ETI_SPEC_AVAIL> {	\
  									\
    static void rotg(Kokkos::Cuda space,				\
		     float& a, float& b, float& c, float& s) {		\
      Kokkos::Profiling::pushRegion("KokkosBlas::rotg[TPL_CUBLAS,float]"); \
      rotg_print_specialization<float>();				\
      KokkosBlas::Impl::CudaBlasSingleton& singleton =			\
	KokkosBlas::Impl::CudaBlasSingleton::singleton();		\
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(					\
				   cublasSetStream(singleton.handle, space.cuda_stream())); \
      cublasSrotg(singleton.handle, &a, &b, &c, &s);			\
      Kokkos::Profiling::popRegion();					\
    }									\
  };

#define KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_CUBLAS(ETI_SPEC_AVAIL, MEMSPACE) \
  template <>								\
  struct Rotg<Kokkos::complex<double>, Kokkos::Cuda, MEMSPACE, true, ETI_SPEC_AVAIL> { \
  									\
    static void rotg(Kokkos::Cuda space,				\
		     Kokkos::complex<double>& a,			\
		     Kokkos::complex<double>& b,			\
		     double& c,						\
		     Kokkos::complex<double>& s) {			\
      Kokkos::Profiling::pushRegion("KokkosBlas::rotg[TPL_CUBLAS,complex<double>]"); \
      rotg_print_specialization<Kokkos::complex<double>>();		\
      KokkosBlas::Impl::CudaBlasSingleton& singleton =			\
	KokkosBlas::Impl::CudaBlasSingleton::singleton();		\
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(					\
				   cublasSetStream(singleton.handle, space.cuda_stream())); \
      cublasZrotg(singleton.handle,					\
		  reinterpret_cast<cuDoubleComplex*>(&a),		\
		  reinterpret_cast<cuDoubleComplex*>(&b),		\
		  &c,							\
		  reinterpret_cast<cuDoubleComplex*>(&s));		\
      Kokkos::Profiling::popRegion();					\
    }									\
  };

#define KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_CUBLAS(ETI_SPEC_AVAIL, MEMSPACE) \
  template <>								\
  struct Rotg<Kokkos::complex<float>, Kokkos::Cuda, MEMSPACE, true, ETI_SPEC_AVAIL> { \
  									\
    static void rotg(Kokkos::Cuda space,				\
		     Kokkos::complex<float>& a,				\
		     Kokkos::complex<float>& b,				\
		     float& c,						\
		     Kokkos::complex<float>& s) {			\
      Kokkos::Profiling::pushRegion("KokkosBlas::rotg[TPL_CUBLAS,complex<float>]"); \
      rotg_print_specialization<Kokkos::complex<float>>();		\
      KokkosBlas::Impl::CudaBlasSingleton& singleton =			\
	KokkosBlas::Impl::CudaBlasSingleton::singleton();		\
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(					\
				   cublasSetStream(singleton.handle, space.cuda_stream())); \
      cublasCrotg(singleton.handle,					\
		  reinterpret_cast<cuComplex*>(&a),			\
		  reinterpret_cast<cuComplex*>(&b),			\
		  &c,							\
		  reinterpret_cast<cuComplex*>(&s));			\
      Kokkos::Profiling::popRegion();					\
    }									\
  };

KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_CUBLAS(true, Kokkos::CudaSpace)
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_CUBLAS(false, Kokkos::CudaSpace)
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_CUBLAS(true, Kokkos::CudaUVMSpace)
KOKKOSBLAS1_DROTG_TPL_SPEC_DECL_CUBLAS(false, Kokkos::CudaUVMSpace)

KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_CUBLAS(true, Kokkos::CudaSpace)
KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_CUBLAS(false, Kokkos::CudaSpace)
KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_CUBLAS(true, Kokkos::CudaUVMSpace)
KOKKOSBLAS1_SROTG_TPL_SPEC_DECL_CUBLAS(false, Kokkos::CudaUVMSpace)

KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_CUBLAS(true, Kokkos::CudaSpace)
KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_CUBLAS(false, Kokkos::CudaSpace)
KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_CUBLAS(true, Kokkos::CudaUVMSpace)
KOKKOSBLAS1_ZROTG_TPL_SPEC_DECL_CUBLAS(false, Kokkos::CudaUVMSpace)

KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_CUBLAS(true, Kokkos::CudaSpace)
KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_CUBLAS(false, Kokkos::CudaSpace)
KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_CUBLAS(true, Kokkos::CudaUVMSpace)
KOKKOSBLAS1_CROTG_TPL_SPEC_DECL_CUBLAS(false, Kokkos::CudaUVMSpace)

}  // namespace Impl
}  // namespace KokkosBlas

#endif

#endif
