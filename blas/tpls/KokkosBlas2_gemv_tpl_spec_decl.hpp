// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS2_GEMV_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS2_GEMV_TPL_SPEC_DECL_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#include "KokkosBlas_Host_tpl.hpp"

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_GEMV_DETERMINE_ARGS(LAYOUTA)                             \
  bool A_is_lr      = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;     \
  const int M       = static_cast<int>(A_is_lr ? A.extent(1) : A.extent(0)); \
  const int N       = static_cast<int>(A_is_lr ? A.extent(0) : A.extent(1)); \
  constexpr int one = 1;                                                     \
  const int LDA     = A_is_lr ? A.stride(0) : A.stride(1);                   \
                                                                             \
  char transa;                                                               \
  if ((trans[0] == 'N') || (trans[0] == 'n'))                                \
    transa = A_is_lr ? 'T' : 'N';                                            \
  else if ((trans[0] == 'T') || (trans[0] == 't'))                           \
    transa = A_is_lr ? 'N' : 'T';                                            \
  else {                                                                     \
    if (A_is_lr)                                                             \
      throw std::runtime_error(                                              \
          "Error: HostBlas::gemv conjugate transpose requires LayoutLeft "   \
          "views.");                                                         \
    transa = 'C';                                                            \
  }

#define KOKKOSBLAS2_DGEMV_BLAS(EXEC_SPACE, LAYOUTA, LAYOUTX, LAYOUTY, ETI_SPEC_AVAIL)                                  \
  template <>                                                                                                          \
  struct GEMV<EXEC_SPACE, Kokkos::View<const double**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
              Kokkos::View<const double*, LAYOUTX, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
              Kokkos::View<double*, LAYOUTY, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true,              \
              ETI_SPEC_AVAIL> {                                                                                        \
    typedef double SCALAR;                                                                                             \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;     \
    typedef Kokkos::View<const SCALAR*, LAYOUTX, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XViewType;      \
    typedef Kokkos::View<SCALAR*, LAYOUTY, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > YViewType;            \
                                                                                                                       \
    static void gemv(const EXEC_SPACE& /* space */, const char trans[], typename AViewType::const_value_type& alpha,   \
                     const AViewType& A, const XViewType& X, typename YViewType::const_value_type& beta,               \
                     const YViewType& Y) {                                                                             \
      Kokkos::Profiling::pushRegion("KokkosBlas::gemv[TPL_BLAS,double]");                                              \
      KOKKOSBLAS2_GEMV_DETERMINE_ARGS(LAYOUTA);                                                                        \
      HostBlas<double>::gemv(transa, M, N, alpha, A.data(), LDA, X.data(), one, beta, Y.data(), one);                  \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

#define KOKKOSBLAS2_SGEMV_BLAS(EXEC_SPACE, LAYOUTA, LAYOUTX, LAYOUTY, ETI_SPEC_AVAIL)                                 \
  template <>                                                                                                         \
  struct GEMV<EXEC_SPACE, Kokkos::View<const float**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
              Kokkos::View<const float*, LAYOUTX, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
              Kokkos::View<float*, LAYOUTY, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true,              \
              ETI_SPEC_AVAIL> {                                                                                       \
    typedef float SCALAR;                                                                                             \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;    \
    typedef Kokkos::View<const SCALAR*, LAYOUTX, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XViewType;     \
    typedef Kokkos::View<SCALAR*, LAYOUTY, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > YViewType;           \
                                                                                                                      \
    static void gemv(const EXEC_SPACE& /* space */, const char trans[], typename AViewType::const_value_type& alpha,  \
                     const AViewType& A, const XViewType& X, typename YViewType::const_value_type& beta,              \
                     const YViewType& Y) {                                                                            \
      Kokkos::Profiling::pushRegion("KokkosBlas::gemv[TPL_BLAS,float]");                                              \
      KOKKOSBLAS2_GEMV_DETERMINE_ARGS(LAYOUTA);                                                                       \
      HostBlas<float>::gemv(transa, M, N, alpha, A.data(), LDA, X.data(), one, beta, Y.data(), one);                  \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS2_ZGEMV_BLAS(EXEC_SPACE, LAYOUTA, LAYOUTX, LAYOUTY, ETI_SPEC_AVAIL)                                \
  template <>                                                                                                        \
  struct GEMV<                                                                                                       \
      EXEC_SPACE,                                                                                                    \
      Kokkos::View<const Kokkos::complex<double>**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,  \
      Kokkos::View<const Kokkos::complex<double>*, LAYOUTX, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,   \
      Kokkos::View<Kokkos::complex<double>*, LAYOUTY, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true,   \
      ETI_SPEC_AVAIL> {                                                                                              \
    typedef Kokkos::complex<double> SCALAR;                                                                          \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;   \
    typedef Kokkos::View<const SCALAR*, LAYOUTX, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XViewType;    \
    typedef Kokkos::View<SCALAR*, LAYOUTY, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > YViewType;          \
                                                                                                                     \
    static void gemv(const EXEC_SPACE& /* space */, const char trans[], typename AViewType::const_value_type& alpha, \
                     const AViewType& A, const XViewType& X, typename YViewType::const_value_type& beta,             \
                     const YViewType& Y) {                                                                           \
      Kokkos::Profiling::pushRegion("KokkosBlas::gemv[TPL_BLAS,complex<double>]");                                   \
      KOKKOSBLAS2_GEMV_DETERMINE_ARGS(LAYOUTA);                                                                      \
      const std::complex<double> alpha_val = alpha, beta_val = beta;                                                 \
      HostBlas<std::complex<double> >::gemv(transa, M, N, alpha_val,                                                 \
                                            reinterpret_cast<const std::complex<double>*>(A.data()), LDA,            \
                                            reinterpret_cast<const std::complex<double>*>(X.data()), one, beta_val,  \
                                            reinterpret_cast<std::complex<double>*>(Y.data()), one);                 \
      Kokkos::Profiling::popRegion();                                                                                \
    }                                                                                                                \
  };

#define KOKKOSBLAS2_CGEMV_BLAS(EXEC_SPACE, LAYOUTA, LAYOUTX, LAYOUTY, ETI_SPEC_AVAIL)                                \
  template <>                                                                                                        \
  struct GEMV<                                                                                                       \
      EXEC_SPACE,                                                                                                    \
      Kokkos::View<const Kokkos::complex<float>**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,   \
      Kokkos::View<const Kokkos::complex<float>*, LAYOUTX, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,    \
      Kokkos::View<Kokkos::complex<float>*, LAYOUTY, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true,    \
      ETI_SPEC_AVAIL> {                                                                                              \
    typedef Kokkos::complex<float> SCALAR;                                                                           \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;   \
    typedef Kokkos::View<const SCALAR*, LAYOUTX, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XViewType;    \
    typedef Kokkos::View<SCALAR*, LAYOUTY, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > YViewType;          \
                                                                                                                     \
    static void gemv(const EXEC_SPACE& /* space */, const char trans[], typename AViewType::const_value_type& alpha, \
                     const AViewType& A, const XViewType& X, typename YViewType::const_value_type& beta,             \
                     const YViewType& Y) {                                                                           \
      Kokkos::Profiling::pushRegion("KokkosBlas::gemv[TPL_BLAS,complex<float>]");                                    \
      KOKKOSBLAS2_GEMV_DETERMINE_ARGS(LAYOUTA);                                                                      \
      const std::complex<float> alpha_val = alpha, beta_val = beta;                                                  \
      HostBlas<std::complex<float> >::gemv(transa, M, N, alpha_val,                                                  \
                                           reinterpret_cast<const std::complex<float>*>(A.data()), LDA,              \
                                           reinterpret_cast<const std::complex<float>*>(X.data()), one, beta_val,    \
                                           reinterpret_cast<std::complex<float>*>(Y.data()), one);                   \
      Kokkos::Profiling::popRegion();                                                                                \
    }                                                                                                                \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS2_DGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_DGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_DGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_DGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS2_SGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_SGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_SGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_SGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS2_ZGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_ZGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_ZGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_ZGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS2_CGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_CGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_CGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_CGEMV_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS2_DGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_DGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_DGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_DGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS2_SGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_SGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_SGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_SGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS2_ZGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_ZGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_ZGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_ZGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS2_CGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_CGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_CGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_CGEMV_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)
#endif

#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS2_DGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_DGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_DGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_DGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS2_SGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_SGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_SGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_SGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS2_ZGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_ZGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_ZGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_ZGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS2_CGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_CGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_CGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_CGEMV_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)
#endif

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_BLAS

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_GEMV_CUBLAS_DETERMINE_ARGS(LAYOUTA)                      \
  bool A_is_lr      = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;     \
  const int M       = static_cast<int>(A_is_lr ? A.extent(1) : A.extent(0)); \
  const int N       = static_cast<int>(A_is_lr ? A.extent(0) : A.extent(1)); \
  constexpr int one = 1;                                                     \
  const int LDA     = A_is_lr ? A.stride(0) : A.stride(1);                   \
                                                                             \
  cublasOperation_t transa;                                                  \
  if ((trans[0] == 'N') || (trans[0] == 'n'))                                \
    transa = A_is_lr ? CUBLAS_OP_T : CUBLAS_OP_N;                            \
  else if ((trans[0] == 'T') || (trans[0] == 't'))                           \
    transa = A_is_lr ? CUBLAS_OP_N : CUBLAS_OP_T;                            \
  else {                                                                     \
    if (A_is_lr)                                                             \
      throw std::runtime_error(                                              \
          "Error: cublas gemv conjugate transpose requires LayoutLeft "      \
          "views.");                                                         \
    transa = CUBLAS_OP_C;                                                    \
  }

#define KOKKOSBLAS2_DGEMV_CUBLAS(LAYOUTA, LAYOUTX, LAYOUTY, ETI_SPEC_AVAIL)                                           \
  template <>                                                                                                         \
  struct GEMV<                                                                                                        \
      Kokkos::Cuda, Kokkos::View<const double**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,    \
      Kokkos::View<const double*, LAYOUTX, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                   \
      Kokkos::View<double*, LAYOUTY, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true, ETI_SPEC_AVAIL> { \
    typedef double SCALAR;                                                                                            \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;  \
    typedef Kokkos::View<const SCALAR*, LAYOUTX, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XViewType;   \
    typedef Kokkos::View<SCALAR*, LAYOUTY, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > YViewType;         \
                                                                                                                      \
    static void gemv(const Kokkos::Cuda& space, const char trans[], typename AViewType::const_value_type& alpha,      \
                     const AViewType& A, const XViewType& X, typename YViewType::const_value_type& beta,              \
                     const YViewType& Y) {                                                                            \
      Kokkos::Profiling::pushRegion("KokkosBlas::gemv[TPL_CUBLAS,double]");                                           \
      KOKKOSBLAS2_GEMV_CUBLAS_DETERMINE_ARGS(LAYOUTA);                                                                \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                      \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                               \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                               \
          cublasDgemv(s.handle, transa, M, N, &alpha, A.data(), LDA, X.data(), one, &beta, Y.data(), one));           \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                              \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS2_SGEMV_CUBLAS(LAYOUTA, LAYOUTX, LAYOUTY, ETI_SPEC_AVAIL)                                          \
  template <>                                                                                                        \
  struct GEMV<                                                                                                       \
      Kokkos::Cuda, Kokkos::View<const float**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,    \
      Kokkos::View<const float*, LAYOUTX, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                   \
      Kokkos::View<float*, LAYOUTY, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true, ETI_SPEC_AVAIL> { \
    typedef float SCALAR;                                                                                            \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
    typedef Kokkos::View<const SCALAR*, LAYOUTX, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XViewType;  \
    typedef Kokkos::View<SCALAR*, LAYOUTY, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > YViewType;        \
                                                                                                                     \
    static void gemv(const Kokkos::Cuda& space, const char trans[], typename AViewType::const_value_type& alpha,     \
                     const AViewType& A, const XViewType& X, typename YViewType::const_value_type& beta,             \
                     const YViewType& Y) {                                                                           \
      Kokkos::Profiling::pushRegion("KokkosBlas::gemv[TPL_CUBLAS,float]");                                           \
      KOKKOSBLAS2_GEMV_CUBLAS_DETERMINE_ARGS(LAYOUTA);                                                               \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                     \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                              \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                              \
          cublasSgemv(s.handle, transa, M, N, &alpha, A.data(), LDA, X.data(), one, &beta, Y.data(), one));          \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                             \
      Kokkos::Profiling::popRegion();                                                                                \
    }                                                                                                                \
  };

#define KOKKOSBLAS2_ZGEMV_CUBLAS(LAYOUTA, LAYOUTX, LAYOUTY, ETI_SPEC_AVAIL)                                            \
  template <>                                                                                                          \
  struct GEMV<                                                                                                         \
      Kokkos::Cuda,                                                                                                    \
      Kokkos::View<const Kokkos::complex<double>**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,  \
      Kokkos::View<const Kokkos::complex<double>*, LAYOUTX, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,   \
      Kokkos::View<Kokkos::complex<double>*, LAYOUTY, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true,   \
      ETI_SPEC_AVAIL> {                                                                                                \
    typedef Kokkos::complex<double> SCALAR;                                                                            \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;   \
    typedef Kokkos::View<const SCALAR*, LAYOUTX, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XViewType;    \
    typedef Kokkos::View<SCALAR*, LAYOUTY, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > YViewType;          \
                                                                                                                       \
    static void gemv(const Kokkos::Cuda& space, const char trans[], typename AViewType::const_value_type& alpha,       \
                     const AViewType& A, const XViewType& X, typename YViewType::const_value_type& beta,               \
                     const YViewType& Y) {                                                                             \
      Kokkos::Profiling::pushRegion("KokkosBlas::gemv[TPL_CUBLAS,complex<double>]");                                   \
      KOKKOSBLAS2_GEMV_CUBLAS_DETERMINE_ARGS(LAYOUTA);                                                                 \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                       \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                                \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasZgemv(                                                                    \
          s.handle, transa, M, N, reinterpret_cast<const cuDoubleComplex*>(&alpha),                                    \
          reinterpret_cast<const cuDoubleComplex*>(A.data()), LDA, reinterpret_cast<const cuDoubleComplex*>(X.data()), \
          one, reinterpret_cast<const cuDoubleComplex*>(&beta), reinterpret_cast<cuDoubleComplex*>(Y.data()), one));   \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                               \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

#define KOKKOSBLAS2_CGEMV_CUBLAS(LAYOUTA, LAYOUTX, LAYOUTY, ETI_SPEC_AVAIL)                                            \
  template <>                                                                                                          \
  struct GEMV<                                                                                                         \
      Kokkos::Cuda,                                                                                                    \
      Kokkos::View<const Kokkos::complex<float>**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,   \
      Kokkos::View<const Kokkos::complex<float>*, LAYOUTX, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,    \
      Kokkos::View<Kokkos::complex<float>*, LAYOUTY, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true,    \
      ETI_SPEC_AVAIL> {                                                                                                \
    typedef Kokkos::complex<float> SCALAR;                                                                             \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;   \
    typedef Kokkos::View<const SCALAR*, LAYOUTX, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XViewType;    \
    typedef Kokkos::View<SCALAR*, LAYOUTY, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > YViewType;          \
                                                                                                                       \
    static void gemv(const Kokkos::Cuda& space, const char trans[], typename AViewType::const_value_type& alpha,       \
                     const AViewType& A, const XViewType& X, typename YViewType::const_value_type& beta,               \
                     const YViewType& Y) {                                                                             \
      Kokkos::Profiling::pushRegion("KokkosBlas::gemv[TPL_CUBLAS,complex<float>]");                                    \
      KOKKOSBLAS2_GEMV_CUBLAS_DETERMINE_ARGS(LAYOUTA);                                                                 \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                       \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                                \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                                \
          cublasCgemv(s.handle, transa, M, N, reinterpret_cast<const cuComplex*>(&alpha),                              \
                      reinterpret_cast<const cuComplex*>(A.data()), LDA, reinterpret_cast<const cuComplex*>(X.data()), \
                      one, reinterpret_cast<const cuComplex*>(&beta), reinterpret_cast<cuComplex*>(Y.data()), one));   \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                               \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

KOKKOSBLAS2_DGEMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_DGEMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_DGEMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_DGEMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS2_SGEMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_SGEMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_SGEMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_SGEMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS2_ZGEMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_ZGEMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_ZGEMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_ZGEMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS2_CGEMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_CGEMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS2_CGEMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS2_CGEMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_GEMV_ROCBLAS_DETERMINE_ARGS(LAYOUT)                      \
  bool A_is_lr      = std::is_same<Kokkos::LayoutRight, LAYOUT>::value;      \
  const int M       = static_cast<int>(A_is_lr ? A.extent(1) : A.extent(0)); \
  const int N       = static_cast<int>(A_is_lr ? A.extent(0) : A.extent(1)); \
  constexpr int one = 1;                                                     \
  const int LDA     = A_is_lr ? A.stride(0) : A.stride(1);                   \
                                                                             \
  rocblas_operation transa;                                                  \
  if ((trans[0] == 'N') || (trans[0] == 'n'))                                \
    transa = A_is_lr ? rocblas_operation_transpose : rocblas_operation_none; \
  else if ((trans[0] == 'T') || (trans[0] == 't'))                           \
    transa = A_is_lr ? rocblas_operation_none : rocblas_operation_transpose; \
  else {                                                                     \
    if (A_is_lr)                                                             \
      throw std::runtime_error(                                              \
          "Error: rocBLAS Xgemv conjugate transpose requires LayoutLeft "    \
          "matrix.");                                                        \
    transa = rocblas_operation_conjugate_transpose;                          \
  }

#define KOKKOSBLAS2_DGEMV_ROCBLAS(LAYOUT, ETI_SPEC_AVAIL)                                                           \
  template <>                                                                                                       \
  struct GEMV<                                                                                                      \
      Kokkos::HIP, Kokkos::View<const double**, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,     \
      Kokkos::View<const double*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                   \
      Kokkos::View<double*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true, ETI_SPEC_AVAIL> { \
    typedef double SCALAR;                                                                                          \
    typedef Kokkos::View<const SCALAR**, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;  \
    typedef Kokkos::View<const SCALAR*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XViewType;   \
    typedef Kokkos::View<SCALAR*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > YViewType;         \
                                                                                                                    \
    static void gemv(const Kokkos::HIP& space, const char trans[], typename AViewType::const_value_type& alpha,     \
                     const AViewType& A, const XViewType& X, typename YViewType::const_value_type& beta,            \
                     const YViewType& Y) {                                                                          \
      Kokkos::Profiling::pushRegion("KokkosBlas::gemv[TPL_ROCBLAS,double]");                                        \
      KOKKOSBLAS2_GEMV_ROCBLAS_DETERMINE_ARGS(LAYOUT);                                                              \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                      \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, space.hip_stream()));                          \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(                                                                            \
          rocblas_dgemv(s.handle, transa, M, N, &alpha, A.data(), LDA, X.data(), one, &beta, Y.data(), one));       \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, NULL));                                        \
      Kokkos::Profiling::popRegion();                                                                               \
    }                                                                                                               \
  };

#define KOKKOSBLAS2_SGEMV_ROCBLAS(LAYOUT, ETI_SPEC_AVAIL)                                                              \
  template <>                                                                                                          \
  struct GEMV<Kokkos::HIP, Kokkos::View<const float**, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
              Kokkos::View<const float*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,               \
              Kokkos::View<float*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true,               \
              ETI_SPEC_AVAIL> {                                                                                        \
    typedef float SCALAR;                                                                                              \
    typedef Kokkos::View<const SCALAR**, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;     \
    typedef Kokkos::View<const SCALAR*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XViewType;      \
    typedef Kokkos::View<SCALAR*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > YViewType;            \
                                                                                                                       \
    static void gemv(const Kokkos::HIP& space, const char trans[], typename AViewType::const_value_type& alpha,        \
                     const AViewType& A, const XViewType& X, typename YViewType::const_value_type& beta,               \
                     const YViewType& Y) {                                                                             \
      Kokkos::Profiling::pushRegion("KokkosBlas::gemv[TPL_ROCBLAS,float]");                                            \
      KOKKOSBLAS2_GEMV_ROCBLAS_DETERMINE_ARGS(LAYOUT);                                                                 \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                         \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, space.hip_stream()));                             \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(                                                                               \
          rocblas_sgemv(s.handle, transa, M, N, &alpha, A.data(), LDA, X.data(), one, &beta, Y.data(), one));          \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, NULL));                                           \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

#define KOKKOSBLAS2_ZGEMV_ROCBLAS(LAYOUT, ETI_SPEC_AVAIL)                                                             \
  template <>                                                                                                         \
  struct GEMV<                                                                                                        \
      Kokkos::HIP,                                                                                                    \
      Kokkos::View<const Kokkos::complex<double>**, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,   \
      Kokkos::View<const Kokkos::complex<double>*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,    \
      Kokkos::View<Kokkos::complex<double>*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true,    \
      ETI_SPEC_AVAIL> {                                                                                               \
    typedef Kokkos::complex<double> SCALAR;                                                                           \
    typedef Kokkos::View<const SCALAR**, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;    \
    typedef Kokkos::View<const SCALAR*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XViewType;     \
    typedef Kokkos::View<SCALAR*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > YViewType;           \
                                                                                                                      \
    static void gemv(const Kokkos::HIP& space, const char trans[], typename AViewType::const_value_type& alpha,       \
                     const AViewType& A, const XViewType& X, typename YViewType::const_value_type& beta,              \
                     const YViewType& Y) {                                                                            \
      Kokkos::Profiling::pushRegion("KokkosBlas::gemv[TPL_ROCBLAS,complex<double>]");                                 \
      KOKKOSBLAS2_GEMV_ROCBLAS_DETERMINE_ARGS(LAYOUT);                                                                \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                        \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, space.hip_stream()));                            \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_zgemv(s.handle, transa, M, N,                                         \
                                                      reinterpret_cast<const rocblas_double_complex*>(&alpha),        \
                                                      reinterpret_cast<const rocblas_double_complex*>(A.data()), LDA, \
                                                      reinterpret_cast<const rocblas_double_complex*>(X.data()), one, \
                                                      reinterpret_cast<const rocblas_double_complex*>(&beta),         \
                                                      reinterpret_cast<rocblas_double_complex*>(Y.data()), one));     \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, NULL));                                          \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS2_CGEMV_ROCBLAS(LAYOUT, ETI_SPEC_AVAIL)                                                            \
  template <>                                                                                                        \
  struct GEMV<                                                                                                       \
      Kokkos::HIP,                                                                                                   \
      Kokkos::View<const Kokkos::complex<float>**, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,   \
      Kokkos::View<const Kokkos::complex<float>*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,    \
      Kokkos::View<Kokkos::complex<float>*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true,    \
      ETI_SPEC_AVAIL> {                                                                                              \
    typedef Kokkos::complex<float> SCALAR;                                                                           \
    typedef Kokkos::View<const SCALAR**, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;   \
    typedef Kokkos::View<const SCALAR*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > XViewType;    \
    typedef Kokkos::View<SCALAR*, LAYOUT, Kokkos::HIP, Kokkos::MemoryTraits<Kokkos::Unmanaged> > YViewType;          \
                                                                                                                     \
    static void gemv(const Kokkos::HIP& space, const char trans[], typename AViewType::const_value_type& alpha,      \
                     const AViewType& A, const XViewType& X, typename YViewType::const_value_type& beta,             \
                     const YViewType& Y) {                                                                           \
      Kokkos::Profiling::pushRegion("KokkosBlas::gemv[TPL_ROCBLAS,complex<float>]");                                 \
      KOKKOSBLAS2_GEMV_ROCBLAS_DETERMINE_ARGS(LAYOUT);                                                               \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                       \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, space.hip_stream()));                           \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_cgemv(s.handle, transa, M, N,                                        \
                                                      reinterpret_cast<const rocblas_float_complex*>(&alpha),        \
                                                      reinterpret_cast<const rocblas_float_complex*>(A.data()), LDA, \
                                                      reinterpret_cast<const rocblas_float_complex*>(X.data()), one, \
                                                      reinterpret_cast<const rocblas_float_complex*>(&beta),         \
                                                      reinterpret_cast<rocblas_float_complex*>(Y.data()), one));     \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, NULL));                                         \
      Kokkos::Profiling::popRegion();                                                                                \
    }                                                                                                                \
  };

KOKKOSBLAS2_DGEMV_ROCBLAS(Kokkos::LayoutLeft, true)
KOKKOSBLAS2_DGEMV_ROCBLAS(Kokkos::LayoutLeft, false)
KOKKOSBLAS2_DGEMV_ROCBLAS(Kokkos::LayoutRight, true)
KOKKOSBLAS2_DGEMV_ROCBLAS(Kokkos::LayoutRight, false)

KOKKOSBLAS2_SGEMV_ROCBLAS(Kokkos::LayoutLeft, true)
KOKKOSBLAS2_SGEMV_ROCBLAS(Kokkos::LayoutLeft, false)
KOKKOSBLAS2_SGEMV_ROCBLAS(Kokkos::LayoutRight, true)
KOKKOSBLAS2_SGEMV_ROCBLAS(Kokkos::LayoutRight, false)

KOKKOSBLAS2_ZGEMV_ROCBLAS(Kokkos::LayoutLeft, true)
KOKKOSBLAS2_ZGEMV_ROCBLAS(Kokkos::LayoutLeft, false)
KOKKOSBLAS2_ZGEMV_ROCBLAS(Kokkos::LayoutRight, true)
KOKKOSBLAS2_ZGEMV_ROCBLAS(Kokkos::LayoutRight, false)

KOKKOSBLAS2_CGEMV_ROCBLAS(Kokkos::LayoutLeft, true)
KOKKOSBLAS2_CGEMV_ROCBLAS(Kokkos::LayoutLeft, false)
KOKKOSBLAS2_CGEMV_ROCBLAS(Kokkos::LayoutRight, true)
KOKKOSBLAS2_CGEMV_ROCBLAS(Kokkos::LayoutRight, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCBLAS

// ONEMKL
#if defined(KOKKOSKERNELS_ENABLE_TPL_MKL) && defined(KOKKOS_ENABLE_SYCL)
#include <mkl.h>
#include <oneapi/mkl/blas.hpp>
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

inline oneapi::mkl::transpose mode_kk_to_onemkl(char mode_kk) {
  switch (toupper(mode_kk)) {
    case 'N': return oneapi::mkl::transpose::nontrans;
    case 'T': return oneapi::mkl::transpose::trans;
    case 'C': return oneapi::mkl::transpose::conjtrans;
    default:;
  }
  throw std::invalid_argument("Invalid mode for oneMKL (should be one of N, T, C)");
}

template <typename T, bool is_complex = false>
struct kokkos_to_std_type_map {
  using type = T;
};

// e.g., map Kokkos::complex<float> to std::complex<float>
template <typename T>
struct kokkos_to_std_type_map<T, true> {
  using type = std::complex<typename KokkosKernels::ArithTraits<T>::mag_type>;
};

#define KOKKOSBLAS2_GEMV_ONEMKL(SCALAR, LAYOUT, ETI_SPEC_AVAIL)                                                         \
  template <>                                                                                                           \
  struct GEMV<                                                                                                          \
      EXEC_SPACE, Kokkos::View<const SCALAR**, LAYOUT, Kokkos::SYCL, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
      Kokkos::View<const SCALAR*, LAYOUT, Kokkos::SYCL, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                      \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::SYCL, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true, ETI_SPEC_AVAIL> {    \
    using mem_traits = Kokkos::MemoryTraits<Kokkos::Unmanaged>;                                                         \
    using AViewType  = Kokkos::View<const SCALAR**, LAYOUT, EXEC_SPACE, mem_traits>;                                    \
    using XViewType  = Kokkos::View<const SCALAR*, LAYOUT, EXEC_SPACE, mem_traits>;                                     \
    using YViewType  = Kokkos::View<SCALAR*, LAYOUT, EXEC_SPACE, mem_traits>;                                           \
                                                                                                                        \
    static void gemv(const EXEC_SPACE& exec, const char kk_trans[], typename AViewType::const_value_type& alpha,        \
                     const AViewType& A, const XViewType& X, typename YViewType::const_value_type& beta,                \
                     const YViewType& Y) {                                                                              \
      if (beta == KokkosKernels::ArithTraits<SCALAR>::zero()) {                                                         \
        Kokkos::deep_copy(Y, KokkosKernels::ArithTraits<SCALAR>::zero());                                               \
      }                                                                                                                 \
                                                                                                                        \
      bool row_major               = std::is_same<Kokkos::LayoutRight, LAYOUT>::value;                                  \
      const std::int64_t M         = A.extent(0);                                                                       \
      const std::int64_t N         = A.extent(1);                                                                       \
      oneapi::mkl::transpose trans = mode_kk_to_onemkl(kk_trans[0]);                                                    \
      const std::int64_t LDA       = row_major ? A.stride(0) : A.stride(1);                                             \
      std::string label            = "KokkosBlas::gemv[TPL_ONEMKL," + KokkosKernels::ArithTraits<SCALAR>::name() + "]"; \
                                                                                                                        \
      Kokkos::Profiling::pushRegion(label);                                                                             \
      using mag_type    = kokkos_to_std_type_map<SCALAR, KokkosKernels::ArithTraits<SCALAR>::is_complex>::type;         \
      const mag_type* a = reinterpret_cast<const mag_type*>(A.data());                                                  \
      const mag_type* x = reinterpret_cast<const mag_type*>(X.data());                                                  \
      mag_type* y       = reinterpret_cast<mag_type*>(Y.data());                                                        \
      if (row_major) {                                                                                                  \
        oneapi::mkl::blas::row_major::gemv(exec.sycl_queue(), trans, M, N, alpha, a, LDA, x, 1, beta, y, 1);            \
      } else {                                                                                                          \
        oneapi::mkl::blas::column_major::gemv(exec.sycl_queue(), trans, M, N, alpha, a, LDA, x, 1, beta, y, 1);         \
      }                                                                                                                 \
      Kokkos::Profiling::popRegion();                                                                                   \
    }                                                                                                                   \
  };

KOKKOSBLAS2_GEMV_ONEMKL(float, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_GEMV_ONEMKL(float, Kokkos::LayoutRight, true)
KOKKOSBLAS2_GEMV_ONEMKL(double, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_GEMV_ONEMKL(double, Kokkos::LayoutRight, true)
KOKKOSBLAS2_GEMV_ONEMKL(Kokkos::complex<float>, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_GEMV_ONEMKL(Kokkos::complex<float>, Kokkos::LayoutRight, true)
KOKKOSBLAS2_GEMV_ONEMKL(Kokkos::complex<double>, Kokkos::LayoutLeft, true)
KOKKOSBLAS2_GEMV_ONEMKL(Kokkos::complex<double>, Kokkos::LayoutRight, true)
}  // namespace Impl
}  // namespace KokkosBlas
#endif

#endif
