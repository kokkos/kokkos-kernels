// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS3_TRSM_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS3_TRSM_TPL_SPEC_DECL_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#include "KokkosBlas_Host_tpl.hpp"

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS3_DTRSM_BLAS(EXEC_SPACE, LAYOUTA, LAYOUTB, ETI_SPEC_AVAIL)                                           \
  template <>                                                                                                          \
  struct TRSM<EXEC_SPACE, Kokkos::View<const double**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
              Kokkos::View<double**, LAYOUTB, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true,             \
              ETI_SPEC_AVAIL> {                                                                                        \
    typedef double SCALAR;                                                                                             \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;     \
    typedef Kokkos::View<SCALAR**, LAYOUTB, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType;           \
                                                                                                                       \
    static void trsm(const EXEC_SPACE& /*space*/, const char side[], const char uplo[], const char trans[],            \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,               \
                     const BViewType& B) {                                                                             \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_BLAS,double]");                                              \
      const int M = static_cast<int>(B.extent(0));                                                                     \
      const int N = static_cast<int>(B.extent(1));                                                                     \
                                                                                                                       \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                                 \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                                 \
                                                                                                                       \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                                 \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                                 \
                                                                                                                       \
      char side_;                                                                                                      \
      char uplo_;                                                                                                      \
                                                                                                                       \
      if (A_is_ll) {                                                                                                   \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                      \
          side_ = 'L';                                                                                                 \
        else                                                                                                           \
          side_ = 'R';                                                                                                 \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                      \
          uplo_ = 'L';                                                                                                 \
        else                                                                                                           \
          uplo_ = 'U';                                                                                                 \
      } else {                                                                                                         \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                      \
          side_ = 'R';                                                                                                 \
        else                                                                                                           \
          side_ = 'L';                                                                                                 \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                      \
          uplo_ = 'U';                                                                                                 \
        else                                                                                                           \
          uplo_ = 'L';                                                                                                 \
      }                                                                                                                \
                                                                                                                       \
      if (A_is_ll)                                                                                                     \
        HostBlas<double>::trsm(side_, uplo_, trans[0], diag[0], M, N, alpha, A.data(), LDA, B.data(), LDB);            \
      else                                                                                                             \
        HostBlas<double>::trsm(side_, uplo_, trans[0], diag[0], N, M, alpha, A.data(), LDA, B.data(), LDB);            \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

#define KOKKOSBLAS3_STRSM_BLAS(EXEC_SPACE, LAYOUTA, LAYOUTB, ETI_SPEC_AVAIL)                                          \
  template <>                                                                                                         \
  struct TRSM<EXEC_SPACE, Kokkos::View<const float**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
              Kokkos::View<float**, LAYOUTB, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true,             \
              ETI_SPEC_AVAIL> {                                                                                       \
    typedef float SCALAR;                                                                                             \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;    \
    typedef Kokkos::View<SCALAR**, LAYOUTB, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType;          \
                                                                                                                      \
    static void trsm(const EXEC_SPACE& /*space*/, const char side[], const char uplo[], const char trans[],           \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,              \
                     const BViewType& B) {                                                                            \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_BLAS,float]");                                              \
      const int M = static_cast<int>(B.extent(0));                                                                    \
      const int N = static_cast<int>(B.extent(1));                                                                    \
                                                                                                                      \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                                \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                                \
                                                                                                                      \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                                \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                                \
                                                                                                                      \
      char side_;                                                                                                     \
      char uplo_;                                                                                                     \
                                                                                                                      \
      if (A_is_ll) {                                                                                                  \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                     \
          side_ = 'L';                                                                                                \
        else                                                                                                          \
          side_ = 'R';                                                                                                \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                     \
          uplo_ = 'L';                                                                                                \
        else                                                                                                          \
          uplo_ = 'U';                                                                                                \
      } else {                                                                                                        \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                     \
          side_ = 'R';                                                                                                \
        else                                                                                                          \
          side_ = 'L';                                                                                                \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                     \
          uplo_ = 'U';                                                                                                \
        else                                                                                                          \
          uplo_ = 'L';                                                                                                \
      }                                                                                                               \
                                                                                                                      \
      if (A_is_ll)                                                                                                    \
        HostBlas<float>::trsm(side_, uplo_, trans[0], diag[0], M, N, alpha, A.data(), LDA, B.data(), LDB);            \
      else                                                                                                            \
        HostBlas<float>::trsm(side_, uplo_, trans[0], diag[0], N, M, alpha, A.data(), LDA, B.data(), LDB);            \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS3_ZTRSM_BLAS(EXEC_SPACE, LAYOUTA, LAYOUTB, ETI_SPEC_AVAIL)                                        \
  template <>                                                                                                       \
  struct TRSM<                                                                                                      \
      EXEC_SPACE,                                                                                                   \
      Kokkos::View<const Kokkos::complex<double>**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<Kokkos::complex<double>**, LAYOUTB, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true, \
      ETI_SPEC_AVAIL> {                                                                                             \
    typedef Kokkos::complex<double> SCALAR;                                                                         \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;  \
    typedef Kokkos::View<SCALAR**, LAYOUTB, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType;        \
                                                                                                                    \
    static void trsm(const EXEC_SPACE& /*space*/, const char side[], const char uplo[], const char trans[],         \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,            \
                     const BViewType& B) {                                                                          \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_BLAS,complex<double>]");                                  \
      const int M = static_cast<int>(B.extent(0));                                                                  \
      const int N = static_cast<int>(B.extent(1));                                                                  \
                                                                                                                    \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                              \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                              \
                                                                                                                    \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                              \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                              \
                                                                                                                    \
      char side_;                                                                                                   \
      char uplo_;                                                                                                   \
                                                                                                                    \
      if (A_is_ll) {                                                                                                \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                   \
          side_ = 'L';                                                                                              \
        else                                                                                                        \
          side_ = 'R';                                                                                              \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                   \
          uplo_ = 'L';                                                                                              \
        else                                                                                                        \
          uplo_ = 'U';                                                                                              \
      } else {                                                                                                      \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                   \
          side_ = 'R';                                                                                              \
        else                                                                                                        \
          side_ = 'L';                                                                                              \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                   \
          uplo_ = 'U';                                                                                              \
        else                                                                                                        \
          uplo_ = 'L';                                                                                              \
      }                                                                                                             \
                                                                                                                    \
      const std::complex<double> alpha_val = alpha;                                                                 \
      if (A_is_ll)                                                                                                  \
        HostBlas<std::complex<double> >::trsm(side_, uplo_, trans[0], diag[0], M, N, alpha_val,                     \
                                              reinterpret_cast<const std::complex<double>*>(A.data()), LDA,         \
                                              reinterpret_cast<std::complex<double>*>(B.data()), LDB);              \
      else                                                                                                          \
        HostBlas<std::complex<double> >::trsm(side_, uplo_, trans[0], diag[0], N, M, alpha_val,                     \
                                              reinterpret_cast<const std::complex<double>*>(A.data()), LDA,         \
                                              reinterpret_cast<std::complex<double>*>(B.data()), LDB);              \
      Kokkos::Profiling::popRegion();                                                                               \
    }                                                                                                               \
  };

#define KOKKOSBLAS3_CTRSM_BLAS(EXEC_SPACE, LAYOUTA, LAYOUTB, ETI_SPEC_AVAIL)                                       \
  template <>                                                                                                      \
  struct TRSM<                                                                                                     \
      EXEC_SPACE,                                                                                                  \
      Kokkos::View<const Kokkos::complex<float>**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<Kokkos::complex<float>**, LAYOUTB, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true, \
      ETI_SPEC_AVAIL> {                                                                                            \
    typedef Kokkos::complex<float> SCALAR;                                                                         \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
    typedef Kokkos::View<SCALAR**, LAYOUTB, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType;       \
                                                                                                                   \
    static void trsm(const EXEC_SPACE& /*space*/, const char side[], const char uplo[], const char trans[],        \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,           \
                     const BViewType& B) {                                                                         \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_BLAS,complex<float>]");                                  \
      const int M = static_cast<int>(B.extent(0));                                                                 \
      const int N = static_cast<int>(B.extent(1));                                                                 \
                                                                                                                   \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                             \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                             \
                                                                                                                   \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                             \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                             \
                                                                                                                   \
      char side_;                                                                                                  \
      char uplo_;                                                                                                  \
                                                                                                                   \
      if (A_is_ll) {                                                                                               \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                  \
          side_ = 'L';                                                                                             \
        else                                                                                                       \
          side_ = 'R';                                                                                             \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                  \
          uplo_ = 'L';                                                                                             \
        else                                                                                                       \
          uplo_ = 'U';                                                                                             \
      } else {                                                                                                     \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                  \
          side_ = 'R';                                                                                             \
        else                                                                                                       \
          side_ = 'L';                                                                                             \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                  \
          uplo_ = 'U';                                                                                             \
        else                                                                                                       \
          uplo_ = 'L';                                                                                             \
      }                                                                                                            \
                                                                                                                   \
      const std::complex<float> alpha_val = alpha;                                                                 \
      if (A_is_ll)                                                                                                 \
        HostBlas<std::complex<float> >::trsm(side_, uplo_, trans[0], diag[0], M, N, alpha_val,                     \
                                             reinterpret_cast<const std::complex<float>*>(A.data()), LDA,          \
                                             reinterpret_cast<std::complex<float>*>(B.data()), LDB);               \
      else                                                                                                         \
        HostBlas<std::complex<float> >::trsm(side_, uplo_, trans[0], diag[0], N, M, alpha_val,                     \
                                             reinterpret_cast<const std::complex<float>*>(A.data()), LDA,          \
                                             reinterpret_cast<std::complex<float>*>(B.data()), LDB);               \
      Kokkos::Profiling::popRegion();                                                                              \
    }                                                                                                              \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS3_STRSM_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_STRSM_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_STRSM_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_STRSM_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS3_CTRSM_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_CTRSM_BLAS(Kokkos::Serial, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_CTRSM_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_CTRSM_BLAS(Kokkos::Serial, Kokkos::LayoutRight, Kokkos::LayoutRight, false)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS3_STRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_STRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_STRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_STRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS3_CTRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_CTRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_CTRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_CTRSM_BLAS(Kokkos::OpenMP, Kokkos::LayoutRight, Kokkos::LayoutRight, false)
#endif

#ifdef KOKKOS_ENABLE_THREADS
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS3_STRSM_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_STRSM_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_STRSM_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_STRSM_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS3_CTRSM_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_CTRSM_BLAS(Kokkos::Threads, Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_CTRSM_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_CTRSM_BLAS(Kokkos::Threads, Kokkos::LayoutRight, Kokkos::LayoutRight, false)
#endif

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_BLAS

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS3_DTRSM_CUBLAS(LAYOUTA, LAYOUTB, ETI_SPEC_AVAIL)                                                     \
  template <>                                                                                                          \
  struct TRSM<                                                                                                         \
      Kokkos::Cuda, Kokkos::View<const double**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,     \
      Kokkos::View<double**, LAYOUTB, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true, ETI_SPEC_AVAIL> { \
    typedef double SCALAR;                                                                                             \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;   \
    typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType;         \
                                                                                                                       \
    static void trsm(const Kokkos::Cuda& space, const char side[], const char uplo[], const char trans[],              \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,               \
                     const BViewType& B) {                                                                             \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_CUBLAS,double]");                                            \
      const int M = static_cast<int>(B.extent(0));                                                                     \
      const int N = static_cast<int>(B.extent(1));                                                                     \
                                                                                                                       \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                                 \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                                 \
                                                                                                                       \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                                 \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                                 \
                                                                                                                       \
      cublasSideMode_t side_;                                                                                          \
      cublasFillMode_t uplo_;                                                                                          \
      cublasOperation_t trans_;                                                                                        \
      cublasDiagType_t diag_;                                                                                          \
                                                                                                                       \
      if (A_is_ll) {                                                                                                   \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                      \
          side_ = CUBLAS_SIDE_LEFT;                                                                                    \
        else                                                                                                           \
          side_ = CUBLAS_SIDE_RIGHT;                                                                                   \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                      \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                              \
        else                                                                                                           \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                              \
      } else {                                                                                                         \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                      \
          side_ = CUBLAS_SIDE_RIGHT;                                                                                   \
        else                                                                                                           \
          side_ = CUBLAS_SIDE_LEFT;                                                                                    \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                      \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                              \
        else                                                                                                           \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                              \
      }                                                                                                                \
                                                                                                                       \
      if ((trans[0] == 'N') || (trans[0] == 'n'))                                                                      \
        trans_ = CUBLAS_OP_N;                                                                                          \
      else if ((trans[0] == 'T') || (trans[0] == 't'))                                                                 \
        trans_ = CUBLAS_OP_T;                                                                                          \
      else                                                                                                             \
        trans_ = CUBLAS_OP_C;                                                                                          \
      if ((diag[0] == 'U') || (diag[0] == 'u'))                                                                        \
        diag_ = CUBLAS_DIAG_UNIT;                                                                                      \
      else                                                                                                             \
        diag_ = CUBLAS_DIAG_NON_UNIT;                                                                                  \
                                                                                                                       \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                       \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                                \
      if (A_is_ll) {                                                                                                   \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                              \
            cublasDtrsm(s.handle, side_, uplo_, trans_, diag_, M, N, &alpha, A.data(), LDA, B.data(), LDB));           \
      } else {                                                                                                         \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                              \
            cublasDtrsm(s.handle, side_, uplo_, trans_, diag_, N, M, &alpha, A.data(), LDA, B.data(), LDB));           \
      }                                                                                                                \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                               \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

#define KOKKOSBLAS3_STRSM_CUBLAS(LAYOUTA, LAYOUTB, ETI_SPEC_AVAIL)                                                    \
  template <>                                                                                                         \
  struct TRSM<                                                                                                        \
      Kokkos::Cuda, Kokkos::View<const float**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >,     \
      Kokkos::View<float**, LAYOUTB, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true, ETI_SPEC_AVAIL> { \
    typedef float SCALAR;                                                                                             \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;  \
    typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType;        \
                                                                                                                      \
    static void trsm(const Kokkos::Cuda& space, const char side[], const char uplo[], const char trans[],             \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,              \
                     const BViewType& B) {                                                                            \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_CUBLAS,float]");                                            \
      const int M = static_cast<int>(B.extent(0));                                                                    \
      const int N = static_cast<int>(B.extent(1));                                                                    \
                                                                                                                      \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                                \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                                \
                                                                                                                      \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                                \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                                \
                                                                                                                      \
      cublasSideMode_t side_;                                                                                         \
      cublasFillMode_t uplo_;                                                                                         \
      cublasOperation_t trans_;                                                                                       \
      cublasDiagType_t diag_;                                                                                         \
                                                                                                                      \
      if (A_is_ll) {                                                                                                  \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                     \
          side_ = CUBLAS_SIDE_LEFT;                                                                                   \
        else                                                                                                          \
          side_ = CUBLAS_SIDE_RIGHT;                                                                                  \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                     \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                             \
        else                                                                                                          \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                             \
      } else {                                                                                                        \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                     \
          side_ = CUBLAS_SIDE_RIGHT;                                                                                  \
        else                                                                                                          \
          side_ = CUBLAS_SIDE_LEFT;                                                                                   \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                     \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                             \
        else                                                                                                          \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                             \
      }                                                                                                               \
                                                                                                                      \
      if ((trans[0] == 'N') || (trans[0] == 'n'))                                                                     \
        trans_ = CUBLAS_OP_N;                                                                                         \
      else if ((trans[0] == 'T') || (trans[0] == 't'))                                                                \
        trans_ = CUBLAS_OP_T;                                                                                         \
      else                                                                                                            \
        trans_ = CUBLAS_OP_C;                                                                                         \
      if ((diag[0] == 'U') || (diag[0] == 'u'))                                                                       \
        diag_ = CUBLAS_DIAG_UNIT;                                                                                     \
      else                                                                                                            \
        diag_ = CUBLAS_DIAG_NON_UNIT;                                                                                 \
                                                                                                                      \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                      \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                               \
      if (A_is_ll) {                                                                                                  \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                             \
            cublasStrsm(s.handle, side_, uplo_, trans_, diag_, M, N, &alpha, A.data(), LDA, B.data(), LDB));          \
      } else {                                                                                                        \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                             \
            cublasStrsm(s.handle, side_, uplo_, trans_, diag_, N, M, &alpha, A.data(), LDA, B.data(), LDB));          \
      }                                                                                                               \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                              \
                                                                                                                      \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS3_ZTRSM_CUBLAS(LAYOUTA, LAYOUTB, ETI_SPEC_AVAIL)                                                    \
  template <>                                                                                                         \
  struct TRSM<                                                                                                        \
      Kokkos::Cuda,                                                                                                   \
      Kokkos::View<const Kokkos::complex<double>**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<Kokkos::complex<double>**, LAYOUTB, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true, \
      ETI_SPEC_AVAIL> {                                                                                               \
    typedef Kokkos::complex<double> SCALAR;                                                                           \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType;  \
    typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType;        \
                                                                                                                      \
    static void trsm(const Kokkos::Cuda& space, const char side[], const char uplo[], const char trans[],             \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,              \
                     const BViewType& B) {                                                                            \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_CUBLAS,complex<double>]");                                  \
      const int M = static_cast<int>(B.extent(0));                                                                    \
      const int N = static_cast<int>(B.extent(1));                                                                    \
                                                                                                                      \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                                \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                                \
                                                                                                                      \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                                \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                                \
                                                                                                                      \
      cublasSideMode_t side_;                                                                                         \
      cublasFillMode_t uplo_;                                                                                         \
      cublasOperation_t trans_;                                                                                       \
      cublasDiagType_t diag_;                                                                                         \
                                                                                                                      \
      if (A_is_ll) {                                                                                                  \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                     \
          side_ = CUBLAS_SIDE_LEFT;                                                                                   \
        else                                                                                                          \
          side_ = CUBLAS_SIDE_RIGHT;                                                                                  \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                     \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                             \
        else                                                                                                          \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                             \
      } else {                                                                                                        \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                     \
          side_ = CUBLAS_SIDE_RIGHT;                                                                                  \
        else                                                                                                          \
          side_ = CUBLAS_SIDE_LEFT;                                                                                   \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                     \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                             \
        else                                                                                                          \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                             \
      }                                                                                                               \
                                                                                                                      \
      if ((trans[0] == 'N') || (trans[0] == 'n'))                                                                     \
        trans_ = CUBLAS_OP_N;                                                                                         \
      else if ((trans[0] == 'T') || (trans[0] == 't'))                                                                \
        trans_ = CUBLAS_OP_T;                                                                                         \
      else                                                                                                            \
        trans_ = CUBLAS_OP_C;                                                                                         \
      if ((diag[0] == 'U') || (diag[0] == 'u'))                                                                       \
        diag_ = CUBLAS_DIAG_UNIT;                                                                                     \
      else                                                                                                            \
        diag_ = CUBLAS_DIAG_NON_UNIT;                                                                                 \
                                                                                                                      \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                      \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                               \
      if (A_is_ll) {                                                                                                  \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasZtrsm(s.handle, side_, uplo_, trans_, diag_, M, N,                     \
                                                     reinterpret_cast<const cuDoubleComplex*>(&alpha),                \
                                                     reinterpret_cast<const cuDoubleComplex*>(A.data()), LDA,         \
                                                     reinterpret_cast<cuDoubleComplex*>(B.data()), LDB));             \
      } else {                                                                                                        \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasZtrsm(s.handle, side_, uplo_, trans_, diag_, N, M,                     \
                                                     reinterpret_cast<const cuDoubleComplex*>(&alpha),                \
                                                     reinterpret_cast<const cuDoubleComplex*>(A.data()), LDA,         \
                                                     reinterpret_cast<cuDoubleComplex*>(B.data()), LDB));             \
      }                                                                                                               \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                              \
                                                                                                                      \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS3_CTRSM_CUBLAS(LAYOUTA, LAYOUTB, ETI_SPEC_AVAIL)                                                   \
  template <>                                                                                                        \
  struct TRSM<                                                                                                       \
      Kokkos::Cuda,                                                                                                  \
      Kokkos::View<const Kokkos::complex<float>**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      Kokkos::View<Kokkos::complex<float>**, LAYOUTB, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true, \
      ETI_SPEC_AVAIL> {                                                                                              \
    typedef Kokkos::complex<float> SCALAR;                                                                           \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
    typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType;       \
                                                                                                                     \
    static void trsm(const Kokkos::Cuda& space, const char side[], const char uplo[], const char trans[],            \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,             \
                     const BViewType& B) {                                                                           \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_CUBLAS,complex<float>]");                                  \
      const int M = static_cast<int>(B.extent(0));                                                                   \
      const int N = static_cast<int>(B.extent(1));                                                                   \
                                                                                                                     \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                               \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                               \
                                                                                                                     \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                               \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                               \
                                                                                                                     \
      cublasSideMode_t side_;                                                                                        \
      cublasFillMode_t uplo_;                                                                                        \
      cublasOperation_t trans_;                                                                                      \
      cublasDiagType_t diag_;                                                                                        \
                                                                                                                     \
      if (A_is_ll) {                                                                                                 \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                    \
          side_ = CUBLAS_SIDE_LEFT;                                                                                  \
        else                                                                                                         \
          side_ = CUBLAS_SIDE_RIGHT;                                                                                 \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                    \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                            \
        else                                                                                                         \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                            \
      } else {                                                                                                       \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                    \
          side_ = CUBLAS_SIDE_RIGHT;                                                                                 \
        else                                                                                                         \
          side_ = CUBLAS_SIDE_LEFT;                                                                                  \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                    \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                            \
        else                                                                                                         \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                            \
      }                                                                                                              \
                                                                                                                     \
      if ((trans[0] == 'N') || (trans[0] == 'n'))                                                                    \
        trans_ = CUBLAS_OP_N;                                                                                        \
      else if ((trans[0] == 'T') || (trans[0] == 't'))                                                               \
        trans_ = CUBLAS_OP_T;                                                                                        \
      else                                                                                                           \
        trans_ = CUBLAS_OP_C;                                                                                        \
      if ((diag[0] == 'U') || (diag[0] == 'u'))                                                                      \
        diag_ = CUBLAS_DIAG_UNIT;                                                                                    \
      else                                                                                                           \
        diag_ = CUBLAS_DIAG_NON_UNIT;                                                                                \
                                                                                                                     \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                     \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                              \
      if (A_is_ll) {                                                                                                 \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasCtrsm(                                                                \
            s.handle, side_, uplo_, trans_, diag_, M, N, reinterpret_cast<const cuComplex*>(&alpha),                 \
            reinterpret_cast<const cuComplex*>(A.data()), LDA, reinterpret_cast<cuComplex*>(B.data()), LDB));        \
      } else {                                                                                                       \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasCtrsm(                                                                \
            s.handle, side_, uplo_, trans_, diag_, N, M, reinterpret_cast<const cuComplex*>(&alpha),                 \
            reinterpret_cast<const cuComplex*>(A.data()), LDA, reinterpret_cast<cuComplex*>(B.data()), LDB));        \
      }                                                                                                              \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                             \
                                                                                                                     \
      Kokkos::Profiling::popRegion();                                                                                \
    }                                                                                                                \
  };

KOKKOSBLAS3_DTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_DTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_DTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_DTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS3_STRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_STRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_STRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_STRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS3_ZTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_ZTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_ZTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_ZTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, false)

KOKKOSBLAS3_CTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, true)
KOKKOSBLAS3_CTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, false)
KOKKOSBLAS3_CTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, true)
KOKKOSBLAS3_CTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

#endif
