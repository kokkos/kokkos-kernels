#ifndef KOKKOSBLAS_GEQRF_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS_GEQRF_TPL_SPEC_DECL_HPP_

#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) && \
    defined(KOKKOSKERNELS_ENABLE_TPL_LAPACKE)
#include "KokkosBlas_Host_tpl.hpp"
#include "KokkosLapack_Host_tpl.hpp"
#include <stdio.h>

namespace KokkosBlas {
namespace Impl {

// FUNCTION

#define KOKKOSBLAS_DGEQRF_LAPACK(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL)           \
  template <class ExecSpace>                                                  \
  struct GEQRF<                                                               \
      Kokkos::View<double**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<double*, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<double*, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef double SCALAR;                                                    \
    typedef int ORDINAL;                                                      \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                   \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        TauViewType;                                                          \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WViewType;                                                            \
                                                                              \
    static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace) { \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosLapack::geqrf[TPL_LAPACK, double]");                         \
      int M           = A.extent(0);                                          \
      int N           = A.extent(1);                                          \
      bool A_is_lr    = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;    \
      const int AST   = A_is_lr ? A.stride(0) : A.stride(1),                  \
                LDA   = AST == 0 ? 1 : AST;                                   \
      const int lwork = workspace.extent(0);                                  \
      HostLapack<SCALAR>::geqrf(A_is_lr, M, N, A.data(), LDA, tau.data(),     \
                                workspace.data(), lwork);                     \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_SGEQRF_LAPACK(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL)            \
  template <class ExecSpace>                                                   \
  struct GEQRF<                                                                \
      Kokkos::View<float**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<float*, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<float*, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef float SCALAR;                                                      \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                     \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                     \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        WViewType;                                                             \
                                                                               \
    static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace) {  \
      Kokkos::Profiling::pushRegion("KokkosLapack::geqrf[TPL_LAPACK, float]"); \
      int M           = A.extent(0);                                           \
      int N           = A.extent(1);                                           \
      bool A_is_lr    = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;     \
      const int AST   = A_is_lr ? A.stride(0) : A.stride(1),                   \
                LDA   = AST == 0 ? 1 : AST;                                    \
      const int lwork = workspace.extent(0);                                   \
      HostLapack<SCALAR>::geqrf(A_is_lr, M, N, A.data(), LDA, tau.data(),      \
                                workspace.data(), lwork);                      \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS_ZGEQRF_LAPACK(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL)           \
  template <class ExecSpace>                                                  \
  struct GEQRF<Kokkos::View<Kokkos::complex<double>**, LAYOUTA,               \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<double>*, LAYOUTA,                \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<double>*, LAYOUTA,                \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               true, ETI_SPEC_AVAIL> {                                        \
    typedef Kokkos::complex<double> SCALAR;                                   \
    typedef std::complex<double> S2;                                          \
    typedef int ORDINAL;                                                      \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                   \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        TauViewType;                                                          \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WViewType;                                                            \
    static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace) { \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosLapack::geqrf[TPL_LAPACK, complex<double>]");                \
      int M           = A.extent(0);                                          \
      int N           = A.extent(1);                                          \
      bool A_is_lr    = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;    \
      const int AST   = A_is_lr ? A.stride(0) : A.stride(1),                  \
                LDA   = AST == 0 ? 1 : AST;                                   \
      const int lwork = workspace.extent(0);                                  \
      HostLapack<S2>::geqrf(A_is_lr, M, N, reinterpret_cast<S2*>(A.data()),   \
                            LDA, reinterpret_cast<S2*>(tau.data()),           \
                            reinterpret_cast<S2*>(workspace.data()), lwork);  \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_CGEQRF_LAPACK(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL)           \
  template <class ExecSpace>                                                  \
  struct GEQRF<Kokkos::View<Kokkos::complex<float>**, LAYOUTA,                \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<float>*, LAYOUTA,                 \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<float>*, LAYOUTA,                 \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               true, ETI_SPEC_AVAIL> {                                        \
    typedef Kokkos::complex<float> SCALAR;                                    \
    typedef std::complex<float> S2;                                           \
    typedef int ORDINAL;                                                      \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                   \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        TauViewType;                                                          \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WViewType;                                                            \
                                                                              \
    static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace) { \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosLapack::geqrf[TPL_LAPACK, complex<float>]");                 \
      int M           = A.extent(0);                                          \
      int N           = A.extent(1);                                          \
      bool A_is_lr    = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;    \
      const int AST   = A_is_lr ? A.stride(0) : A.stride(1),                  \
                LDA   = AST == 0 ? 1 : AST;                                   \
      const int lwork = workspace.extent(0);                                  \
      HostLapack<S2>::geqrf(A_is_lr, M, N, reinterpret_cast<S2*>(A.data()),   \
                            LDA, reinterpret_cast<S2*>(tau.data()),           \
                            reinterpret_cast<S2*>(workspace.data()), lwork);  \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

// WORKSPACE QUERIES

#define KOKKOSBLAS_DGEQRF_WORKSPACE_LAPACK(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL) \
  template <class ExecSpace>                                                  \
  struct GEQRF_WORKSPACE<                                                     \
      Kokkos::View<double**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<double*, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef double SCALAR;                                                    \
    typedef int ORDINAL;                                                      \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                   \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        TauViewType;                                                          \
                                                                              \
    static int64_t geqrf_workspace(AViewType& A, TauViewType& tau) {          \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosLapack::geqrf[TPL_LAPACK, double]");                         \
      int M         = A.extent(0);                                            \
      int N         = A.extent(1);                                            \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;      \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                    \
                LDA = AST == 0 ? 1 : AST;                                     \
      int lwork     = -1;                                                     \
      SCALAR query  = 0;                                                      \
      HostLapack<SCALAR>::geqrf(A_is_lr, M, N, A.data(), LDA, tau.data(),     \
                                &query, lwork);                               \
      Kokkos::Profiling::popRegion();                                         \
      return (int64_t)query;                                                  \
    }                                                                         \
  };

#define KOKKOSBLAS_SGEQRF_WORKSPACE_LAPACK(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL)  \
  template <class ExecSpace>                                                   \
  struct GEQRF_WORKSPACE<                                                      \
      Kokkos::View<float**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<float*, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef float SCALAR;                                                      \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                     \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
                                                                               \
    static int64_t geqrf_workspace(AViewType& A, TauViewType& tau) {           \
      Kokkos::Profiling::pushRegion("KokkosLapack::geqrf[TPL_LAPACK, float]"); \
      int M         = A.extent(0);                                             \
      int N         = A.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      int lwork     = -1;                                                      \
      SCALAR query  = 0;                                                       \
      HostLapack<SCALAR>::geqrf(A_is_lr, M, N, A.data(), LDA, tau.data(),      \
                                &query, lwork);                                \
      Kokkos::Profiling::popRegion();                                          \
      return (int64_t)query;                                                   \
    }                                                                          \
  };

#define KOKKOSBLAS_ZGEQRF_WORKSPACE_LAPACK(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL) \
  template <class ExecSpace>                                                  \
  struct GEQRF_WORKSPACE<                                                     \
      Kokkos::View<Kokkos::complex<double>**, LAYOUTA,                        \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<Kokkos::complex<double>*, LAYOUTA,                         \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef Kokkos::complex<double> SCALAR;                                   \
    typedef std::complex<double> S2;                                          \
    typedef int ORDINAL;                                                      \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                   \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        TauViewType;                                                          \
    static int64_t geqrf_workspace(AViewType& A, TauViewType& tau) {          \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosLapack::geqrf[TPL_LAPACK, complex<double>]");                \
      int M         = A.extent(0);                                            \
      int N         = A.extent(1);                                            \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;      \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                    \
                LDA = AST == 0 ? 1 : AST;                                     \
      int lwork     = -1;                                                     \
      SCALAR query  = 0;                                                      \
      HostLapack<S2>::geqrf(A_is_lr, M, N, reinterpret_cast<S2*>(A.data()),   \
                            LDA, reinterpret_cast<S2*>(tau.data()),           \
                            reinterpret_cast<S2*>(&query), lwork);            \
      Kokkos::Profiling::popRegion();                                         \
      return (int64_t)query.real();                                           \
    }                                                                         \
  };

#define KOKKOSBLAS_CGEQRF_WORKSPACE_LAPACK(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL) \
  template <class ExecSpace>                                                  \
  struct GEQRF_WORKSPACE<                                                     \
      Kokkos::View<Kokkos::complex<float>**, LAYOUTA,                         \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<Kokkos::complex<float>*, LAYOUTA,                          \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef Kokkos::complex<float> SCALAR;                                    \
    typedef std::complex<float> S2;                                           \
    typedef int ORDINAL;                                                      \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                   \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        TauViewType;                                                          \
                                                                              \
    static int64_t geqrf_workspace(AViewType& A, TauViewType& tau) {          \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosLapack::geqrf[TPL_LAPACK, complex<float>]");                 \
      int M         = A.extent(0);                                            \
      int N         = A.extent(1);                                            \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;      \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                    \
                LDA = AST == 0 ? 1 : AST;                                     \
      int lwork     = -1;                                                     \
      SCALAR query  = 0;                                                      \
      HostLapack<S2>::geqrf(A_is_lr, M, N, reinterpret_cast<S2*>(A.data()),   \
                            LDA, reinterpret_cast<S2*>(tau.data()),           \
                            reinterpret_cast<S2*>(&query), lwork);            \
      Kokkos::Profiling::popRegion();                                         \
      return (int64_t)query.real();                                           \
    }                                                                         \
  };

KOKKOSBLAS_DGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_DGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_DGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_DGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS_SGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_SGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_SGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_SGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS_ZGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_ZGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_ZGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_ZGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS_CGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_CGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_CGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_CGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS_DGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_DGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_DGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_DGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace,
                                   false)

KOKKOSBLAS_SGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_SGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_SGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_SGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace,
                                   false)

KOKKOSBLAS_ZGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_ZGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_ZGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_ZGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace,
                                   false)

KOKKOSBLAS_CGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_CGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_CGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_CGEQRF_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::HostSpace,
                                   false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // ENABLE BLAS/LAPACK

// CUSOLVER

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSOLVER
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS_DGEQRF_CUSOLVER(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL)         \
  template <class ExecSpace>                                                  \
  struct GEQRF<                                                               \
      Kokkos::View<double**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<double*, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<double*, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef double SCALAR;                                                    \
    typedef int ORDINAL;                                                      \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                   \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        TauViewType;                                                          \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WViewType;                                                            \
                                                                              \
    static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace) { \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::geqrf[TPL_CUSOLVER, double]");                         \
      int devinfo     = 0;                                                    \
      int M           = A.extent(0);                                          \
      int N           = A.extent(1);                                          \
      bool A_is_lr    = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;    \
      const int AST   = A_is_lr ? A.stride(0) : A.stride(1),                  \
                LDA   = AST == 0 ? 1 : AST;                                   \
      const int lwork = workspace.extent(0);                                  \
      KokkosBlas::Impl::CudaSolverSingleton& s =                              \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                 \
      cusolverDnDgeqrf(s.handle, M, N, A.data(), LDA, tau.data(),             \
                       workspace.data(), lwork, &devinfo);                    \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_SGEQRF_CUSOLVER(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL)         \
  template <class ExecSpace>                                                  \
  struct GEQRF<                                                               \
      Kokkos::View<float**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<float*, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<float*, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef float SCALAR;                                                     \
    typedef int ORDINAL;                                                      \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                   \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        TauViewType;                                                          \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WViewType;                                                            \
                                                                              \
    static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace) { \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::geqrf[TPL_CUSOLVER, single]");                         \
      int devinfo   = 0;                                                      \
      int M         = A.extent(0);                                            \
      int N         = A.extent(1);                                            \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;      \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                    \
                LDA = AST == 0 ? 1 : AST;                                     \
      KokkosBlas::Impl::CudaSolverSingleton& s =                              \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                 \
      const int lwork = workspace.extent(0);                                  \
      cusolverDnSgeqrf(s.handle, M, N, A.data(), LDA, tau.data(),             \
                       workspace.data(), lwork, &devinfo);                    \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_ZGEQRF_CUSOLVER(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL)         \
  template <class ExecSpace>                                                  \
  struct GEQRF<Kokkos::View<Kokkos::complex<double>**, LAYOUTA,               \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<double>*, LAYOUTA,                \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<double>*, LAYOUTA,                \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               true, ETI_SPEC_AVAIL> {                                        \
    typedef Kokkos::complex<double> SCALAR;                                   \
    typedef double PRECISION;                                                 \
    typedef int ORDINAL;                                                      \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                   \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        TauViewType;                                                          \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WViewType;                                                            \
                                                                              \
    static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace) { \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::geqrf[TPL_CUSOLVER, Kokkos::complex<double>]");        \
      int devinfo   = 0;                                                      \
      int M         = A.extent(0);                                            \
      int N         = A.extent(1);                                            \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;      \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                    \
                LDA = AST == 0 ? 1 : AST;                                     \
      KokkosBlas::Impl::CudaSolverSingleton& s =                              \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                 \
      const int lwork = workspace.extent(0);                                  \
      cusolverDnZgeqrf(s.handle, M, N,                                        \
                       reinterpret_cast<cuDoubleComplex*>(A.data()), LDA,     \
                       reinterpret_cast<cuDoubleComplex*>(tau.data()),        \
                       reinterpret_cast<cuDoubleComplex*>(workspace.data()),  \
                       lwork, &devinfo);                                      \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_CGEQRF_CUSOLVER(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL)          \
  template <class ExecSpace>                                                   \
  struct GEQRF<Kokkos::View<Kokkos::complex<float>**, LAYOUTA,                 \
                            Kokkos::Device<ExecSpace, MEMSPACE>,               \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
               Kokkos::View<Kokkos::complex<float>*, LAYOUTA,                  \
                            Kokkos::Device<ExecSpace, MEMSPACE>,               \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
               Kokkos::View<Kokkos::complex<float>*, LAYOUTA,                  \
                            Kokkos::Device<ExecSpace, MEMSPACE>,               \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
               true, ETI_SPEC_AVAIL> {                                         \
    typedef Kokkos::complex<float> SCALAR;                                     \
    typedef float PRECISION;                                                   \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                     \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                     \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        WViewType;                                                             \
                                                                               \
    static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace) {  \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::geqrf[TPL_CUSOLVER, Kokkos::complex<float>]");          \
      int devinfo   = 0;                                                       \
      int M         = A.extent(0);                                             \
      int N         = A.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      int lwork     = workspace.extent(0);                                     \
      KokkosBlas::Impl::CudaSolverSingleton& s =                               \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                  \
      cusolverDnCgeqrf(s.handle, M, N, reinterpret_cast<cuComplex*>(A.data()), \
                       LDA, reinterpret_cast<cuComplex*>(tau.data()),          \
                       reinterpret_cast<cuComplex*>(workspace.data()), lwork,  \
                       &devinfo);                                              \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

// WORKSPACE_QUERIES

#define KOKKOSBLAS_DGEQRF_WORKSPACE_CUSOLVER(LAYOUTA, MEMSPACE,            \
                                             ETI_SPEC_AVAIL)               \
  template <class ExecSpace>                                               \
  struct GEQRF_WORKSPACE<                                                  \
      Kokkos::View<double**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<double*, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      true, ETI_SPEC_AVAIL> {                                              \
    typedef double SCALAR;                                                 \
    typedef int ORDINAL;                                                   \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                \
                         Kokkos::Device<ExecSpace, MEMSPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >         \
        AViewType;                                                         \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                 \
                         Kokkos::Device<ExecSpace, MEMSPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >         \
        TauViewType;                                                       \
                                                                           \
    static int64_t geqrf_workspace(AViewType& A, TauViewType& tau) {       \
      Kokkos::Profiling::pushRegion(                                       \
          "KokkosBlas::geqrf[TPL_CUSOLVER, double]");                      \
      int M         = A.extent(0);                                         \
      int N         = A.extent(1);                                         \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;   \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                 \
                LDA = AST == 0 ? 1 : AST;                                  \
      int lwork     = 0;                                                   \
      KokkosBlas::Impl::CudaSolverSingleton& s =                           \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();              \
      cusolverDnDgeqrf_bufferSize(s.handle, M, N, A.data(), LDA, &lwork);  \
      Kokkos::Profiling::popRegion();                                      \
      return (int64_t)lwork;                                               \
    }                                                                      \
  };

#define KOKKOSBLAS_SGEQRF_WORKSPACE_CUSOLVER(LAYOUTA, MEMSPACE,           \
                                             ETI_SPEC_AVAIL)              \
  template <class ExecSpace>                                              \
  struct GEQRF_WORKSPACE<                                                 \
      Kokkos::View<float**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<float*, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      true, ETI_SPEC_AVAIL> {                                             \
    typedef float SCALAR;                                                 \
    typedef int ORDINAL;                                                  \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,             \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >        \
        AViewType;                                                        \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                \
                         Kokkos::Device<ExecSpace, MEMSPACE>,             \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >        \
        TauViewType;                                                      \
                                                                          \
    static int64_t geqrf_workspace(AViewType& A, TauViewType& tau) {      \
      Kokkos::Profiling::pushRegion(                                      \
          "KokkosBlas::geqrf[TPL_CUSOLVER, single]");                     \
      int M         = A.extent(0);                                        \
      int N         = A.extent(1);                                        \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;  \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                \
                LDA = AST == 0 ? 1 : AST;                                 \
      KokkosBlas::Impl::CudaSolverSingleton& s =                          \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();             \
      int lwork = 0;                                                      \
      cusolverDnSgeqrf_bufferSize(s.handle, M, N, A.data(), LDA, &lwork); \
      Kokkos::Profiling::popRegion();                                     \
      return (int64_t)lwork;                                              \
    }                                                                     \
  };

#define KOKKOSBLAS_ZGEQRF_WORKSPACE_CUSOLVER(LAYOUTA, MEMSPACE,              \
                                             ETI_SPEC_AVAIL)                 \
  template <class ExecSpace>                                                 \
  struct GEQRF_WORKSPACE<                                                    \
      Kokkos::View<Kokkos::complex<double>**, LAYOUTA,                       \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<Kokkos::complex<double>*, LAYOUTA,                        \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      true, ETI_SPEC_AVAIL> {                                                \
    typedef Kokkos::complex<double> SCALAR;                                  \
    typedef double PRECISION;                                                \
    typedef int ORDINAL;                                                     \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                  \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >           \
        AViewType;                                                           \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                   \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >           \
        TauViewType;                                                         \
                                                                             \
    static int64_t geqrf_workspace(AViewType& A, TauViewType& tau) {         \
      Kokkos::Profiling::pushRegion(                                         \
          "KokkosBlas::geqrf[TPL_CUSOLVER, Kokkos::complex<double>]");       \
      int M         = A.extent(0);                                           \
      int N         = A.extent(1);                                           \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;     \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                   \
                LDA = AST == 0 ? 1 : AST;                                    \
      KokkosBlas::Impl::CudaSolverSingleton& s =                             \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                \
      int lwork = 0;                                                         \
      cusolverDnZgeqrf_bufferSize(                                           \
          s.handle, M, N, reinterpret_cast<cuDoubleComplex*>(A.data()), LDA, \
          &lwork);                                                           \
      Kokkos::Profiling::popRegion();                                        \
      return (int64_t)lwork;                                                 \
    }                                                                        \
  };

#define KOKKOSBLAS_CGEQRF_WORKSPACE_CUSOLVER(LAYOUTA, MEMSPACE,                \
                                             ETI_SPEC_AVAIL)                   \
  template <class ExecSpace>                                                   \
  struct GEQRF_WORKSPACE<                                                      \
      Kokkos::View<Kokkos::complex<float>**, LAYOUTA,                          \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<Kokkos::complex<float>*, LAYOUTA,                           \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef Kokkos::complex<float> SCALAR;                                     \
    typedef float PRECISION;                                                   \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTA,                                     \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
                                                                               \
    static int64_t geqrf_workspace(AViewType& A, TauViewType& tau) {           \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::geqrf[TPL_CUSOLVER, Kokkos::complex<float>]");          \
      int M         = A.extent(0);                                             \
      int N         = A.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      int lwork     = 0;                                                       \
      KokkosBlas::Impl::CudaSolverSingleton& s =                               \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                  \
      cusolverDnCgeqrf_bufferSize(s.handle, M, N,                              \
                                  reinterpret_cast<cuComplex*>(A.data()), LDA, \
                                  &lwork);                                     \
      Kokkos::Profiling::popRegion();                                          \
      return (int64_t)lwork;                                                   \
    }                                                                          \
  };

// CUDA Space
KOKKOSBLAS_DGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_DGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_SGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_SGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_ZGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_ZGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_CGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_CGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_DGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     true)
KOKKOSBLAS_DGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     false)

KOKKOSBLAS_SGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     true)
KOKKOSBLAS_SGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     false)

KOKKOSBLAS_ZGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     true)
KOKKOSBLAS_ZGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     false)

KOKKOSBLAS_CGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     true)
KOKKOSBLAS_CGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     false)

// CUDA UVM Space
KOKKOSBLAS_DGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS_DGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS_SGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS_SGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS_ZGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS_ZGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS_CGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS_CGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS_DGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     true)
KOKKOSBLAS_DGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     false)

KOKKOSBLAS_SGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     true)
KOKKOSBLAS_SGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     false)

KOKKOSBLAS_ZGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     true)
KOKKOSBLAS_ZGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     false)

KOKKOSBLAS_CGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     true)
KOKKOSBLAS_CGEQRF_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // IF CUSOLVER && CUBLAS

#endif  // KOKKOSBLAS_GEQRF_TPL_SPEC_DECL_HPP_
