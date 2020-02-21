/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#ifndef KOKKOSBLAS3_TRMM_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS3_TRMM_TPL_SPEC_DECL_HPP_

// Generic Host side BLAS (could be MKL or anything)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#include "KokkosBlas_Host_tpl.hpp"

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS3_DTRMM_BLAS( LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct TRMM< \
     Kokkos::View<const double**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<double**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef double SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType; \
  \
  static void \
  trmm (const char side[], \
        const char uplo[], \
        const char trans[], \
        const char diag[], \
        typename BViewType::const_value_type& alpha, \
        const AViewType& A, \
        const BViewType& B) { \
    \
    Kokkos::Profiling::pushRegion("KokkosBlas::trmm[TPL_BLAS,double]"); \
    const int M = static_cast<int> (B.extent(0)); \
    const int N = static_cast<int> (B.extent(1)); \
    \
    bool A_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTA>::value; \
    bool B_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTB>::value; \
    \
    const int AST = A_is_ll?A.stride(1):A.stride(0), LDA = (AST == 0) ? 1 : AST; \
    const int BST = B_is_ll?B.stride(1):B.stride(0), LDB = (BST == 0) ? 1 : BST; \
    \
    char  side_; \
    char  uplo_; \
    \
    if(A_is_ll) { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = 'L'; \
      else \
        side_ = 'R'; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = 'L'; \
      else \
        uplo_ = 'U'; \
    } \
    else { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = 'R'; \
      else \
        side_ = 'L'; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = 'U'; \
      else \
        uplo_ = 'L'; \
    } \
    \
    if(A_is_ll) \
      HostBlas<double>::trmm(side_, uplo_, trans[0], diag[0], M, N, alpha, A.data(), LDA, B.data(), LDB); \
    else \
      HostBlas<double>::trmm(side_, uplo_, trans[0], diag[0], N, M, alpha, A.data(), LDA, B.data(), LDB); \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSBLAS3_STRMM_BLAS( LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct TRMM< \
     Kokkos::View<const float**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<float**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef float SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType; \
  \
  static void \
  trmm (const char side[], \
        const char uplo[], \
        const char trans[], \
        const char diag[], \
        typename BViewType::const_value_type& alpha, \
        const AViewType& A, \
        const BViewType& B) { \
    \
    Kokkos::Profiling::pushRegion("KokkosBlas::trmm[TPL_BLAS,float]"); \
    const int M = static_cast<int> (B.extent(0)); \
    const int N = static_cast<int> (B.extent(1)); \
    \
    bool A_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTA>::value; \
    bool B_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTB>::value; \
    \
    const int AST = A_is_ll?A.stride(1):A.stride(0), LDA = (AST == 0) ? 1 : AST; \
    const int BST = B_is_ll?B.stride(1):B.stride(0), LDB = (BST == 0) ? 1 : BST; \
    \
    char  side_; \
    char  uplo_; \
    \
    if(A_is_ll) { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = 'L'; \
      else \
        side_ = 'R'; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = 'L'; \
      else \
        uplo_ = 'U'; \
    } \
    else { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = 'R'; \
      else \
        side_ = 'L'; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = 'U'; \
      else \
        uplo_ = 'L'; \
    } \
    \
    if(A_is_ll) \
      HostBlas<float>::trmm(side_, uplo_, trans[0], diag[0], M, N, alpha, A.data(), LDA, B.data(), LDB); \
    else \
      HostBlas<float>::trmm(side_, uplo_, trans[0], diag[0], N, M, alpha, A.data(), LDA, B.data(), LDB); \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSBLAS3_ZTRMM_BLAS( LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct TRMM< \
     Kokkos::View<const Kokkos::complex<double>**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<Kokkos::complex<double>**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef Kokkos::complex<double> SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType; \
  \
  static void \
  trmm (const char side[], \
        const char uplo[], \
        const char trans[], \
        const char diag[], \
        typename BViewType::const_value_type& alpha, \
        const AViewType& A, \
        const BViewType& B) { \
    \
    Kokkos::Profiling::pushRegion("KokkosBlas::trmm[TPL_BLAS,complex<double>]"); \
    const int M = static_cast<int> (B.extent(0)); \
    const int N = static_cast<int> (B.extent(1)); \
    \
    bool A_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTA>::value; \
    bool B_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTB>::value; \
    \
    const int AST = A_is_ll?A.stride(1):A.stride(0), LDA = (AST == 0) ? 1 : AST; \
    const int BST = B_is_ll?B.stride(1):B.stride(0), LDB = (BST == 0) ? 1 : BST; \
    \
    char  side_; \
    char  uplo_; \
    \
    if(A_is_ll) { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = 'L'; \
      else \
        side_ = 'R'; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = 'L'; \
      else \
        uplo_ = 'U'; \
    } \
    else { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = 'R'; \
      else \
        side_ = 'L'; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = 'U'; \
      else \
        uplo_ = 'L'; \
    } \
    \
    const std::complex<double> alpha_val = alpha; \
    if(A_is_ll) \
      HostBlas<std::complex<double> >::trmm(side_, uplo_, trans[0], diag[0], M, N, alpha_val, reinterpret_cast<const std::complex<double>*>(A.data()), LDA, reinterpret_cast<std::complex<double>*>(B.data()), LDB); \
    else \
      HostBlas<std::complex<double> >::trmm(side_, uplo_, trans[0], diag[0], N, M, alpha_val, reinterpret_cast<const std::complex<double>*>(A.data()), LDA, reinterpret_cast<std::complex<double>*>(B.data()), LDB); \
    Kokkos::Profiling::popRegion(); \
  } \
}; \

#define KOKKOSBLAS3_CTRMM_BLAS( LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct TRMM< \
     Kokkos::View<const Kokkos::complex<float>**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<Kokkos::complex<float>**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef Kokkos::complex<float> SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType; \
  \
  static void \
  trmm (const char side[], \
        const char uplo[], \
        const char trans[], \
        const char diag[], \
        typename BViewType::const_value_type& alpha, \
        const AViewType& A, \
        const BViewType& B) { \
    \
    Kokkos::Profiling::pushRegion("KokkosBlas::trmm[TPL_BLAS,complex<float>]"); \
    const int M = static_cast<int> (B.extent(0)); \
    const int N = static_cast<int> (B.extent(1)); \
    \
    bool A_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTA>::value; \
    bool B_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTB>::value; \
    \
    const int AST = A_is_ll?A.stride(1):A.stride(0), LDA = (AST == 0) ? 1 : AST; \
    const int BST = B_is_ll?B.stride(1):B.stride(0), LDB = (BST == 0) ? 1 : BST; \
    \
    char  side_; \
    char  uplo_; \
    \
    if(A_is_ll) { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = 'L'; \
      else \
        side_ = 'R'; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = 'L'; \
      else \
        uplo_ = 'U'; \
    } \
    else { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = 'R'; \
      else \
        side_ = 'L'; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = 'U'; \
      else \
        uplo_ = 'L'; \
    } \
    \
    const std::complex<float> alpha_val = alpha; \
    if(A_is_ll) \
      HostBlas<std::complex<float> >::trmm(side_, uplo_, trans[0], diag[0], M, N, alpha_val, reinterpret_cast<const std::complex<float>*>(A.data()), LDA, reinterpret_cast<std::complex<float>*>(B.data()), LDB); \
    else \
      HostBlas<std::complex<float> >::trmm(side_, uplo_, trans[0], diag[0], N, M, alpha_val, reinterpret_cast<const std::complex<float>*>(A.data()), LDA, reinterpret_cast<std::complex<float>*>(B.data()), LDB); \
    Kokkos::Profiling::popRegion(); \
  } \
};

// Explicitly define the TRMM class for all permutations listed below

//KOKKOSBLAS3_DTRMM_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
KOKKOSBLAS3_DTRMM_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
//KOKKOSBLAS3_DTRMM_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS3_DTRMM_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

//KOKKOSBLAS3_STRMM_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
KOKKOSBLAS3_STRMM_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
//KOKKOSBLAS3_STRMM_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS3_STRMM_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

//KOKKOSBLAS3_ZTRMM_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
KOKKOSBLAS3_ZTRMM_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
//KOKKOSBLAS3_ZTRMM_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS3_ZTRMM_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

//KOKKOSBLAS3_CTRMM_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
KOKKOSBLAS3_CTRMM_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
//KOKKOSBLAS3_CTRMM_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS3_CTRMM_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

}
}
#endif // KOKKOSKERNELS_ENABLE_TPL_BLAS

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include<KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS3_DTRMM_CUBLAS( LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct TRMM< \
     Kokkos::View<const double**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<double**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef double SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType; \
 \
  static void \
  trmm (const char side[], \
        const char uplo[], \
        const char trans[], \
        const char diag[], \
        typename BViewType::const_value_type& alpha, \
        const AViewType& A, \
        const BViewType& B) { \
    \
    Kokkos::Profiling::pushRegion("KokkosBlas::trmm[TPL_CUBLAS,double]"); \
    const int M = static_cast<int> (B.extent(0)); \
    const int N = static_cast<int> (B.extent(1)); \
    \
    bool A_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTA>::value; \
    bool B_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTB>::value; \
    \
    const int AST = A_is_ll?A.stride(1):A.stride(0), LDA = (AST == 0) ? 1 : AST; \
    const int BST = B_is_ll?B.stride(1):B.stride(0), LDB = (BST == 0) ? 1 : BST; \
    \
    cublasSideMode_t  side_; \
    cublasFillMode_t  uplo_; \
    cublasOperation_t trans_; \
    cublasDiagType_t  diag_; \
    \
    if(A_is_ll) { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = CUBLAS_SIDE_LEFT; \
      else \
        side_ = CUBLAS_SIDE_RIGHT; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = CUBLAS_FILL_MODE_LOWER; \
      else \
        uplo_ = CUBLAS_FILL_MODE_UPPER; \
    } \
    else { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = CUBLAS_SIDE_RIGHT; \
      else \
        side_ = CUBLAS_SIDE_LEFT; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = CUBLAS_FILL_MODE_UPPER; \
      else \
        uplo_ = CUBLAS_FILL_MODE_LOWER; \
    } \
    \
    if ((trans[0]=='N')||(trans[0]=='n')) \
      trans_ = CUBLAS_OP_N; \
    else if ((trans[0]=='T')||(trans[0]=='t')) \
      trans_ = CUBLAS_OP_T; \
    else \
      trans_ = CUBLAS_OP_C; \
    if ((diag[0]=='U')||(diag[0]=='u')) \
      diag_ = CUBLAS_DIAG_UNIT; \
    else \
      diag_ = CUBLAS_DIAG_NON_UNIT; \
    \
    KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton(); \
    if(A_is_ll) \
      cublasDtrmm(s.handle, side_, uplo_, trans_, diag_, M, N, &alpha, A.data(), LDA, B.data(), LDB, B.data(), LDB); \
    else \
      cublasDtrmm(s.handle, side_, uplo_, trans_, diag_, N, M, &alpha, A.data(), LDA, B.data(), LDB, B.data(), LDB); \
    \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSBLAS3_STRMM_CUBLAS( LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct TRMM< \
     Kokkos::View<const float**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<float**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef float SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType; \
  \
  static void \
  trmm (const char side[], \
        const char uplo[], \
        const char trans[], \
        const char diag[], \
        typename BViewType::const_value_type& alpha, \
        const AViewType& A, \
        const BViewType& B) { \
    \
    Kokkos::Profiling::pushRegion("KokkosBlas::trmm[TPL_CUBLAS,float]"); \
    const int M = static_cast<int> (B.extent(0)); \
    const int N = static_cast<int> (B.extent(1)); \
    \
    bool A_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTA>::value; \
    bool B_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTB>::value; \
    \
    const int AST = A_is_ll?A.stride(1):A.stride(0), LDA = (AST == 0) ? 1 : AST; \
    const int BST = B_is_ll?B.stride(1):B.stride(0), LDB = (BST == 0) ? 1 : BST; \
    \
    cublasSideMode_t  side_; \
    cublasFillMode_t  uplo_; \
    cublasOperation_t trans_; \
    cublasDiagType_t  diag_; \
    \
    if(A_is_ll) { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = CUBLAS_SIDE_LEFT; \
      else \
        side_ = CUBLAS_SIDE_RIGHT; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = CUBLAS_FILL_MODE_LOWER; \
      else \
        uplo_ = CUBLAS_FILL_MODE_UPPER; \
    } \
    else { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = CUBLAS_SIDE_RIGHT; \
      else \
        side_ = CUBLAS_SIDE_LEFT; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = CUBLAS_FILL_MODE_UPPER; \
      else \
        uplo_ = CUBLAS_FILL_MODE_LOWER; \
    } \
    \
    if ((trans[0]=='N')||(trans[0]=='n')) \
      trans_ = CUBLAS_OP_N; \
    else if ((trans[0]=='T')||(trans[0]=='t')) \
      trans_ = CUBLAS_OP_T; \
    else \
      trans_ = CUBLAS_OP_C; \
    if ((diag[0]=='U')||(diag[0]=='u')) \
      diag_ = CUBLAS_DIAG_UNIT; \
    else \
      diag_ = CUBLAS_DIAG_NON_UNIT; \
    \
    KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton(); \
    if(A_is_ll) \
      cublasStrmm(s.handle, side_, uplo_, trans_, diag_, M, N, &alpha, A.data(), LDA, B.data(), LDB), B.data(), LDB; \
    else \
      cublasStrmm(s.handle, side_, uplo_, trans_, diag_, N, M, &alpha, A.data(), LDA, B.data(), LDB, B.data(), LDB); \
    \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSBLAS3_ZTRMM_CUBLAS( LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct TRMM< \
     Kokkos::View<const Kokkos::complex<double>**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<Kokkos::complex<double>**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef Kokkos::complex<double> SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType; \
  \
  static void \
  trmm (const char side[], \
        const char uplo[], \
        const char trans[], \
        const char diag[], \
        typename BViewType::const_value_type& alpha, \
        const AViewType& A, \
        const BViewType& B) { \
    \
    Kokkos::Profiling::pushRegion("KokkosBlas::trmm[TPL_CUBLAS,complex<double>]"); \
    const int M = static_cast<int> (B.extent(0)); \
    const int N = static_cast<int> (B.extent(1)); \
    \
    bool A_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTA>::value; \
    bool B_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTB>::value; \
    \
    const int AST = A_is_ll?A.stride(1):A.stride(0), LDA = (AST == 0) ? 1 : AST; \
    const int BST = B_is_ll?B.stride(1):B.stride(0), LDB = (BST == 0) ? 1 : BST; \
    \
    cublasSideMode_t  side_; \
    cublasFillMode_t  uplo_; \
    cublasOperation_t trans_; \
    cublasDiagType_t  diag_; \
    \
    if(A_is_ll) { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = CUBLAS_SIDE_LEFT; \
      else \
        side_ = CUBLAS_SIDE_RIGHT; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = CUBLAS_FILL_MODE_LOWER; \
      else \
        uplo_ = CUBLAS_FILL_MODE_UPPER; \
    } \
    else { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = CUBLAS_SIDE_RIGHT; \
      else \
        side_ = CUBLAS_SIDE_LEFT; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = CUBLAS_FILL_MODE_UPPER; \
      else \
        uplo_ = CUBLAS_FILL_MODE_LOWER; \
    } \
    \
    if ((trans[0]=='N')||(trans[0]=='n')) \
      trans_ = CUBLAS_OP_N; \
    else if ((trans[0]=='T')||(trans[0]=='t')) \
      trans_ = CUBLAS_OP_T; \
    else \
      trans_ = CUBLAS_OP_C; \
    if ((diag[0]=='U')||(diag[0]=='u')) \
      diag_ = CUBLAS_DIAG_UNIT; \
    else \
      diag_ = CUBLAS_DIAG_NON_UNIT; \
    \
    KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton(); \
    if(A_is_ll) \
      cublasZtrmm(s.handle, side_, uplo_, trans_, diag_, M, N, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(A.data()), LDA, reinterpret_cast<cuDoubleComplex*>(B.data()), LDB, reinterpret_cast<cuDoubleComplex*>(B.data()), LDB); \
    else \
      cublasZtrmm(s.handle, side_, uplo_, trans_, diag_, N, M, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(A.data()), LDA, reinterpret_cast<cuDoubleComplex*>(B.data()), LDB, reinterpret_cast<cuDoubleComplex*>(B.data()), LDB); \
    \
    Kokkos::Profiling::popRegion(); \
  } \
}; \

#define KOKKOSBLAS3_CTRMM_CUBLAS( LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct TRMM< \
     Kokkos::View<const Kokkos::complex<float>**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<Kokkos::complex<float>**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef Kokkos::complex<float> SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType; \
  \
  static void \
  trmm (const char side[], \
        const char uplo[], \
        const char trans[], \
        const char diag[], \
        typename BViewType::const_value_type& alpha, \
        const AViewType& A, \
        const BViewType& B) { \
    \
    Kokkos::Profiling::pushRegion("KokkosBlas::trmm[TPL_CUBLAS,complex<float>]"); \
    const int M = static_cast<int> (B.extent(0)); \
    const int N = static_cast<int> (B.extent(1)); \
    \
    bool A_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTA>::value; \
    bool B_is_ll = std::is_same<Kokkos::LayoutLeft,LAYOUTB>::value; \
    \
    const int AST = A_is_ll?A.stride(1):A.stride(0), LDA = (AST == 0) ? 1 : AST; \
    const int BST = B_is_ll?B.stride(1):B.stride(0), LDB = (BST == 0) ? 1 : BST; \
    \
    cublasSideMode_t  side_; \
    cublasFillMode_t  uplo_; \
    cublasOperation_t trans_; \
    cublasDiagType_t  diag_; \
    \
    if(A_is_ll) { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = CUBLAS_SIDE_LEFT; \
      else \
        side_ = CUBLAS_SIDE_RIGHT; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = CUBLAS_FILL_MODE_LOWER; \
      else \
        uplo_ = CUBLAS_FILL_MODE_UPPER; \
    } \
    else { \
      if ((side[0]=='L')||(side[0]=='l')) \
        side_ = CUBLAS_SIDE_RIGHT; \
      else \
        side_ = CUBLAS_SIDE_LEFT; \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = CUBLAS_FILL_MODE_UPPER; \
      else \
        uplo_ = CUBLAS_FILL_MODE_LOWER; \
    } \
    \
    if ((trans[0]=='N')||(trans[0]=='n')) \
      trans_ = CUBLAS_OP_N; \
    else if ((trans[0]=='T')||(trans[0]=='t')) \
      trans_ = CUBLAS_OP_T; \
    else \
      trans_ = CUBLAS_OP_C; \
    if ((diag[0]=='U')||(diag[0]=='u')) \
      diag_ = CUBLAS_DIAG_UNIT; \
    else \
      diag_ = CUBLAS_DIAG_NON_UNIT; \
    \
    KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton(); \
    if(A_is_ll) \
      cublasCtrmm(s.handle, side_, uplo_, trans_, diag_, M, N, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(A.data()), LDA, reinterpret_cast<cuComplex*>(B.data()), LDB, reinterpret_cast<cuComplex*>(B.data()), LDB); \
    else \
      cublasCtrmm(s.handle, side_, uplo_, trans_, diag_, N, M, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(A.data()), LDA, reinterpret_cast<cuComplex*>(B.data()), LDB, reinterpret_cast<cuComplex*>(B.data()), LDB); \
    \
    Kokkos::Profiling::popRegion(); \
  } \
};

//KOKKOSBLAS3_DTRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, true)
KOKKOSBLAS3_DTRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
//KOKKOSBLAS3_DTRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS3_DTRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

//KOKKOSBLAS3_DTRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_DTRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, false)
//KOKKOSBLAS3_DTRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_DTRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, false)

//KOKKOSBLAS3_STRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, true)
KOKKOSBLAS3_STRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
//KOKKOSBLAS3_STRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS3_STRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

//KOKKOSBLAS3_STRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_STRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, false)
//KOKKOSBLAS3_STRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_STRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, false)

//KOKKOSBLAS3_ZTRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, true)
KOKKOSBLAS3_ZTRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
//KOKKOSBLAS3_ZTRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS3_ZTRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

//KOKKOSBLAS3_ZTRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_ZTRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, false)
//KOKKOSBLAS3_ZTRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_ZTRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, false)

//KOKKOSBLAS3_CTRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, true)
KOKKOSBLAS3_CTRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
//KOKKOSBLAS3_CTRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS3_CTRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

//KOKKOSBLAS3_CTRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_CTRMM_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, false)
//KOKKOSBLAS3_CTRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_CTRMM_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, false)

}
}
#endif // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

#endif
