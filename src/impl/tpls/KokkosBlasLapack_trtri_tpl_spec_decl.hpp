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

#ifndef KOKKOSBLASLAPACK_TRTRI_TPL_SPEC_DECL_HPP_
#define KOKKOSBLASLAPACK_TRTRI_TPL_SPEC_DECL_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK
#include "KokkosBlas_Host_tpl.hpp" // trtri prototype
#include "KokkosBlas_tpl_spec.hpp"

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLASLAPACK_TRTRI_BLAS(SCALAR_TYPE, BASE_SCALAR_TYPE, LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL) \
template<class ExecSpace> \
struct TRTRI< \
     Kokkos::View<int, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const SCALAR_TYPE**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef SCALAR_TYPE SCALAR; \
typedef Kokkos::View<int, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > RViewType; \
  typedef Kokkos::View<const SCALAR_TYPE**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  \
  static void \
  trtri (const RViewType& R, \
        const char uplo[], \
        const char diag[], \
        const AViewType& A) { \
    \
    Kokkos::Profiling::pushRegion("KokkosBlas::trtri[TPL_BLAS,"#SCALAR_TYPE"]"); \
    const int M = static_cast<int> (A.extent(0)); \
    \
    bool A_is_layout_left = std::is_same<Kokkos::LayoutLeft,LAYOUTA>::value; \
    int matrix_layout_ = A_is_layout_left ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR; \
    \
    const int AST = A_is_layout_left?A.stride(1):A.stride(0), LDA = (AST == 0) ? 1 : AST; \
    \
    char  uplo_; \
    \
    if(A_is_layout_left) { \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = 'L'; \
      else \
        uplo_ = 'U'; \
    } \
    else { \
      if ((uplo[0]=='L')||(uplo[0]=='l')) \
        uplo_ = 'U'; \
      else \
        uplo_ = 'L'; \
    } \
    \
    R() = HostBlas<BASE_SCALAR_TYPE>::trtri(matrix_layout_, uplo_, diag[0], M, reinterpret_cast<const BASE_SCALAR_TYPE *>(A.data()), LDA); \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSBLASLAPACK_DTRTRI_BLAS(LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL) \
KOKKOSBLASLAPACK_TRTRI_BLAS(double, double, LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLASLAPACK_STRTRI_BLAS(LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL) \
KOKKOSBLASLAPACK_TRTRI_BLAS(float, float, LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLASLAPACK_ZTRTRI_BLAS(LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL) \
KOKKOSBLASLAPACK_TRTRI_BLAS(Kokkos::complex<double>, std::complex<double>, LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLASLAPACK_CTRTRI_BLAS(LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL) \
KOKKOSBLASLAPACK_TRTRI_BLAS(Kokkos::complex<float>, std::complex<float>, LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL)
// Explicitly define the TRTRI class for all permutations listed below

//KOKKOSBLASLAPACK_DTRTRI_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
KOKKOSBLASLAPACK_DTRTRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, false)
//KOKKOSBLASLAPACK_DTRTRI_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLASLAPACK_DTRTRI_BLAS(Kokkos::LayoutRight, Kokkos::HostSpace, false)

//KOKKOSBLASLAPACK_STRTRI_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
KOKKOSBLASLAPACK_STRTRI_BLAS(Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
//KOKKOSBLASLAPACK_STRTRI_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLASLAPACK_STRTRI_BLAS(Kokkos::LayoutRight, Kokkos::HostSpace, false)

//KOKKOSBLASLAPACK_ZTRTRI_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
KOKKOSBLASLAPACK_ZTRTRI_BLAS(Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
//KOKKOSBLASLAPACK_ZTRTRI_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLASLAPACK_ZTRTRI_BLAS(Kokkos::LayoutRight, Kokkos::HostSpace, false)

//KOKKOSBLASLAPACK_CTRTRI_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
KOKKOSBLASLAPACK_CTRTRI_BLAS(Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
//KOKKOSBLASLAPACK_CTRTRI_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLASLAPACK_CTRTRI_BLAS(Kokkos::LayoutRight, Kokkos::HostSpace, false)

} // namespace Impl
} // nameSpace KokkosBlas
#endif // KOKKOSKERNELS_ENABLE_TPL_LAPACK

#endif // KOKKOSBLASLAPACK_TRTRI_TPL_SPEC_DECL_HPP_
