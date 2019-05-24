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

#ifndef KOKKOSBLAS1_COPY_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS1_COPY_TPL_SPEC_DECL_HPP_

namespace KokkosBlas {
namespace Impl {
  template<class XV, class YV>
  inline void copy_print_specialization() {
      #ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
        #ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
          printf("KokkosBlas1::copy<> TPL cuBLAS specialization for < %s , %s >\n",typeid(XV).name(),typeid(YV).name());
        #else
          #ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
            printf("KokkosBlas1::copy<> TPL Blas specialization for < %s , %s >\n",typeid(XV).name(),typeid(YV).name());
          #endif        
        #endif
      #endif
  }
}
}

#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#include "KokkosBlas_Host_tpl.hpp"

namespace KokkosBlas {
namespace Impl {


#define KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( LAYOUTX, LAYOUTY, MEMSPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct Copy< \
Kokkos::View<const double*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
Kokkos::View<double*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
1,true, ETI_SPEC_AVAIL > { \
  \
  typedef Kokkos::View<const double*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > XV; \
  typedef Kokkos::View<double*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > YV; \
  typedef typename XV::size_type size_type; \
  \
  static void copy (const XV& X, const YV& Y) \
  { \
    Kokkos::Profiling::pushRegion("KokkosBlas::copy[TPL_BLAS,double]"); \
    const size_type numElems = X.extent(0); \
    if (numElems < static_cast<size_type> (INT_MAX)) { \
      copy_print_specialization<XV,YV>(); \
      const int N   = static_cast<int> (numElems); \
      const int XST = X.stride(0); \
      const int YST = Y.stride(0); \
      const int LDX = (XST == 0) ? 1 : XST; \
      const int LDY = (YST == 0) ? 1 : YST; \
      HostBlas<double>::copy(N, X.data(), LDX, Y.data(), LDY); \
    } else { \
      Copy<XV,YV,1,false,ETI_SPEC_AVAIL>::copy(X,Y); \
    } \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( LAYOUTX, LAYOUTY, MEMSPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct Copy< \
Kokkos::View<const float*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
Kokkos::View<float*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
1,true, ETI_SPEC_AVAIL > { \
  \
  typedef Kokkos::View<const float*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > XV; \
  typedef Kokkos::View<float*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > YV; \
  typedef typename XV::size_type size_type; \
  \
  static void copy (const XV& X, const YV& Y) \
  { \
    Kokkos::Profiling::pushRegion("KokkosBlas::copy[TPL_BLAS,float]"); \
    const size_type numElems = X.extent(0); \
    if (numElems < static_cast<size_type> (INT_MAX)) { \
      copy_print_specialization<XV,YV>(); \
      const int N   = static_cast<int> (numElems); \
      const int XST = X.stride(0); \
      const int YST = Y.stride(0); \
      const int LDX = (XST == 0) ? 1 : XST; \
      const int LDY = (YST == 0) ? 1 : YST; \
      HostBlas<float>::copy(N, X.data(), LDX, Y.data(), LDY);  \
    } else { \
      Copy<XV,YV,1,false,ETI_SPEC_AVAIL>::copy(X,Y); \
    } \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( LAYOUTX, LAYOUTY, MEMSPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct Copy< \
Kokkos::View<const Kokkos::complex<double>*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
Kokkos::View<Kokkos::complex<double>*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
1,true, ETI_SPEC_AVAIL > { \
  \
  typedef Kokkos::View<const Kokkos::complex<double>*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > XV; \
  typedef Kokkos::View<Kokkos::complex<double>*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > YV; \
  typedef typename XV::size_type size_type; \
  \
  static void copy (const XV& X, const YV& Y) \
  { \
    Kokkos::Profiling::pushRegion("KokkosBlas::copy[TPL_BLAS,complex<double>]"); \
    const size_type numElems = X.extent(0); \
    if (numElems < static_cast<size_type> (INT_MAX)) { \
      copy_print_specialization<XV,YV>(); \
      const int N   = static_cast<int> (numElems); \
      const int XST = X.stride(0); \
      const int YST = Y.stride(0); \
      const int LDX = (XST == 0) ? 1 : XST; \
      const int LDY = (YST == 0) ? 1 : YST; \
      HostBlas<std::complex<double> >::copy(N, reinterpret_cast<const std::complex<double>*>(X.data()), LDX, reinterpret_cast<std::complex<double>*>(Y.data()), LDY); \
    } else { \
      Copy<XV,YV,1,false,ETI_SPEC_AVAIL>::copy(X,Y); \
    } \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( LAYOUTX, LAYOUTY, MEMSPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct Copy< \
Kokkos::View<const Kokkos::complex<float>*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
Kokkos::View<Kokkos::complex<float>*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
1,true, ETI_SPEC_AVAIL > { \
  \
  typedef Kokkos::View<const Kokkos::complex<float>*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > XV; \
  typedef Kokkos::View<Kokkos::complex<float>*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > YV; \
  typedef typename XV::size_type size_type; \
  \
  static void copy (const XV& X, const YV& Y) \
  { \
    Kokkos::Profiling::pushRegion("KokkosBlas::copy[TPL_BLAS,complex<float>]"); \
    const size_type numElems = X.extent(0); \
    if (numElems < static_cast<size_type> (INT_MAX)) { \
      copy_print_specialization<XV,YV>(); \
      const int N   = static_cast<int> (numElems); \
      const int XST = X.stride(0); \
      const int YST = Y.stride(0); \
      const int LDX = (XST == 0) ? 1 : XST; \
      const int LDY = (YST == 0) ? 1 : YST; \
      HostBlas<std::complex<float> >::copy(N, reinterpret_cast<const std::complex<float>*>(X.data()), LDX, reinterpret_cast<std::complex<float>*>(Y.data()), LDY); \
    } else { \
      Copy<XV,YV,1,false,ETI_SPEC_AVAIL>::copy(X,Y); \
    } \
    Kokkos::Profiling::popRegion(); \
  } \
};

KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutRight, Kokkos::HostSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutStride, Kokkos::HostSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutStride, Kokkos::HostSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::HostSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::HostSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::HostSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::HostSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::HostSpace, false)

KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutRight, Kokkos::HostSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutStride, Kokkos::HostSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft,  Kokkos::LayoutStride, Kokkos::HostSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::HostSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::HostSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::HostSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::HostSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::HostSpace, false)

KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutRight, Kokkos::HostSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutStride, Kokkos::HostSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutStride, Kokkos::HostSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::HostSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::HostSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::HostSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::HostSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::HostSpace, false)

KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutRight, Kokkos::HostSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutStride, Kokkos::HostSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutStride, Kokkos::HostSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::HostSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::HostSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::HostSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::HostSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_BLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::HostSpace, false)

}
}

#endif

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include<KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( LAYOUTX, LAYOUTY, MEMSPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct Copy< \
Kokkos::View<const double*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
Kokkos::View<double*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
1,true, ETI_SPEC_AVAIL > { \
  \
  typedef Kokkos::View<const double*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > XV; \
  typedef Kokkos::View<double*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > YV; \
  typedef typename XV::size_type size_type; \
  \
  static void copy (const XV& X, const YV& Y) \
  { \
    Kokkos::Profiling::pushRegion("KokkosBlas::copy[TPL_CUBLAS,double]"); \
    const size_type numElems = X.extent(0); \
    if (numElems < static_cast<size_type> (INT_MAX)) { \
      copy_print_specialization<XV,YV>(); \
      const int N   = static_cast<int> (numElems); \
      const int XST = X.stride(0), LDX = (XST == 0) ? 1 : XST; \
      const int YST = Y.stride(0), LDY = (YST == 0) ? 1 : YST; \
      KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton(); \
      cublasDcopy(s.handle, N, X.data(), LDX, Y.data(), LDY); \
    } else { \
      Copy<XV,YV,1,false,ETI_SPEC_AVAIL>::copy(X,Y); \
    } \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( LAYOUTX, LAYOUTY, MEMSPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct Copy< \
Kokkos::View<const float*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
Kokkos::View<float*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
1,true, ETI_SPEC_AVAIL > { \
  \
  typedef Kokkos::View<const float*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > XV; \
  typedef Kokkos::View<float*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > YV; \
  typedef typename XV::size_type size_type; \
  \
  static void copy (const XV& X, const YV& Y) \
  { \
    Kokkos::Profiling::pushRegion("KokkosBlas::copy[TPL_CUBLAS,float]"); \
    const size_type numElems = X.extent(0); \
    if (numElems < static_cast<size_type> (INT_MAX)) { \
      copy_print_specialization<XV,YV>(); \
      const int N   = static_cast<int> (numElems); \
      const int XST = X.stride(0), LDX = (XST == 0) ? 1 : XST; \
      const int YST = Y.stride(0), LDY = (YST == 0) ? 1 : YST; \
      KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton(); \
      cublasScopy(s.handle, N, X.data(), LDX, Y.data(), LDY); \
    } else { \
      Copy<XV,YV,1,false,ETI_SPEC_AVAIL>::copy(X,Y); \
    } \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( LAYOUTX, LAYOUTY, MEMSPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct Copy< \
Kokkos::View<const Kokkos::complex<double>*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
Kokkos::View<Kokkos::complex<double>*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
1,true, ETI_SPEC_AVAIL > { \
  \
  typedef Kokkos::View<const Kokkos::complex<double>*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > XV; \
  typedef Kokkos::View<Kokkos::complex<double>*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > YV; \
  typedef typename XV::size_type size_type; \
  \
  static void copy (const XV& X, const YV& Y) \
  { \
    Kokkos::Profiling::pushRegion("KokkosBlas::copy[TPL_CUBLAS,complex<double>]"); \
    const size_type numElems = X.extent(0); \
    if (numElems < static_cast<size_type> (INT_MAX)) { \
      copy_print_specialization<XV,YV>(); \
      const int N   = static_cast<int> (numElems); \
      const int XST = X.stride(0), LDX = (XST == 0) ? 1 : XST; \
      const int YST = Y.stride(0), LDY = (YST == 0) ? 1 : YST; \
      KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton(); \
      cublasZcopy(s.handle, N, reinterpret_cast<const cuDoubleComplex*>(X.data()), LDX, reinterpret_cast<cuDoubleComplex*>(Y.data()), LDY); \
    } else { \
      Copy<XV,YV,1,false,ETI_SPEC_AVAIL>::copy(X,Y); \
    } \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( LAYOUTX, LAYOUTY, MEMSPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct Copy< \
Kokkos::View<const Kokkos::complex<float>*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
Kokkos::View<Kokkos::complex<float>*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
             Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
1,true, ETI_SPEC_AVAIL > { \
  \
  typedef Kokkos::View<const Kokkos::complex<float>*, LAYOUTX, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > XV; \
  typedef Kokkos::View<Kokkos::complex<float>*, LAYOUTY, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > YV; \
  typedef typename XV::size_type size_type; \
  \
  static void copy (const XV& X, const YV& Y) \
  { \
    Kokkos::Profiling::pushRegion("KokkosBlas::copy[TPL_CUBLAS,complex<float>]"); \
    const size_type numElems = X.extent(0); \
    if (numElems < static_cast<size_type> (INT_MAX)) { \
      copy_print_specialization<XV,YV>(); \
      const int N   = static_cast<int> (numElems); \
      const int XST = X.stride(0), LDX = (XST == 0) ? 1 : XST; \
      const int YST = Y.stride(0), LDY = (YST == 0) ? 1 : YST; \
      KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton(); \
      cublasCcopy(s.handle, N, reinterpret_cast<const cuComplex*>(X.data()), LDX, reinterpret_cast<cuComplex*>(Y.data()), LDY); \
    } else { \
      Copy<XV,YV,1,false,ETI_SPEC_AVAIL>::copy(X,Y); \
    } \
    Kokkos::Profiling::popRegion(); \
  } \
};

KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutRight, Kokkos::CudaSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutStride, Kokkos::CudaSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutStride, Kokkos::CudaSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::CudaSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::CudaSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::CudaSpace, false)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::CudaSpace, true)
KOKKOSBLAS1_DCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::CudaSpace, false)

KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutRight, Kokkos::CudaSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutStride, Kokkos::CudaSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutStride, Kokkos::CudaSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::CudaSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::CudaSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::CudaSpace, false)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::CudaSpace, true)
KOKKOSBLAS1_SCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::CudaSpace, false)

KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutRight, Kokkos::CudaSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutStride, Kokkos::CudaSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutStride, Kokkos::CudaSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::CudaSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::CudaSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::CudaSpace, false)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::CudaSpace, true)
KOKKOSBLAS1_ZCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::CudaSpace, false)

KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutRight, Kokkos::CudaSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutStride, Kokkos::CudaSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutLeft,  Kokkos::LayoutStride, Kokkos::CudaSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::CudaSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutStride, Kokkos::CudaSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutRight, Kokkos::CudaSpace, false)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::CudaSpace, true)
KOKKOSBLAS1_CCOPY_TPL_SPEC_DECL_CUBLAS( Kokkos::LayoutStride, Kokkos::LayoutStride, Kokkos::CudaSpace, false)

}
}

#endif

#endif
