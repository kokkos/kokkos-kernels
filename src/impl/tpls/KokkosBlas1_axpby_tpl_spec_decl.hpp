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

#ifndef KOKKOSBLAS1_AXPBY_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS1_AXPBY_TPL_SPEC_DECL_HPP_

namespace KokkosBlas {
namespace Impl {

#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
extern "C" void daxpy_( const int N, const double alpha,
                                     const double* x, const int x_inc,
                                     const double* y, const int y_inc);

#define KOKKOSBLAS1_DAXPBY_BLAS( LAYOUT, MEMSPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct Axpby< \
     double, \
     Kokkos::View<const double*, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     double, \
     Kokkos::View<double*, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     1, true, ETI_SPEC_AVAIL> { \
  typedef double AV; \
  typedef double BV; \
  typedef Kokkos::View<const double*, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > XV; \
  typedef Kokkos::View<double*, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > YV; \
\
  static void \
  axpby (const AV& alpha, const XV& X, const BV& beta, const YV& Y) { \
    if(X.extent(0) < INT_MAX) { \
      printf("%i %i\n",X.extent(0),Y.extent(0));\
      for(int i=0; i<X.extent(0);i++) printf("%i %lf %lf %lf %lf\n",i,X(i),Y(i),X.data()[i],Y.data()[i]);\
      daxpy_((int)X.extent(0)-1,alpha,X.data(),(int)1,Y.data(),(int)1); \
      printf("Hello\n");\
    } else \
      Axpby<AV,XV,BV,YV,YV::Rank,false,ETI_SPEC_AVAIL>::axpby(alpha,X,beta,Y); \
  } \
};

extern "C" void saxpy_( const int N, const float alpha,
                              const float* x, const int x_inc,
                              const float* y, const int y_inc);

#define KOKKOSBLAS1_SAXPBY_BLAS( LAYOUT, MEMSPACE , ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct Axpby< \
     float, \
     Kokkos::View<const float*, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     float, \
     Kokkos::View<float*, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     1, true, ETI_SPEC_AVAIL> { \
  typedef float AV; \
  typedef float BV; \
  typedef Kokkos::View<const float*, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > XV; \
  typedef Kokkos::View<float*, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > YV; \
\
  static void \
  axpby (const AV& alpha, const XV& X, const BV& beta, const YV& Y) { \
    printf("TPL\n"); \
    if(X.extent(0) < INT_MAX) { \
      saxpy_(X.extent(0),alpha,X.data(),1,Y.data(),1); \
    } else \
      Axpby<AV,XV,BV,YV,YV::Rank,false,ETI_SPEC_AVAIL>::axpby(alpha,X,beta,Y); \
  } \
};

KOKKOSBLAS1_DAXPBY_BLAS( Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_DAXPBY_BLAS( Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_DAXPBY_BLAS( Kokkos::LayoutRight, Kokkos::HostSpace, true )
KOKKOSBLAS1_DAXPBY_BLAS( Kokkos::LayoutRight, Kokkos::HostSpace, false )

KOKKOSBLAS1_SAXPBY_BLAS( Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS1_SAXPBY_BLAS( Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS1_SAXPBY_BLAS( Kokkos::LayoutRight, Kokkos::HostSpace, true )
KOKKOSBLAS1_SAXPBY_BLAS( Kokkos::LayoutRight, Kokkos::HostSpace, false )

#endif // KOKKOSKERNELS_ENABLE_TPL_BLAS
}
}

#endif
