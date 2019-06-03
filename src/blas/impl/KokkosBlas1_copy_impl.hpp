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
#ifndef KOKKOS_BLAS1_IMPL_COPY_HPP_
#define KOKKOS_BLAS1_IMPL_COPY_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>

namespace KokkosBlas {
namespace Impl {

//
// copy
//

// Entry-wise: Y(i,j) = X(i,j).
template<class XMV, class YMV, class SizeType = typename YMV::size_type>
struct MV_Copy_Functor
{
  typedef typename YMV::execution_space execution_space;
  typedef SizeType                            size_type;
  typedef Kokkos::Details::ArithTraits<typename XMV::non_const_value_type> ATS;

  const size_type numCols;
  XMV X_;
  YMV Y_;

  MV_Copy_Functor (const XMV& X, const YMV& Y) :
    numCols (X.extent(1)), X_ (X), Y_ (Y)
  {
    static_assert (Kokkos::Impl::is_view<XMV>::value, "KokkosBlas::Impl::"
                   "MV_Copy_Functor: XMV is not a Kokkos::View.");
    static_assert (Kokkos::Impl::is_view<YMV>::value, "KokkosBlas::Impl::"
                   "MV_Copy_Functor: YMV is not a Kokkos::View.");
    static_assert (XMV::rank == 2, "KokkosBlas::Impl::"
                   "MV_Copy_Functor: XMV is not rank 2");
    static_assert (YMV::rank == 2, "KokkosBlas::Impl::"
                   "MV_Copy_Functor: YMV is not rank 2");
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const size_type& i) const
  {
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
    for (size_type j = 0; j < numCols; ++j) {
      Y_(i,j) = X_(i,j);
    }
  }
};

// Single-vector, entry-wise: R(i) = X(i).
template<class XV, class YV, class SizeType = typename YV::size_type>
struct V_Copy_Functor
{
  typedef typename YV::execution_space execution_space;
  typedef SizeType                            size_type;
  typedef Kokkos::Details::ArithTraits<typename XV::non_const_value_type> ATS;

  XV X_;
  YV Y_;

  V_Copy_Functor (const XV& X, const YV& Y) : X_ (X), Y_ (Y)
  {
    static_assert (Kokkos::Impl::is_view<XV>::value, "KokkosBlas::Impl::"
                   "V_Copy_Functor: XV is not a Kokkos::View.");
    static_assert (Kokkos::Impl::is_view<YV>::value, "KokkosBlas::Impl::"
                   "V_Copy_Functor: YV is not a Kokkos::View.");
    static_assert (XV::rank == 1, "KokkosBlas::Impl::"
                   "V_Copy_Functor: XV is not rank 1");
    static_assert (YV::rank == 1, "KokkosBlas::Impl::"
                   "V_Copy_Functor: YV is not rank 1");
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const size_type& i) const
  {
    Y_(i) = X_(i);
  }
};

// Invoke the "generic" (not unrolled) multivector functor that
// computes entry-wise absolute value.
template<class XMV, class YMV, class SizeType>
void
MV_Copy_Generic (const XMV& X, const YMV& Y)
{
  static_assert (Kokkos::Impl::is_view<XMV>::value, "KokkosBlas::Impl::"
                 "MV_Copy_Generic: XMV is not a Kokkos::View.");
  static_assert (Kokkos::Impl::is_view<YMV>::value, "KokkosBlas::Impl::"
                 "MV_Copy_Generic: YMV is not a Kokkos::View.");
  static_assert (XMV::rank == 2, "KokkosBlas::Impl::"
                 "MV_Copy_Generic: XMV is not rank 2");
  static_assert (YMV::rank == 2, "KokkosBlas::Impl::"
                 "MV_Copy_Generic: YMV is not rank 2");

  typedef typename XMV::execution_space execution_space;
  const SizeType numRows = X.extent(0);
  Kokkos::RangePolicy<execution_space, SizeType> policy (0, numRows);

  MV_Copy_Functor<XMV, YMV, SizeType> op (X, Y);
  Kokkos::parallel_for ("KokkosBlas::Copy::S1", policy, op);
}

// Variant of MV_Copy_Generic for single vectors (1-D Views) X and Y.
template<class XV, class YV, class SizeType>
void
V_Copy_Generic (const XV& X, const YV& Y)
{
  static_assert (Kokkos::Impl::is_view<XV>::value, "KokkosBlas::Impl::"
                 "V_Copy_Generic: XV is not a Kokkos::View.");
  static_assert (Kokkos::Impl::is_view<YV>::value, "KokkosBlas::Impl::"
                 "V_Copy_Generic: YV is not a Kokkos::View.");
  static_assert (XV::rank == 1, "KokkosBlas::Impl::"
                 "V_Copy_Generic: XV is not rank 1");
  static_assert (YV::rank == 1, "KokkosBlas::Impl::"
                 "V_Copy_Generic: YV is not rank 1");

  typedef typename XV::execution_space execution_space;
  const SizeType numRows = X.extent(0);
  Kokkos::RangePolicy<execution_space, SizeType> policy (0, numRows);

  V_Copy_Functor<XV, YV, SizeType> op (X, Y);
  Kokkos::parallel_for ("KokkosBlas::Copy::S3", policy, op);

}

}
}
#endif // KOKKOS_BLAS1_MV_IMPL_COPY_HPP_
