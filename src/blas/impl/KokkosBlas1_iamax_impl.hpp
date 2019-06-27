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
#ifndef KOKKOSBLAS1_IAMAX_IMPL_HPP_
#define KOKKOSBLAS1_IAMAX_IMPL_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <KokkosBlas1_iamax_spec.hpp>

namespace KokkosBlas {
namespace Impl {

/// \brief Iamax functor for single vectors.
///
/// \tparam RV 0-D output View
/// \tparam XV 1-D input View
/// \tparam MagType Magnitude type
/// \tparam SizeType Index type.  Use int (32 bits) if possible.
template<class RV, class XV, class MagType, class SizeType = typename XV::size_type>
struct V_Iamax_Functor
{
  typedef SizeType                                                 size_type;
  typedef MagType                                                  mag_type;
  typedef typename XV::non_const_value_type                        xvalue_type;
  typedef Kokkos::Details::InnerProductSpaceTraits<xvalue_type>    IPT;
  typedef typename Kokkos::MaxLoc<mag_type,size_type>::value_type  maxloc_type;
  
  typename XV::const_type m_x;

  V_Iamax_Functor (const XV& x) :
    m_x (x)
  {
    static_assert (Kokkos::Impl::is_view<RV>::value,
                   "KokkosBlas::Impl::V_Iamax_Functor: "
                   "R is not a Kokkos::View.");
    static_assert (Kokkos::Impl::is_view<XV>::value,
                   "KokkosBlas::Impl::V_Iamax_Functor: "
                   "X is not a Kokkos::View.");
    static_assert (Kokkos::Impl::is_same<typename RV::value_type,
                   typename RV::non_const_value_type>::value,
                   "KokkosBlas::Impl::V_Iamax_Functor: R is const.  "
                   "It must be nonconst, because it is an output argument "
                   "(we have to be able to write to its entries).");
    static_assert (RV::rank == 0 && XV::rank == 1,
                   "KokkosBlas::Impl::V_Iamax_Functor: "
                   "RV must have rank 0 and XV must have rank 1.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_type & i, maxloc_type & lmaxloc ) const {
	mag_type val = IPT::norm (m_x(i));
    if( val > lmaxloc.val ) { lmaxloc.val = val; lmaxloc.loc = i; }
  }
};

/// \brief Column-wise Iamax functor for multivectors.
///
/// \tparam RV 1-D output View
/// \tparam XMV 2-D input View
/// \tparam MagType Magnitude type
/// \tparam SizeType Index type.  Use int (32 bits) if possible.
template<class RV, class XMV, class MagType, class SizeType = typename XMV::size_type>
struct MV_Iamax_FunctorVector
{
  typedef typename XMV::execution_space                         execution_space;
  typedef typename XMV::memory_space                            memory_space;
  typedef SizeType                                              size_type;
  typedef MagType                                               mag_type;
  typedef typename XMV::non_const_value_type                    xvalue_type;
  typedef Kokkos::Details::InnerProductSpaceTraits<xvalue_type> IPT;
  typedef Kokkos::MaxLoc<mag_type,size_type>                         maxloc_reducer;
  typedef typename Kokkos::MaxLoc<mag_type,size_type>::value_type    maxloc_type;
  typedef typename Kokkos::TeamPolicy<execution_space>::member_type  member_type;

  typename XMV::const_type m_x;
  typename RV::non_const_type m_r;
  
  MV_Iamax_FunctorVector (const RV& r, const XMV& x) :
    m_r (r), m_x (x)
  {
    static_assert (Kokkos::Impl::is_view<RV>::value,
                   "KokkosBlas::Impl::MV_Iamax_FunctorVector: "
                   "R is not a Kokkos::View.");
    static_assert (Kokkos::Impl::is_view<XMV>::value,
                   "KokkosBlas::Impl::MV_Iamax_FunctorVector: "
                   "X is not a Kokkos::View.");
    static_assert (Kokkos::Impl::is_same<typename RV::value_type,
                   typename RV::non_const_value_type>::value,
                   "KokkosBlas::Impl::MV_Iamax_FunctorVector: "
                   "R is const.  It must be nonconst, because it is an output "
                   "argument (we must be able to write to its entries).");
    static_assert (RV::rank == 1 && XMV::rank == 2,
                   "KokkosBlas::Impl::MV_Iamax_FunctorVector: "
                   "RV must have rank 1 and XMV must have rank 2.");
    static_assert (Kokkos::Impl::MemorySpaceAccess< typename XMV::device_type::memory_space, typename RV::device_type::memory_space >::accessible,
                   "KokkosBlas::Impl::MV_Iamax_FunctorVector: "
                   "RV and XMV must have the same memory space if RV is 1-D view.");
  }

    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type &teamMember) const
    {
      const int lid = teamMember.league_rank();// teamId
      const int tid = teamMember.team_rank(); // threadId

      maxloc_type col_maxloc;
      Kokkos::parallel_reduce( Kokkos::TeamThreadRange(teamMember, m_x.extent(0)), [&] (const int& i, maxloc_type& thread_lmaxloc) {
        mag_type val = IPT::norm (m_x(i,lid));
        if( val > thread_lmaxloc.val ) { thread_lmaxloc.val = val; thread_lmaxloc.loc = i; }      
      }, maxloc_reducer(col_maxloc));
      if (tid==0) m_r(lid) = col_maxloc.loc;
    }
};


/// \brief Find the index of the element with the maximum magnitude of the single vector (1-D
///   View) X, and store the result in the 0-D View r.
template<class RV, class XV, class SizeType>
void
V_Iamax_Invoke (const RV& r, const XV& X)
{
  typedef typename XV::execution_space execution_space;
  typedef Kokkos::Details::ArithTraits<typename XV::non_const_value_type> AT;
  typedef typename AT::mag_type mag_type;

  typedef typename Kokkos::MaxLoc<mag_type,SizeType>::value_type maxloc_type;
  typedef Kokkos::MaxLoc<mag_type,SizeType> maxloc_reducer;

  const SizeType numRows = static_cast<SizeType> (X.extent(0));

  // Avoid MaxLoc Reduction if this is a zero length view
  if( numRows == 0 ) {
    Kokkos::deep_copy(r,0);
    return;
  }

  Kokkos::RangePolicy<execution_space, SizeType> policy (0, numRows);

  typedef V_Iamax_Functor<RV, XV, mag_type, SizeType> functor_type;
  functor_type op (X);
  maxloc_type maxloc;
  Kokkos::parallel_reduce ("KokkosBlas::Iamax::S0", policy, op, maxloc_reducer(maxloc)); 

  if(Kokkos::Impl::MemorySpaceAccess< typename maxloc_reducer::result_view_type::memory_space, typename RV::device_type::memory_space >::accessible) {
    r() = maxloc.loc;
  }
  else {
    typename RV::value_type r_val = maxloc.loc;
    Kokkos::View<typename RV::value_type, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > r_loc(&r_val);
    Kokkos::deep_copy(r,r_loc);
  }
}


/// \brief Find the index of the element with the maximum magnitude of the columns of the
///   multivector (2-D View) X, and store result(s) in the 1-D View r.
template<class RV, class XMV, class SizeType>
void
MV_Iamax_Invoke (const RV& r, const XMV& X)
{
  typedef typename XMV::execution_space execution_space;
  typedef Kokkos::Details::ArithTraits<typename XMV::non_const_value_type> AT;
  typedef typename AT::mag_type mag_type;

  const SizeType numRows = static_cast<SizeType> (X.extent(0));

  // Avoid MaxLoc Reduction if this is a zero length view
  if( numRows == 0 ) {
    Kokkos::deep_copy(r,0);
    return;
  }

  typedef Kokkos::TeamPolicy<execution_space>  team_policy;
  const team_policy policy( r.extent(0), Kokkos::AUTO );

  // If the input multivector (2-D View) has only one column, invoke
  // the single-vector version of the kernel.
  if (X.extent(1) == 1) {
    typedef Kokkos::View<typename RV::non_const_value_type,
                         typename RV::array_layout,
                         typename RV::device_type,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>> RV0D;
    RV0D r_0(r, 0);
    auto X_0 = Kokkos::subview (X, Kokkos::ALL (), 0);
    typedef decltype (X_0) XV1D;
    V_Iamax_Invoke<RV0D, XV1D, SizeType> (r_0, X_0);
  }
  else {
    typedef MV_Iamax_FunctorVector<RV, XMV, mag_type, SizeType> functor_type;
    functor_type op (r,X);
    Kokkos::parallel_for ("KokkosBlas::Iamax::S1", policy, op);
  }
}

} // namespace Impl
} // namespace KokkosBlas

#endif // KOKKOSBLAS1_IAMAX_IMPL_HPP_
