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

#ifndef KOKKOSBLAS2_TEAM_GEMV_SPEC_HPP_
#define KOKKOSBLAS2_TEAM_GEMV_SPEC_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_InnerProductSpaceTraits.hpp>
#include <KokkosBlas2_team_gemv_impl.hpp>

namespace KokkosBlas {

template <typename MemberType, typename ArgTrans,
          typename ArgAlgo = Algo::Gemv::Default>
struct TeamGemv {
  template <typename ScalarType, typename AViewType, typename xViewType,
            typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& /*member*/,
                                           const ScalarType /*alpha*/,
                                           const AViewType& /*A*/,
                                           const xViewType& /*x*/,
                                           const ScalarType /*beta*/,
                                           const yViewType& /*y*/);
};

template <typename MemberType, typename ArgTrans, typename ArgAlgo>
struct TeamVectorGemv {
  template <typename ScalarType, typename AViewType, typename xViewType,
            typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& /*member*/,
                                           const ScalarType /*alpha*/,
                                           const AViewType& /*A*/,
                                           const xViewType& /*x*/,
                                           const ScalarType /*beta*/,
                                           const yViewType& /*y*/);
};

///
/// NT
///

template <typename MemberType>
struct TeamGemv<MemberType, Trans::NoTranspose, Algo::Gemv::Unblocked> {
  template <typename ScalarType, typename AViewType, typename xViewType,
            typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType& member, const ScalarType alpha, const AViewType& A,
      const xViewType& x, const ScalarType beta, const yViewType& y) {
    static_assert(AViewType::Rank == 2,
                  "KokkosBlas::TeamGemv requires rank-2 A matrix");
    return Impl::TeamGemvInternal<Algo::Gemv::Unblocked>::invoke(
        member, A.extent(0), A.extent(1), alpha, A.data(), A.stride_0(),
        A.stride_1(), x.data(), x.stride_0(), beta, y.data(), y.stride_0());
  }
};

template <typename MemberType>
struct TeamGemv<MemberType, Trans::NoTranspose, Algo::Gemv::Blocked> {
  template <typename ScalarType, typename AViewType, typename xViewType,
            typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType& member, const ScalarType alpha, const AViewType& A,
      const xViewType& x, const ScalarType beta, const yViewType& y) {
    static_assert(AViewType::Rank == 2,
                  "KokkosBlas::TeamGemv requires rank-2 A matrix");
    return Impl::TeamGemvInternal<Algo::Gemv::Blocked>::invoke(
        member, A.extent(0), A.extent(1), alpha, A.data(), A.stride_0(),
        A.stride_1(), x.data(), x.stride_0(), beta, y.data(), y.stride_0());
  }
};

///
/// T
///

template <typename MemberType>
struct TeamGemv<MemberType, Trans::Transpose, Algo::Gemv::Unblocked> {
  template <typename ScalarType, typename AViewType, typename xViewType,
            typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType& member, const ScalarType alpha, const AViewType& A,
      const xViewType& x, const ScalarType beta, const yViewType& y) {
    static_assert(AViewType::Rank == 2,
                  "BLAS TeamGemv requires rank-2 A matrix");
    return Impl::TeamGemvInternal<Algo::Gemv::Unblocked>::invoke(
        member, A.extent(1), A.extent(0), alpha, A.data(), A.stride_1(),
        A.stride_0(), x.data(), x.stride_0(), beta, y.data(), y.stride_0());
  }
};

template <typename MemberType>
struct TeamGemv<MemberType, Trans::Transpose, Algo::Gemv::Blocked> {
  template <typename ScalarType, typename AViewType, typename xViewType,
            typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType& member, const ScalarType alpha, const AViewType& A,
      const xViewType& x, const ScalarType beta, const yViewType& y) {
    static_assert(AViewType::Rank == 2,
                  "BLAS TeamGemv requires rank-2 A matrix");
    return Impl::TeamGemvInternal<Algo::Gemv::Blocked>::invoke(
        member, A.extent(1), A.extent(0), alpha, A.data(), A.stride_1(),
        A.stride_0(), x.data(), x.stride_0(), beta, y.data(), y.stride_0());
  }
};

///
/// CT
///

template <typename MemberType>
struct TeamGemv<MemberType, Trans::ConjTranspose, Algo::Gemv::Unblocked> {
  template <typename ScalarType, typename AViewType, typename xViewType,
            typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType& member, const ScalarType alpha, const AViewType& A,
      const xViewType& x, const ScalarType beta, const yViewType& y) {
    static_assert(AViewType::Rank == 2,
                  "BLAS TeamGemv requires rank-2 A matrix");
    return Impl::TeamGemvInternal<Algo::Gemv::Unblocked>::invoke(
        member, Impl::OpConj{}, A.extent(1), A.extent(0), alpha, A.data(),
        A.stride_1(), A.stride_0(), x.data(), x.stride_0(), beta, y.data(),
        y.stride_0());
  }
};

template <typename MemberType>
struct TeamGemv<MemberType, Trans::ConjTranspose, Algo::Gemv::Blocked> {
  template <typename ScalarType, typename AViewType, typename xViewType,
            typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType& member, const ScalarType alpha, const AViewType& A,
      const xViewType& x, const ScalarType beta, const yViewType& y) {
    static_assert(AViewType::Rank == 2,
                  "BLAS TeamGemv requires rank-2 A matrix");
    return Impl::TeamGemvInternal<Algo::Gemv::Blocked>::invoke(
        member, Impl::OpConj{}, A.extent(1), A.extent(0), alpha, A.data(),
        A.stride_1(), A.stride_0(), x.data(), x.stride_0(), beta, y.data(),
        y.stride_0());
  }
};

///
/// NT
///

template <typename MemberType>
struct TeamVectorGemv<MemberType, Trans::NoTranspose, Algo::Gemv::Unblocked> {
  template <typename ScalarType, typename AViewType, typename xViewType,
            typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType& member, const ScalarType alpha, const AViewType& A,
      const xViewType& x, const ScalarType beta, const yViewType& y) {
    static_assert(AViewType::Rank == 2,
                  "Batched TeamVectorGemv requires rank-2 A matrix");
    return Impl::TeamVectorGemvInternal<Algo::Gemv::Unblocked>::invoke(
        member, A.extent(0), A.extent(1), alpha, A.data(), A.stride_0(),
        A.stride_1(), x.data(), x.stride_0(), beta, y.data(), y.stride_0());
  }
};

///
/// T
///

template <typename MemberType>
struct TeamVectorGemv<MemberType, Trans::Transpose, Algo::Gemv::Unblocked> {
  template <typename ScalarType, typename AViewType, typename xViewType,
            typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType& member, const ScalarType alpha, const AViewType& A,
      const xViewType& x, const ScalarType beta, const yViewType& y) {
    static_assert(AViewType::Rank == 2,
                  "Batched TeamVectorGemv requires rank-2 A matrix");
    return Impl::TeamVectorGemvInternal<Algo::Gemv::Unblocked>::invoke(
        member, A.extent(1), A.extent(0), alpha, A.data(), A.stride_1(),
        A.stride_0(), x.data(), x.stride_0(), beta, y.data(), y.stride_0());
  }
};

///
/// CT
///

template <typename MemberType>
struct TeamVectorGemv<MemberType, Trans::ConjTranspose, Algo::Gemv::Unblocked> {
  template <typename ScalarType, typename AViewType, typename xViewType,
            typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType& member, const ScalarType alpha, const AViewType& A,
      const xViewType& x, const ScalarType beta, const yViewType& y) {
    static_assert(AViewType::Rank == 2,
                  "Batched TeamVectorGemv requires rank-2 A matrix");
    return Impl::TeamVectorGemvInternal<Algo::Gemv::Unblocked>::invoke(
        member, Impl::OpConj{}, A.extent(1), A.extent(0), alpha, A.data(),
        A.stride_1(), A.stride_0(), x.data(), x.stride_0(), beta, y.data(),
        y.stride_0());
  }
};

}  // namespace KokkosBlas

#endif
