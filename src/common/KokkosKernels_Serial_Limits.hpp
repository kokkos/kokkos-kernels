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

#ifndef KOKKOSKERNELS_SERIAL_LIMITS_HPP
#define KOKKOSKERNELS_SERIAL_LIMITS_HPP

namespace KokkosKernels {

/// \brief Threshold sizes for shifting Serial vs non-Serial execution spaces 
//         for dot, axpby, spmv routines

template < typename iType >
struct ThresholdSizes {

  static_assert( std::is_integral<iType>::value, "KokkosKernels::ThresholdSizes Error: Must be templated on integral-type" );

  static constexpr iType dot_serial_limit = 20000;
  static constexpr iType spmv_serial_limit = 200;
  static constexpr iType axpby_serial_limit = 2000;

};

/// \brief GetSmallProblemDeviceType: Helper routine to return proper Device type
///        (i.e. Kokkos::Serial execution space) when running a small problem size
namespace Impl {
template < class T >
struct GetSmallProblemDeviceType {
  typedef T type;
};

#ifdef KOKKOS_ENABLE_SERIAL
template < class ExecSpace >
struct GetSmallProblemDeviceType< Kokkos::Device< ExecSpace, Kokkos::HostSpace > >
{
  typedef Kokkos::Device< Kokkos::Serial, Kokkos::HostSpace > type;
};

template <>
struct GetSmallProblemDeviceType< Kokkos::Serial >
{
  typedef Kokkos::Device< Kokkos::Serial, Kokkos::HostSpace > type;
};

#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct GetSmallProblemDeviceType< Kokkos::OpenMP >
{
  typedef Kokkos::Device< Kokkos::Serial, Kokkos::HostSpace > type;
};
#endif

#ifdef KOKKOS_ENABLE_THREADS
template <>
struct GetSmallProblemDeviceType< Kokkos::Threads >
{
  typedef Kokkos::Device< Kokkos::Serial, Kokkos::HostSpace > type;
};
#endif

template <>
struct GetSmallProblemDeviceType< Kokkos::HostSpace >
{
  typedef Kokkos::Device< Kokkos::Serial, Kokkos::HostSpace > type;
};

#endif

} // namespace Impl
} // namespace KokkosKernels

#endif

