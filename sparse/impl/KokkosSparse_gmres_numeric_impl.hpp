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

#ifndef KOKKOSSPARSE_IMPL_GMRES_NUMERIC_HPP_
#define KOKKOSSPARSE_IMPL_GMRES_NUMERIC_HPP_

/// \file KokkosSparse_gmres_numeric_impl.hpp
/// \brief Implementation(s) of the numeric phase of GMRES.

#include <KokkosKernels_config.h>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_gmres_handle.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <KokkosSparse_spadd.hpp>
#include <KokkosSparse_Utils.hpp>
#include <KokkosSparse_SortCrs.hpp>
#include <KokkosKernels_Utils.hpp>

#include <limits>

//#define NUMERIC_OUTPUT_INFO

namespace KokkosSparse {
namespace Impl {
namespace Experimental {

template <class GmresHandle>
struct GmresWrap {
  //
  // Useful types
  //
  using execution_space         = typename GmresHandle::execution_space;
  using index_t                 = typename GmresHandle::nnz_lno_t;
  using size_type               = typename GmresHandle::size_type;
  using scalar_t                = typename GmresHandle::nnz_scalar_t;
  using HandleDeviceEntriesType = typename GmresHandle::nnz_lno_view_t;
  using HandleDeviceRowMapType  = typename GmresHandle::nnz_row_view_t;
  using HandleDeviceValueType   = typename GmresHandle::nnz_value_view_t;
  using karith                  = typename Kokkos::ArithTraits<scalar_t>;
  using policy_type             = typename GmresHandle::TeamPolicy;
  using member_type             = typename policy_type::member_type;
  using range_policy            = typename GmresHandle::RangePolicy;

  /**
   * The main gmres numeric function.
   */
  template <class KHandle, class ARowMapType, class AEntriesType,
            class AValuesType, class LRowMapType, class LEntriesType,
            class LValuesType, class URowMapType, class UEntriesType,
            class UValuesType>
  static void gmres_numeric(KHandle& kh, GmresHandle& thandle,
                           const ARowMapType& A_row_map,
                           const AEntriesType& A_entries,
                           const AValuesType& A_values, LRowMapType& L_row_map,
                           LEntriesType& L_entries, LValuesType& L_values,
                           URowMapType& U_row_map, UEntriesType& U_entries,
                           UValuesType& U_values, bool deterministic) {
  }  // end gmres_numeric

};  // struct GmresWrap

}  // namespace Experimental
}  // namespace Impl
}  // namespace KokkosSparse

#endif
