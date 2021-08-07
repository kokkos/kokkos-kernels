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
// Questions? Contact Jennifer Loe (jloe@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>

// This function creates a diagonal sparse matrix for testing matrix operations.
// The elements on the diagonal are 1, 2, ..., n-1, n.
// If "invert" is true, it will return the inverse of the above diagonal matrix.
//
template <typename crsMat_t>
crsMat_t kk_generate_diag_matrix(typename crsMat_t::const_ordinal_type n,
                                 const bool invert = false){
  typedef typename crsMat_t::ordinal_type ot;
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type   cols_view_t;
  typedef typename crsMat_t::values_type::non_const_type values_view_t;

  typedef typename row_map_view_t::non_const_value_type size_type;
  typedef typename cols_view_t::non_const_value_type lno_t;
  typedef typename values_view_t::non_const_value_type scalar_t;

  row_map_view_t rowmap_view("rowmap_view", n+1);
  cols_view_t columns_view("colsmap_view", n);
  values_view_t values_view("values_view", n);

  {
    typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view (rowmap_view);
    typename cols_view_t::HostMirror hc = Kokkos::create_mirror_view (columns_view);
    typename values_view_t::HostMirror hv = Kokkos::create_mirror_view (values_view);

    for (lno_t i = 0; i <= n; ++i){
      hr(i) = size_type(i);
    }

    for (ot i = 0; i < n; ++i){
      hc(i) = lno_t(i);
      if(invert){
        hv(i) = scalar_t(1.0)/(scalar_t(i + 1));
      }
      else{
        hv(i) = scalar_t(i + 1);
      }
    }
    Kokkos::deep_copy (rowmap_view , hr);
    Kokkos::deep_copy (columns_view , hc);
    Kokkos::deep_copy (values_view , hv);
  }

  graph_t static_graph (columns_view, rowmap_view);
  crsMat_t crsmat("CrsMatrix", n, values_view, static_graph);
  return crsmat;
}
