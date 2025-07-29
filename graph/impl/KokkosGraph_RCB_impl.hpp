//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOSGRAPH_RCB_IMPL_HPP
#define KOKKOSGRAPH_RCB_IMPL_HPP

#include "Kokkos_Core.hpp"
#include "KokkosKernels_Utils.hpp"
#include <vector>
#include <algorithm>

namespace KokkosGraph {
namespace Impl {

template <typename perm_view_type>
struct FillOneIncrementFunctor {
  using ordinal_t = typename perm_view_type::value_type;
  perm_view_type A;

  FillOneIncrementFunctor(perm_view_type &A_) : A(A_) {}
  KOKKOS_INLINE_FUNCTION void operator()(ordinal_t i) const { A(i) = i; }
};

template <typename reducer_type, typename view_type, typename ordinal_t>
struct MinMaxReducerFunctor {
  using reducer_value_type = typename reducer_type::value_type;
  view_type A;
  MinMaxReducerFunctor(const view_type &A_) : A(A_) {}
  KOKKOS_INLINE_FUNCTION void operator()(ordinal_t i, reducer_value_type &lminmax) const {
    typename view_type::value_type val = A(i);
    if (val < lminmax.min_val) lminmax.min_val = val;
    if (val > lminmax.max_val) lminmax.max_val = val;
  }
};

template <typename perm_view_type1, typename perm_view_type2, typename coors_view_type>
struct UpdatePermAndMeshFunctor {
  using ordinal_t = typename perm_view_type1::value_type;
  perm_view_type1 reverse_perm_bisect;  // a subview
  perm_view_type1 prev_reverse_perm;    // a subview
  perm_view_type2 perm;                 // a full-length view
  perm_view_type2 reverse_perm;         // a full-length view
  coors_view_type coordinates_orig;     // a subview
  coors_view_type coordinates_new;      // a subview
  ordinal_t p1_size;
  ordinal_t offset;
  ordinal_t N;  // length of reverse_perm_bisect
  ordinal_t ndim;

  UpdatePermAndMeshFunctor(const perm_view_type1 &reverse_perm_bisect_, const perm_view_type1 &prev_reverse_perm_,
                           perm_view_type2 &perm_, perm_view_type2 &reverse_perm_,
                           const coors_view_type &coordinates_orig_, coors_view_type &coordinates_new_,
                           const ordinal_t &p1_size_, const ordinal_t &offset_)
      : reverse_perm_bisect(reverse_perm_bisect_),
        prev_reverse_perm(prev_reverse_perm_),
        perm(perm_),
        reverse_perm(reverse_perm_),
        coordinates_orig(coordinates_orig_),
        coordinates_new(coordinates_new_),
        p1_size(p1_size_),
        offset(offset_) {
    N    = static_cast<ordinal_t>(reverse_perm_bisect.extent(0));
    ndim = static_cast<ordinal_t>(coordinates_orig.extent(1));
  }
  KOKKOS_INLINE_FUNCTION void operator()(ordinal_t i) const {
    // orig_lcl_idx: 0 --> (N-1)
    ordinal_t orig_lcl_idx = reverse_perm_bisect(i);

    // new_lcl_idx: 0 --> (N-1)
    ordinal_t new_lcl_idx;
    if (i < p1_size) {
      new_lcl_idx = i;
    } else {
      new_lcl_idx = (N - 1 - i) + p1_size;
    }
    // Calculate new_gbl_idx by adding an offset
    ordinal_t new_gbl_idx = new_lcl_idx + offset;

    // Retrieve gbl_orig_idx
    ordinal_t gbl_orig_idx = prev_reverse_perm(orig_lcl_idx);

    // Update perm at gbl_orig_idx location
    perm(gbl_orig_idx) = new_gbl_idx;

    // Update reverse_perm at new_gbl_idx location
    reverse_perm(new_gbl_idx) = gbl_orig_idx;

    // Update coordinates with bisecting results
    for (ordinal_t j = 0; j < ndim; j++) {
      coordinates_new(new_lcl_idx, j) = coordinates_orig(orig_lcl_idx, j);
    }
  }
};

template <typename view_type, typename value_type>
void find_min_max(const view_type &A, value_type &min_val, value_type &max_val) {
  using execution_space = typename view_type::device_type::execution_space;
  using reducer_type    = Kokkos::MinMax<value_type>;
  size_t n_elements     = static_cast<size_t>(A.extent(0));
  if (n_elements == 0) {
    min_val = static_cast<value_type>(0);
    max_val = static_cast<value_type>(0);
    return;
  }
  typename reducer_type::value_type result;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<execution_space>(0, n_elements),
                          MinMaxReducerFunctor<reducer_type, view_type, size_t>(A), reducer_type(result));
  min_val = result.min_val;
  max_val = result.max_val;
}

/**
 * @brief Bisect and assign partition indices to coordinate list
 */
template <typename coors_view_type, typename value_type, typename index_view_type, typename ordinal_type>
inline void bisect(const coors_view_type &coors_1d, const value_type &init_min_val, const value_type &init_max_val,
                   index_view_type &reverse_perm_bisect, ordinal_type &p1_size, ordinal_type &p2_size) {
  ordinal_type N     = static_cast<ordinal_type>(coors_1d.extent(0));
  value_type min_val = init_min_val;
  value_type max_val = init_max_val;
  value_type p1_weight, p2_weight;
  value_type prev_weight_ratio = 0;
  value_type curr_weight_ratio;
  while (1) {
    value_type mid_point = (max_val + min_val) / 2.0;
    p1_size              = 0;
    p2_size              = 0;
    // Use one array "reverse_perm_bisect" to store indices of both partitions
    for (ordinal_type i = 0; i < N; i++) {
      if (coors_1d(i) > mid_point) {  // partition 1: store forward
        reverse_perm_bisect(p1_size) = i;
        p1_size++;
      } else {  // partition 2: store backward
        reverse_perm_bisect(N - 1 - p2_size) = i;
        p2_size++;
      }
    }
    p1_weight = static_cast<value_type>(p1_size);
    p2_weight = static_cast<value_type>(p2_size);

    curr_weight_ratio = std::max(p1_weight, p2_weight) / std::min(p1_weight, p2_weight);

    if (curr_weight_ratio < 1.1)
      break;
    else if (curr_weight_ratio == prev_weight_ratio)
      break;
    else {
      // Update min_val or max_val to calculate a new mid_point
      // Idea: shift mid_point to the heavier partition
      if (p1_weight > p2_weight)
        min_val = mid_point;
      else
        max_val = mid_point;
      prev_weight_ratio = curr_weight_ratio;
    }
  }
}

}  // namespace Impl
}  // namespace KokkosGraph
#endif
