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

#ifndef KOKKOSGRAPH_RCB_HPP
#define KOKKOSGRAPH_RCB_HPP

#include "KokkosGraph_RCB_impl.hpp"

namespace KokkosGraph {
namespace Experimental {

// The recursive coordinate bisection (RCB) algorithm partitions the graph
// according to the coordinates of the mesh points. This function returns
// a vector containing sizes of partitions, the coordinate list (organized in RCB
// order), a permutation array describing the mapping from the original order
// to RCB order, and a reverse permutation array describing the mapping from
// the RCB order to the original order

template <typename coors_view_type, typename perm_view_type, typename ordinal_type>
std::vector<ordinal_type>
recursive_coordinate_bisection(coors_view_type &coordinates, perm_view_type &perm, perm_view_type &reverse_perm, const ordinal_type &n_levels) {
  using execution_space = typename coors_view_type::device_type::execution_space;
  using scalar_t        = typename coors_view_type::value_type;

  if (n_levels < 2) {
    std::ostringstream os;
    os << "KokkosGraph::Experimental::recursive_coordinate_bisection only works with more than 1 level of bisection (i.e., 2 partitions).";
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  ordinal_type N    = static_cast<ordinal_type>(coordinates.extent(0));
  ordinal_type ndim = static_cast<ordinal_type>(coordinates.extent(1));

  // Allocate coordinates views on device memory
  coors_view_type coordinates_bisect (Kokkos::view_alloc(Kokkos::WithoutInitializing, "coordinates_bisect"), N, ndim);
  perm_view_type  reverse_perm_bisect(Kokkos::view_alloc(Kokkos::WithoutInitializing, "reverse_perm_bisect"), N);
  perm_view_type  prev_reverse_perm  (Kokkos::view_alloc(Kokkos::WithoutInitializing, "prev_reverse_perm"), N);

  // Create host mirrors of device views
  typename coors_view_type::HostMirror h_coordinates         = Kokkos::create_mirror_view( coordinates );
  typename perm_view_type::HostMirror  h_perm                = Kokkos::create_mirror_view( perm );
  typename perm_view_type::HostMirror  h_reverse_perm        = Kokkos::create_mirror_view( reverse_perm );
  typename perm_view_type::HostMirror  h_reverse_perm_bisect = Kokkos::create_mirror_view( reverse_perm_bisect );

  // Copy coordinates from device memory to host memory because bisecting is currently executed on host
  Kokkos::deep_copy( h_coordinates, coordinates );

  // Initialize
  Kokkos::parallel_for(Kokkos::RangePolicy<execution_space, ordinal_type>(0, N), Impl::FillOneIncrementFunctor<perm_view_type>(perm));
  Kokkos::deep_copy( reverse_perm, perm );

  ordinal_type n_partitions = 1; // number of partitions at the previous level (initial value is 1, i.e., starting with the entire mesh points)
  ordinal_type max_n_partitions = static_cast <ordinal_type>(std::pow(2, n_levels - 1));
  std::vector<ordinal_type> partition_sizes(max_n_partitions); // contain the number of basis functions (or elements) per partition in the previous level
  partition_sizes[0] = N; // starting with the entire mesh points
  std::vector<ordinal_type> partition_sizes_tmp(max_n_partitions);

  // Start RCB
  for (ordinal_type lvl = 1; lvl < n_levels; lvl++)  { // skip level 0 and start from level 1
    ordinal_type coordinates_offset = 0; // always start from beginning of the mesh points
    ordinal_type cnt_partitions = 0;

    // Keep a copy of reverse permutation list
    Kokkos::deep_copy( prev_reverse_perm, reverse_perm );

    for (ordinal_type p = 0; p < n_partitions; p++) { // go through each partition of previous level and do bisecting
      if (p > 0) {
        // Calculate coordinates offset of the current partition based on the previous partition
        coordinates_offset += partition_sizes[p - 1];
      }

      ordinal_type N0 = partition_sizes[p]; // partiion size (or length)
      ordinal_type p1_size = 0;
      ordinal_type p2_size = 0;
      auto sub_coordinates           = Kokkos::subview(coordinates,           Kokkos::make_pair(coordinates_offset, coordinates_offset + N0), Kokkos::ALL());
      auto sub_h_coordinates         = Kokkos::subview(h_coordinates,         Kokkos::make_pair(coordinates_offset, coordinates_offset + N0), Kokkos::ALL());
      auto sub_coordinates_bisect    = Kokkos::subview(coordinates_bisect,    Kokkos::make_pair(coordinates_offset, coordinates_offset + N0), Kokkos::ALL());
      auto sub_reverse_perm_bisect   = Kokkos::subview(reverse_perm_bisect,   Kokkos::make_pair(coordinates_offset, coordinates_offset + N0));
      auto sub_prev_reverse_perm     = Kokkos::subview(prev_reverse_perm,     Kokkos::make_pair(coordinates_offset, coordinates_offset + N0));
      auto sub_h_reverse_perm_bisect = Kokkos::subview(h_reverse_perm_bisect, Kokkos::make_pair(coordinates_offset, coordinates_offset + N0));

      // Find min, max, and span of each dimension
      scalar_t x_min, x_max, y_min, y_max, z_min, z_max;
      scalar_t x_span = 0.0;
      scalar_t y_span = 0.0;
      scalar_t z_span = 0.0;

      auto x_coors = Kokkos::subview (sub_coordinates, Kokkos::ALL(), 0);
      Impl::find_min_max(x_coors, x_min, x_max);
      x_span = x_max - x_min;

      if (ndim > 1) {
        auto y_coors = Kokkos::subview (sub_coordinates, Kokkos::ALL(), 1);
        Impl::find_min_max(y_coors, y_min, y_max);
        y_span = y_max - y_min;
      }

      if (ndim > 2) {
        auto z_coors = Kokkos::subview (sub_coordinates, Kokkos::ALL(), 2);
        Impl::find_min_max(z_coors, z_min, z_max);
        z_span = z_max - z_min;
      }

      // Bisect partition on the most elongated dimension (host execution, for now)
      if ((x_span >= y_span) && (x_span >= z_span)) {
        auto h_x_coors = Kokkos::subview (sub_h_coordinates, Kokkos::ALL(), 0);
        Impl::bisect(h_x_coors, x_min, x_max, sub_h_reverse_perm_bisect, p1_size, p2_size);
      }
      else if ((y_span >= x_span) && (y_span >= z_span)) {
        auto h_y_coors = Kokkos::subview (sub_h_coordinates, Kokkos::ALL(), 1);
        Impl::bisect(h_y_coors, y_min, y_max, sub_h_reverse_perm_bisect, p1_size, p2_size);
      }
      else {
        auto h_z_coors = Kokkos::subview (sub_h_coordinates, Kokkos::ALL(), 2);
        Impl::bisect(h_z_coors, z_min, z_max, sub_h_reverse_perm_bisect, p1_size, p2_size);
      }

      Kokkos::deep_copy( sub_reverse_perm_bisect, sub_h_reverse_perm_bisect );

      // Update global permutation and reverse permutation lists and shuffle coordinates using bisecting results
      Impl::UpdatePermAndMeshFunctor<decltype(sub_reverse_perm_bisect),
                                     perm_view_type,
                                     decltype(sub_coordinates)> func(sub_reverse_perm_bisect,
                                                                     sub_prev_reverse_perm,
                                                                     perm,
                                                                     reverse_perm,
                                                                     sub_coordinates,
                                                                     sub_coordinates_bisect,
                                                                     p1_size, coordinates_offset);
      Kokkos::RangePolicy<execution_space, ordinal_type> policy(0, N0);
      Kokkos::parallel_for(policy, func);

      //std::cout << "    Level " << lvl << ", bisecting partition " << p << " (size: " << N0 << ") of level " << (lvl - 1) << ": 1st sub-partition's size " << p1_size << ", 2nd sub-partition's size " << p2_size << std::endl;

      if (p1_size != 0) {
        partition_sizes_tmp[cnt_partitions] = p1_size;
        cnt_partitions++;
      }
      if (p2_size != 0) {
        partition_sizes_tmp[cnt_partitions] = p2_size;
        cnt_partitions++;
      }
    } // end Partition loop

    // Update coordinates
    Kokkos::deep_copy( coordinates,   coordinates_bisect );
    Kokkos::deep_copy( h_coordinates, coordinates_bisect );

    // Update the number of partitions of this level (used for bisections in the next level)
    n_partitions = cnt_partitions;

    // Update partition sizes of this level (used for bisections in the next level)
    std::copy(partition_sizes_tmp.begin(), partition_sizes_tmp.end(), partition_sizes.begin());
  } // end Level loop

  if (n_partitions < max_n_partitions)
    partition_sizes.resize(n_partitions);

  return partition_sizes;
}

}  // namespace Experimental
}  // namespace KokkosGraph

#endif
