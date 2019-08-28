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

#ifndef KOKKOSSPARSE_IMPL_SPTRSV_SYMBOLIC_HPP_
#define KOKKOSSPARSE_IMPL_SPTRSV_SYMBOLIC_HPP_

/// \file Kokkos_Sparse_impl_sptrsv_symbolic.hpp
/// \brief Implementation(s) of sparse triangular solve.

#include <KokkosKernels_config.h>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_sptrsv_handle.hpp>

//#define LVL_OUTPUT_INFO

namespace KokkosSparse {
namespace Impl {
namespace Experimental {


template < class TriSolveHandle, class RowMapType, class EntriesType >
void lower_tri_symbolic ( TriSolveHandle &thandle, const RowMapType drow_map, const EntriesType dentries) {

 if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP ||
      thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1 )
/*   || thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHED_TP2 )*/
 {
  // Scheduling currently compute on host - need host copy of all views

  typedef typename TriSolveHandle::size_type size_type;

  typedef typename TriSolveHandle::nnz_lno_view_t  DeviceEntriesType;
  typedef typename TriSolveHandle::nnz_lno_view_t::HostMirror HostEntriesType;

  typedef typename TriSolveHandle::signed_nnz_lno_view_t DeviceSignedEntriesType;
  typedef typename TriSolveHandle::signed_nnz_lno_view_t::HostMirror HostSignedEntriesType;

  typedef typename TriSolveHandle::signed_integral_t signed_integral_t;

  size_type nrows = thandle.get_nrows();

  auto row_map = Kokkos::create_mirror_view(drow_map);
  Kokkos::deep_copy(row_map, drow_map);

  auto entries = Kokkos::create_mirror_view(dentries);
  Kokkos::deep_copy(entries, dentries);
  
  DeviceEntriesType dnodes_per_level = thandle.get_nodes_per_level();
  HostEntriesType nodes_per_level = Kokkos::create_mirror_view(dnodes_per_level);
  Kokkos::deep_copy(nodes_per_level, dnodes_per_level);

  DeviceEntriesType dnodes_grouped_by_level = thandle.get_nodes_grouped_by_level();
  HostEntriesType nodes_grouped_by_level = Kokkos::create_mirror_view(dnodes_grouped_by_level);
  Kokkos::deep_copy(nodes_grouped_by_level, dnodes_grouped_by_level);

  DeviceSignedEntriesType dlevel_list = thandle.get_level_list();
  HostSignedEntriesType level_list = Kokkos::create_mirror_view(dlevel_list);
  Kokkos::deep_copy(level_list, dlevel_list);

  HostSignedEntriesType previous_level_list( Kokkos::ViewAllocateWithoutInitializing("previous_level_list"), nrows );
  Kokkos::deep_copy( previous_level_list, signed_integral_t(-1) );


  // node 0 is trivially independent in lower tri solve, start with it in level 0
  size_type level = 0;
  auto starting_node = 0;
  level_list(starting_node) = 0;
  size_type node_count = 1; //lower tri: starting with node 0 already in level 0

  nodes_per_level(0) = 1;
  nodes_grouped_by_level(0) = starting_node;

  while (node_count < nrows) {

    for ( size_type row = 1; row < nrows; ++row ) { // row 0 already included
      if ( level_list(row) == -1 ) { // unmarked
        bool is_root = true;
        signed_integral_t ptrstart = row_map(row);
        signed_integral_t ptrend   = row_map(row+1);

        for (signed_integral_t offset = ptrstart; offset < ptrend; ++offset) {
          size_type col = entries(offset);
          if ( previous_level_list(col) == -1 && col != row ) { // unmarked
            if ( col < row ) {
              is_root = false;
              break;
            }
          }
        } // end for offset , i.e. cols of this row

        if ( is_root == true ) {
          level_list(row) = level;
          nodes_per_level(level) += 1;
          nodes_grouped_by_level(node_count) = row;
          node_count += 1;
        }

      } // end if
    } // end for row

    //Kokkos::deep_copy(previous_level_list, level_list);
    for ( size_type i = 0; i < nrows; ++i ) {
      previous_level_list(i) = level_list(i);
    }

    level += 1;
  } // end while

  thandle.set_symbolic_complete();
  thandle.set_num_levels(level);

  // Output check
#ifdef LVL_OUTPUT_INFO
  std::cout << "  set symbolic complete: " << thandle.is_symbolic_complete() << std::endl;
  std::cout << "  set num levels: " << thandle.get_num_levels() << std::endl;

  std::cout << "  lower_tri_symbolic result: " << std::endl;
  for ( size_type i = 0; i < node_count; ++i )
  { std::cout << "node: " << i << "  level_list = " << level_list(i) << std::endl; }

  for ( size_type i = 0; i < level; ++i )
  { std::cout << "level: " << i << "  nodes_per_level = " << nodes_per_level(i) << std::endl; }

  for ( size_type i = 0; i < node_count; ++i )
  { std::cout << "i: " << i << "  nodes_grouped_by_level = " << nodes_grouped_by_level(i) << std::endl; }
#endif

  Kokkos::deep_copy(dnodes_grouped_by_level, nodes_grouped_by_level);
  Kokkos::deep_copy(dnodes_per_level, nodes_per_level);
  Kokkos::deep_copy(dlevel_list, level_list);
 }
#ifdef KOKKOSKERNELS_ENABLE_SUPERNODAL
 else if (thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_NAIVE ||
          thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_ETREE) {
  typedef typename TriSolveHandle::size_type size_type;

  typedef typename TriSolveHandle::nnz_lno_view_t  DeviceEntriesType;
  typedef typename TriSolveHandle::nnz_lno_view_t::HostMirror HostEntriesType;

  typedef typename TriSolveHandle::signed_nnz_lno_view_t DeviceSignedEntriesType;
  typedef typename TriSolveHandle::signed_nnz_lno_view_t::HostMirror HostSignedEntriesType;

  typedef typename TriSolveHandle::signed_integral_t signed_integral_t;

  typedef typename TriSolveHandle::supercols_t             supercols_t;
  typedef typename TriSolveHandle::supercols_t::HostMirror supercols_host_t;


  // rowptr: pointer to begining of each row (CRS)
  auto row_map = Kokkos::create_mirror_view(drow_map);
  Kokkos::deep_copy(row_map, drow_map);

  // # of nodes per level
  DeviceEntriesType dnodes_per_level = thandle.get_nodes_per_level ();
  HostEntriesType nodes_per_level = Kokkos::create_mirror_view (dnodes_per_level);

  // node ids in each level
  DeviceEntriesType dnodes_grouped_by_level = thandle.get_nodes_grouped_by_level ();
  HostEntriesType nodes_grouped_by_level = Kokkos::create_mirror_view (dnodes_grouped_by_level);

  // map node id to level that this node belongs to
  DeviceSignedEntriesType dlevel_list = thandle.get_level_list ();
  HostSignedEntriesType level_list = Kokkos::create_mirror_view (dlevel_list);

  // type of kernels used at each level
  int size_tol = thandle.get_supernode_size_tol();
  supercols_host_t kernel_type_by_level = thandle.get_kernel_type_host ();

  // # of supernodal columns
  size_type nsuper = thandle.get_num_supernodes ();
  const int* supercols = thandle.get_supercols_host ();

  // workspace
  signed_integral_t max_lwork = 0;
  supercols_host_t work_offset_host = thandle.get_work_offset_host ();
  if (thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_NAIVE) {
    // >> Naive (sequential) version: going through supernodal column one at a time from 1 to nsuper
    // Set number of level equal to be the number of supernodal columns
    thandle.set_num_levels (nsuper);

    // Set up level sets: going through supernodal column one at a time from 1 to nsuper
    for (size_type s = 0; s < nsuper; s++) {
      nodes_per_level (s) = 1;           // # of nodes per level
      nodes_grouped_by_level (s) = s;    // only one task per level (task id)
      level_list (s) = s;                // map task id to level

      // local/max workspace size
      size_type row = supercols[s];
      signed_integral_t lwork = row_map (row+1) - row_map(row);
      if (max_lwork < lwork) {
        max_lwork = lwork;
      }

      // kernel type
      if (lwork < size_tol) {
        kernel_type_by_level (s) = 0;
      } else {
        kernel_type_by_level (s) = 2;
      }
      work_offset_host (s) = 0;
    }
  } else {
    /* initialize the ready tasks with leaves */
    const int *parents = thandle.get_etree_parents ();
    int *check = (int*)calloc(nsuper, sizeof(int));
    for (size_type s = 0; s < nsuper; s++) {
      if (parents[s] >= 0) {
        check[parents[s]] ++;
      }
    }

    signed_integral_t num_done = 0;
    signed_integral_t level = 0;
    #define profile_supernodal_etree
    #ifdef profile_supernodal_etree
    // min, max, tot size of supernodes
    signed_integral_t max_nsrow = 0;
    signed_integral_t min_nsrow = 0;
    signed_integral_t tot_nsrow = 0;

    signed_integral_t max_nscol = 0;
    signed_integral_t min_nscol = 0;
    signed_integral_t tot_nscol = 0;

    // min, max, tot num of leaves
    signed_integral_t max_nleave = 0;
    signed_integral_t min_nleave = 0;
    signed_integral_t tot_nleave = 0;
    #endif
    while (num_done < nsuper) {
      nodes_per_level (level) = 0; 
      // look for ready-tasks
      signed_integral_t lwork = 0;
      signed_integral_t num_leave = 0;
      signed_integral_t avg_nsrow = 0;
      for (size_type s = 0; s < nsuper; s++) {
        if (check[s] == 0) {
          //printf( " %d: ready[%d]=%d\n",level, num_done+num_leave, s );
          nodes_per_level (level) ++; 
          nodes_grouped_by_level (num_done + num_leave) = s;
          level_list (s) = level;

          // work offset
          work_offset_host (s) = lwork;
 
          // update workspace size
          size_type row = supercols[s];
          signed_integral_t nsrow = row_map (row+1) - row_map(row);
          lwork += nsrow;

          // total supernode size
          avg_nsrow += supercols[s+1]-supercols[s];

          #ifdef profile_supernodal_etree
          // gather static if requested
          signed_integral_t nscol = supercols[s+1] - supercols[s];
          if (tot_nscol == 0) {
            max_nscol = nscol;
            min_nscol = nscol;

            max_nsrow = nsrow;
            min_nsrow = nsrow;
          } else {
            if (max_nscol < nscol) {
              max_nscol = nscol;
            }
            if (min_nscol > nscol) {
              min_nscol = nscol;
            }

            if (max_nsrow < nsrow) {
              max_nsrow = nsrow;
            }
            if (min_nsrow > nsrow) {
              min_nsrow = nsrow;
            }
          }
          tot_nsrow += nsrow;
          tot_nscol += nscol;
          #endif

          num_leave ++;
        }
      }
      //printf( " lwork = %d\n",lwork );
      if (lwork > max_lwork) {
        max_lwork = lwork;
      }

      // average supernode size at this level
      avg_nsrow /= num_leave;
      // kernel type
      if (avg_nsrow < size_tol) {
        kernel_type_by_level (level) = 0;
      } else {
        kernel_type_by_level (level) = 2;
      }
      #ifdef profile_supernodal_etree
      if (level == 0) {
        max_nleave = num_leave;
        min_nleave = num_leave;
      } else {
        if (max_nleave < num_leave) {
          max_nleave = num_leave;
        }
        if (min_nleave > num_leave) {
          min_nleave = num_leave;
        }
      }
      tot_nleave += num_leave;
      #endif

      // free the dependency
      for (signed_integral_t task = 0; task < num_leave; task++) {
        size_type s = nodes_grouped_by_level (num_done + task);
        check[s] = -1;
        //printf( " %d: check[%d]=%d ",level,s,check[s]);
        if (parents[s] >= 0) {
          check[parents[s]] --;
          //printf( " -> check[%d]=%d",parents[s],check[parents[s]]);
        }
        //printf( "\n" );
      }
      num_done += num_leave;
      //printf( " level=%d: num_done=%d / %d\n",level,num_done,nsuper );
      level ++;
    }
    #ifdef profile_supernodal_etree
    std::cout << "   * supernodal rows: min = " << min_nsrow  << "\t max = " << max_nsrow  << "\t avg = " << tot_nsrow/nsuper << std::endl;
    std::cout << "   * supernodal cols: min = " << min_nscol  << "\t max = " << max_nscol  << "\t avg = " << tot_nscol/nsuper << std::endl;
    std::cout << "   * numer of leaves: min = " << min_nleave << "\t max = " << max_nleave << "\t avg = " << tot_nleave/level << std::endl;
    #endif
    // Set number of level equal to be the number of supernodal columns
    thandle.set_num_levels (level);
    free(check);
  }
  // workspace size
  thandle.set_workspace_size (max_lwork);
  // workspace offset initialized to be zero
  supercols_t work_offset = thandle.get_work_offset ();
  Kokkos::deep_copy (work_offset, work_offset_host);

  // kernel types
  supercols_t dkernel_type_by_level = thandle.get_kernel_type ();
  Kokkos::deep_copy (dkernel_type_by_level, kernel_type_by_level);

  // deep copy to device (of scheduling info)
  Kokkos::deep_copy (dnodes_grouped_by_level, nodes_grouped_by_level);
  Kokkos::deep_copy (dnodes_per_level, nodes_per_level);
  Kokkos::deep_copy (dlevel_list, level_list);

  thandle.set_symbolic_complete();
 }
#endif
} // end lowertri_level_sched


template < class TriSolveHandle, class RowMapType, class EntriesType >
void upper_tri_symbolic ( TriSolveHandle &thandle, const RowMapType drow_map, const EntriesType dentries ) {

 if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP ||
      thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1 )
/*   || thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHED_TP2 )*/
 {
  // Scheduling currently compute on host - need host copy of all views

  typedef typename TriSolveHandle::size_type size_type;

  typedef typename TriSolveHandle::nnz_lno_view_t  DeviceEntriesType;
  typedef typename TriSolveHandle::nnz_lno_view_t::HostMirror HostEntriesType;

  typedef typename TriSolveHandle::signed_nnz_lno_view_t DeviceSignedEntriesType;
  typedef typename TriSolveHandle::signed_nnz_lno_view_t::HostMirror HostSignedEntriesType;

  typedef typename TriSolveHandle::signed_integral_t signed_integral_t;

  size_type nrows = thandle.get_nrows();

  auto row_map = Kokkos::create_mirror_view(drow_map);
  Kokkos::deep_copy(row_map, drow_map);

  auto entries = Kokkos::create_mirror_view(dentries);
  Kokkos::deep_copy(entries, dentries);
  
  DeviceEntriesType dnodes_per_level = thandle.get_nodes_per_level();
  HostEntriesType nodes_per_level = Kokkos::create_mirror_view(dnodes_per_level);
  Kokkos::deep_copy(nodes_per_level, dnodes_per_level);

  DeviceEntriesType dnodes_grouped_by_level = thandle.get_nodes_grouped_by_level();
  HostEntriesType nodes_grouped_by_level = Kokkos::create_mirror_view(dnodes_grouped_by_level);
  Kokkos::deep_copy(nodes_grouped_by_level, dnodes_grouped_by_level);

  DeviceSignedEntriesType dlevel_list = thandle.get_level_list();
  HostSignedEntriesType level_list = Kokkos::create_mirror_view(dlevel_list);
  Kokkos::deep_copy(level_list, dlevel_list);

  HostSignedEntriesType previous_level_list( Kokkos::ViewAllocateWithoutInitializing("previous_level_list"), nrows);
  Kokkos::deep_copy( previous_level_list, signed_integral_t(-1) );


  // final row is trivially independent in upper tri solve, start with it in level 0
  size_type level = 0;
  auto starting_node = nrows - 1;
  level_list(starting_node) = 0;
  size_type node_count = 1; //upper tri: starting with node n already in level 0

  nodes_per_level(0) = 1;
  nodes_grouped_by_level(0) = starting_node;

  while (node_count < nrows) {

    for ( signed_integral_t row = nrows-2; row >= 0; --row ) { // row 0 already included
      if ( level_list(row) == -1 ) { // unmarked
        bool is_root = true;
        signed_integral_t ptrstart = row_map(row);
        signed_integral_t ptrend   = row_map(row+1);

        for (signed_integral_t offset = ptrend-1; offset >= ptrstart; --offset) {
          signed_integral_t col = entries(offset);
          if ( previous_level_list(col) == -1 && col != row ) { // unmarked
            if ( col > row ) {
              is_root = false;
              break;
            }
          }
        } // end for offset , i.e. cols of this row

        if ( is_root == true ) {
          level_list(row) = level;
          nodes_per_level(level) += 1;
          nodes_grouped_by_level(node_count) = row;
          node_count += 1;
        }

      } // end if
    } // end for row

    //Kokkos::deep_copy(previous_level_list, level_list);
    for ( size_type i = 0; i < nrows; ++i ) {
      previous_level_list(i) = level_list(i);
    }

    level += 1;
  } // end while

  thandle.set_symbolic_complete();
  thandle.set_num_levels(level);

  // Output check
#ifdef LVL_OUTPUT_INFO
  std::cout << "  set symbolic complete: " << thandle.is_symbolic_complete() << std::endl;
  std::cout << "  set num levels: " << thandle.get_num_levels() << std::endl;

  std::cout << "  upper_tri_symbolic result: " << std::endl;
  for ( size_type i = 0; i < node_count; ++i )
  { std::cout << "node: " << i << "  level_list = " << level_list(i) << std::endl; }

  for ( size_type i = 0; i < level; ++i )
  { std::cout << "level: " << i << "  nodes_per_level = " << nodes_per_level(i) << std::endl; }

  for ( size_type i = 0; i < node_count; ++i )
  { std::cout << "i: " << i << "  nodes_grouped_by_level = " << nodes_grouped_by_level(i) << std::endl; }
#endif

  Kokkos::deep_copy(dnodes_grouped_by_level, nodes_grouped_by_level);
  Kokkos::deep_copy(dnodes_per_level, nodes_per_level);
  Kokkos::deep_copy(dlevel_list, level_list);
 }
#ifdef KOKKOSKERNELS_ENABLE_SUPERNODAL
 else if (thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_NAIVE ||
          thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_ETREE) {
  typedef typename TriSolveHandle::size_type size_type;

  typedef typename TriSolveHandle::nnz_lno_view_t  DeviceEntriesType;
  typedef typename TriSolveHandle::nnz_lno_view_t::HostMirror HostEntriesType;

  typedef typename TriSolveHandle::signed_nnz_lno_view_t DeviceSignedEntriesType;
  typedef typename TriSolveHandle::signed_nnz_lno_view_t::HostMirror HostSignedEntriesType;

  typedef typename TriSolveHandle::signed_integral_t signed_integral_t;

  typedef typename TriSolveHandle::supercols_t             supercols_t;
  typedef typename TriSolveHandle::supercols_t::HostMirror supercols_host_t;


  // rowptr: pointer to begining of each row (CRS)
  auto row_map = Kokkos::create_mirror_view(drow_map);
  Kokkos::deep_copy(row_map, drow_map);

  // # of nodes per level
  DeviceEntriesType dnodes_per_level = thandle.get_nodes_per_level ();
  HostEntriesType nodes_per_level = Kokkos::create_mirror_view (dnodes_per_level);

  // node ids in each level
  DeviceEntriesType dnodes_grouped_by_level = thandle.get_nodes_grouped_by_level ();
  HostEntriesType nodes_grouped_by_level = Kokkos::create_mirror_view (dnodes_grouped_by_level);

  // type of kernels used at each level
  int size_tol = thandle.get_supernode_size_tol();
  supercols_host_t kernel_type_by_level = thandle.get_kernel_type_host ();

  // map node id to level that this node belongs to
  DeviceSignedEntriesType dlevel_list = thandle.get_level_list ();
  HostSignedEntriesType level_list = Kokkos::create_mirror_view (dlevel_list);

  // # of supernodal columns
  size_type nsuper = thandle.get_num_supernodes ();
  const int* supercols = thandle.get_supercols_host ();

  // workspace
  signed_integral_t max_lwork = 0;
  supercols_host_t work_offset_host = thandle.get_work_offset_host ();
  if (thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_NAIVE) {
    // >> Naive (sequential) version: going through supernodal column one at a time from 1 to nsuper
    // Set number of level equal to be the number of supernodal columns
    thandle.set_num_levels (nsuper);

    // Set up level sets: going through supernodal column one at a time from 1 to nsuper
    for (size_type s = 0; s < nsuper; s++) {
      nodes_per_level (s) = 1;                  // # of nodes per level
      nodes_grouped_by_level (s) = nsuper-1-s;  // only one task per level (task id)
      level_list (nsuper-1-s) = s;              // map task id to level

      size_type row = supercols[s];
      signed_integral_t lwork = row_map (row+1) - row_map(row);
      if (max_lwork < lwork) {
        max_lwork = lwork;
      }
      work_offset_host (s) = 0;

      if (lwork < size_tol) {
        kernel_type_by_level (s) = 0;
      } else {
        kernel_type_by_level (s) = 2;
      }
    }
  }
  else {
    /* schduling from bottom to top (as for L-solve) *
     * then reverse it for U-solve                   */

    /* initialize the ready tasks with leaves */
    const int *parents = thandle.get_etree_parents ();
    int *check  = (int*)calloc(nsuper, sizeof(int));
    for (size_type s = 0; s < nsuper; s++) {
      if (parents[s] >= 0) {
        check[parents[s]] ++;
      }
    }
    //printf( " Init:\n" );
    //for (size_type s = 0; s <nsuper; s++) printf( " check[%d] = %d\n",s,check[s] );

    size_type nrows = thandle.get_nrows();
    HostEntriesType inverse_nodes_per_level ("nodes_per_level", nrows);
    HostEntriesType inverse_nodes_grouped_by_level ("nodes_grouped_by_level", nrows);

    signed_integral_t num_done = 0;
    signed_integral_t level = 0;
    #ifdef profile_supernodal_etree
    // min, max, tot size of supernodes
    signed_integral_t max_nsrow = 0;
    signed_integral_t min_nsrow = 0;
    signed_integral_t tot_nsrow = 0;

    signed_integral_t max_nscol = 0;
    signed_integral_t min_nscol = 0;
    signed_integral_t tot_nscol = 0;

    // min, max, tot num of leaves
    signed_integral_t max_nleave = 0;
    signed_integral_t min_nleave = 0;
    signed_integral_t tot_nleave = 0;
    #endif
    while (num_done < nsuper) {
      nodes_per_level (level) = 0; 
      // look for ready-tasks
      signed_integral_t lwork = 0;
      signed_integral_t num_leave = 0;
      for (size_type s = 0; s < nsuper; s++) {
        if (check[s] == 0) {
          inverse_nodes_per_level (level) ++; 
          inverse_nodes_grouped_by_level (num_done + num_leave) = s;
          //printf( " level=%d: %d/%d: s=%d\n",level, num_done+num_leave,nsuper, s );

          // work offset
          work_offset_host (s) = lwork;
 
          // update workspace size
          size_type row = supercols[s];
          signed_integral_t nsrow = row_map (row+1) - row_map(row);
          //printf( " %d %d %d %d\n",num_done+num_leave, level, nsrow, supercols[s+1]-supercols[s] );
          lwork += nsrow;

          #ifdef profile_supernodal_etree
          // gather static if requested
          signed_integral_t nscol = supercols[s+1] - supercols[s];
          if (tot_nscol == 0) {
            max_nscol = nscol;
            min_nscol = nscol;

            max_nsrow = nsrow;
            min_nsrow = nsrow;
          } else {
            if (max_nscol < nscol) {
              max_nscol = nscol;
            }
            if (min_nscol > nscol) {
              min_nscol = nscol;
            }

            if (max_nsrow < nsrow) {
              max_nsrow = nsrow;
            }
            if (min_nsrow > nsrow) {
              min_nsrow = nsrow;
            }
          }
          tot_nsrow += nsrow;
          tot_nscol += nscol;
          #endif

          num_leave ++;
        }
      }
      //printf( " lwork = %d\n",lwork );
      if (lwork > max_lwork) {
        max_lwork = lwork;
      }
      #ifdef profile_supernodal_etree
      if (level == 0) {
        max_nleave = num_leave;
        min_nleave = num_leave;
      } else {
        if (max_nleave < num_leave) {
          max_nleave = num_leave;
        }
        if (min_nleave > num_leave) {
          min_nleave = num_leave;
        }
      }
      tot_nleave += num_leave;
      #endif

      // free the dependency
      for (signed_integral_t task = 0; task < num_leave; task++) {
        size_type s = inverse_nodes_grouped_by_level (num_done + task);
        check[s] = -1;
        //printf( " %d: check[%d]=%d ",level,s,check[s]);
        if (parents[s] >= 0) {
          check[parents[s]] --;
          //printf( " -> check[%d]=%d",parents[s],check[parents[s]]);
        }
        //printf( "\n" );
      }
      num_done += num_leave;
      //printf( " level=%d: num_done=%d / %d\n",level,num_done,nsuper );
      level ++;
    }
    free(check);
    #ifdef profile_supernodal_etree
    std::cout << "   * supernodal rows: min = " << min_nsrow  << "\t max = " << max_nsrow  << "\t avg = " << tot_nsrow/nsuper << std::endl;
    std::cout << "   * supernodal cols: min = " << min_nscol  << "\t max = " << max_nscol  << "\t avg = " << tot_nscol/nsuper << std::endl;
    std::cout << "   * numer of leaves: min = " << min_nleave << "\t max = " << max_nleave << "\t avg = " << tot_nleave/level << std::endl;
    #endif

    // now invert the lists
    num_done = 0;
    signed_integral_t num_level = level;
    for (level = 0; level < num_level; level ++) {
      signed_integral_t num_leave = inverse_nodes_per_level (num_level - level - 1);
      nodes_per_level (level) = num_leave;
      //printf( " -> nodes_per_level(%d -> %d) = %d\n",num_level-level-1, level, num_leave );

      signed_integral_t avg_nsrow = 0;
      for (signed_integral_t task = 0; task < num_leave; task++) {
        signed_integral_t s = inverse_nodes_grouped_by_level (nsuper - num_done - 1);

        nodes_grouped_by_level (num_done) = s;
        level_list (s) = level;
        //printf( " -> level=%d: %d->%d: s=%d\n",level, nsuper-num_done-1, num_done, s );
        num_done ++;
        avg_nsrow += supercols[s+1]-supercols[s];
      }

      // average supernodal size at this level
      avg_nsrow /= num_leave;
      // kernel type
      if (avg_nsrow < size_tol) {
        kernel_type_by_level (level) = 0;
      } else {
        kernel_type_by_level (level) = 2;
      }
    }

    // Set number of levels
    thandle.set_num_levels (num_level);
  }
  // workspace size
  thandle.set_workspace_size (max_lwork);
  // workspace offset initialized to be zero
  supercols_t work_offset = thandle.get_work_offset ();
  Kokkos::deep_copy (work_offset, work_offset_host);

  // kernel type
  supercols_t dkernel_type_by_level = thandle.get_kernel_type ();
  Kokkos::deep_copy (dkernel_type_by_level, kernel_type_by_level);

  // deep copy to device (info about scheduling)
  Kokkos::deep_copy (dnodes_grouped_by_level, nodes_grouped_by_level);
  Kokkos::deep_copy (dnodes_per_level, nodes_per_level);
  Kokkos::deep_copy (dlevel_list, level_list);

  thandle.set_symbolic_complete ();
 }
#endif
} // end uppertri_level_sched


} // namespace Experimental
} // namespace Impl
} // namespace KokkosSparse

#endif
