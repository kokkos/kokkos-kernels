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

#ifndef KOKKOS_EXAMPLE_FENL_HPP
#define KOKKOS_EXAMPLE_FENL_HPP

#include <stdlib.h>
#include <BoxElemPart.hpp>
#include <WrapMPI.hpp>

namespace Kokkos {
namespace Example {
namespace FENL {

struct Perf {
  size_t global_elem_count ;
  size_t global_node_count ;
  size_t newton_iter_count ;
  size_t cg_iter_count ;
  double map_ratio ;
  double fill_node_set ;
  double scan_node_count ;
  double fill_graph_entries ;
  double sort_graph_entries ;
  double fill_element_graph ;
  double create_sparse_matrix ;
  double fill_time ;
  double bc_time ;
  double matvec_time ;
  double cg_time ;
  double newton_residual ;
  double error_max ;

};

template < class Device , BoxElemPart::ElemOrder ElemOrder >
Perf fenl(
  MPI_Comm comm ,
  const int use_print ,
  const int use_trials ,
  const int use_atomic ,
  const int global_elems[] );

} /* namespace FENL */
} /* namespace Example */
} /* namespace Kokkos */

#endif /* #ifndef KOKKOS_EXAMPLE_FENL_HPP */

