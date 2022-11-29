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

#ifndef KOKKOS_EXAMPLE_WRAP_MPI
#define KOKKOS_EXAMPLE_WRAP_MPI

#include <Kokkos_Macros.hpp>
#include <string>

#if defined( KOKKOS_ENABLE_MPI )

#include <mpi.h>

namespace Kokkos {
namespace Example {

inline
double all_reduce( double value , MPI_Comm comm )
{
  double local = value ;
  MPI_Allreduce( & local , & value , 1 , MPI_DOUBLE , MPI_SUM , comm );
  return value ;
}

inline
double all_reduce_max( double value , MPI_Comm comm )
{
  double local = value ;
  MPI_Allreduce( & local , & value , 1 , MPI_DOUBLE , MPI_MAX , comm );
  return value ;
}

} // namespace Example
} // namespace Kokkos

#elif ! defined( KOKKOS_ENABLE_MPI )

/* Wrap the the MPI_Comm type and heavily used MPI functions
 * to reduce the number of '#if defined( KOKKOS_ENABLE_MPI )'
 * blocks which have to be sprinkled throughout the examples.
 */

typedef int MPI_Comm ;

inline int MPI_Comm_size( MPI_Comm , int * size ) { *size = 1 ; return 0 ; }
inline int MPI_Comm_rank( MPI_Comm , int * rank ) { *rank = 0 ; return 0 ; }
inline int MPI_Barrier( MPI_Comm ) { return 0; }

namespace Kokkos {
namespace Example {

inline
double all_reduce( double value , MPI_Comm ) { return value ; }

inline
double all_reduce_max( double value , MPI_Comm ) { return value ; }

} // namespace Example
} // namespace Kokkos

#endif /* ! defined( KOKKOS_ENABLE_MPI ) */
#endif /* #ifndef KOKKOS_EXAMPLE_WRAP_MPI */

