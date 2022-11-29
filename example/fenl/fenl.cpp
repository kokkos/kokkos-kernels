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
#include <HexElement.hpp>
#include <fenl_impl.hpp>

namespace Kokkos {
namespace Example {
namespace FENL {

#if defined( KOKKOS_ENABLE_THREADS )

template
Perf fenl< Kokkos::Threads , Kokkos::Example::BoxElemPart::ElemLinear >(
  MPI_Comm comm ,
  const int use_print ,
  const int use_trials ,
  const int use_atomic ,
  const int global_elems[] );


template
Perf fenl< Kokkos::Threads , Kokkos::Example::BoxElemPart::ElemQuadratic >(
  MPI_Comm comm ,
  const int use_print ,
  const int use_trials ,
  const int use_atomic ,
  const int global_elems[] );

#endif


#if defined (KOKKOS_ENABLE_OPENMP)

template
Perf fenl< Kokkos::OpenMP , Kokkos::Example::BoxElemPart::ElemLinear >(
  MPI_Comm comm ,
  const int use_print ,
  const int use_trials ,
  const int use_atomic ,
  const int global_elems[] );


template
Perf fenl< Kokkos::OpenMP , Kokkos::Example::BoxElemPart::ElemQuadratic >(
  MPI_Comm comm ,
  const int use_print ,
  const int use_trials ,
  const int use_atomic ,
  const int global_elems[] );

#endif

#if defined( KOKKOS_ENABLE_CUDA )

template
Perf fenl< Kokkos::Cuda , Kokkos::Example::BoxElemPart::ElemLinear >(
  MPI_Comm comm ,
  const int use_print ,
  const int use_trials ,
  const int use_atomic ,
  const int global_elems[] );


template
Perf fenl< Kokkos::Cuda , Kokkos::Example::BoxElemPart::ElemQuadratic >(
  MPI_Comm comm ,
  const int use_print ,
  const int use_trials ,
  const int use_atomic ,
  const int global_elems[] );

#endif


} /* namespace FENL */
} /* namespace Example */
} /* namespace Kokkos */

