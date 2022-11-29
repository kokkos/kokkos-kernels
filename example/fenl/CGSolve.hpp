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

#ifndef KOKKOS_EXAMPLE_CG_SOLVE
#define KOKKOS_EXAMPLE_CG_SOLVE

#include <cmath>
#include <limits>
#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include <KokkosSparse_spmv.hpp>
#include <Kokkos_Timer.hpp>

#include <WrapMPI.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Example {

template <class ImportType, class SparseMatrixType, class VectorType,
          class TagType = void>
struct CGSolve;

template <class ImportType, class SparseMatrixType, class VectorType>
struct CGSolve<ImportType, SparseMatrixType, VectorType,
               typename std::enable_if<(Kokkos::is_view<VectorType>::value &&
                                        VectorType::rank == 1)>::type> {
  typedef typename VectorType::value_type scalar_type;
  typedef typename VectorType::execution_space execution_space;

  size_t iteration;
  double iter_time;
  double matvec_time;
  double norm_res;

  CGSolve(const ImportType& import, const SparseMatrixType& A,
          const VectorType& b, const VectorType& x,
          const size_t maximum_iteration = 200,
          const double tolerance = std::numeric_limits<double>::epsilon())
      : iteration(0), iter_time(0), matvec_time(0), norm_res(0) {
    const size_t count_owned = import.count_owned;
    const size_t count_total = import.count_owned + import.count_receive;

    // Need input vector to matvec to be owned + received
    VectorType pAll("cg::p", count_total);

    VectorType p =
        Kokkos::subview(pAll, std::pair<size_t, size_t>(0, count_owned));
    VectorType r("cg::r", count_owned);
    VectorType Ap("cg::Ap", count_owned);

    /* r = b - A * x ; */

    /* p  = x       */ Kokkos::deep_copy(p, x);
    /* import p     */ import(pAll);
    /* Ap = A * p   */ KokkosSparse::spmv("N", 1.0, A, pAll, 0.0, Ap);
    /* b - Ap => r  */ KokkosBlas::update(1.0, b, -1.0, Ap, 0.0, r);
    /* p  = r       */ Kokkos::deep_copy(p, r);

    double old_rdot =
        Kokkos::Example::all_reduce(KokkosBlas::dot(r, r), import.comm);

    norm_res  = sqrt(old_rdot);
    iteration = 0;

    Kokkos::Timer wall_clock;
    Kokkos::Timer timer;

    while (tolerance < norm_res && iteration < maximum_iteration) {
      /* pAp_dot = dot( p , Ap = A * p ) */

      timer.reset();
      /* import p    */ import(pAll);
      /* Ap = A * p  */ KokkosSparse::spmv("N", 1.0, A, pAll, 0.0, Ap);
      execution_space().fence();
      matvec_time += timer.seconds();

      const double pAp_dot =
          Kokkos::Example::all_reduce(KokkosBlas::dot(p, Ap), import.comm);
      const double alpha = old_rdot / pAp_dot;

      /* x +=  alpha * p ;  */ KokkosBlas::axpby(alpha, p, 1.0, x);
      /* r += -alpha * Ap ; */ KokkosBlas::axpby(-alpha, Ap, 1.0, r);

      const double r_dot =
          Kokkos::Example::all_reduce(KokkosBlas::dot(r, r), import.comm);
      const double beta = r_dot / old_rdot;

      /* p = r + beta * p ; */ KokkosBlas::axpby(1.0, r, beta, p);

      norm_res = std::sqrt(old_rdot = r_dot);

      ++iteration;
    }

    execution_space().fence();
    iter_time = wall_clock.seconds();
  }
};

}  // namespace Example
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #ifndef KOKKOS_EXAMPLE_CG_SOLVE */
