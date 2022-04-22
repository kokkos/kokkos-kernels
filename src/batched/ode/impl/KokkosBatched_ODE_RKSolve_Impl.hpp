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

#ifndef __KOKKOSBATCHED_ODE_RKSOLVE_IMPL_HPP__
#define __KOKKOSBATCHED_ODE_RKSOLVE_IMPL_HPP__

#include <Kokkos_ArithTraits.hpp>
#include "Kokkos_Layout.hpp"
#include "Kokkos_MemoryTraits.hpp"
#include "Kokkos_NumericTraits.hpp"

#include <KokkosBatched_ODE_Args.hpp>
#include <KokkosBatched_ODE_RungeKuttaTables.hpp>
#include <KokkosBatched_ODE_SolverStatus.hpp>

//#include <type_traits>

namespace KokkosBatched {
namespace Experimental {
namespace ODE {

//=====================================================================
// Functions used in RKSolve:
//=====================================================================

KOKKOS_FORCEINLINE_FUNCTION double tol(const double y, const double y0,
                                       const double absTol,
                                       const double relTol) {
  return absTol + relTol * Kokkos::fmax(Kokkos::fabs(y), Kokkos::fabs(y0));
}

template <typename View>
KOKKOS_FUNCTION bool isfinite(View &y, const unsigned ndofs) {
  bool is_finite = true;
  for (unsigned i = 0; i < ndofs; ++i) {
    if (!Kokkos::isfinite(y[i])) {
      is_finite = false;
      break;
    }
  }
  return is_finite;
}

//=====================================================================
// RKSolve internal:
//=====================================================================
//
// template <typename TableType>
//
// Use only NumEquations and derivatives from ode in solver...
struct SerialRKSolveInternal {
  template <typename ODEType, typename TableType, typename ViewTypeA,
            typename ViewTypeB>
  KOKKOS_FUNCTION static void step(const ODEType &ode, const TableType &table,
                                   const double t0, const double dt,
                                   ViewTypeA &y, ViewTypeA &y0,
                                   ViewTypeA &ytemp, ViewTypeB &kstack,
                                   const ODEArgs &args, double &err) {
    const int ndofs              = static_cast<int>(y.extent(0));
    static constexpr int nstages = TableType::nstages;

    for (int j = 0; j < nstages; ++j) {
      const int offset = (j + 1) * j / 2;
      for (int n = 0; n < ndofs; ++n) {
        double coeff = 0.0;
        for (int k = 0; k < j; ++k) {  // lower diagonal matrix
          coeff += table.a[k + offset] * kstack(k, n);
        }

        ytemp[n] = y0[n] + dt * coeff;
      }
      auto ksub = Kokkos::subview(kstack, j, Kokkos::ALL);
      ode.derivatives(t0 + table.c[j] * dt, ytemp, ksub);
    }

    for (int n = 0; n < ndofs; ++n) {
      double coeff = 0.0;
      double errJ  = 0.0;
      for (int k = 0; k < nstages; ++k) {
        coeff += table.b[k] * kstack(k, n);
        errJ += table.e[k] * kstack(k, n);
      }
      y[n] = y0[n] + dt * coeff;
      errJ *= dt;
      err = Kokkos::fmax(
          err, Kokkos::fabs(errJ) / tol(y[n], y0[n], args.absTol, args.relTol));
    }
  }
};

//=====================================================================
// RKSolve impl:
//=====================================================================

template <typename TableType>
template <typename ODEType, typename ViewTypeA, typename ViewTypeB>
KOKKOS_FUNCTION ODESolverStatus SerialRKSolve<TableType>::invoke(
    const ODEType &ode, const ODEArgs &args_, ViewTypeA &y, ViewTypeA &y0,
    ViewTypeA &dydt, ViewTypeA &ytemp, ViewTypeB &kstack, double tstart,
    double tend) {
  using Kokkos::fmax;
  using Kokkos::fmin;
  using Kokkos::pow;

  TableType table;

  double epsilon = Kokkos::Experimental::epsilon<double>::value;
  // args checks for valid arguments.
  ODEArgs args(args_);
  args.absTol      = args.absTol > epsilon ? args.absTol : epsilon;
  args.minStepSize = args.minStepSize > epsilon ? args.minStepSize : epsilon;
  const int ndofs  = static_cast<int>(y.extent(0));
  // TODO: Add a whole bunch of checks here to make sure view dimensions line
  // up.
  // Check: size y = size y0 = size ytemp = size dydt

  // TODO: should this be handled with an assert?
  // assert(ode.num_equations() == ndofs, "Mismatched number of dofs in ode
  // solver");
  // if(ode.num_equations != ndofs)
  //{
  // throw std::runtime_error("Mismatched number of dofs in ode solver.");
  //}
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  static_assert(
      Kokkos::is_view<ViewTypeA>::value,
      "KokkosBatched::ODE::SerialRKSolve: ViewTypeA is not a Kokkos::View.");
  static_assert(
      Kokkos::is_view<ViewTypeB>::value,
      "KokkosBatched::ODE::SerialRKSolve: ViewTypeB is not a Kokkos::View.");
  static_assert(
      ViewTypeA::Rank == 1,
      "KokkosBatched::ODE::SerialRKSolve: ViewTypeA must have rank 1.");
  static_assert(
      ViewTypeB::Rank == 2,
      "KokkosBatched::ODE::SerialRKSolve: ViewTypeB must have rank 2.");

  // Check compatibility of dimensions at run time.
  if (y.extent(0) != y0.extent(0) || y.extent(0) != dydt.extent(0) ||
      y.extent(0) != ytemp.extent(0)) {
    KOKKOS_IMPL_DO_NOT_USE_PRINTF(
        "KokkosBatched::ODE::SerialRKSolve: Dimensions of y, y0, ytemp, dydt "
        "do not match: y: %d, y0: %d, "
        "ytemp: %d, dydt: %d\n",
        (int)y.extent(0), (int)y0.extent(0), (int)ytemp.extent(0),
        (int)dydt.extent(0));
    return 1;
  }
  if (ode.num_equations != ndofs) {
    KOKKOS_IMPL_DO_NOT_USE_PRINTF(
        "KokkosBatched::ODE::SerialRKSolve: Mismatched number of dofs."
        "y.extent(0) = %d, ode.num_equations = %d.",
        (int)y.extent(0), (int)ode.num_equations);
    return 1;
  }
    if (TableType::nstages != k.extent(0) || nodfs != k.extent(1){
    KOKKOS_IMPL_DO_NOT_USE_PRINTF(
        "KokkosBatched::ODE::SerialRKSolve: Wrong dimensions of k. k: %d x %d",
        (int)k.extent(0), (int)k.extent(1));
    return 1;
    }
#endif

  double t0 = tstart;

  for (int i = 0; i < ndofs; ++i) {
    y0[i] = y[i];  // why??
  }

  if (!isfinite(y0, ndofs)) {
    return ODESolverStatus::NONFINITE_STATE;
  }

  const double pFactor = -1.0 / table.order;

  // Compute starting time step length:
  double dt = (tend - t0) / args.num_substeps;

  // Main time-stepping loop:
  for (int n = 0; n < args.maxSubSteps; ++n) {
    ode.derivatives(t0, y0, dydt);

    // Limit dt to not exceed t_end
    if (t0 + dt > tend) {
      dt = tend - t0;
    }

    double err = 0.0;
    // Start iterative approach with time step adaptation
    do {
      err = 0.0;
      SerialRKSolveInternal::step(ode, table, t0, dt, y, y0, ytemp, kstack,
                                  args, err);

      // Reduce dt for large error
      if (err > 1 && args.is_adaptive) {
        dt *= fmax(0.2, 0.8 * pow(err, pFactor));

        if (dt < args.minStepSize) {
          return ODESolverStatus::MINIMUM_TIMESTEP_REACHED;
        }
      }

    } while (err > 1 && args.is_adaptive);

    t0 += dt;

    for (int i = 0; i < ndofs; ++i) {
      y0[i] = y[i];  // why again??
    }

    if (t0 >= tend) {
      auto status = !isfinite(y, ndofs) ? ODESolverStatus::NONFINITE_STATE
                                        : ODESolverStatus::SUCCESS;
      return status;
    }

    // Increase dt for small error
    if (err < 0.5 && args.is_adaptive) {
      dt *= fmin(10.0, fmax(2.0, 0.9 * pow(err, pFactor)));
    }
  }
  return ODESolverStatus::FAILED_TO_CONVERGE;
}

}  // namespace ODE
}  // namespace Experimental
}  // namespace KokkosBatched

#endif
