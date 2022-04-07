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

#ifndef __KOKKOSBATCHED_ODE_RKSOLVERS_HPP__
#define __KOKKOSBATCHED_ODE_RKSOLVERS_HPP__

#include <Kokkos_ArithTraits.hpp>
#include "Kokkos_Layout.hpp"
#include "Kokkos_MemoryTraits.hpp"

#include <KokkosBatched_ODE_Args.h>
#include <KokkosBatched_ODE_RungeKuttaTables.h>
#include <KokkosBatched_ODE_SolverEndStatus.h>
#include <KokkosBatched_ODE_AllocationState.h>

#include <type_traits>

namespace KokkosBatched {
namespace ode {

KOKKOS_FORCEINLINE_FUNCTION double tol(const double y, const double y0,
                                       const double absTol,
                                       const double relTol) {
  return absTol + relTol * Kokkos::fmax(Kokkos::fabs(y), Kokkos::fabs(y0));
}

template <typename View>
KOKKOS_FUNCTION bool isfinite(View& y, const unsigned ndofs) {
  bool is_finite = true;
  for (unsigned i = 0; i < ndofs; ++i) {
    if (!Kokkos::isfinite(y[i])) {
      is_finite = false;
      break;
    }
  }
  return is_finite;
}
template <typename TableType>
struct RungeKuttaSolver {
  const TableType table;
  const SolverControls controls;
  static constexpr int nstages = TableType::n;

  RungeKuttaSolver(const ODEArgs& args) : controls(args) {}

  template <typename ODEType, typename StateType>
  KOKKOS_FUNCTION ODESolverStatus solve(const ODEType& ode, double tstart,
                                        double tend, StateType& s) const {
    using Kokkos::fmax;
    using Kokkos::fmin;
    using Kokkos::pow;

    const int ndofs = s.ndofs();

    // TODO: should this be handled with an assert?
    // assert(ode.num_equations() == ndofs, "Mismatched number of dofs in ode
    // solver");
    // if(ode.num_equations != ndofs)
    //{
    // throw std::runtime_error("Mismatched number of dofs in ode solver.");
    //}

    double t0 = tstart;

    for (int i = 0; i < ndofs; ++i) {
      s.y0[i] = s.y[i];
    }

    if (!isfinite(s.y0, ndofs)) {
      return ODESolverStatus::NONFINITE_STATE;
    }

    const double pFactor = -1.0 / table.order;

    double dt = (tend - t0) / controls.num_substeps;

    // Main time-stepping loop:
    for (int n = 0; n < controls.maxSubSteps; ++n) {
      ode.derivatives(t0, s.y0, s.dydt);

      // Limit dt to not exceed t_end
      if (t0 + dt > tend) {
        dt = tend - t0;
      }

      double err = 0.0;
      // Start iterative approach with time step adaptation
      do {
        err = 0.0;
        step(ode, t0, dt, s, err);

        // Reduce dt for large error
        if (err > 1 && controls.is_adaptive) {
          dt *= fmax(0.2, 0.8 * pow(err, pFactor));

          if (dt < controls.minStepSize) {
            return ODESolverStatus::MINIMUM_TIMESTEP_REACHED;
          }
        }

      } while (err > 1 && controls.is_adaptive);

      t0 += dt;

      for (int i = 0; i < ndofs; ++i) {
        s.y0[i] = s.y[i];
      }

      if (t0 >= tend) {
        auto status = !isfinite(s.y, ndofs) ? ODESolverStatus::NONFINITE_STATE
                                            : ODESolverStatus::SUCCESS;
        return status;
      }

      // Increase dt for small error
      if (err < 0.5 && controls.is_adaptive) {
        dt *= fmin(10.0, fmax(2.0, 0.9 * pow(err, pFactor)));
      }
    }
    return ODESolverStatus::FAILED_TO_CONVERGE;
  }

  template <typename ODEType, typename StateType>
  KOKKOS_FUNCTION void step(const ODEType& ode, const double t0,
                            const double dt, StateType& s, double& err) const {
    const int ndofs = s.ndofs();

    for (int j = 0; j < nstages; ++j) {
      const int offset = (j + 1) * j / 2;
      for (int n = 0; n < ndofs; ++n) {
        double coeff = 0.0;
        for (int k = 0; k < j; ++k) {  // lower diagonal matrix
          coeff += table.a[k + offset] * s.k(k, n);
        }

        s.ytemp[n] = s.y0[n] + dt * coeff;
      }
      auto ksub = Kokkos::subview(s.k, j, Kokkos::ALL);
      ode.derivatives(t0 + table.c[j] * dt, s.ytemp, ksub);
    }

    for (int n = 0; n < ndofs; ++n) {
      double coeff = 0.0;
      double errJ  = 0.0;
      for (int k = 0; k < nstages; ++k) {
        coeff += table.b[k] * s.k(k, n);
        errJ += table.e[k] * s.k(k, n);
      }
      s.y[n] = s.y0[n] + dt * coeff;
      errJ *= dt;
      err = Kokkos::fmax(
          err, Kokkos::fabs(errJ) /
                   tol(s.y[n], s.y0[n], controls.absTol, controls.relTol));
    }
  }
};
}  // namespace ode
}  // namespace KokkosBatched

#endif
