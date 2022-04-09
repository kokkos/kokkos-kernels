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

#ifndef __KOKKOSBATCHED_ODE_RKSOLVE_HPP__
#define __KOKKOSBATCHED_ODE_RKSOLVE_HPP__

#include <Kokkos_ArithTraits.hpp>
#include "Kokkos_Layout.hpp"
#include "Kokkos_MemoryTraits.hpp"

#include <KokkosBatched_ODE_Args.h>
#include <KokkosBatched_ODE_RungeKuttaTables.h>
#include <KokkosBatched_ODE_SolverEndStatus.h>

#include <type_traits>

namespace KokkosBatched {
namespace Experimental {
namespace ODE {

template <typename TableType>
struct SerialRKSolve {
  // Type of Runge-Kutta table.
  // Current methods supported are:
  // RKEH, RK12, BS, RKF45, CashKarp, DormandPrince
  const TableType table;
  static constexpr int nstages = TableType::n;

  // Internal solver options. Initialized via ODEArgs passed to
  // the RungeKuttaSolver constructor.
  const SolverControls controls;

  // Initializes RK Solver with paremeters.
  // Passing the ODEArgs to Solver Controls verifies that all
  // settings are withing acceptable tolerances.
  SerialRKSolve(const ODEArgs& args) : controls(args) {}

  template <typename ODEType, typename ViewTypeA, typename ViewTypeB>
  KOKKOS_FUNCTION ODESolverStatus invoke(const ODEType& ode, ViewTypeA y,
                                         ViewTypeA y0, ViewTypeA dydt,
                                         ViewTypeA ytemp, ViewTypeB kstack,
                                         double tstart, double tend) const;

  template <typename ODEType, typename ViewTypeA, typename ViewTypeB>
  KOKKOS_FUNCTION void step(const ODEType& ode, const double t0,
                            const double dt, ViewTypeA y, ViewTypeA y0,
                            ViewTypeA ytemp, ViewTypeB kstack,
                            double& err) const;
};
}  // namespace ODE
}  // namespace Experimental
}  // namespace KokkosBatched

#include "KokkosBatched_ODE_RKSolve_Impl.hpp"

#endif
