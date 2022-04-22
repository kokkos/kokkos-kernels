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

#ifndef __KOKKOSBATCHED_ODE_ARGS_HPP__
#define __KOKKOSBATCHED_ODE_ARGS_HPP__

#include <limits>

namespace KokkosBatched {
namespace Experimental {
namespace ODE {

enum class ODESolverType { RKEH, RK12, RKBS, RKF45, CashKarp, DOPRI5 };

struct ODEArgs {
  double absTol    = 1e-12;   // Absolute tolerance
  double relTol    = 1e-6;    // Relative tolerance
  int maxSubSteps  = 100000;  // Max number of time steps.
  int num_substeps = 10;      // Starting (=Minimum) number of time steps.

  // Minimum time step length:
  // Initially set to an unrealistic number.
  // Will be set to std::numeric_limits<double>::epsilon() if not set by user
  double minStepSize = std::numeric_limits<double>::lowest();

  // int order            = 0;  // CVODE
  // int multistep_method = 2;  // CV_BDF;    // CVODE
  // int iteration_method = 2;  // CV_NEWTON; // CVODE

  // int matrix_solver = 0;  // CVODE and LSODE
  // int band          = 0;  // CVODE and LSODE

  // Use adaptive time stepping?
  bool is_adaptive = true;
};

}  // namespace ODE
}  // namespace Experimental
}  // namespace KokkosBatched

#endif
