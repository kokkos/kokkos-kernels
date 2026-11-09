// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSODE_NEWTON_HPP
#define KOKKOSODE_NEWTON_HPP

/// \author Luc Berger-Vergiat (lberge@sandia.gov)
/// \file KokkosODE_Newton.hpp

#include "Kokkos_Core.hpp"

#include "KokkosODE_Types.hpp"
#include "KokkosODE_Newton_impl.hpp"

namespace KokkosODE {
namespace Experimental {

/// \brief Newton solver for non-linear system of equations
///
/// \param sys [in]: the nonlinear system to solve
/// \param params [in]: Newton solver parameters
/// \param J [out]: storage for the Jacobian, extent (neqs, neqs)
/// \param tmp [out]: temporary storage, extent (neqs, neqs + 4) when the
/// scalar type is at least as large as int, or (neqs, 2 * (neqs + 5)) for
/// smaller scalar types such as half_t (see
/// KokkosODE::Impl::newton_tmp_num_cols). Each row of tmp, or tmp as a whole,
/// must be contiguous in memory.
/// \param y0 [in/out]: initial guess, set to the solution on output
/// \param rhs [out]: storage for the residual, extent neqs
/// \param update [out]: storage for the Newton update, extent neqs
/// \param scale [in]: scaling factors used to compute the norm of the update
/// \param newton_iterations [out]: number of iterations performed
struct Newton {
  template <class system_type, class mat_type, class ini_vec_type, class rhs_vec_type, class update_type,
            class scale_type>
  KOKKOS_FUNCTION static newton_solver_status Solve(const system_type& sys, const Newton_params& params,
                                                    const mat_type& J, const mat_type& tmp, const ini_vec_type& y0,
                                                    const rhs_vec_type& rhs, const update_type& update,
                                                    const scale_type& scale, int& newton_iterations) {
    return KokkosODE::Impl::NewtonSolve(sys, params, J, tmp, y0, rhs, update, scale, newton_iterations);
  }
};

}  // namespace Experimental
}  // namespace KokkosODE

#endif  // KOKKOSODE_NEWTON_HPP
