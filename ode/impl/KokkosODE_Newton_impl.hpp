// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSODE_NEWTON_IMPL_HPP
#define KOKKOSODE_NEWTON_IMPL_HPP

#include "Kokkos_Core.hpp"
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_Gesv.hpp"
#include "KokkosBlas1_nrm2.hpp"
#include "KokkosBlas1_scal.hpp"
#include "KokkosBlas1_axpby.hpp"
#include "KokkosKernels_InnerProductSpaceTraits.hpp"

#include "KokkosBatched_Getrf.hpp"
#include "KokkosBatched_Getrs.hpp"
#include "KokkosODE_Types.hpp"
#include "KokkosKernels_Utils.hpp"

namespace KokkosODE {
namespace Impl {

// Number of columns required for NewtonSolve's tmp argument, which must have
// extent (neqs, newton_tmp_num_cols<scalar_type>(neqs)). tmp is used either as
// the workspace for SerialGesv (neqs x (neqs + 4) scalars) or to hold neqs int
// pivots for SerialGetrf/SerialGetrs. The pivots are written contiguously
// starting at an int-aligned address in tmp's first row (or tmp's entire span,
// if contiguous). When scalar_type is smaller than int (e.g. half_t), the
// first row must be widened so the pivots and alignment padding fit there.
template <class scalar_type>
KOKKOS_INLINE_FUNCTION constexpr int newton_tmp_num_cols(int neqs) {
  return sizeof(scalar_type) < sizeof(int) ? 2 * (neqs + 5) : neqs + 4;
}

template <class system_type, class mat_type, class ini_vec_type, class rhs_vec_type, class update_type,
          class scale_type>
KOKKOS_FUNCTION KokkosODE::Experimental::newton_solver_status NewtonSolve(
    system_type& sys, const KokkosODE::Experimental::Newton_params& params, mat_type& J, mat_type& tmp,
    ini_vec_type& y0, rhs_vec_type& rhs, update_type& update, const scale_type& scale, int& newton_iterations) {
  using newton_solver_status = KokkosODE::Experimental::newton_solver_status;
  using value_type           = typename ini_vec_type::non_const_value_type;

  // Use double for scalar convergence checks (norm, rate, tol).
  // When value_type is half_t, half_t-precision norms and rates are
  // unreliable: the rate can round to exactly 1.0, triggering false
  // divergence.  Computing norms in double makes the rate computation
  // accurate while the vectors themselves remain in the original type.
  using norm_type    = double;
  using value_traits = KokkosKernels::ArithTraits<value_type>;
  // Use the scalar type's epsilon for the divergence tolerance so that
  // half_t retains a lenient tol (matching pre-double-norm behavior),
  // while float/double keep their stricter, precision-appropriate tol.
  const norm_type tol = Kokkos::max(10 * static_cast<norm_type>(value_traits::eps()) / params.rel_tol,
                                    Kokkos::min(0.03, Kokkos::sqrt(params.rel_tol)));
  sys.residual(y0, rhs);
  const norm_type norm0 = static_cast<norm_type>(KokkosBlas::serial_nrm2(rhs));
  norm_type norm        = KokkosKernels::ArithTraits<norm_type>::zero();
  norm_type norm_old    = KokkosKernels::ArithTraits<norm_type>::zero();
  norm_type rate        = KokkosKernels::ArithTraits<norm_type>::zero();

  // LBV - 07/24/2023: for now assume that we take
  // a full Newton step. Eventually this value can
  // be computed using a line search algorithm to
  // improve convergence for difficult problems.
  const value_type alpha = KokkosKernels::ArithTraits<value_type>::one();

  // Iterate until maxIts or the tolerance is reached
  newton_iterations = 0;
  for (int it = 0; it < params.max_iters; ++it) {  // handle.maxIters; ++it) {
    newton_iterations++;

    // compute initial rhs
    sys.residual(y0, rhs);

    // Solve the following linearized
    // problem at each iteration: J*update=-rhs
    // with J=du/dx, rhs=f(u_n+update)-f(u_n)

    // compute LHS
    sys.jacobian(y0, J);

    // Solve the linear problem J*update = rhs using LU with partial
    // (row) pivoting. The static pivoting implemented in
    // KokkosBatched::SerialGesv uses a greedy row-to-column assignment
    // that can fail spuriously on well-conditioned sparse systems
    // (e.g. chemistry Jacobians with one dense row), so it is only kept
    // as a fallback for when tmp can't hold the pivots.
    int linSolverStat    = 0;
    using mat_value_type = typename mat_type::non_const_value_type;
    // The pivots are stored in tmp, starting at the first int-aligned address
    // (tmp may be less aligned than int, e.g. for half_t). They must fit in a
    // contiguous run of tmp's memory starting at tmp.data(): either tmp's
    // whole span, or its first row/column if only that dimension has stride 1.
    constexpr size_t piv_align_pad =
        (alignof(int) > alignof(mat_value_type)) ? (alignof(int) - alignof(mat_value_type)) : size_t(0);
    size_t tmp_contiguous_len = 0;
    if (tmp.span_is_contiguous()) {
      tmp_contiguous_len = tmp.span();
    } else if (tmp.stride(1) == 1) {
      tmp_contiguous_len = tmp.extent(1);
    } else if (tmp.stride(0) == 1) {
      tmp_contiguous_len = tmp.extent(0);
    }
    if (sys.neqs * sizeof(int) + piv_align_pad <= tmp_contiguous_len * sizeof(mat_value_type)) {
      Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::AnonymousSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> piv(
          KokkosKernels::Impl::alignPtrTo<int>(tmp.data()), sys.neqs);
      for (int idx = 0; idx < sys.neqs; ++idx) update(idx) = rhs(idx);
      linSolverStat = KokkosBatched::SerialGetrf<KokkosBatched::Algo::Getrf::Unblocked>::invoke(J, piv);
      if (linSolverStat == 0) {
        linSolverStat = KokkosBatched::SerialGetrs<KokkosBatched::Trans::NoTranspose,
                                                   KokkosBatched::Algo::Getrs::Unblocked>::invoke(J, piv, update);
      }
    } else {
      linSolverStat = KokkosBatched::SerialGesv<KokkosBatched::Gesv::StaticPivoting>::invoke(J, update, rhs, tmp);
    }

    // Return before touching y0 if the linear solve failed: applying the
    // (stale or partial) update would corrupt the iterate.
    if (linSolverStat != 0) {
      return newton_solver_status::LIN_SOLVE_FAIL;
    }

    KokkosBlas::SerialScale::invoke(-1, update);

    // Compute the rms norm of the scaled update and check for divergence
    // before applying the update to y0
    norm_type norm_new = 0;
    for (int idx = 0; idx < sys.neqs; ++idx) {
      double ui = static_cast<double>(update(idx));
      double si = static_cast<double>(scale(idx));
      norm_new += (ui * ui) / (si * si);
    }
    norm_new = Kokkos::sqrt(norm_new / sys.neqs);
    if ((it > 0) && norm_old > KokkosKernels::ArithTraits<norm_type>::zero()) {
      rate = norm_new / norm_old;
      if ((rate >= 1) || Kokkos::pow(rate, params.max_iters - it) / (1 - rate) * norm_new > tol) {
        return newton_solver_status::NLS_DIVERGENCE;
      }
    }

    // update solution // x = x + alpha*update
    KokkosBlas::serial_axpy(alpha, update, y0);
    norm = static_cast<norm_type>(KokkosBlas::serial_nrm2(rhs));

    if ((it > 0) && norm_old > KokkosKernels::ArithTraits<norm_type>::zero() &&
        ((norm_new == 0) || ((rate / (1 - rate)) * norm_new < tol))) {
      return newton_solver_status::NLS_SUCCESS;
    }

    if ((norm < (params.rel_tol * norm0)) ||
        (it > 0 ? static_cast<norm_type>(KokkosBlas::serial_nrm2(update)) < params.abs_tol : false)) {
      return newton_solver_status::NLS_SUCCESS;
    }

    norm_old = norm_new;
  }
  return newton_solver_status::MAX_ITER;
}

}  // namespace Impl
}  // namespace KokkosODE

#endif  // KOKKOSODE_NEWTON_IMPL_HPP
