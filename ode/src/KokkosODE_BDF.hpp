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

#ifndef KOKKOSODE_BDF_HPP
#define KOKKOSODE_BDF_HPP

/// \author Luc Berger-Vergiat (lberge@sandia.gov)
/// \file KokkosODE_BDF.hpp

#include "Kokkos_Core.hpp"
#include "KokkosODE_Types.hpp"
#include "KokkosODE_RungeKutta.hpp"

#include "KokkosODE_BDF_impl.hpp"

namespace KokkosODE {
namespace Experimental {

enum BDF_type : int {
  BDF1 = 0,
  BDF2 = 1,
  BDF3 = 2,
  BDF4 = 3,
  BDF5 = 4,
  BDF6 = 5
};

template <BDF_type T>
struct BDF_coeff_helper {
  using table_type = void;
};

template <>
struct BDF_coeff_helper<BDF_type::BDF1> {
  using table_type = KokkosODE::Impl::BDF_table<1>;
};

template <>
struct BDF_coeff_helper<BDF_type::BDF2> {
  using table_type = KokkosODE::Impl::BDF_table<2>;
};

template <>
struct BDF_coeff_helper<BDF_type::BDF3> {
  using table_type = KokkosODE::Impl::BDF_table<3>;
};

template <>
struct BDF_coeff_helper<BDF_type::BDF4> {
  using table_type = KokkosODE::Impl::BDF_table<4>;
};

template <>
struct BDF_coeff_helper<BDF_type::BDF5> {
  using table_type = KokkosODE::Impl::BDF_table<5>;
};

template <>
struct BDF_coeff_helper<BDF_type::BDF6> {
  using table_type = KokkosODE::Impl::BDF_table<6>;
};

template <BDF_type T>
struct BDF {
  using table_type = typename BDF_coeff_helper<T>::table_type;

  template <class ode_type, class vec_type, class mv_type, class mat_type,
            class scalar_type>
  KOKKOS_FUNCTION static void Solve(
      const ode_type& ode, const scalar_type t_start, const scalar_type t_end,
      const int num_steps, const vec_type& y0, const vec_type& y,
      const vec_type& rhs, const vec_type& update, const mv_type& y_vecs,
      const mv_type& kstack, const mat_type& temp, const mat_type& jac) {
    const table_type table;

    const double dt = (t_end - t_start) / num_steps;
    double t        = t_start;

    // Load y0 into y_vecs(:, 0)
    for (int eqIdx = 0; eqIdx < ode.neqs; ++eqIdx) {
      y_vecs(eqIdx, 0) = y0(eqIdx);
    }

    // Compute initial start-up history vectors
    // Using a non adaptive explicit method.
    const int init_steps = table.order - 1;
    if (num_steps < init_steps) {
      return;
    }
    KokkosODE::Experimental::ODE_params params(table.order - 1);
    for (int stepIdx = 0; stepIdx < init_steps; ++stepIdx) {
      KokkosODE::Experimental::RungeKutta<RK_type::RKF45>::Solve(
          ode, params, t, t + dt, y0, y, update, kstack);

      for (int eqIdx = 0; eqIdx < ode.neqs; ++eqIdx) {
        y_vecs(eqIdx, stepIdx + 1) = y(eqIdx);
        y0(eqIdx)                  = y(eqIdx);
      }
      t += dt;
    }

    for (int stepIdx = init_steps; stepIdx < num_steps; ++stepIdx) {
      KokkosODE::Impl::BDFStep(ode, table, t, dt, y0, y, rhs, update, y_vecs,
                               temp, jac);

      // Update history
      for (int eqIdx = 0; eqIdx < ode.neqs; ++eqIdx) {
        y0(eqIdx) = y(eqIdx);
        for (int orderIdx = 0; orderIdx < table.order - 1; ++orderIdx) {
          y_vecs(eqIdx, orderIdx) = y_vecs(eqIdx, orderIdx + 1);
        }
        y_vecs(eqIdx, table.order - 1) = y(eqIdx);
      }
      t += dt;
    }
  } // Solve()

  template <class ode_type, class mat_type, class vec_type, class scalar_type>
  KOKKOS_FUNCTION static void SolveODE(
      const ode_type& ode, const KokkosODE::Experimental::ODE_params& params,
      const scalar_type t_start, const scalar_type t_end, const vec_type& y0,
      const vec_type& y, const mat_type& buffer) {
    const table_type table;

    (void) ode, params, t_start, t_end, y0, y, buffer;

  } // SolveODE
};

}  // namespace Experimental
}  // namespace KokkosODE

#endif  // KOKKOSODE_BDF_HPP
