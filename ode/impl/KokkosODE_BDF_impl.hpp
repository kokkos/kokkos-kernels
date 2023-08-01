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

#ifndef KOKKOSBLAS_BDF_IMPL_HPP
#define KOKKOSBLAS_BDF_IMPL_HPP

#include "Kokkos_Core.hpp"

#include "KokkosODE_Newton.hpp"

namespace KokkosODE {
namespace Impl {

template <int order>
struct BDF_table {};

template <>
struct BDF_table<1> {
  static constexpr int order = 1;
  Kokkos::Array<double, 2> coefficients{{-1.0, 1.0}};
};

template <>
struct BDF_table<2> {
  static constexpr int order = 2;
  Kokkos::Array<double, 3> coefficients{{-4.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0}};
};

template <>
struct BDF_table<3> {
  static constexpr int order = 3;
  Kokkos::Array<double, 4> coefficients{
      {-18.0 / 11.0, 9.0 / 11.0, -2.0 / 11.0, 6.0 / 11.0}};
};

template <>
struct BDF_table<4> {
  static constexpr int order = 4;
  Kokkos::Array<double, 5> coefficients{
      {-48.0 / 25.0, 36.0 / 25.0, -16.0 / 25.0, 3.0 / 25.0, 12.0 / 25.0}};
};

template <>
struct BDF_table<5> {
  static constexpr int order = 5;
  Kokkos::Array<double, 6> coefficients{{-300.0 / 137.0, 300.0 / 137.0,
                                         -200.0 / 137.0, 75.0 / 137.0,
                                         -12.0 / 137.0, 60.0 / 137.0}};
};

template <>
struct BDF_table<6> {
  static constexpr int order = 6;
  Kokkos::Array<double, 7> coefficients{
      {-360.0 / 147.0, 450.0 / 147.0, -400.0 / 147.0, 225.0 / 147.0,
       -72.0 / 147.0, 10.0 / 147.0, 60.0 / 147.0}};
};

template <class system_type, class table_type, class mv_type>
struct BDF_system_wrapper {
  const system_type mySys;
  const int neqs;
  const table_type table;
  const int order = table.order;

  double t, dt;
  mv_type yn;

  KOKKOS_FUNCTION
  BDF_system_wrapper(const system_type& mySys_, const table_type& table_,
                     const double t_, const double dt_, const mv_type& yn_)
      : mySys(mySys_),
        neqs(mySys_.neqs),
        table(table_),
        t(t_),
        dt(dt_),
        yn(yn_) {}

  template <class vec_type>
  KOKKOS_FUNCTION void residual(const vec_type& y, const vec_type& f) const {
    // f = f(t+dt, y)
    mySys.evaluate_function(t, dt, y, f);

    for (int eqIdx = 0; eqIdx < neqs; ++eqIdx) {
      f(eqIdx) = y(eqIdx) - table.coefficients[order] * dt * f(eqIdx);
      for (int orderIdx = 0; orderIdx < order; ++orderIdx) {
        f(eqIdx) +=
            table.coefficients[order - 1 - orderIdx] * yn(eqIdx, orderIdx);
      }
    }
  }

  template <class vec_type, class mat_type>
  KOKKOS_FUNCTION void jacobian(const vec_type& y, const mat_type& jac) const {
    mySys.evaluate_jacobian(t, dt, y, jac);

    for (int rowIdx = 0; rowIdx < neqs; ++rowIdx) {
      for (int colIdx = 0; colIdx < neqs; ++colIdx) {
        jac(rowIdx, colIdx) =
            -table.coefficients[order] * dt * jac(rowIdx, colIdx);
      }
      jac(rowIdx, rowIdx) += 1.0;
    }
  }
};

template <class ode_type, class table_type, class vec_type, class mv_type,
          class mat_type, class scalar_type>
KOKKOS_FUNCTION void BDFStep(ode_type& ode, const table_type& table,
                             scalar_type t, scalar_type dt,
                             const vec_type& y_old, const vec_type& y_new,
                             const vec_type& rhs, const vec_type& update,
                             const mv_type& y_vecs, const mat_type& temp,
                             const mat_type& jac) {
  using newton_params = KokkosODE::Experimental::Newton_params;

  BDF_system_wrapper sys(ode, table, t, dt, y_vecs);
  const newton_params param(50, 1e-14, 1e-12);

  // first set y_new = y_old
  for (int eqIdx = 0; eqIdx < sys.neqs; ++eqIdx) {
    y_new(eqIdx) = y_old(eqIdx);
  }

  // solver the nonlinear problem
  {
    KokkosODE::Experimental::Newton::Solve(sys, param, jac, temp, y_new, rhs,
                                           update);
  }

}  // BDFStep

}  // namespace Impl
}  // namespace KokkosODE

#endif  // KOKKOSBLAS_BDF_IMPL_HPP
