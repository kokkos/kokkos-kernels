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
#include "KokkosBlas2_serial_gemv.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"

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

  template <class system_type, class subview_type>
struct BDF_system_wrapper2 {
  const system_type mySys;
  const int neqs;
  const subview_type psi, d;

  double t, dt, c = 0;

  KOKKOS_FUNCTION
  BDF_system_wrapper2(const system_type& mySys_,
		      const subview_type& psi_, const subview_type& d_,
		      const double t_, const double dt_)
      : mySys(mySys_),
        neqs(mySys_.neqs),
	psi(psi_),
	d(d_),
        t(t_),
        dt(dt_) {}

  template <class vec_type>
  KOKKOS_FUNCTION void residual(const vec_type& y, const vec_type& f) const {
    // f = f(t+dt, y)
    mySys.evaluate_function(t, dt, y, f);

    // rhs = higher order terms + y_{n+1}^i - y_n - dt*f
    for (int eqIdx = 0; eqIdx < neqs; ++eqIdx) {
      f(eqIdx) =  psi(eqIdx) + d(eqIdx) - c * f(eqIdx);
    }
  }

  template <class vec_type, class mat_type>
  KOKKOS_FUNCTION void jacobian(const vec_type& y, const mat_type& jac) const {
    mySys.evaluate_jacobian(t, dt, y, jac);

    // J = I - dt*(dy/dy)
    for (int rowIdx = 0; rowIdx < neqs; ++rowIdx) {
      for (int colIdx = 0; colIdx < neqs; ++colIdx) {
        jac(rowIdx, colIdx) = -dt * jac(rowIdx, colIdx);
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

template <class mat_type, class scalar_type>
KOKKOS_FUNCTION void compute_coeffs(const int order,
				    const scalar_type factor,
				    const mat_type& coeffs) {
  coeffs(0, 0) = 1.0;
  for(int colIdx = 0; colIdx < order; ++colIdx) {
    coeffs(0, colIdx + 1) = 1.0;
    for(int rowIdx = 0; rowIdx < order; ++rowIdx) {
      coeffs(rowIdx + 1, colIdx + 1) = ((rowIdx - factor*(colIdx + 1.0)) / (rowIdx + 1.0)) * coeffs(rowIdx, colIdx + 1);
    }
  }
}

template <class mat_type, class scalar_type>
KOKKOS_FUNCTION void update_D(const int order, const scalar_type factor,
			      const mat_type& coeffs,
			      const mat_type& tempD, const mat_type& D) {
  std::cout << "Extract subviews from D and tempD" << std::endl;
  auto subD     = Kokkos::subview(D,     Kokkos::ALL(), Kokkos::pair<int, int>(0, order + 1));
  auto subTempD = Kokkos::subview(tempD, Kokkos::ALL(), Kokkos::pair<int, int>(0, order + 1));

  compute_coeffs(order, factor, coeffs);
  auto R = Kokkos::subview(coeffs, Kokkos::pair<int, int>(0, order + 1), Kokkos::pair<int, int>(0, order + 1));
  KokkosBatched::SerialGemm<KokkosBatched::Trans::Transpose,
			    KokkosBatched::Trans::NoTranspose,
			    KokkosBatched::Algo::Gemm::Blocked>::invoke(1.0, R, subD, 0.0, subTempD);

  compute_coeffs(order, 1.0, coeffs);
  auto U = Kokkos::subview(coeffs, Kokkos::pair<int, int>(0, order + 1), Kokkos::pair<int, int>(0, order + 1));
  KokkosBatched::SerialGemm<KokkosBatched::Trans::Transpose,
			    KokkosBatched::Trans::NoTranspose,
			    KokkosBatched::Algo::Gemm::Blocked>::invoke(1.0, U, subTempD, 0.0, subD);
}

template <class ode_type, class vec_type,
          class mat_type, class scalar_type>
KOKKOS_FUNCTION void BDFStep(ode_type& ode, scalar_type t, scalar_type& dt,
			     scalar_type t_end, int& order, int& num_equal_steps,
			     const int max_newton_iters,
			     const scalar_type atol, const scalar_type rtol,
			     const scalar_type min_factor,
                             const vec_type& y_old, const vec_type& y_new,
                             const vec_type& rhs, const vec_type& update,
                             const mat_type& temp, const mat_type& temp2) {
  using newton_params = KokkosODE::Experimental::Newton_params;
  // Kokkos::Array<double, 6> kappa{{0., -0.1850, -1/9      , -0.0823000, -0.0415000, 0.}};
  // alpha(i) = (1 - kappa(i)) * gamma(i)
  // error_const(i) = kappa(i) * gamma(i) + 1 / (i + 1)
  // gamma(i) = sum_{k=0}^i(1.0 / k); gamma(0) = 0;
  Kokkos::Array<double, 6> alpha{{0., 1.185  , 1.66666667, 1.98421667, 2.16979167, 2.28333333}};
  Kokkos::Array<double, 6> error_const{{1.        , 0.315     , 0.16666667, 0.09911667, 0.11354167, 0.16666667}};

  // Extract columns of temp to form temporary
  // subviews to operate on.
  const int numRows = temp.extent_int(0); const int numCols = temp.extent_int(1);
  std::cout << "numRows: " << numRows << ", numCols: " << numCols << std::endl;
  std::cout << "Extract subview from temp" << std::endl;
  int offset = 0;
  auto D         = Kokkos::subview(temp, Kokkos::ALL(), Kokkos::pair<int, int>(offset, offset + 8));
  offset += 8;
  auto tempD     = Kokkos::subview(temp, Kokkos::ALL(), Kokkos::pair<int, int>(offset, offset + 8));
  offset += 8;
  auto scale     = Kokkos::subview(temp, Kokkos::ALL(), offset + 1); ++offset; // Scaling coefficients for error calculation
  auto y_predict = Kokkos::subview(temp, Kokkos::ALL(), offset + 1); ++offset; // Initial guess for y_{n+1}
  auto d         = Kokkos::subview(temp, Kokkos::ALL(), offset + 1); ++offset; // Newton update: y_{n+1}^i - y_n
  auto psi       = Kokkos::subview(temp, Kokkos::ALL(), offset + 1); ++offset; // Higher order terms contribution to rhs
  auto error     = Kokkos::subview(temp, Kokkos::ALL(), offset + 1); ++offset; // Error estimate
  auto jac       = Kokkos::subview(temp, Kokkos::ALL(), Kokkos::pair<int, int>(offset, offset + ode.neqs)); // Jacobian matrix
  offset += ode.neqs;
  auto tmp_gesv  = Kokkos::subview(temp, Kokkos::ALL(),
				   Kokkos::pair<int, int>(offset, offset + ode.neqs + 4)); // Buffer space for gesv calculation
  offset += ode.neqs + 4;

  std::cout << "Extract subview from temp2" << std::endl;
  std::cout << "  temp2: (" << temp2.extent(0) << ", " << temp2.extent(1) << ")" << std::endl;
  auto coeffs = Kokkos::subview(temp2, Kokkos::ALL(), Kokkos::pair<int, int>(0, order + 1));
  auto gamma  = Kokkos::subview(temp2, Kokkos::ALL(), 6);
  gamma(0) = 0.0; gamma(1) = 1.0; gamma(2) = 1.5; gamma(3) = 1.83333333; gamma(4) = 2.08333333; gamma(5) = 2.28333333;

  BDF_system_wrapper2 sys(ode, psi, d, t, dt);
  const newton_params param(max_newton_iters, atol, rtol);

  scalar_type max_step = Kokkos::ArithTraits<scalar_type>::max();
  scalar_type min_step = Kokkos::ArithTraits<scalar_type>::min();
  if(dt > max_step) {
    update_D(order, max_step / dt, coeffs, tempD, D);
    dt = max_step;
    num_equal_steps = 0;
  } else if(dt < min_step) {
    update_D(order, min_step / dt, coeffs, tempD, D);
    dt = min_step;
    num_equal_steps = 0;
  }

  // first set y_new = y_old
  for (int eqIdx = 0; eqIdx < sys.neqs; ++eqIdx) {
    y_new(eqIdx) = y_old(eqIdx);
  }

  std::cout << "Hello 1" << std::endl;

  double t_new = 0;
  bool step_accepted = false;
  while (!step_accepted) {
    if(dt < min_step) {return ;}
    t_new = t + dt;

    if(t_new > t_end) {
      t_new = t_end;
      update_D(order, (t_new - t) / dt, coeffs, tempD, D);
      num_equal_steps = 0;
    }
    dt = t_new - t;

    std::cout << "Hello 2" << std::endl;

    for(int eqIdx = 0; eqIdx < sys.neqs; ++eqIdx) {
      for(int orderIdx = 0; orderIdx < order + 1; ++orderIdx) {
	y_predict(eqIdx) += D(eqIdx, orderIdx);
      }
      scale(eqIdx) = atol + rtol*Kokkos::abs(y_predict(eqIdx));
    }

    // Compute psi, the sum of the higher order
    // contribution to the residual
    auto subD     = Kokkos::subview(D, Kokkos::ALL(), Kokkos::pair<int, int>(1, order + 1));
    auto subGamma = Kokkos::subview(gamma, Kokkos::pair<int, int>(1, order + 1));
    KokkosBlas::Experimental::serial_gemv('N', 1.0 / alpha[order], subD, subGamma, 0.0, psi);

    std::cout << "Hello 3" << std::endl;

    sys.c = dt / alpha[order];
    sys.jacobian(y_new, jac);
    std::cout << "c=" << sys.c << ", psi=" << psi(0) << ", y_predict=" << y_predict(0) << std::endl;
    std::cout << "jac: " << jac(0, 0) << std::endl;
    Kokkos::deep_copy(y_new, y_predict);
    KokkosODE::Experimental::newton_solver_status newton_status = 
      KokkosODE::Experimental::Newton::Solve(sys, param, jac, tmp_gesv, y_new, rhs,
					     update);

    std::cout << "Newton Status: " << (newton_status == KokkosODE::Experimental::newton_solver_status::NLS_SUCCESS ? "converged" : "not converged") << std::endl;
    if(newton_status == KokkosODE::Experimental::newton_solver_status::MAX_ITER) {
      dt = 0.5*dt;
      update_D(order, 0.5, coeffs, tempD, D);
      num_equal_steps = 0;

    } else {
      // Estimate the solution error
      scalar_type safety = 0.9*(2*max_newton_iters + 1) / (2*max_newton_iters + param.iters);
      scalar_type error_norm = 0;
      for(int eqIdx = 0; eqIdx < sys.neqs; ++eqIdx) {
	scale(eqIdx) = atol + rtol*Kokkos::abs(y_new(eqIdx));
	error(eqIdx) = error_const[order]*d(eqIdx) / scale(eqIdx);
	error_norm += error(eqIdx)*error(eqIdx);
      }
      error_norm = Kokkos::sqrt(error_norm);

      // Check error norm and adapt step size or accept step
      if(error_norm > 1) {
	scalar_type factor = Kokkos::max(min_factor, safety*Kokkos::pow(error_norm, -1.0 / (order + 1)));
	dt = factor*dt;
	update_D(order, factor, coeffs, tempD, D);
	num_equal_steps = 0;
      } else {
	step_accepted = true;
      }
    }

    std::cout << "Hello 5" << std::endl;
  } // while(!step_accepted)

}  // BDFStep

}  // namespace Impl
}  // namespace KokkosODE

#endif  // KOKKOSBLAS_BDF_IMPL_HPP
