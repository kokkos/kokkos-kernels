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

#ifndef __KOKKOSBATCHED_ODE_TESTPROBLEMS_HPP__
#define __KOKKOSBATCHED_ODE_TESTPROBLEMS_HPP__

#include <Kokkos_ArithTraits.hpp>

namespace KokkosBatched {
namespace Experimental {
namespace ODE {

// Note: The name `DegreeNPoly` indicates that the SOLUTION
// is a degree N poly, not the ODE. :)

struct DegreeOnePoly {
  // Soln: y = t + 1
  DegreeOnePoly(int neqs_) : neqs(neqs_) {}

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double /*t*/, View1& /*y*/,
                                   View2& dydt) const {
    for (int i = 0; i < neqs; i++) {
      dydt[i] = 1;
    }
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 1.0; }
  KOKKOS_FUNCTION double expected_val(const double t, const int /*n*/) const {
    return t + 1.0;
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }
  const int neqs;
};

struct DegreeTwoPoly {
  // Soln: y = (1/2)t^2 + t + 1
  DegreeTwoPoly(int neqs_) : neqs(neqs_) {}
  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double t, View1& /*y*/, View2& dydt) const {
    for (int i = 0; i < neqs; i++) {
      dydt[i] = t + 1;
    }
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 1.0; }
  KOKKOS_FUNCTION double expected_val(const double t, const int /*n*/) const {
    return 0.5 * t * t + t + 1.0;
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }
  const int neqs;
};

struct DegreeThreePoly {
  // Soln: y = (1/3)t^3 + (1/2)t^2 + t + 1
  DegreeThreePoly(int neqs_) : neqs(neqs_) {}
  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double t, View1& /*y*/, View2& dydt) const {
    for (int i = 0; i < neqs; i++) {
      dydt[i] = t * t + t + 1;
    }
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 1.0; }
  KOKKOS_FUNCTION double expected_val(const double t, const int /*n*/) const {
    return (1. / 3) * t * t * t + (1. / 2) * t * t + t + 1;
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }
  const int neqs;
};

struct DegreeFivePoly {
  // Soln: y = (1/5)t^5 + (1/4)t^4 + (1/3)t^3 + (1/2)t^2 + t + 1
  DegreeFivePoly(int neqs_) : neqs(neqs_) {}

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double t, View1& /*y*/, View2& dydt) const {
    for (int i = 0; i < neqs; i++) {
      // dydt = t^4 + t^3 + t^2 + t + 1
      dydt[i] = t * t * t * t + t * t * t + t * t + t + 1;
    }
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 1.0; }
  KOKKOS_FUNCTION double expected_val(const double t, const int /*n*/) const {
    return (1. / 5) * t * t * t * t * t + (1. / 4) * t * t * t * t +
           (1. / 3) * t * t * t + (1. / 2) * t * t + t + 1;
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }
  const int neqs;
};

struct Exponential {
  // Soln: y = e^(rate*t)
  Exponential(int neqs_, double rate_) : neqs(neqs_), rate(rate_) {}

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double /*t*/, View1& y, View2& dydt) const {
    for (int i = 0; i < neqs; i++) {
      dydt[i] = rate * y[i];
    }
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 1.0; }
  KOKKOS_FUNCTION double expected_val(const double t, const int /*n*/) const {
    return Kokkos::exp(rate * t);
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }
  const int neqs;
  const double rate;
};

// Example 8.1 from Leveque
// Solution has two time scales and approaches cos(t) exponentially
struct CosExp {
  CosExp(int neqs_, double lambda_, double t0_, double eta_)
      : neqs(neqs_), lambda(lambda_), t0(t0_), eta(eta_) {}

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double t, View1& y, View2& dydt) const {
    for (int i = 0; i < neqs; i++) {
      dydt[i] = lambda * (y[i] - Kokkos::cos(t)) - Kokkos::sin(t);
    }
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 10.0; }
  KOKKOS_FUNCTION double expected_val(const double t, const int /*n*/) const {
    return Kokkos::exp(lambda * (t - t0)) * (eta - Kokkos::cos(t0)) +
           Kokkos::cos(t);
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }
  const int neqs;
  const double lambda;
  const double t0;
  const double eta;
};

// y'' + c * y' + k * y = 0
// y(0) = 1
// y'(0) = 0
// lambda1 = (- c + sqrt( c * c - 4 * k)) / 2
// lambda2 = (- c - sqrt( c * c - 4 * k)) / 2

// choice of c = 1001, k = 1000 -> stiffness ratio of 1e3
// Solution is for case of real distinct eigenvalues
struct SpringMassDamper {
  SpringMassDamper(int neqs_, double c_, double k_)
      : neqs(neqs_),
        c(c_),
        k(k_),
        lambda1((-c + Kokkos::pow(c * c - 4. * k, 0.5)) / 2.),
        lambda2((-c - Kokkos::pow(c * c - 4. * k, 0.5)) / 2.) {}

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double /*t*/, View1& y, View2& dydt) const {
    dydt[0] = y[1];
    dydt[1] = -k * y[0] - c * y[1];
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 1.0; }
  KOKKOS_FUNCTION double expected_val(const double t, const int n) const {
    using Kokkos::exp;

    const double det = lambda1 - lambda2;
    double val       = 0;
    if (n == 0) {
      val = -(lambda2 / det) * exp(lambda1 * t) +
            (lambda1 / det) * exp(lambda2 * t);
    } else {
      val = -(lambda2 * lambda1 / det) * exp(lambda1 * t) +
            (lambda1 * lambda2 / det) * exp(lambda2 * t);
    }
    return val;
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }

  const int neqs;
  const double c;
  const double k;
  const double lambda1;
  const double lambda2;
};

// Example 7.9 in LeVeque
// Stiff chemical decay process w/ three dofs where
// A -K1-> B -K2-> C

// Eigenvalues are -K2, -K1, 0
// Solution is of the form y_j = cj1 * exp(-K1 * t) + cj2 * exp(-K2 * t) + cj3
// for j = 0, 1, 2
struct StiffChemicalDecayProcess {
  StiffChemicalDecayProcess(int neqs_, double K1_, double K2_)
      : neqs(neqs_), K1(K1_), K2(K2_) {}

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double /*t*/, View1& y, View2& dydt) const {
    dydt[0] = -K1 * y[0];
    dydt[1] = K1 * y[0] - K2 * y[1];
    dydt[2] = K2 * y[1];
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 0.2; }
  KOKKOS_FUNCTION double expected_val(const double t, const int n) const {
    using Kokkos::exp;

    const double C21 = y1_init * K1 / (K2 - K1);
    const double C22 = y2_init - C21;

    double val = 0.0;
    if (n == 0) {
      val = y1_init * exp(-K1 * t);
    } else if (n == 1) {
      val = C21 * exp(-K1 * t) + C22 * exp(-K2 * t);
    } else {
      const double C31 = -K2 * C21 / K1;
      const double C32 = -C22;
      const double C33 = y1_init + y2_init + y3_init;
      val              = C31 * exp(-K1 * t) + C32 * exp(-K2 * t) + C33;
    }
    return val;
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }
  const int neqs;
  const double y1_init = 3.0;
  const double y2_init = 4.0;
  const double y3_init = 2.0;
  const double K1;
  const double K2;
};

// Particle starts at (1,0) on the unit circle w/ 0' = rate, anti-clockwise
struct Tracer {
  Tracer(int neqs_, double rate_) : neqs(neqs_), rate(rate_) {}

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double /*t*/, View1& y, View2& dydt) const {
    for (int i = 0; i < neqs; i += 2) {
      const double R = Kokkos::sqrt(y[i] * y[i] + y[i + 1] * y[i + 1]);
      dydt[i]        = -rate * y[i + 1] / R;
      dydt[i + 1]    = rate * y[i] / R;
    }
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 2.0 * pi; }
  KOKKOS_FUNCTION double expected_val(const double t, const int n) const {
    const double theta = rate * t;
    double val         = 0.0;
    if (n % 2 == 0) {
      val = Kokkos::cos(theta);
    } else {
      val = Kokkos::sin(theta);
    }
    return val;
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }
  const int neqs;
  const double pi = 4.0 * Kokkos::atan(1.0);
  const double rate;
};

// Enright et al. 1975 Problem B5, linear, complex eigenvalues
// Eigenvalues -10 +- i * alpha, -4, -1, -1/2, -0.1
struct EnrightB5 {
  EnrightB5(int neqs_, double alpha_ = 100.0) : neqs(neqs_), alpha(alpha_) {}

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double /*t*/, View1& y, View2& dydt) const {
    dydt[0] = -10. * y[0] + alpha * y[1];
    dydt[1] = -alpha * y[0] - 10. * y[1];
    dydt[2] = -4. * y[2];
    dydt[3] = -y[3];
    dydt[4] = -0.5 * y[4];
    dydt[5] = -0.1 * y[5];
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 1.0; }
  KOKKOS_FUNCTION double expected_val(const double t, const int n) const {
    using Kokkos::cos;
    using Kokkos::exp;
    using Kokkos::sin;

    double val = 0;

    const double c1 = 1.0;
    const double c2 = -1.0;

    const double a[2] = {0, 1};
    const double b[2] = {-1, 0};

    if (n < 2) {
      val = exp(-10. * t) *
            (c1 * (a[n] * cos(alpha * t) - b[n] * sin(alpha * t)) +
             c2 * (a[n] * sin(alpha * t) + b[n] * cos(alpha * t)));
    } else if (n == 2) {
      val = exp(-4. * t);
    } else if (n == 3) {
      val = exp(-t);
    } else if (n == 4) {
      val = exp(-0.5 * t);
    } else {
      val = exp(-0.1 * t);
    }

    return val;
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }
  const int neqs;
  const double alpha;
};

// Enright et al. 1975 Problem C1, nonlinear coupling
struct EnrightC1 {
  EnrightC1(int neqs_) : neqs(neqs_) {}

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double /*t*/, View1& y, View2& dydt) const {
    dydt[0] = -y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];
    dydt[1] = -10. * y[1] + 10. * (y[2] * y[2] + y[3] * y[3]);
    dydt[2] = -40. * y[2] + 40. * y[3] * y[3];
    dydt[3] = -100.0 * y[3] + 2.;
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 20.0; }
  KOKKOS_FUNCTION double expected_val(const double /*t*/,
                                      const int /*n*/) const {
    // IC at t = 0
    return 1.0;
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }
  const int neqs;
};

// Enright et al. 1975 Problem C5, nonlinear coupling
struct EnrightC5 {
  EnrightC5(int neqs_, const double beta_ = 20.0) : neqs(neqs_), beta(beta_) {}

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double /*t*/, View1& y, View2& dydt) const {
    dydt[0] = -y[0] + 2.;
    dydt[1] = -10. * y[1] + beta * y[0] * y[0];
    dydt[2] = -40. * y[2] + 4. * beta * (y[0] * y[0] + y[1] * y[1]);
    dydt[3] =
        -100.0 * y[3] + 10. * beta * (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 20.0; }
  KOKKOS_FUNCTION double expected_val(const double /*t*/,
                                      const int /*n*/) const {
    // IC at t = 0
    return 1.0;
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }
  const int neqs;
  const double beta;
};

// Enright et al. 1975 Problem D2, nonlinear w/ real eigenvalues
// see paper for the range of eigenvalues
struct EnrightD2 {
  EnrightD2(int neqs_) : neqs(neqs_) {}

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double /*t*/, View1& y, View2& dydt) const {
    dydt[0] = -0.04 * y[0] + 0.01 * y[1] * y[2];
    dydt[1] = 400.0 * y[0] - 100.0 * y[1] * y[2] - 3000. * y[1] * y[1];
    dydt[2] = 30. * y[1] * y[1];
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 1.0; }  // decrease from 40.0
  KOKKOS_FUNCTION double expected_val(const double /*t*/, const int n) const {
    // IC at t = 0
    double val = 0.0;
    if (n == 0) {
      val = 1.0;
    }
    return val;
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }
  const int neqs;
};

// Enright et al. 1975 Problem D4, nonlinear w/ real eigenvalues
struct EnrightD4 {
  EnrightD4(int neqs_) : neqs(neqs_) {}

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void derivatives(double /*t*/, View1& y, View2& dydt) const {
    dydt[0] = -0.013 * y[0] - 1000. * y[0] * y[2];
    dydt[1] = -2500. * y[1] * y[2];
    dydt[2] = dydt[0] + dydt[1];
  }

  KOKKOS_FUNCTION double tstart() const { return 0.0; }
  KOKKOS_FUNCTION double tend() const { return 1.0; }  // decrease from 50.0
  KOKKOS_FUNCTION double expected_val(const double /*t*/, const int n) const {
    // IC at t = 0
    double val = 0.0;
    if (n < 2) {
      val = 1.0;
    }
    return val;
  }
  KOKKOS_FUNCTION int num_equations() const { return neqs; }
  const int neqs;
};

}  // namespace ODE
}  // namespace Experimental
}  // namespace KokkosBatched

#endif
