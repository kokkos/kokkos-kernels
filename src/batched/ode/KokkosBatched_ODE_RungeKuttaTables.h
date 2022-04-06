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

#ifndef __KOKKOSBATCHED_ODE_RUNGEKUTTATABLES_HPP__
#define __KOKKOSBATCHED_ODE_RUNGEKUTTATABLES_HPP__

#include <Kokkos_Array.hpp>

namespace KokkosBatched {
namespace ode {
//=====================================================================
// Generalized RK Explicit ODE solver with embedded error estimation
//=====================================================================

// Methods supported:
// Euler-Heun Method (RKEH)
// Fehlberg 1-2 (RK12)
// Bogacki-Shampine (BS)
// Fehlberg Method (RKF45)
// Cash-Karp Method
// Dormand-Prince Method

struct RKEH  // Euler Huen Method
{
  // Follows form of Butcher Tableau

  // c0| a00
  // c2| a10 a11
  // c3| a20 a21 a22
  // c4| a30 a31 a32
  // . | .   .   .
  // . | .   .       .
  // . | .   .          .
  // cs| as0 as1 . . . . . .  ass
  //--------------------------------
  //   | b0  b1  b2  b3 . . . bs
  //
  // And is always in lower triangular form for explicit methods

  static constexpr int n     = 2;  // total dimensions, nxn system
  static constexpr int order = 2;
  Kokkos::Array<double, (n * n + n) / 2> a{
      {0.0, 1.0, 0.0}};  //(n*n+n)/2 size of lower triangular matrix
  Kokkos::Array<double, n> b{{0.5, 0.5}};
  Kokkos::Array<double, n> c{{0.0, 1.0}};
  Kokkos::Array<double, n> e{{-0.5, 0.5}};
};

struct RK12  // Known as Fehlberg 1-2 method
{
  static constexpr int n     = 3;
  static constexpr int order = 2;
  Kokkos::Array<double, (n * n + n) / 2> a{
      {0.0, 0.5, 0.0, 1.0 / 256.0, 255.0 / 256.0, 0.0}};
  Kokkos::Array<double, n> b{{1.0 / 512.0, 255.0 / 256.0, 1. / 512}};
  Kokkos::Array<double, n> c{{0.0, 1.0 / 2.0, 1.0}};
  Kokkos::Array<double, n> e{{1.0 / 256.0 - 1.0 / 512.0, 0.0, -1.0 / 512.0}};
};

struct BS  // Bogacki-Shampine method
{
  static constexpr int n     = 4;
  static constexpr int order = 3;
  Kokkos::Array<double, (n * n + n) / 2> a{{0.0, 0.5, 0.0, 0.0, 3.0 / 4.0, 0.0,
                                            2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0,
                                            0.0}};
  Kokkos::Array<double, n> b{{2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0}};
  Kokkos::Array<double, n> c{{0.0, 0.5, 0.75, 1.0}};
  Kokkos::Array<double, n> e{{2.0 / 9.0 - 7.0 / 24.0, 1.0 / 3.0 - 0.25,
                              4.0 / 9.0 - 1.0 / 3.0, -1.0 / 8.0}};
};

struct RKF45  // Fehlberg Method
{
  static constexpr int n     = 6;
  static constexpr int order = 5;
  Kokkos::Array<double, (n * n + n) / 2> a{{0.0,
                                            0.25,
                                            0.0,
                                            3.0 / 32.0,
                                            9.0 / 32.0,
                                            0.0,
                                            1932.0 / 2197.0,
                                            -7200.0 / 2197.0,
                                            7296.0 / 2197.0,
                                            0.0,
                                            439.0 / 216.0,
                                            -8.0,
                                            3680.0 / 513.0,
                                            -845.0 / 4104.0,
                                            0.0,
                                            -8.0 / 27.0,
                                            2.0,
                                            -3544.0 / 2565.0,
                                            1859.0 / 4104.0,
                                            -11.0 / 40.0,
                                            0.0}};
  Kokkos::Array<double, n> b{{16.0 / 135.0, 0.0, 6656.0 / 12825.0,
                              28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0}};
  Kokkos::Array<double, n> c{{0.0, 0.25, 3.0 / 8.0, 12.0 / 13.0, 1.0, 0.5}};
  Kokkos::Array<double, n> e{
      {16.0 / 135.0 - 25.0 / 216.0, 0.0, 6656.0 / 12825.0 - 1408.0 / 2565.0,
       28561.0 / 56430.0 - 2197.0 / 4104.0, -9.0 / 50.0 + 0.2, 2.0 / 55.0}};
};

struct CashKarp {
  static constexpr int n     = 6;
  static constexpr int order = 5;
  Kokkos::Array<double, (n * n + n) / 2> a{{0.0,
                                            0.2,
                                            0.0,
                                            3.0 / 40.0,
                                            9.0 / 40.0,
                                            0.0,
                                            0.3,
                                            -0.9,
                                            1.2,
                                            0.0,
                                            -11.0 / 54.0,
                                            2.5,
                                            -70.0 / 27.0,
                                            35.0 / 27.0,
                                            0.0,
                                            1631.0 / 55296.0,
                                            175.0 / 512.0,
                                            575.0 / 13824.0,
                                            44275.0 / 110592.0,
                                            253.0 / 4096.0,
                                            0.0}};
  Kokkos::Array<double, n> b{
      {37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0}};
  Kokkos::Array<double, n> c{{0.0, 0.2, 0.3, 0.6, 1.0, 7.0 / 8.0}};
  Kokkos::Array<double, n> e{{37.0 / 378.0 - 2825.0 / 27648.0, 0.0,
                              250.0 / 621.0 - 18575.0 / 48384.0,
                              125.0 / 594.0 - 13525.0 / 55296.0,
                              -277.0 / 14336.0, 512.0 / 1771.0 - 0.25}};
};

struct DormandPrince  // Referred to as DOPRI5
{
  static constexpr int n     = 7;
  static constexpr int order = 5;
  Kokkos::Array<double, (n * n + n) / 2> a{{0.0,
                                            0.2,
                                            0.0,
                                            3.0 / 40.0,
                                            9.0 / 40.0,
                                            0.0,
                                            44.0 / 45.0,
                                            -56.0 / 15.0,
                                            32.0 / 9.0,
                                            0.0,
                                            19372.0 / 6561.0,
                                            -25360.0 / 2187.0,
                                            64448.0 / 6561.0,
                                            -212.0 / 729.0,
                                            0.0,
                                            9017.0 / 3168.0,
                                            -355.0 / 33.0,
                                            46732.0 / 5247.0,
                                            49.0 / 176.0,
                                            -5103.0 / 18656.0,
                                            0.0,
                                            35.0 / 384.0,
                                            0.0,
                                            500.0 / 1113.0,
                                            125.0 / 192.0,
                                            -2187.0 / 6784.0,
                                            11.0 / 84.0,
                                            0.0}};
  Kokkos::Array<double, n> b{{35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0,
                              -2187.0 / 6784.0, 11.0 / 84.0, 0.0}};
  Kokkos::Array<double, n> c{{0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0}};
  Kokkos::Array<double, n> e{
      {35.0 / 384.0 - 5179.0 / 57600.0, 0.0, 500.0 / 1113.0 - 7571.0 / 16695.0,
       125.0 / 192.0 - 393.0 / 640.0, -2187.0 / 6784.0 + 92097.0 / 339200.0,
       11.0 / 84.0 - 187.0 / 2100.0, -1.0 / 40.0}};
};
}  // namespace ode
}  // namespace KokkosBatched

#endif
