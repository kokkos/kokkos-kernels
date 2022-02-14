/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/
#ifndef SIERRA_tftk_DeviceODESolvers_h
#define SIERRA_tftk_DeviceODESolvers_h

#include <Kokkos_Macros.hpp>
#include <Kokkos_ArithTraits.hpp>
#include "Kokkos_Layout.hpp"
#include "Kokkos_MemoryTraits.hpp"

//#include "tftk_util/tftk_KokkosTypes.h"
#include "tftk_KokkosTypes.h"
#include <tftk_ODEArgs.h>
#include <tftk_RungeKuttaTables.h>
#include <tftk_DeviceODESolverStatus.h>
#include <tftk_DeviceODESolverState.h>

//#include "stk_util/util/ReportHandler.hpp"

#include <type_traits>

namespace tftk
{
namespace ode
{

KOKKOS_FORCEINLINE_FUNCTION double
tol(const double y, const double y0, const double absTol, const double relTol)
{
  return absTol +
      relTol *
      Kokkos::Experimental::fmax(Kokkos::Experimental::fabs(y), Kokkos::Experimental::fabs(y0));
}

template <typename View> KOKKOS_FUNCTION bool isfinite(View & y, const unsigned ndofs)
{
  bool is_finite = true;
  for (unsigned i = 0; i < ndofs; ++i)
  {
    if (!Kokkos::Experimental::isfinite(y[i]))
    {
      is_finite = false;
      break;
    }
  }
  return is_finite;
}
template <typename TableType> struct RungeKuttaSolver
{
  static constexpr int nstages = TableType::n;

  RungeKuttaSolver(const ODEArgs & args) : controls(args) {}

  template <typename ODEType, typename StateType>
  KOKKOS_FUNCTION ODESolverStatus solve(
      const ODEType & ode, double tstart, double tend, StateType & s) const
  {
    using Kokkos::Experimental::fmax;
    using Kokkos::Experimental::fmin;
    using Kokkos::Experimental::pow;

    const int ndofs = s.ndofs();
    
    //TODO: should this be handled with an assert?
   // assert(ode.num_equations() == ndofs, "Mismatched number of dofs in ode solver");
    //if(ode.num_equations != ndofs)
    //{
     // throw std::runtime_error("Mismatched number of dofs in ode solver.");
    //}

    double t0 = tstart;

    for (int i = 0; i < ndofs; ++i)
    {
      s.y0[i] = s.y[i];
    }

    if (!isfinite(s.y0, ndofs))
    {
      return ODESolverStatus::NONFINITE_STATE;
    }

    const double pFactor = -1.0 / table.order;

    double dt = (tend - t0) / controls.num_substeps;

    for (int n = 0; n < controls.maxSubSteps; ++n)
    {
      ode.derivatives(t0, s.y0, s.dydt);

      // Limit dt to not exceed t_end
      if (t0 + dt > tend)
      {
        dt = tend - t0;
      }

      double err = 0.0;
      // Start iterative approach with time step adaptation
      do
      {
        err = 0.0;
        step(ode, t0, dt, s, err);

        // Reduce dt for large error
        if (err > 1 && controls.is_adaptive)
        {
          dt *= fmax(0.2, 0.8 * pow(err, pFactor));

          if (dt < controls.minStepSize)
          {
            return ODESolverStatus::MINIMUM_TIMESTEP_REACHED;
          }
        }

      } while (err > 1 && controls.is_adaptive);

      t0 += dt;

      for (int i = 0; i < ndofs; ++i)
      {
        s.y0[i] = s.y[i];
      }

      if (t0 >= tend)
      {
        auto status =
            !isfinite(s.y, ndofs) ? ODESolverStatus::NONFINITE_STATE : ODESolverStatus::SUCCESS;
        return status;
      }

      // Increase dt for small error
      if (err < 0.5 && controls.is_adaptive)
      {
        dt *= fmin(10.0, fmax(2.0, 0.9 * pow(err, pFactor)));
      }
    }
    return ODESolverStatus::FAILED_TO_CONVERGE;
  }

  template <typename ODEType, typename StateType>
  KOKKOS_FUNCTION void
  step(const ODEType & ode, const double t0, const double dt, StateType & s, double & err) const
  {
    const int ndofs = s.ndofs();

    for (int j = 0; j < nstages; ++j)
    {
      const int offset = (j + 1) * j / 2;
      for (int n = 0; n < ndofs; ++n)
      {
        double coeff = 0.0;
        for (int k = 0; k < j; ++k)
        { // lower diagonal matrix
          coeff += table.a[k + offset] * s.k(k, n);
        }

        s.ytemp[n] = s.y0[n] + dt * coeff;
      }
      auto ksub = Kokkos::subview(s.k, j, Kokkos::ALL);
      ode.derivatives(t0 + table.c[j] * dt, s.ytemp, ksub);
    }

    for (int n = 0; n < ndofs; ++n)
    {
      double coeff = 0.0;
      double errJ = 0.0;
      for (int k = 0; k < nstages; ++k)
      {
        coeff += table.b[k] * s.k(k, n);
        errJ += table.e[k] * s.k(k, n);
      }
      s.y[n] = s.y0[n] + dt * coeff;
      errJ *= dt;
      err = Kokkos::Experimental::fmax(err,
          Kokkos::Experimental::fabs(errJ) /
              tol(s.y[n], s.y0[n], controls.absTol, controls.relTol));
    }
  }

  const TableType table;
  const SolverControls controls;
};
} // namespace ode
} // namespace tftk

#endif
