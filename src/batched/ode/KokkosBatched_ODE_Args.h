/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/
#ifndef __KOKKOSBATCHED_ODE_ARGS_HPP__
#define __KOKKOSBATCHED_ODE_ARGS_HPP__

#include <limits>

namespace KokkosBatched {
namespace ode {

enum class ODESolverType {
  CVODE,
  LSODE,
  RKEH,
  RK12,
  RKBS,
  RKF45,
  CashKarp,
  DOPRI5
};

struct ODEArgs {
  ODESolverType solverType = ODESolverType::CVODE;

  double absTol    = 1e-12;
  double relTol    = 1e-6;
  int maxSubSteps  = 100000;
  int num_substeps = 10;

  // Initially set to an unrealistic number.
  // Will be set to 10*std::numeric_limits<double>::epsilon() if not set by user
  double minStepSize = std::numeric_limits<double>::lowest();

  int order            = 0;  // CVODE
  int multistep_method = 2;  // CV_BDF;    // CVODE
  int iteration_method = 2;  // CV_NEWTON; // CVODE

  int matrix_solver = 0;  // CVODE and LSODE
  int band          = 0;  // CVODE and LSODE

  bool defaultValues = true;
  bool verified      = false;
  bool is_adaptive   = true;

  void verify(std::ostream& outputStream, const std::string& infoLine);
};

struct SolverControls {
  SolverControls(const ODEArgs& args)
      : absTol(args.absTol > std::numeric_limits<double>::epsilon()
                   ? args.absTol
                   : std::numeric_limits<double>::epsilon()),
        relTol(args.relTol),
        minStepSize(args.minStepSize > std::numeric_limits<double>::epsilon()
                        ? args.minStepSize
                        : std::numeric_limits<double>::epsilon()),
        maxSubSteps(args.maxSubSteps),
        num_substeps(args.num_substeps),
        is_adaptive(args.is_adaptive) {}

  const double absTol;
  const double relTol;
  const double minStepSize;
  const int maxSubSteps;
  const int num_substeps;
  const bool is_adaptive;
};
}  // namespace ode
}  // namespace KokkosBatched

#endif
