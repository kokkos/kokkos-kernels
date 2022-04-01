/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/
#ifndef SIERRA_tftk_DeviceODESolverStatus_h
#define SIERRA_tftk_DeviceODESolverStatus_h

#include <ostream>
namespace KokkosBatched {
namespace ode {
enum class ODESolverStatus {
  SUCCESS = 0,
  FAILED_TO_CONVERGE,
  MINIMUM_TIMESTEP_REACHED,
  NONFINITE_STATE
};

// std::ostream& operator<<(std::ostream& os, ODESolverStatus status);
}  // namespace ode
}  // namespace KokkosBatched
#endif
