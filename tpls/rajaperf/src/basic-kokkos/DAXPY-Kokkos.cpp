//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

struct DaxpyFunctor {
  Real_ptr x;
  Real_ptr y;
  Real_type a;
  DaxpyFunctor(Real_ptr m_x, Real_ptr m_y, Real_type m_a) : DAXPY_FUNCTOR_CONSTRUCT {  }
  void operator()(Index_type i) const { DAXPY_BODY; }
};

void DAXPY::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  DAXPY_DATA_SETUP;

  // Declare KokkosViews for the pointers that will be wrapped.
  // Get pointer names in the KERNEL_NAME.hpp file
  // Wrap pointers x and y in separate KokkosViews
  // This is a one dimension array
  // One dimensional arrays are indexed to iend (RAJAPerfSuite convention)
  // New template-based machinery in /rajaperf/src/common/RAJAPerfSuite.hpp 
  
  auto x_view = getViewFromPointer(x, iend);

  auto y_view = getViewFromPointer(y, iend);



  auto daxpy_lam = [=](Index_type i) {
                     DAXPY_BODY;
                   };



#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Kokkos_Lambda: {
	  
	  Kokkos::fence();

      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        Kokkos::parallel_for("DAXPY-Kokkos Kokkos_Lambda",
                             Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                            // Increment y_view (pointer wrapped in KokksView)
                            // by product of a and ith entry of x_view
                            // DAXPY_BODY substituted with the
                            // calculation defined in DAXPY.hpp
							KOKKOS_LAMBDA(Index_type i) { y_view[i] += a * x_view[i];}
                             );
      }
	  // Kokkos fence
	  Kokkos::fence();

      stopTimer();
      
      break;
    }
/*    case Kokkos_Functor: {
      DaxpyFunctor daxpy_functor_instance(y,x,a);                                
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        Kokkos::parallel_for("DAXPY-Kokkos Kokkos_Functor", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                             daxpy_functor_instance);
      }

      stopTimer();
     

   
      break;
    }
*/
    default : {
      std::cout << "\n  DAXPY : Unknown variant id = " << vid << std::endl;
    }

  }

  // Moving all data (i.e., pointer, KokkosView-wrapped ponter) back to the host from the device

  moveDataToHostFromKokkosView(x, x_view, iend);
  
  moveDataToHostFromKokkosView(y, y_view, iend);





#endif // RUN_KOKKOS
}

} // end namespace basic
} // end namespace rajaperf
