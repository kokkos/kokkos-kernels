//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "IF_QUAD.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{



// Kokkos-ify here

void IF_QUAD::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  IF_QUAD_DATA_SETUP;

  // Instantiating views using getViewFromPointer for the IF_QUAD definition

 auto a_view = getViewFromPointer(a,  iend);
 auto b_view = getViewFromPointer(b,  iend);
 auto c_view = getViewFromPointer(c,  iend);
 auto x1_view = getViewFromPointer(x1,  iend);
 auto x2_view = getViewFromPointer(x2,  iend);


  auto ifquad_lam = [=](Index_type i) {
                      IF_QUAD_BODY;
                    };




#if defined(RUN_KOKKOS)

  switch ( vid ) {




    case Kokkos_Lambda : {

	  Kokkos::fence();
      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

/*        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), ifquad_lam);
*/
	// Translation 
	Kokkos::parallel_for("IF_QUAD_Kokkos Kokkos_Lambda", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),

		KOKKOS_LAMBDA (Index_type i) {

    Real_type s = b_view[i]*b_view[i] - 4.0*a_view[i]*c_view[i];
    if ( s >= 0 ) { 
      s = sqrt(s); 
      x2_view[i] = (-b_view[i]+s)/(2.0*a_view[i]);
      x1_view[i] = (-b_view[i]-s)/(2.0*a_view[i]);
    } else { 
      x2_view[i] = 0.0;
      x1_view[i] = 0.0;


        }});

      }

	  Kokkos::fence();
      stopTimer();

      break;

    }

    default : {
      std::cout << "\n  IF_QUAD : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

  moveDataToHostFromKokkosView(a, a_view, iend);
  moveDataToHostFromKokkosView(b, b_view, iend);
  moveDataToHostFromKokkosView(c, c_view, iend);
  moveDataToHostFromKokkosView(x1, x1_view, iend);
  moveDataToHostFromKokkosView(x2, x2_view, iend);


}

} // end namespace basic
} // end namespace rajaperf
