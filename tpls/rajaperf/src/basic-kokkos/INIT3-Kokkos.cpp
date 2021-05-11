//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT3.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{


void INIT3::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();
  
  INIT3_DATA_SETUP;

  // Instantiating Views using getViewFromPointer for the INIT3 definition
  // (i.e., INIT3.hpp)
  
  // The pointer is the first argument, and the last index, denoted by iend, is
  // your second argument
  //
  auto out1_view = getViewFromPointer(out1, iend);
  auto out2_view = getViewFromPointer(out2, iend);
  auto out3_view = getViewFromPointer(out3, iend);
  auto in1_view  = getViewFromPointer(in1, iend);
  auto in2_view  = getViewFromPointer(in2, iend);

  // Next step, integrate the INIT3_BODY into the Kokkos parallel expression

  auto init3_lam = [=](Index_type i) {
                     INIT3_BODY;
                   };

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {
       
      startTimer();
      for(RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for(Index_type i = ibegin; i < iend; ++i) {
	  INIT3_BODY;
        }

      }
      stopTimer();

      break;
}


    case Lambda_Seq : {
      
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        
        for (Index_type i = ibegin; i < iend; ++i) {
	  init3_lam(i);
        }


    }
    stopTimer();  
   
    break;
}

// Nota bene -- Conversion of Raja code begins here
    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

//        RAJA::forall<RAJA::simd_exec>(
//          RAJA::RangeSegment(ibegin, iend), init3_lam);
          
         // Kokkos translation making INIT3_BODY explicit
        Kokkos::parallel_for("INIT3-Kokkos Kokkos_Lambda", 
                              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
		                      KOKKOS_LAMBDA(Index_type i) {
                              //INIT3_BODY definition:
                              // out1[i] = out2[i] = out3[i] = - in1[i] - in2[i] ;
                                out1_view[i] = out2_view[i] = out3_view[i] = - in1_view[i] - in2_view[i];
                                });
      }
      Kokkos::fence();
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  INIT3 : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

   moveDataToHostFromKokkosView(out1, out1_view, iend);
   moveDataToHostFromKokkosView(out2, out2_view, iend);
   moveDataToHostFromKokkosView(out3, out3_view, iend);
   moveDataToHostFromKokkosView(in1, in1_view, iend);
   moveDataToHostFromKokkosView(in2, in2_view, iend);




}

} // end namespace basic
} // end namespace rajaperf
