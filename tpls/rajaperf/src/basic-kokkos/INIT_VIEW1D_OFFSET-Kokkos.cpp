//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D_OFFSET.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{



void INIT_VIEW1D_OFFSET::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getRunSize()+1;

  INIT_VIEW1D_OFFSET_DATA_SETUP;

  auto a_view = getViewFromPointer(a, iend);


#if defined(RUN_KOKKOS)


  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          INIT_VIEW1D_OFFSET_BODY;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_Seq : {

      auto initview1doffset_base_lam = [=](Index_type i) {
                                         INIT_VIEW1D_OFFSET_BODY;
                                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          initview1doffset_base_lam(i);
        }

      }
      stopTimer();

      break;
    }

	// Conversion of Raja code to Kokkos starts here
	//
    case Kokkos_Lambda : {

      //INIT_VIEW1D_OFFSET_VIEW_RAJA;

      /*auto initview1doffset_lam = [=](Index_type i) {
                                    INIT_VIEW1D_OFFSET_BODY_RAJA;
                                  };

*/
 
      // Set a fence
      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

//        RAJA::forall<RAJA::simd_exec>(
//          RAJA::RangeSegment(ibegin, iend), initview1doffset_lam);
          Kokkos::parallel_for("INIT_VIEW1D_OFFSET_Kokkos Kokkos_Lambda", 
                               Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend), 
                               KOKKOS_LAMBDA (Index_type i) {
                               //INIT_VIEW1D_OFFSET_BODY_RAJA
                               //Instead, use the INIT_VIEW1D_OFFSET_BODY
                               //definition:
                               //a[i-ibegin] = i * v;
                               a_view[i-ibegin] = i * v;
                               });


      }
      Kokkos::fence();
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  INIT_VIEW1D_OFFSET : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

  // Move data from Kokkos View back to Host
  moveDataToHostFromKokkosView(a, a_view, iend);



}

} // end namespace basic
} // end namespace rajaperf
