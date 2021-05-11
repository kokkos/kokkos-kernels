//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void INIT_VIEW1D::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  INIT_VIEW1D_DATA_SETUP;

  // Declare a Kokkos View that will be used to wrap a pointer 
  auto a_view = getViewFromPointer(a, iend);




#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          INIT_VIEW1D_BODY;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_Seq : {

      auto initview1d_base_lam = [=](Index_type i) {
                                   INIT_VIEW1D_BODY;
                                 };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          initview1d_base_lam(i);
        }

      }
      stopTimer();

      break;
    }

    // AJP began modificaiton here
    case Kokkos_Lambda : {

      //INIT_VIEW1D_VIEW_RAJA;

     /* auto initview1d_lam = [=](Index_type i) {
                              INIT_VIEW1D_BODY_RAJA;
       
                         };
*/
      // fence needed to ensure upstream operations are complete before timer
      // start
      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

//        RAJA::forall<RAJA::simd_exec>(
//          RAJA::RangeSegment(ibegin, iend), initview1d_lam);
         //Kokkos translation
         Kokkos::parallel_for("INIT_VIEW1D_Kokkos Kokkos_Lambda",
                              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin,iend),
                              KOKKOS_LAMBDA (Index_type i) {
                              //INIT_VIEW1D_BODY_RAJA
                              //Instead, use the INIT_VIEW1D_BODY definition
                              //with Kokkos View
                              //a[i] = (i+1) * v;
                              a_view[i] = (i + 1) * v;

                              });

      }

      Kokkos::fence();
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  INIT_VIEW1D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

  moveDataToHostFromKokkosView(a, a_view, iend);


}

} // end namespace basic
} // end namespace rajaperf
