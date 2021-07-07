//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MULADDSUB.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{


void MULADDSUB::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  MULADDSUB_DATA_SETUP;


  // Define Kokkos Views that will wrap pointers defined in MULADDSUB.hpp
  auto out1_view = getViewFromPointer(out1, iend);
  auto out2_view = getViewFromPointer(out2, iend);
  auto out3_view = getViewFromPointer(out3, iend);
  auto in1_view  = getViewFromPointer(in1, iend);
  auto in2_view  = getViewFromPointer(in2, iend);




  auto mas_lam = [=](Index_type i) {
                   MULADDSUB_BODY;
                 };


#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          MULADDSUB_BODY;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          mas_lam(i);
        }

      }
      stopTimer();

      break;
    }

    case Kokkos_Lambda : {

      // Set fence to ensure upstream calculations have completed
      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

//        RAJA::forall<RAJA::simd_exec>(
//          RAJA::RangeSegment(ibegin, iend), mas_lam);
//
//		Kokkos translation
//		If SIMD really matters , consider using Kokkos SIMD
		Kokkos::parallel_for("MULTISUB-KokkosSeq Kokkos_Lambda",
                             Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                             KOKKOS_LAMBDA(Index_type i) {
                             //MULADDSUB_BODY definition:
                             //out1[i] = in1[i] * in2[i] ;
                             //out2[i] = in1[i] + in2[i] ;    
                             //out3[i] = in1[i] - in2[i] ;    
                             // WITH KOKKOS VIEWS
                             out1_view[i] = in1_view[i] * in2_view[i] ;
                             out2_view[i] = in1_view[i] + in2_view[i] ;    
                             out3_view[i] = in1_view[i] - in2_view[i] ;    
                             });

      }
      Kokkos::fence();
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  MULADDSUB : Unknown variant id = " << vid << std::endl;
    }

  }
#endif // RUN_KOKKOS
  moveDataToHostFromKokkosView(out1, out1_view, iend);
  moveDataToHostFromKokkosView(out2, out2_view, iend);
  moveDataToHostFromKokkosView(out3, out3_view, iend);
  moveDataToHostFromKokkosView(out3, out3_view, iend);
  moveDataToHostFromKokkosView(in1, in1_view, iend);
  moveDataToHostFromKokkosView(in2, in2_view, iend);



}

} // end namespace basic
} // end namespace rajaperf
