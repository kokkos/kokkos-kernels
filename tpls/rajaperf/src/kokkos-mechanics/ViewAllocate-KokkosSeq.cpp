//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ViewAllocate.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace kokkos_mechanics
{


// Kokkos-ify here
//void ViewAllocate::runSeqVariant(VariantID vid)

void ViewAllocate::runKokkosSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type data_size = getRunSize();



#if defined(RUN_KOKKOS)

  switch ( vid ) {

  // AJP added (following DAXPY example) --

//#if defined(RUN_KOKKOS)
//#if defined(RUN_OPENMP)


#if defined(RUN_RAJA_SEQ)     

    case Kokkos_Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

/*        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), ifquad_lam);
*/
	// Test host case / CPU
	Kokkos::View<float* , Kokkos::HostSpace>
	   kk_view("kk_view", data_size);

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\n  ViewAllocate : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS




}

} // end namespace basic
} // end namespace rajaperf
