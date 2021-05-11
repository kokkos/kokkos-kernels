//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ViewStreamAdd.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace kokkos_mechanics
{


// Kokkos-ify here
//void ViewStreamAdd::runSeqVariant(VariantID vid)

void ViewStreamAdd::runKokkosSeqVariant(VariantID vid)
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
        Kokkos::parallel_for("perfsuite.kokkos_mechanics.view_stream_add.seq.lambda",Kokkos::RangePolicy<Kokkos::Serial>(0,data_size), [=](int i) {
  h_c[i] = h_a[i] + h_b[i];
			});

    }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\n  ViewStreamAdd : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS




}

} // end namespace basic
} // end namespace rajaperf
