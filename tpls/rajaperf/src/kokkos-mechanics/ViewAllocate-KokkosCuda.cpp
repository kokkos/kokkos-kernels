//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ViewAllocate.hpp"

#include "RAJA/RAJA.hpp"
#if defined (RAJA_ENABLE_CUDA)

#include <iostream>

namespace rajaperf 
{
namespace kokkos_mechanics
{


// Kokkos-ify here

void ViewAllocate::runKokkosCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type data_size = getRunSize();



#if defined(RUN_KOKKOS)

  switch ( vid ) {

  // AJP added (following DAXPY example) --

//#if defined(RUN_KOKKOS)
//#if defined(RUN_OPENMP)



    case Kokkos_Lambda_CUDA : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

	// Test Device case / GPU
	Kokkos::View<float* , Kokkos::CudaSpace>
	   kk_view("kk_view", data_size);

      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  ViewAllocate : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS




}

} // end namespace basic
} // end namespace rajaperf
#endif // RAJA_ENABLE_CUDA
