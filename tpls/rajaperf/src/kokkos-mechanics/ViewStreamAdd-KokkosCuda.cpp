//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ViewStreamAdd.hpp"

#include "RAJA/RAJA.hpp"
#if defined (RAJA_ENABLE_CUDA)

#include <iostream>

namespace rajaperf 
{
namespace kokkos_mechanics
{


// Kokkos-ify here

void ViewStreamAdd::runKokkosCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type data_size = getRunSize();



#if defined(RUN_KOKKOS)


  Kokkos::View<float*, Kokkos::CudaSpace> d_a("device_a",getRunSize());
  Kokkos::View<float*, Kokkos::CudaSpace> d_b("device_b",getRunSize());
  Kokkos::View<float*, Kokkos::CudaSpace> d_c("device_c",getRunSize());

  Kokkos::deep_copy(d_a,h_a);
  Kokkos::deep_copy(d_b,h_b);
  Kokkos::deep_copy(d_c,h_c);

  switch ( vid ) {

  // AJP added (following DAXPY example) --

//#if defined(RUN_KOKKOS)
//#if defined(RUN_OPENMP)



    case Kokkos_Lambda_CUDA : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

	// Test Device case / GPU
        Kokkos::parallel_for("perfsuite.kokkos_mechanics.view_stream_add.cuda.lambda",Kokkos::RangePolicy<Kokkos::Cuda>(0,data_size), [=] __device__ (int i) {
  d_c[i] = d_a[i] + d_b[i];
			});

      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  ViewStreamAdd : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS




}

} // end namespace basic
} // end namespace rajaperf
#endif // RAJA_ENABLE_CUDA
