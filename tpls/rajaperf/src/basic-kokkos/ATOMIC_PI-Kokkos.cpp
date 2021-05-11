//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ATOMIC_PI.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

void ATOMIC_PI::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ATOMIC_PI_DATA_SETUP;

  // Declare Kokkos View that will wrap the pointer defined in ATOMIC_PI.hpp
  auto pi_view = getViewFromPointer(pi, 1);

#if defined(RUN_KOKKOS)

  switch (vid) {

  case Base_Seq: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      *pi = m_pi_init;
      for (Index_type i = ibegin; i < iend; ++i) {
        double x = (double(i) + 0.5) * dx;
        *pi += dx / (1.0 + x * x);
      }
      *pi *= 4.0;
    }
    stopTimer();

    break;
  }

  case Lambda_Seq: {

    auto atomicpi_base_lam = [=](Index_type i) {
      double x = (double(i) + 0.5) * dx;
      *pi += dx / (1.0 + x * x);
    };

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      *pi = m_pi_init;
      for (Index_type i = ibegin; i < iend; ++i) {
        atomicpi_base_lam(i);
      }
      *pi *= 4.0;
    }
    stopTimer();

    break;
  }

  case Kokkos_Lambda: {
    // Ensure all upstream calculations have been completed before starting
    // the timer
    Kokkos::fence();
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Here, making a pointer of pi defined in ATOMIC_PI.hpp; we will use a
      // KokkosView instead
      // *pi = m_pi_init;
      //      RAJA::forall<RAJA::loop_exec>( RAJA::RangeSegment(ibegin, iend),
      //          [=](Index_type i) {
      //            double x = (double(i) + 0.5) * dx;
      //            RAJA::atomicAdd<RAJA::seq_atomic>(pi, dx / (1.0 + x * x));
      //        });
      //
      // Initializing a value, pi, on the host
      *pi = m_pi_init;
      // This is an assignment statement! Not a declaration.
      // David made this assignment because of the structure of the
      // computation.
      // We're moving the data in the pointer to the device (GPU)
      // IT IS IMPORTANT TO REALISE WHEN YOUR VARIABLE / DATA ARE BEING
      // REINITIALIZED
      pi_view = getViewFromPointer(pi, 1);

      Kokkos::parallel_for(
          "ATOMIC_PI-Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            // Original ATOMIC_PI kernel reference implementation
            // defined in ATOMIC_PI.hpp
            double x = (double(i) + 0.5) * dx;
            // Make a reference to the 0th element of a 1D view with one
            // element
            // Atomic operation is an uninterruptable, single operation; e.g.,
            // addition, multiplication, division, etc. All of these atomic
            // operations are architecture dependent. Atomics are advantageous
            // from a correctness point of view
            Kokkos::atomic_add(&pi_view(0), dx / (1.0 + x * x));
          });
      // Moving the data on the device (held in the KokkosView) BACK to the
      // pointer, pi.
      moveDataToHostFromKokkosView(pi, pi_view, 1);
      *pi *= 4.0;
      //*m_pi += *pi;
      //*pi *= 4.0;
      // pi_view *= 4.0;
    }

    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  ATOMIC_PI : Unknown variant id = " << vid << std::endl;
  }
  }
#endif // RUN_KOKKOS

  // moveDataToHostFromKokkosView(pi, pi_view, 1);
}

} // end namespace basic
} // end namespace rajaperf
