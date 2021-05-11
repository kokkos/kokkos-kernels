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

namespace rajaperf 
{
namespace basic
{


void ATOMIC_PI::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ATOMIC_PI_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        *pi = m_pi_init;
        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          double x = (double(i) + 0.5) * dx;
          #pragma omp atomic
          *pi += dx / (1.0 + x * x); 
        }
        *pi *= 4.0;

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto atomicpi_base_lam = [=](Index_type i) {
                                 double x = (double(i) + 0.5) * dx;
                                 #pragma omp atomic
                                 *pi += dx / (1.0 + x * x);
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        *pi = m_pi_init;
        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          atomicpi_base_lam(i);
        }
        *pi *= 4.0;

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        *pi = m_pi_init;
        RAJA::forall<RAJA::omp_parallel_for_exec>( 
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            double x = (double(i) + 0.5) * dx;
            RAJA::atomicAdd<RAJA::omp_atomic>(pi, dx / (1.0 + x * x));
        });
        *pi *= 4.0;

      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  ATOMIC_PI : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
