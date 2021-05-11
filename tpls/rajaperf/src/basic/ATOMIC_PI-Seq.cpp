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


void ATOMIC_PI::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ATOMIC_PI_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        *pi = m_pi_init;
        for (Index_type i = ibegin; i < iend; ++i ) {
          double x = (double(i) + 0.5) * dx;
          *pi += dx / (1.0 + x * x);
        }
        *pi *= 4.0;

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto atomicpi_base_lam = [=](Index_type i) {
                                 double x = (double(i) + 0.5) * dx;
                                 *pi += dx / (1.0 + x * x);
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        *pi = m_pi_init;
        for (Index_type i = ibegin; i < iend; ++i ) {
          atomicpi_base_lam(i);
        }
        *pi *= 4.0;

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
   
        *pi = m_pi_init;
        RAJA::forall<RAJA::loop_exec>( RAJA::RangeSegment(ibegin, iend), 
          [=](Index_type i) {
            double x = (double(i) + 0.5) * dx;
            RAJA::atomicAdd<RAJA::seq_atomic>(pi, dx / (1.0 + x * x));
        });
        *pi *= 4.0;

      }
      stopTimer();

      break;
    }
#endif

    default : {
      std::cout << "\n  ATOMIC_PI : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace basic
} // end namespace rajaperf
