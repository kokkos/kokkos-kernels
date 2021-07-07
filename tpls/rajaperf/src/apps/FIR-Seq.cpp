//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf 
{
namespace apps
{


void FIR::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize() - m_coefflen;

  FIR_COEFF;

  FIR_DATA_SETUP;

  Real_type coeff[FIR_COEFFLEN];
  std::copy(std::begin(coeff_array), std::end(coeff_array), std::begin(coeff));

  auto fir_lam = [=](Index_type i) {
                   FIR_BODY;
                 };
  
  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          FIR_BODY;
        }

      }
      stopTimer();

      break;
    } 

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
           fir_lam(i);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), fir_lam);

      }
      stopTimer(); 

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\n  FIR : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace apps
} // end namespace rajaperf
