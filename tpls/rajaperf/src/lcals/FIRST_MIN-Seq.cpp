//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_MIN.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void FIRST_MIN::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  FIRST_MIN_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        FIRST_MIN_MINLOC_INIT;

        for (Index_type i = ibegin; i < iend; ++i ) {
          FIRST_MIN_BODY;
        }

        m_minloc = RAJA_MAX(m_minloc, mymin.loc);

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto firstmin_base_lam = [=](Index_type i) -> Real_type {
                                 return x[i];
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        FIRST_MIN_MINLOC_INIT;

        for (Index_type i = ibegin; i < iend; ++i ) {
          if ( firstmin_base_lam(i) < mymin.val ) { \
            mymin.val = x[i]; \
            mymin.loc = i; \
          }
        }

        m_minloc = RAJA_MAX(m_minloc, mymin.loc);

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceMinLoc<RAJA::seq_reduce, Real_type, Index_type> loc(
                                                        m_xmin_init, m_initloc);

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          FIRST_MIN_BODY_RAJA;
        });

        m_minloc = RAJA_MAX(m_minloc, loc.getLoc());

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\n  FIRST_MIN : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace lcals
} // end namespace rajaperf
