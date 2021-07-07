//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_1D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void HYDRO_1D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  HYDRO_1D_DATA_SETUP;

  auto hydro1d_lam = [=](Index_type i) {
                       HYDRO_1D_BODY;
                     };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          HYDRO_1D_BODY;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          hydro1d_lam(i);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), hydro1d_lam);

      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  HYDRO_1D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace lcals
} // end namespace rajaperf
