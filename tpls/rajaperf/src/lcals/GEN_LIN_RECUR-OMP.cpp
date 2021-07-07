//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void GEN_LIN_RECUR::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  GEN_LIN_RECUR_DATA_SETUP;

  auto genlinrecur_lam1 = [=](Index_type k) {
                            GEN_LIN_RECUR_BODY1;
                          };
  auto genlinrecur_lam2 = [=](Index_type i) {
                            GEN_LIN_RECUR_BODY2;
                          };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type k = 0; k < N; ++k ) {
          GEN_LIN_RECUR_BODY1;
        }

        #pragma omp parallel for
        for (Index_type i = 1; i < N+1; ++i ) {
          GEN_LIN_RECUR_BODY2;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type k = 0; k < N; ++k ) {
          genlinrecur_lam1(k);
        }

        #pragma omp parallel for
        for (Index_type i = 1; i < N+1; ++i ) {
          genlinrecur_lam2(i);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(0, N), genlinrecur_lam1);

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(1, N+1), genlinrecur_lam2);

      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  GEN_LIN_RECUR : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace lcals
} // end namespace rajaperf
