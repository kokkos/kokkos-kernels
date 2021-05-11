//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_1D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{

 
void POLYBENCH_JACOBI_1D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps= getRunReps();

  POLYBENCH_JACOBI_1D_DATA_SETUP;

  auto poly_jacobi1d_lam1 = [=] (Index_type i) {
                              POLYBENCH_JACOBI_1D_BODY1;
                            };
  auto poly_jacobi1d_lam2 = [=] (Index_type i) {
                              POLYBENCH_JACOBI_1D_BODY2;
                            };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            POLYBENCH_JACOBI_1D_BODY1;
          }
          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            POLYBENCH_JACOBI_1D_BODY2;
          }

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET;

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            poly_jacobi1d_lam1(i);
          }
          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            poly_jacobi1d_lam2(i);
          }

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET;

      break;
    }

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::forall<RAJA::omp_parallel_for_exec> (RAJA::RangeSegment{1, N-1},
            poly_jacobi1d_lam1
          );

          RAJA::forall<RAJA::omp_parallel_for_exec> (RAJA::RangeSegment{1, N-1},
            poly_jacobi1d_lam2
          );

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET;

      break;
    }

    default : {
      std::cout << "\n  POLYBENCH_JACOBI_1D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
