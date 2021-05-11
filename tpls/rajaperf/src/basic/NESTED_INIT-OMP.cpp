//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

//#define USE_OMP_COLLAPSE
#undef USE_OMP_COLLAPSE


void NESTED_INIT::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
 
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  auto nestedinit_lam = [=](Index_type i, Index_type j, Index_type k) {
                          NESTED_INIT_BODY;
                        };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if defined(USE_OMP_COLLAPSE)
          #pragma omp parallel for collapse(3)
#else
          #pragma omp parallel for
#endif
          for (Index_type k = 0; k < nk; ++k ) {
            for (Index_type j = 0; j < nj; ++j ) {
              for (Index_type i = 0; i < ni; ++i ) {
                NESTED_INIT_BODY;
              }
            }
          }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if defined(USE_OMP_COLLAPSE)
          #pragma omp parallel for collapse(3)
#else
          #pragma omp parallel for
#endif
          for (Index_type k = 0; k < nk; ++k ) {
            for (Index_type j = 0; j < nj; ++j ) {
              for (Index_type i = 0; i < ni; ++i ) {
                nestedinit_lam(i, j, k);
              }
            }
          }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

#if defined(USE_OMP_COLLAPSE)
      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    RAJA::ArgList<2, 1, 0>,  // k, j, i
            RAJA::statement::Lambda<0>
          >
        >;
#else
      using EXEC_POL = 
        RAJA::KernelPolicy<
          RAJA::statement::For<2, RAJA::omp_parallel_for_exec,  // k
            RAJA::statement::For<1, RAJA::loop_exec,            // j
              RAJA::statement::For<0, RAJA::loop_exec,          // i
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                                 RAJA::RangeSegment(0, nj),
                                                 RAJA::RangeSegment(0, nk)),
                                nestedinit_lam
                              );

      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
