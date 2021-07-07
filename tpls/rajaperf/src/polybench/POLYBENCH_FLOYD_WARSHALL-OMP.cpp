//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

//#define USE_OMP_COLLAPSE
#undef USE_OMP_COLLAPSE

//#define USE_RAJA_OMP_COLLAPSE
#undef USE_RAJA_OMP_COLLAPSE

namespace rajaperf 
{
namespace polybench
{

 
void POLYBENCH_FLOYD_WARSHALL::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps= getRunReps();

  POLYBENCH_FLOYD_WARSHALL_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < N; ++k) {
#if defined(USE_OMP_COLLAPSE)
          #pragma omp parallel for collapse(2)
#else
          #pragma omp parallel for
#endif
          for (Index_type i = 0; i < N; ++i) {  
            for (Index_type j = 0; j < N; ++j) {
              POLYBENCH_FLOYD_WARSHALL_BODY;
            }
          }
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto poly_floydwarshall_base_lam = [=](Index_type k, Index_type i, 
                                             Index_type j) {
                                           POLYBENCH_FLOYD_WARSHALL_BODY;
                                         };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < N; ++k) {
#if defined(USE_OMP_COLLAPSE)
          #pragma omp parallel for collapse(2)
#else
          #pragma omp parallel for
#endif
          for (Index_type i = 0; i < N; ++i) {
            for (Index_type j = 0; j < N; ++j) {
              poly_floydwarshall_base_lam(k, i, j);
            }
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_FLOYD_WARSHALL_VIEWS_RAJA; 

      auto poly_floydwarshall_lam = [=](Index_type k, Index_type i, 
                                        Index_type j) {
                                      POLYBENCH_FLOYD_WARSHALL_BODY_RAJA;
                                     };

#if defined(USE_RAJA_OMP_COLLAPSE)
      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                      RAJA::ArgList<1, 2>,
              RAJA::statement::Lambda<0>
            >
          >
        >;
#else
      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                                 RAJA::RangeSegment{0, N},
                                                 RAJA::RangeSegment{0, N}),
          poly_floydwarshall_lam 
        );

      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
