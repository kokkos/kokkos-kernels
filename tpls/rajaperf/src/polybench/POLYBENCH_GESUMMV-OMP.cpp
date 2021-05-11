//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GESUMMV.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{

 
void POLYBENCH_GESUMMV::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps= getRunReps();

  POLYBENCH_GESUMMV_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_GESUMMV_BODY1;
          for (Index_type j = 0; j < N; ++j ) {
            POLYBENCH_GESUMMV_BODY2;
          }
          POLYBENCH_GESUMMV_BODY3;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto poly_gesummv_base_lam2 = [=](Index_type i, Index_type j,
                                        Real_type& tmpdot, Real_type& ydot) {
                                      POLYBENCH_GESUMMV_BODY2;
                                    };
      auto poly_gesummv_base_lam3 = [=](Index_type i,
                                        Real_type& tmpdot, Real_type& ydot) {
                                      POLYBENCH_GESUMMV_BODY3;
                                    };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_GESUMMV_BODY1;
          for (Index_type j = 0; j < N; ++j ) {
            poly_gesummv_base_lam2(i, j, tmpdot, ydot);
          }
          poly_gesummv_base_lam3(i, tmpdot, ydot);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_GESUMMV_VIEWS_RAJA;

      auto poly_gesummv_lam1 = [=](Real_type& tmpdot, Real_type& ydot) {
                                   POLYBENCH_GESUMMV_BODY1_RAJA;
                                  };
      auto poly_gesummv_lam2 = [=](Index_type i, Index_type j,
                                   Real_type& tmpdot, Real_type& ydot) {
                                   POLYBENCH_GESUMMV_BODY2_RAJA;
                                  };
      auto poly_gesummv_lam3 = [=](Index_type i,
                                   Real_type& tmpdot, Real_type& ydot) {
                                   POLYBENCH_GESUMMV_BODY3_RAJA;
                                  };

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::Lambda<0, RAJA::Params<0,1>>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0, 1>, RAJA::Params<0,1>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0,1>>
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple( RAJA::RangeSegment{0, N},
                            RAJA::RangeSegment{0, N} ),
          RAJA::make_tuple(static_cast<Real_type>(0.0), 
                           static_cast<Real_type>(0.0)),

          poly_gesummv_lam1,
          poly_gesummv_lam2,
          poly_gesummv_lam3
        );

      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  POLYBENCH_GESUMMV : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
