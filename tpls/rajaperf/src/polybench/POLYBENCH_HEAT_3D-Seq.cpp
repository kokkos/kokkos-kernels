//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_HEAT_3D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{


void POLYBENCH_HEAT_3D::runSeqVariant(VariantID vid)
{
  const Index_type run_reps= getRunReps();

  POLYBENCH_HEAT_3D_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) { 

          for (Index_type i = 1; i < N-1; ++i ) { 
            for (Index_type j = 1; j < N-1; ++j ) { 
              for (Index_type k = 1; k < N-1; ++k ) { 
                POLYBENCH_HEAT_3D_BODY1;
              }
            }
          }

          for (Index_type i = 1; i < N-1; ++i ) { 
            for (Index_type j = 1; j < N-1; ++j ) { 
              for (Index_type k = 1; k < N-1; ++k ) { 
                POLYBENCH_HEAT_3D_BODY2;
              }
            }
          }

        }

      }
      stopTimer();

      POLYBENCH_HEAT_3D_DATA_RESET;

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto poly_heat3d_base_lam1 = [=](Index_type i, Index_type j, 
                                       Index_type k) {
                                     POLYBENCH_HEAT_3D_BODY1;
                                   };
      auto poly_heat3d_base_lam2 = [=](Index_type i, Index_type j, 
                                       Index_type k) {
                                     POLYBENCH_HEAT_3D_BODY2;
                                   };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          for (Index_type i = 1; i < N-1; ++i ) {
            for (Index_type j = 1; j < N-1; ++j ) {
              for (Index_type k = 1; k < N-1; ++k ) {
                poly_heat3d_base_lam1(i, j, k);
              }
            }
          }

          for (Index_type i = 1; i < N-1; ++i ) {
            for (Index_type j = 1; j < N-1; ++j ) {
              for (Index_type k = 1; k < N-1; ++k ) {
                poly_heat3d_base_lam2(i, j, k);
              }
            }
          }

        }

      }
      stopTimer();

      POLYBENCH_HEAT_3D_DATA_RESET;

      break;
    }

    case RAJA_Seq : {

      POLYBENCH_HEAT_3D_VIEWS_RAJA;

      auto poly_heat3d_lam1 = [=](Index_type i, Index_type j, Index_type k) {
                                POLYBENCH_HEAT_3D_BODY1_RAJA;
                              };
      auto poly_heat3d_lam2 = [=](Index_type i, Index_type j, Index_type k) {
                                POLYBENCH_HEAT_3D_BODY2_RAJA;
                              };

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
            >
          >,
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<1>
              >
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1}),

            poly_heat3d_lam1,
            poly_heat3d_lam2
          );

        }

      }
      stopTimer();

      POLYBENCH_HEAT_3D_DATA_RESET;

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\n  POLYBENCH_HEAT_3D : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace polybench
} // end namespace rajaperf
