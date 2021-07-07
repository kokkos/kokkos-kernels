//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMM.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{


void POLYBENCH_GEMM::runSeqVariant(VariantID vid)
{
  const Index_type run_reps= getRunReps();

  POLYBENCH_GEMM_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = 0; i < ni; ++i ) { 
          for (Index_type j = 0; j < nj; ++j ) {
            POLYBENCH_GEMM_BODY1;
            POLYBENCH_GEMM_BODY2;
            for (Index_type k = 0; k < nk; ++k ) {
               POLYBENCH_GEMM_BODY3;
            }
            POLYBENCH_GEMM_BODY4;
          }
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto poly_gemm_base_lam2 = [=](Index_type i, Index_type j) {
                                   POLYBENCH_GEMM_BODY2;
                                 };
      auto poly_gemm_base_lam3 = [=](Index_type i, Index_type j, Index_type k,
                                     Real_type& dot) {
                                   POLYBENCH_GEMM_BODY3;
                                  };
      auto poly_gemm_base_lam4 = [=](Index_type i, Index_type j,
                                     Real_type& dot) {
                                   POLYBENCH_GEMM_BODY4;
                                  };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = 0; i < ni; ++i ) {
          for (Index_type j = 0; j < nj; ++j ) {
            POLYBENCH_GEMM_BODY1;
            poly_gemm_base_lam2(i, j);
            for (Index_type k = 0; k < nk; ++k ) {
              poly_gemm_base_lam3(i, j, k, dot);
            }
            poly_gemm_base_lam4(i, j, dot);
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      POLYBENCH_GEMM_VIEWS_RAJA;

      auto poly_gemm_lam1 = [=](Real_type& dot) {
                                POLYBENCH_GEMM_BODY1_RAJA;
                               };
      auto poly_gemm_lam2 = [=](Index_type i, Index_type j) {
                                POLYBENCH_GEMM_BODY2_RAJA;
                               };
      auto poly_gemm_lam3 = [=](Index_type i, Index_type j, Index_type k, 
                                Real_type& dot) {
                                POLYBENCH_GEMM_BODY3_RAJA;
                               };
      auto poly_gemm_lam4 = [=](Index_type i, Index_type j,
                                Real_type& dot) {
                                POLYBENCH_GEMM_BODY4_RAJA;
                               };

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0, RAJA::Params<0>>,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>>,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<2, RAJA::Segs<0,1,2>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<3, RAJA::Segs<0,1>, RAJA::Params<0>>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
     
          RAJA::make_tuple( RAJA::RangeSegment{0, ni},
                            RAJA::RangeSegment{0, nj},
                            RAJA::RangeSegment{0, nk} ),
          RAJA::tuple<Real_type>{0.0},  // variable for dot

          poly_gemm_lam1,
          poly_gemm_lam2,
          poly_gemm_lam3,
          poly_gemm_lam4

        );

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\n  POLYBENCH_GEMM : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace polybench
} // end namespace rajaperf
