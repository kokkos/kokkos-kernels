//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ATAX.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>


namespace rajaperf
{
namespace polybench
{


void POLYBENCH_ATAX::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps= getRunReps();

  POLYBENCH_ATAX_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_ATAX_BODY1;
          for (Index_type j = 0; j < N; ++j ) {
            POLYBENCH_ATAX_BODY2;
          }
          POLYBENCH_ATAX_BODY3;
        }

        #pragma omp parallel for
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_ATAX_BODY4;
          for (Index_type i = 0; i < N; ++i ) {
            POLYBENCH_ATAX_BODY5;
          }
          POLYBENCH_ATAX_BODY6;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto poly_atax_base_lam2 = [=] (Index_type i, Index_type j, 
                                      Real_type &dot) {
                                   POLYBENCH_ATAX_BODY2;
                                 };
      auto poly_atax_base_lam3 = [=] (Index_type i, 
                                      Real_type &dot) {
                                   POLYBENCH_ATAX_BODY3;
                                  };
      auto poly_atax_base_lam5 = [=] (Index_type i, Index_type j , 
                                      Real_type &dot) {
                                   POLYBENCH_ATAX_BODY5;
                                  };
      auto poly_atax_base_lam6 = [=] (Index_type j, 
                                      Real_type &dot) {
                                   POLYBENCH_ATAX_BODY6;
                                  };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_ATAX_BODY1;
          for (Index_type j = 0; j < N; ++j ) {
            poly_atax_base_lam2(i, j, dot);
          }
          poly_atax_base_lam3(i, dot);
        }

        #pragma omp parallel for
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_ATAX_BODY4;
          for (Index_type i = 0; i < N; ++i ) {
            poly_atax_base_lam5(i, j, dot);
          }
          poly_atax_base_lam6(j, dot);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_ATAX_VIEWS_RAJA;

      auto poly_atax_lam1 = [=] (Index_type i, Real_type &dot) {
                              POLYBENCH_ATAX_BODY1_RAJA;
                             };
      auto poly_atax_lam2 = [=] (Index_type i, Index_type j, Real_type &dot) {
                              POLYBENCH_ATAX_BODY2_RAJA;
                             };
      auto poly_atax_lam3 = [=] (Index_type i, Real_type &dot) {
                              POLYBENCH_ATAX_BODY3_RAJA;
                             };
      auto poly_atax_lam4 = [=] (Index_type j, Real_type &dot) {
                              POLYBENCH_ATAX_BODY4_RAJA;
                             };
      auto poly_atax_lam5 = [=] (Index_type i, Index_type j , Real_type &dot) {
                              POLYBENCH_ATAX_BODY5_RAJA;
                             };
      auto poly_atax_lam6 = [=] (Index_type j, Real_type &dot) {
                              POLYBENCH_ATAX_BODY6_RAJA;
                             };
      using EXEC_POL1 =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
          >
        >;

      using EXEC_POL2 =
        RAJA::KernelPolicy<
          RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
            RAJA::statement::Lambda<0, RAJA::Segs<1>, RAJA::Params<0>>,
            RAJA::statement::For<0, RAJA::loop_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<1>, RAJA::Params<0>>
          >
        >;

      
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
       
        RAJA::kernel_param<EXEC_POL1>(
          RAJA::make_tuple(RAJA::RangeSegment{0, N},
                           RAJA::RangeSegment{0, N}),
          RAJA::tuple<Real_type>{0.0},

          poly_atax_lam1,
          poly_atax_lam2,
          poly_atax_lam3

        );

        RAJA::kernel_param<EXEC_POL2>(
          RAJA::make_tuple(RAJA::RangeSegment{0, N},
                           RAJA::RangeSegment{0, N}),
          RAJA::tuple<Real_type>{0.0},

          poly_atax_lam4,
          poly_atax_lam5,
          poly_atax_lam6

        ); 

      }
      stopTimer();
      
      break;
    }

    default : {
      std::cout << "\n  POLYBENCH_ATAX : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
