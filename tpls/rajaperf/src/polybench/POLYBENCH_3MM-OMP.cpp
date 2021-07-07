//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_3MM.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>
#include <cstring>


#define USE_OMP_COLLAPSE
//#undef USE_OMP_COLLAPSE

#define USE_RAJA_OMP_COLLAPSE
//#undef USE_RAJA_OMP_COLLAPSE


namespace rajaperf 
{
namespace polybench
{

  
void POLYBENCH_3MM::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  POLYBENCH_3MM_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if defined(USE_OMP_COLLAPSE)
        #pragma omp parallel for collapse(2)
#else
        #pragma omp parallel for  
#endif
        for (Index_type i = 0; i < ni; i++ )  {
          for (Index_type j = 0; j < nj; j++) {
            POLYBENCH_3MM_BODY1;
            for (Index_type k = 0; k < nk; k++) {
              POLYBENCH_3MM_BODY2;
            }
            POLYBENCH_3MM_BODY3;
          }
        }

#if defined(USE_OMP_COLLAPSE)
        #pragma omp parallel for collapse(2)
#else
        #pragma omp parallel for  
#endif
        for (Index_type j = 0; j < nj; j++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY4;
            for (Index_type m = 0; m < nm; m++) {
              POLYBENCH_3MM_BODY5;
            }
            POLYBENCH_3MM_BODY6;
          }
        }

#if defined(USE_OMP_COLLAPSE)
        #pragma omp parallel for collapse(2)
#else
        #pragma omp parallel for  
#endif
        for (Index_type i = 0; i < ni; i++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY7;
            for (Index_type j = 0; j < nj; j++) {
              POLYBENCH_3MM_BODY8;
            }
            POLYBENCH_3MM_BODY9;
          }
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto poly_3mm_base_lam2 = [=] (Index_type i, Index_type j, Index_type k,
                                     Real_type &dot) {
                                  POLYBENCH_3MM_BODY2;
                                };
      auto poly_3mm_base_lam3 = [=] (Index_type i, Index_type j,
                                     Real_type &dot) {
                                  POLYBENCH_3MM_BODY3;
                                };
      auto poly_3mm_base_lam5 = [=] (Index_type j, Index_type l, Index_type m,
                                     Real_type &dot) {
                                   POLYBENCH_3MM_BODY5;
                                };
      auto poly_3mm_base_lam6 = [=] (Index_type j, Index_type l,
                                     Real_type &dot) {
                                  POLYBENCH_3MM_BODY6;
                                };
      auto poly_3mm_base_lam8 = [=] (Index_type i, Index_type l, Index_type j,
                                     Real_type &dot) {
                                  POLYBENCH_3MM_BODY8;
                                };
      auto poly_3mm_base_lam9 = [=] (Index_type i, Index_type l,
                                     Real_type &dot) {
                                  POLYBENCH_3MM_BODY9;
                                };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if defined(USE_OMP_COLLAPSE)
        #pragma omp parallel for collapse(2)
#else
        #pragma omp parallel for
#endif
        for (Index_type i = 0; i < ni; i++ )  {
          for (Index_type j = 0; j < nj; j++) {
            POLYBENCH_3MM_BODY1;
            for (Index_type k = 0; k < nk; k++) {
              poly_3mm_base_lam2(i, j, k, dot);
            }
            poly_3mm_base_lam3(i, j, dot);
          }
        }

#if defined(USE_OMP_COLLAPSE)
        #pragma omp parallel for collapse(2)
#else
        #pragma omp parallel for
#endif
        for (Index_type j = 0; j < nj; j++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY4;
            for (Index_type m = 0; m < nm; m++) {
              poly_3mm_base_lam5(j, l, m, dot);
            }
            poly_3mm_base_lam6(j, l, dot);
          }
        }

#if defined(USE_OMP_COLLAPSE)
        #pragma omp parallel for collapse(2)
#else
        #pragma omp parallel for
#endif
        for (Index_type i = 0; i < ni; i++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY7;
            for (Index_type j = 0; j < nj; j++) {
              poly_3mm_base_lam8(i, l, j, dot);
            }
            poly_3mm_base_lam9(i, l, dot);
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_3MM_VIEWS_RAJA;

      auto poly_3mm_lam1 = [=] (Real_type &dot) {
                                  POLYBENCH_3MM_BODY1_RAJA;
                                };
      auto poly_3mm_lam2 = [=] (Index_type i, Index_type j, Index_type k, 
                                Real_type &dot) {
                                  POLYBENCH_3MM_BODY2_RAJA;
                                };
      auto poly_3mm_lam3 = [=] (Index_type i, Index_type j,
                                Real_type &dot) {
                                  POLYBENCH_3MM_BODY3_RAJA;
                                };
      auto poly_3mm_lam4 = [=] (Real_type &dot) {
                                  POLYBENCH_3MM_BODY4_RAJA;
                                };
      auto poly_3mm_lam5 = [=] (Index_type j, Index_type l, Index_type m, 
                                Real_type &dot) {
                                  POLYBENCH_3MM_BODY5_RAJA;
                                };
      auto poly_3mm_lam6 = [=] (Index_type j, Index_type l,
                                Real_type &dot) {
                                  POLYBENCH_3MM_BODY6_RAJA;
                                };
      auto poly_3mm_lam7 = [=] (Real_type &dot) {
                                  POLYBENCH_3MM_BODY7_RAJA;
                                };
      auto poly_3mm_lam8 = [=] (Index_type i, Index_type l, Index_type j, 
                                Real_type &dot) {
                                  POLYBENCH_3MM_BODY8_RAJA;
                                };
      auto poly_3mm_lam9 = [=] (Index_type i, Index_type l,
                                Real_type &dot) {
                                  POLYBENCH_3MM_BODY9_RAJA;
                                };

#if defined(USE_RAJA_OMP_COLLAPSE)
      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    RAJA::ArgList<0, 1>,
            RAJA::statement::Lambda<0, RAJA::Params<0>>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1,2>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0,1>, RAJA::Params<0>>
          >
        >;
#else
      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0, RAJA::Params<0>>,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<1, RAJA::Segs<0,1,2>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0,1>, RAJA::Params<0>>
            >
          >
        >;
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nj},
                           RAJA::RangeSegment{0, nk}),
          RAJA::tuple<Real_type>{0.0},

          poly_3mm_lam1,
          poly_3mm_lam2,
          poly_3mm_lam3

        );

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                           RAJA::RangeSegment{0, nl},
                           RAJA::RangeSegment{0, nm}),
          RAJA::tuple<Real_type>{0.0},

          poly_3mm_lam4,
          poly_3mm_lam5,
          poly_3mm_lam6

        );

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nl},
                           RAJA::RangeSegment{0, nj}),
          RAJA::tuple<Real_type>{0.0},

          poly_3mm_lam7,
          poly_3mm_lam8,
          poly_3mm_lam9

        );

      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  POLYBENCH_2MM : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
