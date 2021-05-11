//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FDTD_2D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{


void POLYBENCH_FDTD_2D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  POLYBENCH_FDTD_2D_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (t = 0; t < tsteps; ++t) {

          #pragma omp parallel for
          for (Index_type j = 0; j < ny; j++) {
            POLYBENCH_FDTD_2D_BODY1;
          }
          #pragma omp parallel for
          for (Index_type i = 1; i < nx; i++) {
            for (Index_type j = 0; j < ny; j++) {
              POLYBENCH_FDTD_2D_BODY2;
            }
          }
          #pragma omp parallel for
          for (Index_type i = 0; i < nx; i++) {
            for (Index_type j = 1; j < ny; j++) {
              POLYBENCH_FDTD_2D_BODY3;
            }
          }
          #pragma omp parallel for
          for (Index_type i = 0; i < nx - 1; i++) {
            for (Index_type j = 0; j < ny - 1; j++) {
              POLYBENCH_FDTD_2D_BODY4;
            }
          }

        }  // tstep loop

      }  // run_reps
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      //
      // Note: first lambda must use capture by reference so that the
      //       scalar variable 't' used in it is updated for each
      //       t-loop iteration.
      //
      auto poly_fdtd2d_base_lam1 = [&](Index_type j) {
                                     POLYBENCH_FDTD_2D_BODY1;
                                   };
      auto poly_fdtd2d_base_lam2 = [=](Index_type i, Index_type j) {
                                     POLYBENCH_FDTD_2D_BODY2;
                                   };
      auto poly_fdtd2d_base_lam3 = [=](Index_type i, Index_type j) {
                                     POLYBENCH_FDTD_2D_BODY3;
                                   };
      auto poly_fdtd2d_base_lam4 = [=](Index_type i, Index_type j) {
                                     POLYBENCH_FDTD_2D_BODY4;
                                   };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (t = 0; t < tsteps; ++t) {

          #pragma omp parallel for
          for (Index_type j = 0; j < ny; j++) {
            poly_fdtd2d_base_lam1(j);
          }
          #pragma omp parallel for
          for (Index_type i = 1; i < nx; i++) {
            for (Index_type j = 0; j < ny; j++) {
              poly_fdtd2d_base_lam2(i, j);
            }
          }
          #pragma omp parallel for
          for (Index_type i = 0; i < nx; i++) {
            for (Index_type j = 1; j < ny; j++) {
              poly_fdtd2d_base_lam3(i, j);
            }
          }
          #pragma omp parallel for
          for (Index_type i = 0; i < nx - 1; i++) {
            for (Index_type j = 0; j < ny - 1; j++) {
              poly_fdtd2d_base_lam4(i, j);
            }
          }

        }  // tstep loop

      }  // run_reps
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_FDTD_2D_VIEWS_RAJA;

      //
      // Note: first lambda must use capture by reference so that the
      //       scalar variable 't' used in it is updated for each
      //       t-loop iteration.
      //
      auto poly_fdtd2d_lam1 = [&](Index_type j) {
                                POLYBENCH_FDTD_2D_BODY1_RAJA;
                              };
      auto poly_fdtd2d_lam2 = [=](Index_type i, Index_type j) {
                                POLYBENCH_FDTD_2D_BODY2_RAJA;
                              };
      auto poly_fdtd2d_lam3 = [=](Index_type i, Index_type j) {
                                POLYBENCH_FDTD_2D_BODY3_RAJA;
                              };
      auto poly_fdtd2d_lam4 = [=](Index_type i, Index_type j) {
                                POLYBENCH_FDTD_2D_BODY4_RAJA;
                              };

      using EXEC_POL1 = RAJA::omp_parallel_for_exec;

      using EXEC_POL234 =  
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (t = 0; t < tsteps; ++t) {

          RAJA::forall<EXEC_POL1>( RAJA::RangeSegment(0, ny),
            poly_fdtd2d_lam1
          );

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{1, nx},
                             RAJA::RangeSegment{0, ny}),
            poly_fdtd2d_lam2
          );

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{0, nx},
                             RAJA::RangeSegment{1, ny}),
            poly_fdtd2d_lam3
          );

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{0, nx-1},
                             RAJA::RangeSegment{0, ny-1}),
            poly_fdtd2d_lam4
          );

        }  // tstep loop

      } // run_reps
      stopTimer();

      break;
    }

    default : {
      std::cout << "\nPOLYBENCH_FDTD_2D  Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
