//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

#include "POLYBENCH_FDTD_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define POLYBENCH_FDTD_2D_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(hz, m_hz, m_nx * m_ny, did, hid); \
  allocAndInitOpenMPDeviceData(ex, m_ex, m_nx * m_ny, did, hid); \
  allocAndInitOpenMPDeviceData(ey, m_ey, m_nx * m_ny, did, hid); \
  allocAndInitOpenMPDeviceData(fict, m_fict, m_tsteps, did, hid);


#define POLYBENCH_FDTD_2D_TEARDOWN_OMP_TARGET  \
  getOpenMPDeviceData(m_hz, hz, m_nx * m_ny, hid, did); \
  deallocOpenMPDeviceData(ex, did); \
  deallocOpenMPDeviceData(ey, did); \
  deallocOpenMPDeviceData(fict, did);


void POLYBENCH_FDTD_2D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_FDTD_2D_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    POLYBENCH_FDTD_2D_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (t = 0; t < tsteps; ++t) {

        #pragma omp target is_device_ptr(ey,fict) device( did )
        #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
        for (Index_type j = 0; j < ny; j++) {
          POLYBENCH_FDTD_2D_BODY1;
        }

        #pragma omp target is_device_ptr(ey,hz) device( did )
        #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
        for (Index_type i = 1; i < nx; i++) {
          for (Index_type j = 0; j < ny; j++) {
            POLYBENCH_FDTD_2D_BODY2;
          }
        }

        #pragma omp target is_device_ptr(ex,hz) device( did )
        #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
        for (Index_type i = 0; i < nx; i++) {
          for (Index_type j = 1; j < ny; j++) {
            POLYBENCH_FDTD_2D_BODY3;
          }
        }

        #pragma omp target is_device_ptr(ex,ey,hz) device( did )
        #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
        for (Index_type i = 0; i < nx - 1; i++) {
          for (Index_type j = 0; j < ny - 1; j++) {
            POLYBENCH_FDTD_2D_BODY4;
          }
        }

      }  // tstep loop

    }
    stopTimer();

    POLYBENCH_FDTD_2D_TEARDOWN_OMP_TARGET;

  } else if (vid == RAJA_OpenMPTarget) {

    POLYBENCH_FDTD_2D_DATA_SETUP_OMP_TARGET;

    POLYBENCH_FDTD_2D_VIEWS_RAJA;

    using EXEC_POL1 = RAJA::omp_target_parallel_for_exec<threads_per_team>;

    using EXEC_POL234 =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>,
          RAJA::statement::Lambda<0>
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (t = 0; t < tsteps; ++t) {

        RAJA::forall<EXEC_POL1>( RAJA::RangeSegment(0, ny),
         [=] (Index_type j) {
           POLYBENCH_FDTD_2D_BODY1_RAJA;
        });

        RAJA::kernel<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{1, nx},
                           RAJA::RangeSegment{0, ny}),
          [=] (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY2_RAJA;
          }
        );

        RAJA::kernel<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nx},
                           RAJA::RangeSegment{1, ny}),
          [=] (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY3_RAJA;
          }
        );

        RAJA::kernel<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nx-1},
                           RAJA::RangeSegment{0, ny-1}),
          [=] (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY4_RAJA;
          }
        );

      }  // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_FDTD_2D_TEARDOWN_OMP_TARGET;

  } else {
      std::cout << "\n  POLYBENCH_FDTD_2D : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
  
