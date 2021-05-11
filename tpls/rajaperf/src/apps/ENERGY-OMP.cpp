//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{


void ENERGY::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ENERGY_DATA_SETUP;
  
  auto energy_lam1 = [=](Index_type i) {
                       ENERGY_BODY1;
                     };
  auto energy_lam2 = [=](Index_type i) {
                       ENERGY_BODY2;
                     };
  auto energy_lam3 = [=](Index_type i) {
                       ENERGY_BODY3;
                     };
  auto energy_lam4 = [=](Index_type i) {
                       ENERGY_BODY4;
                     };
  auto energy_lam5 = [=](Index_type i) {
                       ENERGY_BODY5;
                     };
  auto energy_lam6 = [=](Index_type i) {
                       ENERGY_BODY6;
                     };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
        {
          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            ENERGY_BODY1;
          }

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            ENERGY_BODY2;
          }

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            ENERGY_BODY3;
          }

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            ENERGY_BODY4;
          }

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            ENERGY_BODY5;
          }

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            ENERGY_BODY6;
          }

        } // end omp parallel region

      }
      stopTimer();

      break;
    }
  
    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
        {
          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            energy_lam1(i);
          }

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            energy_lam2(i);
          }

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            energy_lam3(i);
          }

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            energy_lam4(i);
          }

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            energy_lam5(i);
          }

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            energy_lam6(i);
          }

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), energy_lam1);

          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), energy_lam2);

          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), energy_lam3);
  
          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), energy_lam4);
  
          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), energy_lam5);
  
          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), energy_lam6);
  
        }); // end omp parallel region

      }
      stopTimer();
      break;
    }

    default : {
      std::cout << "\n  ENERGY : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace apps
} // end namespace rajaperf
