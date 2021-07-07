//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#include <limits>
#include <iostream>

namespace rajaperf 
{
namespace basic
{


void REDUCE3_INT::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  REDUCE3_INT_DATA_SETUP;

  //Declare KokkosView that will wrap the pointer

  auto vec_view = getViewFromPointer(vec, iend); 

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Int_type vsum = m_vsum_init;
        Int_type vmin = m_vmin_init;
        Int_type vmax = m_vmax_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          REDUCE3_INT_BODY;
        }

        m_vsum += vsum;
        m_vmin = RAJA_MIN(m_vmin, vmin);
        m_vmax = RAJA_MAX(m_vmax, vmax);

      }
      stopTimer();

      break;
    }

    case Lambda_Seq : {

      auto init3_base_lam = [=](Index_type i) -> Int_type {
                              return vec[i];
                            };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Int_type vsum = m_vsum_init;
        Int_type vmin = m_vmin_init;
        Int_type vmax = m_vmax_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          vsum += init3_base_lam(i);
          vmin = RAJA_MIN(vmin, init3_base_lam(i));
          vmax = RAJA_MAX(vmax, init3_base_lam(i));
        }

        m_vsum += vsum;
        m_vmin = RAJA_MIN(m_vmin, vmin);
        m_vmax = RAJA_MAX(m_vmax, vmax);

      }
      stopTimer();

      break;
    }

    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
/*
        RAJA::ReduceSum<RAJA::seq_reduce, Int_type> vsum(m_vsum_init);
        RAJA::ReduceMin<RAJA::seq_reduce, Int_type> vmin(m_vmin_init);
        RAJA::ReduceMax<RAJA::seq_reduce, Int_type> vmax(m_vmax_init);

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          REDUCE3_INT_BODY_RAJA;
        });

        m_vsum += static_cast<Int_type>(vsum.get());
        m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
        m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));
*/
		// These values are initilized elsewhere by RPS
        // These variables were declared to Kokkos-ify the parallel_reduce
        // construct:
#ifndef RAJA_ENABLE_TARGET_OPENMP
		Int_type max_value = m_vmax_init;
		Int_type min_value = m_vmin_init;
		Int_type sum = m_vsum_init;


		parallel_reduce("REDUCE3-Kokkos Kokkos_Lambda",
                         Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
			             KOKKOS_LAMBDA(const int64_t i, Int_type& tl_max, Int_type& tl_min, Int_type& tl_sum){
                             Int_type vec_i = vec_view[i];
		                         if (vec_i > tl_max) tl_max = vec_i;
		                         if (vec_i < tl_min) tl_min = vec_i;
                              tl_sum += vec_i;
                              },
                              Kokkos::Max<Int_type>(max_value),
                              Kokkos::Min<Int_type>(min_value),
                              sum);
            m_vsum += static_cast<Int_type>(sum);
            m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(min_value));
            m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(max_value));
#endif	
      }
      Kokkos::fence();
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  REDUCE3_INT : Unknown variant id = " << vid << std::endl;
    }

  }
#endif // RUN_KOKKOS

  moveDataToHostFromKokkosView(vec, vec_view, iend);
}

} // end namespace basic
} // end namespace rajaperf
