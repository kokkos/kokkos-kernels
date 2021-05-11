//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

//
// Function used in TRAP_INT loop.
//
RAJA_INLINE
// 
KOKKOS_FUNCTION
Real_type trap_int_func(Real_type x,
                        Real_type y,
                        Real_type xp,
                        Real_type yp)
{
   Real_type denom = (x - xp)*(x - xp) + (y - yp)*(y - yp);
   denom = 1.0/sqrt(denom);
   return denom;
}


void TRAP_INT::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  TRAP_INT_DATA_SETUP;

// Declare KokkosViews that will wrap a pointer - not relevant in this case
// ...?



#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type sumx = m_sumx_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          TRAP_INT_BODY;
        }

        m_sumx += sumx * h;

      }
      stopTimer();

      break;
    }

    case Lambda_Seq : {

      auto trapint_base_lam = [=](Index_type i) -> Real_type {
                                Real_type x = x0 + i*h;
                                return trap_int_func(x, y, xp, yp);
                              };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type sumx = m_sumx_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          sumx += trapint_base_lam(i);
        }

        m_sumx += sumx * h;

      }
      stopTimer();

      break;
    }

    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

//        RAJA::ReduceSum<RAJA::seq_reduce, Real_type> sumx(m_sumx_init);

//       RAJA::forall<RAJA::loop_exec>(
//          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
//          TRAP_INT_BODY;
//
//          Begin Kokkos translation
//          A RAJA reduce translates into a Kokkoss::parallel_reduce
//          To perform the translation:
		// Declare and initialize variables
		// To perform a reduction, you need: 1) an initial value; 2) iterate
		// over an iterable; 3) to be able to extract the result at the end of
		// the reduction (in this case, trap_integral_val)

		Real_type trap_integral_val = m_sumx_init;

		Kokkos::parallel_reduce("TRAP_INT_Kokkos Kokkos_Lambda",
                                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                                KOKKOS_LAMBDA(const int64_t i, Real_type& sumx) {TRAP_INT_BODY},
                                trap_integral_val
                                );

        m_sumx += static_cast<Real_type>(trap_integral_val) * h;

      }
      Kokkos::fence();
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  TRAP_INT : Unknown variant id = " << vid << std::endl;
    }

  }
#endif //RUN_KOKKOS
}

} // end namespace basic
} // end namespace rajaperf
