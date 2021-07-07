//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SORTPAIRS.hpp"

#include "RAJA/RAJA.hpp"

#include <algorithm>
#include <vector>
#include <utility>
#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void SORTPAIRS::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  SORTPAIRS_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        using pair_type = std::pair<Real_type, Real_type>;

        std::vector<pair_type> vector_of_pairs;
        vector_of_pairs.reserve(iend-ibegin);

        for (Index_type iemp = ibegin; iemp < iend; ++iemp) {
          vector_of_pairs.emplace_back(x[iend*irep + iemp], i[iend*irep + iemp]);
        }

        std::sort(vector_of_pairs.begin(), vector_of_pairs.end(),
            [](pair_type const& lhs, pair_type const& rhs) {
              return lhs.first < rhs.first;
            });

        for (Index_type iemp = ibegin; iemp < iend; ++iemp) {
          pair_type& pair = vector_of_pairs[iemp - ibegin];
          x[iend*irep + iemp] = pair.first;
          i[iend*irep + iemp] = pair.second;
        }

      }
      stopTimer();

      break;
    }
#if defined(RUN_RAJA_SEQ)
    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::sort_pairs<RAJA::loop_exec>(SORTPAIRS_RAJA_ARGS);

      }
      stopTimer();

      break;
    }
#endif

    default : {
      std::cout << "\n  SORTPAIRS : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace algorithm
} // end namespace rajaperf
