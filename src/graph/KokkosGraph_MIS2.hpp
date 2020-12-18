/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Brian Kelley (bmkelle@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef _KOKKOSGRAPH_DISTANCE2_MIS_HPP
#define _KOKKOSGRAPH_DISTANCE2_MIS_HPP

#include "KokkosGraph_Distance2MIS_impl.hpp"

namespace KokkosGraph{

enum MIS2_Algorithm
{
  MIS2_QUALITY,       //KokkosKernels highest quality (but slow) algorithm, based on ECL-MIS
  MIS2_BELL,          //Bell/Dalton/Olson original algorithm
  MIS2_RANDOMIZED,    //Above plus randomization every round
  MIS2_WORKLIST,      //Above plus worklists
  MIS2_PACKEDSTATUS,  //Above plus packing the (status, random, index) tuple into a single value
  MIS2_FAST           //KokkosKernels fastest algorithm (Above plus TeamPolicy if degree is high enough)
};

namespace Experimental{

// Compute a distance-2 maximal independent set, given a symmetric CRS graph.
// Returns a list of the vertices in the set.
//
// Column indices >= num_verts are ignored.

template <typename device_t, typename rowmap_t, typename colinds_t, typename lno_view_t = typename colinds_t::non_const_type>
lno_view_t
graph_d2_mis(const rowmap_t& rowmap, const colinds_t& colinds, MIS2_Algorithm algo = MIS2_FAST, int* numRounds = nullptr)
{
  if(rowmap.extent(0) <= 1)
  {
    //zero vertices means the MIS is empty.
    return lno_view_t();
  }
  switch(algo)
  {
    case MIS2_QUALITY:
    {
      Impl::D2_MIS_FixedPriority<device_t, rowmap_t, colinds_t, lno_view_t> mis(rowmap, colinds);
      return mis.compute(numRounds);
    }
    case MIS2_FAST:
    {
      Impl::D2_MIS_RandomPriority<device_t, rowmap_t, colinds_t, lno_view_t, true> mis(rowmap, colinds);
      return mis.compute(numRounds);
    }
    case MIS2_BELL:
    {
      Impl::D2_MIS_Bell<device_t, rowmap_t, colinds_t, lno_view_t> mis;
      return mis.compute(rowmap, colinds, numRounds);
    }
    case MIS2_RANDOMIZED:
    {
      Impl::D2_MIS_Randomized<device_t, rowmap_t, colinds_t, lno_view_t> mis;
      return mis.compute(rowmap, colinds, numRounds);
    }
    case MIS2_WORKLIST:
    {
      Impl::D2_MIS_Worklist<device_t, rowmap_t, colinds_t, lno_view_t> mis;
      return mis.compute(rowmap, colinds, numRounds);
    }
    case MIS2_PACKEDSTATUS:
    {
      Impl::D2_MIS_RandomPriority<device_t, rowmap_t, colinds_t, lno_view_t, false> mis(rowmap, colinds);
      return mis.compute(numRounds);
    }
    default:
      throw std::runtime_error("MIS2 alg not imlemented yet");
  }
  throw std::invalid_argument("graph_d2_mis: invalid algorithm");
}

template <typename device_t, typename rowmap_t, typename colinds_t, typename labels_t = typename colinds_t::non_const_type>
labels_t
graph_mis2_coarsen(const rowmap_t& rowmap, const colinds_t& colinds, typename colinds_t::non_const_value_type& numClusters, MIS2_Algorithm algo = MIS2_FAST)
{
  if(rowmap.extent(0) <= 1)
  {
    //there are no vertices to label
    return labels_t();
  }
  labels_t mis2 = graph_d2_mis<device_t, rowmap_t, colinds_t, labels_t>(rowmap, colinds, algo);
  numClusters = mis2.extent(0);
  Impl::D2_MIS_Coarsening<device_t, rowmap_t, colinds_t, labels_t> coarsening(rowmap, colinds, mis2);
  return coarsening.compute();
}

inline const char* mis2_algorithm_name(MIS2_Algorithm algo)
{
  switch(algo)
  {
    case MIS2_QUALITY:
      return "MIS2_QUALITY";
    case MIS2_BELL:
      return "MIS2_BELL";
    case MIS2_RANDOMIZED:
      return "MIS2_RANDOMIZED";
    case MIS2_WORKLIST:
      return "MIS2_WORKLIST";
    case MIS2_PACKEDSTATUS:
      return "MIS2_PACKEDSTATUS";
    case MIS2_FAST:
      return "MIS2_FAST";
  }
  return "*** Invalid MIS2 algo enum value.\n";
}

}  // end namespace Experimental
}  // end namespace KokkosGraph

#endif
