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

#ifndef _KOKKOSGRAPH_DISTANCE2_MIS_IMPL_HPP
#define _KOKKOSGRAPH_DISTANCE2_MIS_IMPL_HPP

#include "Kokkos_Core.hpp"
#include "Kokkos_Bitset.hpp"
#include "KokkosKernels_Utils.hpp"
#include "Kokkos_Random.hpp"
#include <cstdint>

namespace KokkosGraph {
namespace Experimental {
namespace Impl {

template<typename device_t, typename rowmap_t, typename entries_t, typename lno_view_t, bool allowTeams>
struct D2_MIS_RandomPriority
{
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;
  using bitset_t = Kokkos::Bitset<device_t>;
  using const_bitset_t = Kokkos::ConstBitset<device_t>;
  using size_type = typename rowmap_t::non_const_value_type;
  using lno_t = typename entries_t::non_const_value_type;
  //The type of status/priority values.
  using status_t = typename std::make_unsigned<lno_t>::type;
  using status_view_t = Kokkos::View<status_t*, mem_space>;
  using range_pol = Kokkos::RangePolicy<exec_space>;
  using team_pol = Kokkos::TeamPolicy<exec_space>;
  using team_mem = typename team_pol::member_type;
  using all_worklists_t = Kokkos::View<lno_t**, Kokkos::LayoutLeft, mem_space>;
  using worklist_t = Kokkos::View<lno_t*, Kokkos::LayoutLeft, mem_space>;

  // Priority values 0 and max are special, they mean the vertex is
  // in the independent set or eliminated from consideration, respectively.
  // Values in between represent a priority for being added to the set,
  // based on degree and vertex ID as a tiebreak
  //   (higher priority = less preferred to being in the independent set)

  static constexpr status_t IN_SET = 0;
  static constexpr status_t OUT_SET = ~IN_SET;

  D2_MIS_RandomPriority(const rowmap_t& rowmap_, const entries_t& entries_)
    : rowmap(rowmap_), entries(entries_), numVerts(rowmap.extent(0) - 1)
  {
    status_t i = numVerts + 1;
    nvBits = 0;
    while(i)
    {
      i >>= 1;
      nvBits++;
    }
    //Each value in rowStatus represents the status and priority of each row.
    //Each value in colStatus represents the lowest nonzero priority of any row adjacent to the column.
    //  This counts up monotonically as vertices are eliminated (given status OUT_SET)
    rowStatus = status_view_t(Kokkos::ViewAllocateWithoutInitializing("RowStatus"), numVerts);
    colStatus = status_view_t(Kokkos::ViewAllocateWithoutInitializing("ColStatus"), numVerts);
    allWorklists = Kokkos::View<lno_t**, Kokkos::LayoutLeft, mem_space>(Kokkos::ViewAllocateWithoutInitializing("AllWorklists"), numVerts, 3);
  }

  struct RefreshRowStatus
  {
    RefreshRowStatus(const status_view_t& rowStatus_, const worklist_t& worklist_, lno_t nvBits_, int round)
      : rowStatus(rowStatus_), worklist(worklist_), nvBits(nvBits_)
    {
      hashedRound = KokkosKernels::Impl::xorshiftHash<status_t>(round);
    }

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      //Combine vertex and round to get some pseudorandom priority bits that change each round
      status_t priority = KokkosKernels::Impl::xorshiftHash<status_t>(KokkosKernels::Impl::xorshiftHash<status_t>(i) ^ hashedRound);
      //Generate unique status per row, with IN_SET < status < OUT_SET,
      status_t newStatus = (status_t) (i + 1) | (priority << nvBits);
      if(newStatus == OUT_SET)
        newStatus--;
      rowStatus(i) = newStatus;
    }

    status_view_t rowStatus;
    worklist_t worklist;
    int nvBits;
    uint32_t hashedRound;
  };

  struct RefreshColStatus
  {
    RefreshColStatus(const status_view_t& colStatus_, const worklist_t& worklist_, const status_view_t& rowStatus_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_, lno_t worklistLen_)
      : colStatus(colStatus_), worklist(worklist_), rowStatus(rowStatus_), rowmap(rowmap_), entries(entries_), nv(nv_), worklistLen(worklistLen_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      //iterate over {i} union the neighbors of i, to find
      //minimum status.
      status_t s = rowStatus(i);
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      for(size_type j = rowBegin; j < rowEnd; j++)
      {
        lno_t nei = entries(j);
        if(nei < nv && nei != i)
        {
          status_t neiStat = rowStatus(nei);
          if(neiStat < s)
            s = neiStat;
        }
      }
      if(s == IN_SET)
        s = OUT_SET;
      colStatus(i) = s;
    }

    KOKKOS_INLINE_FUNCTION void operator()(const team_mem& t) const
    {
      using MinReducer = Kokkos::Min<status_t>;
      lno_t w = t.league_rank() * t.team_size() + t.team_rank();
      if(w >= worklistLen)
        return;
      lno_t i = worklist(w);
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      lno_t rowLen = rowEnd - rowBegin;
      //iterate over {i} union the neighbors of i, to find
      //minimum status.
      status_t s;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, rowLen + 1),
      [&](lno_t j, status_t& ls)
      {
        lno_t nei = (j == rowLen) ? i : entries(rowBegin + j);
        if(nei < nv)
        {
          status_t neiStat = rowStatus(nei);
          if(neiStat < ls)
            ls = neiStat;
        }
      }, MinReducer(s));
      Kokkos::single(Kokkos::PerThread(t),
      [&]()
      {
        if(s == IN_SET)
          s = OUT_SET;
        colStatus(i) = s;
      });
    }

    status_view_t colStatus;
    worklist_t worklist;
    status_view_t rowStatus;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
    lno_t worklistLen;
  };

  struct DecideSetFunctor
  {
    DecideSetFunctor(const status_view_t& rowStatus_, const status_view_t& colStatus_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_, const worklist_t& worklist_, lno_t worklistLen_)
      : rowStatus(rowStatus_), colStatus(colStatus_), rowmap(rowmap_), entries(entries_), nv(nv_), worklist(worklist_), worklistLen(worklistLen_)
    {}

    //Enum values to be used as flags, so that the team policy version can
    //express the neighbor checking as an OR-reduction
    enum
    {
      NEI_OUT_SET = 1,
      NEI_DIFFERENT_STATUS = 2
    };

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      //Processing row i.
      status_t s = rowStatus(i);
      if(s == IN_SET || s == OUT_SET)
        return;
      //s is the status which must be the minimum among all neighbors
      //to decide that i is IN_SET.
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      bool neiOut = false;
      bool neiMismatchS = false;
      for(size_type j = rowBegin; j <= rowEnd; j++)
      {
        lno_t nei = (j == rowEnd) ? i : entries(j);
        if(nei >= nv)
          continue;
        status_t neiStat = colStatus(nei);
        if(neiStat == OUT_SET)
        {
          neiOut = true;
          break;
        }
        else if(neiStat != s)
        {
          neiMismatchS = true;
        }
      }
      if(neiOut)
      {
        //In order to make future progress, need to update the
        //col statuses for all neighbors of i.
        rowStatus(i) = OUT_SET;
      }
      else if(!neiMismatchS)
      {
        //all neighboring col statuses match s, therefore s is the minimum status among all d2 neighbors
        rowStatus(i) = IN_SET;
      }
    }

    KOKKOS_INLINE_FUNCTION void operator()(const team_mem& t) const
    {
      using OrReducer = Kokkos::BOr<int>;
      lno_t w = t.league_rank() * t.team_size() + t.team_rank();
      if(w >= worklistLen)
        return;
      lno_t i = worklist(w);
      //Processing row i.
      status_t s = rowStatus(i);
      if(s == IN_SET || s == OUT_SET)
        return;
      //s is the status which must be the minimum among all neighbors
      //to decide that i is IN_SET.
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      lno_t rowLen = rowEnd - rowBegin;
      int flags = 0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, rowLen + 1),
      [&](lno_t j, int& lflags)
      {
        lno_t nei = (j == rowLen) ? i : entries(rowBegin + j);
        if(nei >= nv)
          return;
        status_t neiStat = colStatus(nei);
        if(neiStat == OUT_SET)
          lflags |= NEI_OUT_SET;
        else if(neiStat != s)
          lflags |= NEI_DIFFERENT_STATUS;
      }, OrReducer(flags));
      Kokkos::single(Kokkos::PerThread(t),
      [&]()
      {
        if(flags & NEI_OUT_SET)
        {
          //In order to make future progress, need to update the
          //col statuses for all neighbors of i.
          rowStatus(i) = OUT_SET;
        }
        else if(!(flags & NEI_DIFFERENT_STATUS))
        {
          //all neighboring col statuses match s, therefore s is the minimum status among all d2 neighbors
          rowStatus(i) = IN_SET;
        }
      });
    }

    status_view_t rowStatus;
    status_view_t colStatus;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
    worklist_t worklist;
    lno_t worklistLen;
  };

  struct CountInSet
  {
    CountInSet(const status_view_t& rowStatus_)
      : rowStatus(rowStatus_)
    {}
    KOKKOS_INLINE_FUNCTION void operator()(lno_t i, lno_t& lNumInSet) const
    {
      if(rowStatus(i) == IN_SET)
        lNumInSet++;
    }
    status_view_t rowStatus;
  };

  struct CompactInSet
  {
    CompactInSet(const status_view_t& rowStatus_, const lno_view_t& setList_)
      : rowStatus(rowStatus_), setList(setList_)
    {}
    KOKKOS_INLINE_FUNCTION void operator()(lno_t i, lno_t& lNumInSet, bool finalPass) const
    {
      if(rowStatus(i) == IN_SET)
      {
        if(finalPass)
          setList(lNumInSet) = i;
        lNumInSet++;
      }
    }
    status_view_t rowStatus;
    lno_view_t setList;
  };

  struct MaskedWorklist
  {
    MaskedWorklist(const lno_view_t& mask_, const lno_view_t& worklist_)
      : mask(mask_), worklist(worklist_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i, lno_t& lNumInList, bool finalPass) const
    {
      if(mask(i) < 0)
      {
        if(finalPass)
          worklist(lNumInList) = i;
        lNumInList++;
      }
    }
    lno_view_t mask;
    lno_view_t worklist;
  };

  struct CompactWorklistFunctor
  {
    CompactWorklistFunctor(const worklist_t& src_, const worklist_t& dst_, const status_view_t& status_)
      : src(src_), dst(dst_), status(status_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w, lno_t& lNumInSet, bool finalPass) const
    {
      lno_t i = src(w);
      status_t s = status(i);
      if(s != IN_SET && s != OUT_SET)
      {
        //next worklist needs to contain i
        if(finalPass)
          dst(lNumInSet) = i;
        lNumInSet++;
      }
    }

    worklist_t src;
    worklist_t dst;
    status_view_t status;
  };

  lno_view_t compute(int* numRounds)
  {
    //Initialize first worklist to 0...numVerts
    worklist_t rowWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 0);
    worklist_t colWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 1);
    KokkosKernels::Impl::sequential_fill(rowWorklist);
    KokkosKernels::Impl::sequential_fill(colWorklist);
    worklist_t thirdWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 2);
    auto execSpaceEnum = KokkosKernels::Impl::kk_get_exec_space_type<exec_space>();
    bool useTeams = allowTeams && KokkosKernels::Impl::kk_is_gpu_exec_space<exec_space>() && (entries.extent(0) / numVerts >= 16);
    int vectorLength = KokkosKernels::Impl::kk_get_suggested_vector_size(numVerts, entries.extent(0), execSpaceEnum);
    int round = 0;
    lno_t rowWorkLen = numVerts;
    lno_t colWorkLen = numVerts;
    int refreshColTeamSize = 0;
    int decideSetTeamSize = 0;
    if(useTeams)
    {
      team_pol dummyPolicy(1, 1, vectorLength);
      //Compute the recommended team size for RefreshColStatus and DecideSetFunctor (will be constant)
      {
        RefreshColStatus refreshCol(colStatus, colWorklist, rowStatus, rowmap, entries, numVerts, colWorkLen);
        refreshColTeamSize = dummyPolicy.team_size_max(refreshCol, Kokkos::ParallelForTag());
      }
      {
        DecideSetFunctor decideSet(rowStatus, colStatus, rowmap, entries, numVerts, rowWorklist, rowWorkLen);
        decideSetTeamSize = dummyPolicy.team_size_max(decideSet, Kokkos::ParallelForTag());
      }
    }
    while(true)
    {
      //Compute new row statuses
      Kokkos::parallel_for(range_pol(0, rowWorkLen), RefreshRowStatus(rowStatus, rowWorklist, nvBits, round));
      //Compute new col statuses
      {
        RefreshColStatus refreshCol(colStatus, colWorklist, rowStatus, rowmap, entries, numVerts, colWorkLen);
        if(useTeams)
          Kokkos::parallel_for(team_pol((colWorkLen + refreshColTeamSize - 1) / refreshColTeamSize, refreshColTeamSize, vectorLength), refreshCol);
        else
          Kokkos::parallel_for(range_pol(0, colWorkLen), refreshCol);
      }
      //Decide row statuses where enough information is available
      {
        DecideSetFunctor decideSet(rowStatus, colStatus, rowmap, entries, numVerts, rowWorklist, rowWorkLen);
        if(useTeams)
          Kokkos::parallel_for(team_pol((rowWorkLen + decideSetTeamSize - 1) / decideSetTeamSize, decideSetTeamSize, vectorLength), decideSet);
        else
          Kokkos::parallel_for(range_pol(0, rowWorkLen), decideSet);
      }
      round++;
      //Compact row worklist
      Kokkos::parallel_scan(range_pol(0, rowWorkLen), CompactWorklistFunctor(rowWorklist, thirdWorklist, rowStatus), rowWorkLen);
      if(rowWorkLen == 0)
        break;
      std::swap(rowWorklist, thirdWorklist);
      //Compact col worklist
      Kokkos::parallel_scan(range_pol(0, colWorkLen), CompactWorklistFunctor(colWorklist, thirdWorklist, colStatus), colWorkLen);
      std::swap(colWorklist, thirdWorklist);
    }
    if(numRounds)
      *numRounds = round;
    //now that every vertex has been decided IN_SET/OUT_SET,
    //build a compact list of the vertices which are IN_SET.
    lno_t numInSet = 0;
    Kokkos::parallel_reduce(range_pol(0, numVerts), CountInSet(rowStatus), numInSet);
    lno_view_t setList(Kokkos::ViewAllocateWithoutInitializing("D2MIS"), numInSet);
    Kokkos::parallel_scan(range_pol(0, numVerts), CompactInSet(rowStatus, setList));
    return setList;
  }

  //Compute with an initial mask: vertices with mask value < 0 are completely ignored
  lno_view_t compute(const lno_view_t& mask)
  {
    //Initialize first worklist to 0...numVerts
    worklist_t rowWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 0);
    worklist_t colWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 1);
    lno_t rowWorkLen = numVerts;
    lno_t colWorkLen = numVerts;
    //Row worklist: initially only the non-masked vertices
    Kokkos::parallel_scan(range_pol(0, numVerts), MaskedWorklist(mask, rowWorklist), rowWorkLen);
    //TODO: set up col worklist as the union of neighbors row worklist. But all columns is correct for now.
    KokkosKernels::Impl::sequential_fill(colWorklist);
    //Need to fill rowStatus with OUT_SET initially so that vertices not in the worklist don't affect algorithm
    Kokkos::deep_copy(rowStatus, ~(status_t(0)));
    worklist_t thirdWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 2);
    auto execSpaceEnum = KokkosKernels::Impl::kk_get_exec_space_type<exec_space>();
    bool useTeams = allowTeams && KokkosKernels::Impl::kk_is_gpu_exec_space<exec_space>() && (entries.extent(0) / numVerts >= 16);
    int vectorLength = KokkosKernels::Impl::kk_get_suggested_vector_size(numVerts, entries.extent(0), execSpaceEnum);
    int round = 0;
    int refreshColTeamSize = 0;
    int decideSetTeamSize = 0;
    if(useTeams)
    {
      team_pol dummyPolicy(1, 1, vectorLength);
      //Compute the recommended team size for RefreshColStatus and DecideSetFunctor (will be constant)
      {
        RefreshColStatus refreshCol(colStatus, colWorklist, rowStatus, rowmap, entries, numVerts, colWorkLen);
        refreshColTeamSize = dummyPolicy.team_size_max(refreshCol, Kokkos::ParallelForTag());
      }
      {
        DecideSetFunctor decideSet(rowStatus, colStatus, rowmap, entries, numVerts, rowWorklist, rowWorkLen);
        decideSetTeamSize = dummyPolicy.team_size_max(decideSet, Kokkos::ParallelForTag());
      }
    }
    while(true)
    {
      //Compute new row statuses
      Kokkos::parallel_for(range_pol(0, rowWorkLen), RefreshRowStatus(rowStatus, rowWorklist, nvBits, round));
      //Compute new col statuses
      {
        RefreshColStatus refreshCol(colStatus, colWorklist, rowStatus, rowmap, entries, numVerts, colWorkLen);
        if(useTeams)
          Kokkos::parallel_for(team_pol((colWorkLen + refreshColTeamSize - 1) / refreshColTeamSize, refreshColTeamSize, vectorLength), refreshCol);
        else
          Kokkos::parallel_for(range_pol(0, colWorkLen), refreshCol);
      }
      //Decide row statuses where enough information is available
      {
        DecideSetFunctor decideSet(rowStatus, colStatus, rowmap, entries, numVerts, rowWorklist, rowWorkLen);
        if(useTeams)
          Kokkos::parallel_for(team_pol((rowWorkLen + decideSetTeamSize - 1) / decideSetTeamSize, decideSetTeamSize, vectorLength), decideSet);
        else
          Kokkos::parallel_for(range_pol(0, rowWorkLen), decideSet);
      }
      round++;
      //Compact row worklist
      Kokkos::parallel_scan(range_pol(0, rowWorkLen), CompactWorklistFunctor(rowWorklist, thirdWorklist, rowStatus), rowWorkLen);
      if(rowWorkLen == 0)
        break;
      std::swap(rowWorklist, thirdWorklist);
      //Compact col worklist
      Kokkos::parallel_scan(range_pol(0, colWorkLen), CompactWorklistFunctor(colWorklist, thirdWorklist, colStatus), colWorkLen);
      std::swap(colWorklist, thirdWorklist);
    }
    //now that every vertex has been decided IN_SET/OUT_SET,
    //build a compact list of the vertices which are IN_SET.
    lno_t numInSet = 0;
    Kokkos::parallel_reduce(range_pol(0, numVerts), CountInSet(rowStatus), numInSet);
    lno_view_t setList(Kokkos::ViewAllocateWithoutInitializing("D2MIS"), numInSet);
    Kokkos::parallel_scan(range_pol(0, numVerts), CompactInSet(rowStatus, setList));
    return setList;
  }

  rowmap_t rowmap;
  entries_t entries;
  lno_t numVerts;
  status_view_t rowStatus;
  status_view_t colStatus;
  all_worklists_t allWorklists;
  //The number of bits required to represent vertex IDs, in the ECL-MIS tiebreak scheme:
  //  ceil(log_2(numVerts + 1))
  int nvBits;
};

//    UNUSED CODE
//    Version of RefreshRowStatus, which does linear interpolation between a degree-based score and a random score.
//    By gradually increasing the interpolation coefficient in favor of random, the MIS can converge much faster than
//    constant priorities.
//
//    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
//    {
//      lno_t i = worklist(w);
//      int degBits = sizeof(status_t) * 8 - nvBits;
//      if(degBits == 0)
//      {
//        //no space to store degree information. Algorithm will still work but will
//        //probably produce a lower quality MIS.
//        rowStatus(i) = i + 1;
//        return;
//      }
//      //Combine vertex and round to get some pseudorandom priority bits that change each round
//      status_t maxDegRange = (((status_t) 1) << degBits) - 2;
//      lno_t deg = rowmap(i + 1) - rowmap(i);
//      //Compute degree-based score and random score
//      float degScore = (float) (deg - minDeg) * invDegRange;
//      float randScore = (xorshiftHash(i + hashedRound) & 0xFFFF) / 65536.f;
//      //Then linearly interpolate using k
//      float finalScore = k * randScore + (1.f - k) * degScore;
//      rowStatus(i) = (status_t) (i + 1) + (((status_t) (finalScore * maxDegRange)) << nvBits);
//    }
//    */

template<typename device_t, typename rowmap_t, typename entries_t, typename lno_view_t>
struct D2_MIS_FixedPriority
{
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;
  using bitset_t = Kokkos::Bitset<device_t>;
  using const_bitset_t = Kokkos::ConstBitset<device_t>;
  using size_type = typename rowmap_t::non_const_value_type;
  using lno_t = typename entries_t::non_const_value_type;
  //The type of status/priority values.
  using status_t = typename std::make_unsigned<lno_t>::type;
  using status_view_t = Kokkos::View<status_t*, mem_space>;
  using range_pol = Kokkos::RangePolicy<exec_space>;

  // Priority values 0 and max are special, they mean the vertex is
  // in the independent set or eliminated from consideration, respectively.
  // Values in between represent a priority for being added to the set,
  // based on degree and vertex ID as a tiebreak
  //   (higher priority = less preferred to being in the independent set)

  static constexpr status_t IN_SET = 0;
  static constexpr status_t OUT_SET = ~IN_SET;

  D2_MIS_FixedPriority(const rowmap_t& rowmap_, const entries_t& entries_)
    : rowmap(rowmap_), entries(entries_), numVerts(rowmap.extent(0) - 1), colUpdateBitset(numVerts),
    worklist1(Kokkos::ViewAllocateWithoutInitializing("WL1"), numVerts),
    worklist2(Kokkos::ViewAllocateWithoutInitializing("WL2"), numVerts)
  {
    status_t i = numVerts + 1;
    nvBits = 0;
    while(i)
    {
      i >>= 1;
      nvBits++;
    }
    //Each value in rowStatus represents the status and priority of each row.
    //Each value in colStatus represents the lowest nonzero priority of any row adjacent to the column.
    //  This counts up monotonically as vertices are eliminated (given status OUT_SET)
    rowStatus = status_view_t(Kokkos::ViewAllocateWithoutInitializing("RowStatus"), numVerts);
    colStatus = status_view_t(Kokkos::ViewAllocateWithoutInitializing("ColStatus"), numVerts);
    KokkosKernels::Impl::graph_min_max_degree<device_t, lno_t, rowmap_t>(rowmap, minDegree, maxDegree);
    //Compute row statuses 
    Kokkos::parallel_for(range_pol(0, numVerts), InitRowStatus(rowStatus, rowmap, numVerts, nvBits, minDegree, maxDegree));
    //Compute col statuses
    Kokkos::parallel_for(range_pol(0, numVerts), InitColStatus(colStatus, rowStatus, rowmap, entries, numVerts));
  }

  struct InitRowStatus
  {
    InitRowStatus(const status_view_t& rowStatus_, const rowmap_t& rowmap_, lno_t nv_, lno_t nvBits_, lno_t minDeg_, lno_t maxDeg_)
      : rowStatus(rowStatus_), rowmap(rowmap_), nv(nv_), nvBits(nvBits_), minDeg(minDeg_), maxDeg(maxDeg_), invDegRange(1.f / (maxDeg - minDeg)) {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i) const
    {
      //Generate unique status per row, with IN_SET < status < OUT_SET,
      int degBits = sizeof(status_t) * 8 - nvBits;
      if(degBits == 0)
      {
        //no space to store degree information. Algorithm will still work but will
        //probably produce a lower quality MIS.
        rowStatus(i) = i + 1;
        return;
      }
      status_t maxDegRange = (((status_t) 1) << degBits) - 2;
      lno_t deg = rowmap(i + 1) - rowmap(i);
      float degScore = (float) (deg - minDeg) * invDegRange;
      rowStatus(i) = (status_t) (i + 1) + (((status_t) (degScore * maxDegRange)) << nvBits);
    }

    status_view_t rowStatus;
    rowmap_t rowmap;
    lno_t nv;
    int nvBits;
    lno_t minDeg;
    lno_t maxDeg;
    float invDegRange;
  };

  struct InitColStatus
  {
    InitColStatus(const status_view_t& colStatus_, const status_view_t& rowStatus_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_)
      : colStatus(colStatus_), rowStatus(rowStatus_), rowmap(rowmap_), entries(entries_), nv(nv_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i) const
    {
      //iterate over {i} union the neighbors of i, to find
      //minimum status.
      status_t s = rowStatus(i);
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      for(size_type j = rowBegin; j < rowEnd; j++)
      {
        lno_t nei = entries(j);
        if(nei != i && nei < nv)
        {
          status_t neiStat = rowStatus(nei);
          if(neiStat < s)
            s = neiStat;
        }
      }
      colStatus(i) = s;
    }

    status_view_t colStatus;
    status_view_t rowStatus;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
  };

  struct IterateStatusFunctor
  {
    IterateStatusFunctor(const status_view_t& rowStatus_, const status_view_t& colStatus_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_, const lno_view_t& worklist_, const bitset_t& colUpdateBitset_)
      : rowStatus(rowStatus_), colStatus(colStatus_), rowmap(rowmap_), entries(entries_), nv(nv_), worklist(worklist_), colUpdateBitset(colUpdateBitset_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      //Processing row i.
      status_t s = rowStatus(i);
      //s is the status which must be the minimum among all neighbors
      //to decide that i is IN_SET.
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      bool neiOut = false;
      bool neiMismatchS = false;
      for(size_type j = rowBegin; j <= rowEnd; j++)
      {
        lno_t nei = (j == rowEnd) ? i : entries(j);
        if(nei >= nv)
          continue;
        status_t neiStat = colStatus(nei);
        if(neiStat == OUT_SET)
        {
          neiOut = true;
          break;
        }
        else if(neiStat != s)
        {
          neiMismatchS = true;
        }
      }
      bool statusChanged = neiOut || !neiMismatchS;
      if(neiOut)
      {
        //In order to make future progress, need to update the
        //col statuses for all neighbors of i which have status s.
        //This will increase the minimum to the next smallest row,
        //so that another nearby vertex can be added to the set.
        rowStatus(i) = OUT_SET;
      }
      else if(!neiMismatchS)
      {
        rowStatus(i) = IN_SET;
      }
      if(statusChanged)
      {
        for(size_type j = rowBegin; j <= rowEnd; j++)
        {
          lno_t nei = (j == rowEnd) ? i : entries(j);
          if(nei < nv && colStatus(nei) == s)
            colUpdateBitset.set(nei);
        }
      }
      //else: still undecided
    }

    status_view_t rowStatus;
    status_view_t colStatus;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
    lno_view_t worklist;
    bitset_t colUpdateBitset;
  };

  struct UpdateWorklistFunctor
  {
    UpdateWorklistFunctor(const status_view_t& rowStatus_, const lno_view_t& oldWorklist_, const lno_view_t& newWorklist_)
      : rowStatus(rowStatus_), oldWorklist(oldWorklist_), newWorklist(newWorklist_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w, lno_t& lcount, bool finalPass) const
    {
      //processing row i
      lno_t i = oldWorklist(w);
      //Bit i will be set when it's decided IN_SET/OUT_SET.
      //If clear, vertex i needs to be processed still.
      status_t s = rowStatus(i);
      if(s != IN_SET && s != OUT_SET)
      {
        if(finalPass)
          newWorklist(lcount) = i;
        lcount++;
      }
    }

    status_view_t rowStatus;
    lno_view_t oldWorklist;
    lno_view_t newWorklist;
  };

  struct ColRefreshWorklist
  {
    ColRefreshWorklist(const bitset_t& colUpdateBitset_, const lno_view_t& refreshList_)
      : colUpdateBitset(colUpdateBitset_), refreshList(refreshList_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i, lno_t& lindex, bool finalPass) const
    {
      if(colUpdateBitset.test(i))
      {
        if(finalPass)
        {
          refreshList(lindex) = i;
          colUpdateBitset.reset(i);
        }
        lindex++;
      }
    }

    bitset_t colUpdateBitset;
    lno_view_t refreshList;
  };

  struct RefreshColStatus
  {
    RefreshColStatus(const lno_view_t& worklist_, const status_view_t& rowStatus_, const status_view_t& colStatus_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_)
      : worklist(worklist_), rowStatus(rowStatus_), colStatus(colStatus_), rowmap(rowmap_), entries(entries_), nv(nv_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t col = worklist(w);
      status_t minNeiStat = OUT_SET;
      size_type rowBegin = rowmap(col);
      size_type rowEnd = rowmap(col + 1);
      for(size_type j = rowBegin; j <= rowEnd; j++)
      {
        lno_t nei = (j == rowEnd) ? col : entries(j);
        if(nei >= nv)
          continue;
        status_t neiStat = rowStatus(nei);
        if(neiStat < minNeiStat)
          minNeiStat = neiStat;
      }
      if(minNeiStat == IN_SET)
        minNeiStat = OUT_SET;
      colStatus(col) = minNeiStat;
    }

    lno_view_t worklist;
    status_view_t rowStatus;
    status_view_t colStatus;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
  };

  struct CountInSet
  {
    CountInSet(const status_view_t& rowStatus_)
      : rowStatus(rowStatus_)
    {}
    KOKKOS_INLINE_FUNCTION void operator()(lno_t i, lno_t& lNumInSet) const
    {
      if(rowStatus(i) == IN_SET)
        lNumInSet++;
    }
    status_view_t rowStatus;
  };

  struct CompactInSet
  {
    CompactInSet(const status_view_t& rowStatus_, const lno_view_t& setList_)
      : rowStatus(rowStatus_), setList(setList_)
    {}
    KOKKOS_INLINE_FUNCTION void operator()(lno_t i, lno_t& lNumInSet, bool finalPass) const
    {
      if(rowStatus(i) == IN_SET)
      {
        if(finalPass)
          setList(lNumInSet) = i;
        lNumInSet++;
      }
    }
    status_view_t rowStatus;
    lno_view_t setList;
  };

  lno_view_t compute(int* numRounds)
  {
    //Initialize first worklist to 0...numVerts
    KokkosKernels::Impl::sequential_fill(worklist1);
    lno_t workRemain = numVerts;
    int numIter = 0;
    while(workRemain)
    {
      //do another iteration
      Kokkos::parallel_for(range_pol(0, workRemain),
          IterateStatusFunctor(rowStatus, colStatus, rowmap, entries, numVerts, worklist1, colUpdateBitset));
      //And refresh the column statuses using the other worklist.
      lno_t colsToRefresh;
      Kokkos::parallel_scan(range_pol(0, numVerts),
          ColRefreshWorklist(colUpdateBitset, worklist2), colsToRefresh);
      Kokkos::parallel_for(range_pol(0, colsToRefresh),
          RefreshColStatus(worklist2, rowStatus, colStatus, rowmap, entries, numVerts));
      //then build the next worklist with a scan. Also get the length of the next worklist.
      lno_t newWorkRemain = 0;
      Kokkos::parallel_scan(range_pol(0, workRemain),
          UpdateWorklistFunctor(rowStatus, worklist1, worklist2),
          newWorkRemain);
      //Finally, flip the worklists
      std::swap(worklist1, worklist2);
      workRemain = newWorkRemain;
      numIter++;
    }
    if(numRounds)
      *numRounds = numIter;
    //now that every vertex has been decided IN_SET/OUT_SET,
    //build a compact list of the vertices which are IN_SET.
    lno_t numInSet = 0;
    Kokkos::parallel_reduce(range_pol(0, numVerts), CountInSet(rowStatus), numInSet);
    lno_view_t setList(Kokkos::ViewAllocateWithoutInitializing("D2MIS"), numInSet);
    Kokkos::parallel_scan(range_pol(0, numVerts), CompactInSet(rowStatus, setList));
    return setList;
  }

  rowmap_t rowmap;
  entries_t entries;
  lno_t numVerts;
  status_view_t rowStatus;
  status_view_t colStatus;
  //The number of bits required to represent vertex IDs, in the ECL-MIS tiebreak scheme:
  //  ceil(log_2(numVerts + 1))
  int nvBits;
  lno_t minDegree;
  lno_t maxDegree;
  //Bitset representing columns whose status needs to be recomputed
  //These bits are cleared after each refresh.
  bitset_t colUpdateBitset;
  lno_view_t worklist1;
  lno_view_t worklist2;
};

template<typename device_t, typename rowmap_t, typename entries_t, typename labels_t>
struct D2_MIS_Coarsening
{
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;
  using bitset_t = Kokkos::Bitset<device_t>;
  using const_bitset_t = Kokkos::ConstBitset<device_t>;
  using size_type = typename rowmap_t::non_const_value_type;
  using lno_t = typename entries_t::non_const_value_type;
  using lno_view_t = typename entries_t::non_const_type;
  //The type of status/priority values.
  using status_t = typename std::make_unsigned<lno_t>::type;
  using status_view_t = Kokkos::View<status_t*, mem_space>;
  using range_pol = Kokkos::RangePolicy<exec_space>;

  D2_MIS_Coarsening(const rowmap_t& rowmap_, const entries_t& entries_, const labels_t& mis2_)
    : rowmap(rowmap_), entries(entries_), mis2(mis2_),
      numVerts(rowmap.extent(0) - 1),
      labels(Kokkos::ViewAllocateWithoutInitializing("Cluster Labels"), numVerts)
  {
    Kokkos::deep_copy(labels, (lno_t) -1);
  }

  //Phase 1 (over 0...numClusters) labels roots and immediate neighbors of roots.
  struct Phase1Functor
  {
    Phase1Functor(const rowmap_t& rowmap_, const entries_t& entries_, const labels_t& mis2_, lno_t numVerts_, const labels_t& labels_)
      : rowmap(rowmap_), entries(entries_), mis2(mis2_), numVerts(numVerts_), labels(labels_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i) const
    {
      lno_t root = mis2(i);
      size_type rowBegin = rowmap(root);
      size_type rowEnd = rowmap(root + 1);
      labels(root) = i;
      for(size_type j = rowBegin; j < rowEnd; j++)
      {
        lno_t nei = entries(j);
        if(nei != root && nei < numVerts)
        {
          labels(nei) = i;
        }
      }
    }

    rowmap_t rowmap;
    entries_t entries;
    labels_t mis2;
    lno_t numVerts;
    labels_t labels;
  };

  //Phase 2 (over 0...numVerts) joins unlabeled vertices to the smallest adjacent cluster
  struct Phase2Functor
  {
    Phase2Functor(const rowmap_t& rowmap_, const entries_t& entries_, const labels_t& mis2_, lno_t numVerts_, const labels_t& labels_)
      : rowmap(rowmap_), entries(entries_), mis2(mis2_), numVerts(numVerts_), labels(labels_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i) const
    {
      using unsigned_lno_t = typename std::make_unsigned<lno_t>::type;
      if(labels(i) != (lno_t) -1)
        return;
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      lno_t cluster = -1;
      uint32_t minScore = ~(uint32_t) 0;
      for(size_type j = rowBegin; j < rowEnd; j++)
      {
        lno_t nei = entries(j);
        if(nei == i || nei >= numVerts)
          continue;
        lno_t neiCluster = labels(nei);
        if(neiCluster != -1 && neiCluster != cluster)
        {
          //check if this cluster is smaller
          uint32_t score = KokkosKernels::Impl::xorshiftHash<unsigned_lno_t>(i + KokkosKernels::Impl::xorshiftHash<unsigned_lno_t>(neiCluster));
          if(score < minScore)
          {
            cluster = neiCluster;
            minScore = score;
          }
        }
      }
      labels(i) = cluster;
    }

    rowmap_t rowmap;
    entries_t entries;
    labels_t mis2;
    lno_t numVerts;
    labels_t labels;
  };

  labels_t compute()
  {
    lno_t numClusters = mis2.extent(0);
    Kokkos::parallel_for(range_pol(0, numClusters), Phase1Functor(rowmap, entries, mis2, numVerts, labels));
    Kokkos::parallel_for(range_pol(0, numVerts), Phase2Functor(rowmap, entries, mis2, numVerts, labels));
    return labels;
  }

  rowmap_t rowmap;
  entries_t entries;
  labels_t mis2;
  lno_t numVerts;
  labels_t labels;
};

template<typename device_t, typename rowmap_t, typename entries_t, typename labels_t>
struct D2_MIS_Aggregation
{
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;
  using bitset_t = Kokkos::Bitset<device_t>;
  using const_bitset_t = Kokkos::ConstBitset<device_t>;
  using size_type = typename rowmap_t::non_const_value_type;
  using lno_t = typename entries_t::non_const_value_type;
  using lno_view_t = typename entries_t::non_const_type;
  //The type of status/priority values.
  using status_t = typename std::make_unsigned<lno_t>::type;
  using status_view_t = Kokkos::View<status_t*, mem_space>;
  using range_pol = Kokkos::RangePolicy<exec_space>;

  D2_MIS_Aggregation(const rowmap_t& rowmap_, const entries_t& entries_)
    : rowmap(rowmap_), entries(entries_), numVerts(rowmap.extent(0) - 1),
      labels(Kokkos::ViewAllocateWithoutInitializing("AggregateLabels"), numVerts)
  {
    Kokkos::deep_copy(labels, (lno_t) -1);
  }

  void phase1()
  {
    //Compute an MIS-2
    D2_MIS_RandomPriority<device_t, rowmap_t, entries_t, labels_t, true> d2mis(rowmap, entries);
    auto mis2 = d2mis.compute(nullptr);
    auto rowmap_ = rowmap;
    auto entries_ = entries;
    auto numVerts_ = numVerts;
    auto labels_ = labels;
    //Construct initial aggregates using roots and all direct neighbors
    Kokkos::parallel_for(range_pol(0, mis2.extent(0)),
      KOKKOS_LAMBDA(lno_t agg)
      {
        lno_t root = mis2(agg);
        size_type rowBegin = rowmap_(root);
        size_type rowEnd = rowmap_(root + 1);
        labels_(root) = agg;
        for(size_type j = rowBegin; j < rowEnd; j++)
        {
          lno_t nei = entries_(j);
          if(nei < numVerts_)
            labels_(nei) = agg;
        }
      });
    numAggs = mis2.extent(0);
  }

  void phase2()
  {
    int iter = 0;
    auto rowmap_ = rowmap;
    auto entries_ = entries;
    auto numVerts_ = numVerts;
    auto labels_ = labels;
    labels_t candAggSizes(Kokkos::ViewAllocateWithoutInitializing("Phase2 Candidate Agg Sizes"), numVerts);
    auto numAggs_ = numAggs;
    while(true)
    {
      //Compute a new MIS-2 from only unaggregated nodes
      D2_MIS_RandomPriority<device_t, rowmap_t, entries_t, labels_t, true> d2mis(rowmap, entries);
      auto mis2 = d2mis.compute(labels);
      std::cout << "Iter " << iter++ << ": phase2 aggregation MIS contains " << mis2.extent(0) << " root candidates.\n";
      if(mis2.extent(0) == 0)
        break;
      lno_t numCandRoots = mis2.extent(0);
      //Compute the sizes of would-be aggregates.
      Kokkos::parallel_for(range_pol(0, numCandRoots),
        KOKKOS_LAMBDA(lno_t i)
        {
          lno_t candRoot = mis2(i);
          //Count the number of non-aggregated neighbors, including self
          lno_t aggSize = 1;
          size_type rowBegin = rowmap_(candRoot);
          size_type rowEnd = rowmap_(candRoot + 1);
          for(size_type j = rowBegin; j < rowEnd; j++)
          {
            lno_t nei = entries_(j);
            if(nei == candRoot || nei >= numVerts_)
              continue;
            if(labels_(nei) == -1)
              aggSize++;
          }
          candAggSizes(i) = aggSize;
        });
      //Now, filter out the candidate aggs which are big enough, and create those aggregates.
      //Using a scan for this assigns IDs deterministically (unlike an atomic counter).
      lno_t numNewAggs;
      Kokkos::parallel_scan(range_pol(0, numCandRoots),
        KOKKOS_LAMBDA(lno_t i, lno_t& lid, bool finalPass)
        {
          lno_t aggSize = candAggSizes(i);
          if(aggSize < 3)
            return;
          if(finalPass)
          {
            //Build the aggregate
            lno_t root = mis2(i);
            lno_t aggID = numAggs_ + lid;
            labels_(root) = aggID;
            size_type rowBegin = rowmap_(root);
            size_type rowEnd = rowmap_(root + 1);
            for(size_type j = rowBegin; j < rowEnd; j++)
            {
              lno_t nei = entries_(j);
              if(nei == root || nei >= numVerts_)
                continue;
              if(labels_(nei) == -1)
                labels_(nei) = aggID;
            }
          }
          lid++;
        }, numNewAggs);
      std::cout << "                           Of those, " << numNewAggs << " became new aggregates.\n";
      if(numNewAggs == 0)
        break;
      numAggs_ += numNewAggs;
    }
    numAggs = numAggs_;
  }

  void phase3()
  {
    //Phase3 is cleanup. All aggregates have already been created, but some vertices might be unaggregated.
    //Compute the current size of each aggregate, and then join each unaggregated node to the smallest neighboring aggregate.
    auto rowmap_ = rowmap;
    auto entries_ = entries;
    auto numVerts_ = numVerts;
    auto labels_ = labels;
    labels_t aggSizes(Kokkos::ViewAllocateWithoutInitializing("Phase3 Agg Sizes"), numAggs);
    Kokkos::parallel_for(range_pol(0, numVerts),
      KOKKOS_LAMBDA(lno_t i)
      {
        lno_t agg = labels_(i);
        if(agg != -1)
          Kokkos::atomic_increment(&aggSizes(agg));
      });
    //Now, join vertices to aggregates
    Kokkos::parallel_for(range_pol(0, numVerts),
      KOKKOS_LAMBDA(lno_t i)
      {
        lno_t agg = labels_(i);
        if(agg != -1)
          return;
        lno_t smallestAgg = 0;
        lno_t smallestAggSize = numVerts_ + 1;
        size_type rowBegin = rowmap_(i);
        size_type rowEnd = rowmap_(i + 1);
        for(size_type j = rowBegin; j < rowEnd; j++)
        {
          lno_t nei = entries_(j);
          if(nei == i || nei >= numVerts_)
            continue;
          lno_t neiAgg = labels_(nei);
          if(neiAgg == -1)
            continue;
          lno_t neiAggSize = aggSizes(neiAgg);
          if(neiAggSize < smallestAggSize)
          {
            smallestAgg = neiAgg;
            smallestAggSize = neiAggSize;
          }
        }
        labels_(i) = smallestAgg;
      });
  }

  void compute()
  {
    //Pseudocode:
    //
    //  -Phase 1: compute MIS-2, construct an aggregate from each in-set point and its neighbors
    //  -Phase 2: Until no new aggregates can be formed this way:
    //    -Compute an MIS-2 that excludes all aggregated nodes
    //    -For each in-set point:
    //      -Count unaggregated neighbors.
    //      -If total agg size would be >= 3, make the aggregate.
    //  -Phase 3: join still unaggregated nodes to a neighboring aggregate
    //    -Ideally, the smallest neighboring aggregate.
    //    -To remain deterministic, could simply use the agg sizes from end of phase 2 and not update them during phase 3.
    phase1();
    phase2();
    phase3();
  }

  rowmap_t rowmap;
  entries_t entries;
  labels_t mis2;
  lno_t numVerts;
  lno_t numAggs;
  labels_t labels;
};

template<typename device_t, typename rowmap_t, typename entries_t, typename labels_t>
struct D2_MIS_Bell
{
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;
  using bitset_t = Kokkos::Bitset<device_t>;
  using const_bitset_t = Kokkos::ConstBitset<device_t>;
  using size_type = typename rowmap_t::non_const_value_type;
  using lno_t = typename entries_t::non_const_value_type;
  //The type of status/priority values.
  using status_t = typename std::make_unsigned<lno_t>::type;
  using status_view_t = Kokkos::View<status_t*, mem_space>;
  using range_pol = Kokkos::RangePolicy<exec_space>;
  using pool_t = Kokkos::Random_XorShift64_Pool<exec_space>;
  using gen_t = typename pool_t::generator_type;

  using lno_view_t = Kokkos::View<lno_t*, mem_space>;
  using char_view = Kokkos::View<int8_t*, mem_space>;
  using unsigned_view = Kokkos::View<uint32_t*, mem_space>;

  labels_t compute(const rowmap_t& rowmap, const entries_t& entries, int* numRounds)
  {
    lno_t numVerts = rowmap.extent(0) - 1;
    char_view state("is membership", numVerts);
    lno_t unassigned_total = numVerts;
    unsigned_view randoms(Kokkos::ViewAllocateWithoutInitializing("randomized"), numVerts);
    pool_t rand_pool(rand());
    Kokkos::parallel_for("create random entries", range_pol(0, numVerts),
    KOKKOS_LAMBDA(lno_t i)
    {
      gen_t generator = rand_pool.get_state();
      randoms(i) = generator.urand();
      rand_pool.free_state(generator);
    });
    char_view tuple_state(Kokkos::ViewAllocateWithoutInitializing("tuple state"), numVerts);
    unsigned_view tuple_rand(Kokkos::ViewAllocateWithoutInitializing("tuple rand"), numVerts);
    char_view tuple_state_update(Kokkos::ViewAllocateWithoutInitializing("tuple state"), numVerts);
    unsigned_view tuple_rand_update(Kokkos::ViewAllocateWithoutInitializing("tuple rand"), numVerts);
    lno_view_t tuple_idx(Kokkos::ViewAllocateWithoutInitializing("tuple index"), numVerts);
    lno_view_t tuple_idx_update(Kokkos::ViewAllocateWithoutInitializing("tuple index"), numVerts);
    int iter = 0;
    while (unassigned_total > 0)
    {
      Kokkos::parallel_for(range_pol(0, numVerts),
      KOKKOS_LAMBDA(lno_t i)
      {
        tuple_state(i) = state(i);
        tuple_rand(i) = randoms(i);
        tuple_idx(i) = i;
      });
      for (int k = 0; k < 2; k++) {
        Kokkos::parallel_for(range_pol(0, numVerts),
        KOKKOS_LAMBDA(lno_t i)
        {
          int max_state = tuple_state(i);
          uint32_t max_rand = tuple_rand(i);
          lno_t max_idx = tuple_idx(i);
          for (size_type j = rowmap(i); j < rowmap(i + 1); j++) {
              lno_t v = entries(j);
              bool is_max = false;
              if (tuple_state(v) > max_state) {
                  is_max = true;
              }
              else if (tuple_state(v) == max_state) {
                if (tuple_rand(v) > max_rand) {
                  is_max = true;
                }
                else if (tuple_rand(v) == max_rand) {
                  if (tuple_idx(v) > max_idx) {
                      is_max = true;
                  }
                }
              }
              if (is_max) {
                max_state = tuple_state(v);
                max_rand = tuple_rand(v);
                max_idx = tuple_idx(v);
              }
          }
          tuple_state_update(i) = max_state;
          tuple_rand_update(i) = max_rand;
          tuple_idx_update(i) = max_idx;
        });
        Kokkos::parallel_for(range_pol(0, numVerts),
        KOKKOS_LAMBDA(const lno_t i){
          tuple_state(i) = tuple_state_update(i);
          tuple_rand(i) = tuple_rand_update(i);
          tuple_idx(i) = tuple_idx_update(i);
        });
      }
      Kokkos::parallel_reduce(range_pol(0, numVerts),
      KOKKOS_LAMBDA(const lno_t i, lno_t& thread_sum){
        if (state(i) == 0) {
          if (tuple_idx(i) == i) {
            //vertex i has max status in neighborhood so is in set
            state(i) = 1;
          }
          else if(tuple_state(i) == 1) {
            //vertex i is out of set (within neighborhood of another)
            state(i) = -1;
          }
        }
        if (state(i) == 0) {
          //Vertex i is still undecided
          thread_sum++;
        }
      }, unassigned_total);
      iter++;
    }
    if(numRounds)
      *numRounds = iter;
    lno_t numInSet = 0;
    Kokkos::parallel_reduce(range_pol(0, numVerts),
    KOKKOS_LAMBDA(lno_t i, lno_t& lcount)
    {
      if(state(i) == 1)
        lcount++;
    }, numInSet);
    labels_t setList(Kokkos::ViewAllocateWithoutInitializing("D2MIS"), numInSet);
    Kokkos::parallel_scan(range_pol(0, numVerts),
    KOKKOS_LAMBDA(lno_t i, lno_t& lcount, bool finalPass)
    {
      if(state(i) == 1)
      {
        if(finalPass)
          setList(lcount) = i;
        lcount++;
      }
    });
    return setList;
  }
};

template<typename device_t, typename rowmap_t, typename entries_t, typename labels_t>
struct D2_MIS_Randomized
{
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;
  using bitset_t = Kokkos::Bitset<device_t>;
  using const_bitset_t = Kokkos::ConstBitset<device_t>;
  using size_type = typename rowmap_t::non_const_value_type;
  using lno_t = typename entries_t::non_const_value_type;
  //The type of status/priority values.
  using status_t = typename std::make_unsigned<lno_t>::type;
  using status_view_t = Kokkos::View<status_t*, mem_space>;
  using range_pol = Kokkos::RangePolicy<exec_space>;
  using pool_t = Kokkos::Random_XorShift64_Pool<exec_space>;
  using gen_t = typename pool_t::generator_type;

  using hash_t = typename std::make_unsigned<lno_t>::type;
  using hash_view_t = Kokkos::View<hash_t*, mem_space>;
  using unsigned_view = Kokkos::View<uint32_t*, mem_space>;
  using lno_view_t = Kokkos::View<lno_t*, mem_space>;
  using char_view = Kokkos::View<int8_t*, mem_space>;

  labels_t compute(const rowmap_t& rowmap, const entries_t& entries, int* numRounds)
  {
    lno_t numVerts = rowmap.extent(0) - 1;
    char_view state("is membership", numVerts);
    lno_t unassigned_total = numVerts;
    char_view tuple_state(Kokkos::ViewAllocateWithoutInitializing("tuple state"), numVerts);
    unsigned_view tuple_rand(Kokkos::ViewAllocateWithoutInitializing("tuple rand"), numVerts);
    char_view tuple_state_update(Kokkos::ViewAllocateWithoutInitializing("tuple state"), numVerts);
    unsigned_view tuple_rand_update(Kokkos::ViewAllocateWithoutInitializing("tuple rand"), numVerts);
    lno_view_t tuple_idx(Kokkos::ViewAllocateWithoutInitializing("tuple index"), numVerts);
    lno_view_t tuple_idx_update(Kokkos::ViewAllocateWithoutInitializing("tuple index"), numVerts);
    hash_t round = 0;
    pool_t rand_pool(rand());
    while (unassigned_total > 0)
    {
    //  hash_t hashedRound = KokkosKernels::Impl::xorshiftHash(round);
      Kokkos::parallel_for(range_pol(0, numVerts),
      KOKKOS_LAMBDA(lno_t i)
      {
        tuple_state(i) = state(i);
        gen_t generator = rand_pool.get_state();
        tuple_rand(i) = generator.urand();
        rand_pool.free_state(generator);
        tuple_idx(i) = i;
      });
      for (int k = 0; k < 2; k++) {
        Kokkos::parallel_for(range_pol(0, numVerts),
        KOKKOS_LAMBDA(lno_t i)
        {
          int max_state = tuple_state(i);
          uint32_t max_rand = tuple_rand(i);
          lno_t max_idx = tuple_idx(i);
          for (size_type j = rowmap(i); j < rowmap(i + 1); j++) {
              lno_t v = entries(j);
              bool is_max = false;
              if (tuple_state(v) > max_state) {
                  is_max = true;
              }
              else if (tuple_state(v) == max_state) {
                if (tuple_rand(v) > max_rand) {
                  is_max = true;
                }
                else if (tuple_rand(v) == max_rand) {
                  if (tuple_idx(v) > max_idx) {
                      is_max = true;
                  }
                }
              }
              if (is_max) {
                max_state = tuple_state(v);
                max_rand = tuple_rand(v);
                max_idx = tuple_idx(v);
              }
          }
          tuple_state_update(i) = max_state;
          tuple_rand_update(i) = max_rand;
          tuple_idx_update(i) = max_idx;
        });
      Kokkos::parallel_for(range_pol(0, numVerts),
      KOKKOS_LAMBDA(const lno_t i){
        tuple_state(i) = tuple_state_update(i);
        tuple_rand(i) = tuple_rand_update(i);
        tuple_idx(i) = tuple_idx_update(i);
      });
    }
      Kokkos::parallel_reduce(range_pol(0, numVerts),
      KOKKOS_LAMBDA(const lno_t i, lno_t& thread_sum){
        if (state(i) == 0) {
          if (tuple_idx(i) == i) {
            //vertex i has max status in neighborhood so is in set
            state(i) = 1;
          }
          else if(tuple_state(i) == 1) {
            //vertex i is out of set (within neighborhood of another)
            state(i) = -1;
          }
        }
        if (state(i) == 0) {
          //Vertex i is still undecided
          thread_sum++;
        }
      }, unassigned_total);
      round++;
    }
    if(numRounds)
      *numRounds = round;
    lno_t numInSet = 0;
    Kokkos::parallel_reduce(range_pol(0, numVerts),
    KOKKOS_LAMBDA(lno_t i, lno_t& lcount)
    {
      if(state(i) == 1)
        lcount++;
    }, numInSet);
    labels_t setList(Kokkos::ViewAllocateWithoutInitializing("D2MIS"), numInSet);
    Kokkos::parallel_scan(range_pol(0, numVerts),
    KOKKOS_LAMBDA(lno_t i, lno_t& lcount, bool finalPass)
    {
      if(state(i) == 1)
      {
        if(finalPass)
          setList(lcount) = i;
        lcount++;
      }
    });
    return setList;
  }
};

template<typename device_t, typename rowmap_t, typename entries_t, typename labels_t>
struct D2_MIS_Worklist
{
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;
  using bitset_t = Kokkos::Bitset<device_t>;
  using const_bitset_t = Kokkos::ConstBitset<device_t>;
  using size_type = typename rowmap_t::non_const_value_type;
  using lno_t = typename entries_t::non_const_value_type;
  //The type of status/priority values.
  using status_t = typename std::make_unsigned<lno_t>::type;
  using status_view_t = Kokkos::View<status_t*, mem_space>;
  using range_pol = Kokkos::RangePolicy<exec_space>;

  using hash_t = typename std::make_unsigned<lno_t>::type;
  using hash_view_t = Kokkos::View<hash_t*, mem_space>;
  using lno_view_t = Kokkos::View<lno_t*, mem_space>;
  using char_view = Kokkos::View<int8_t*, mem_space>;
  using all_worklists_t = Kokkos::View<lno_t**, Kokkos::LayoutLeft, mem_space>;

  labels_t compute(const rowmap_t& rowmap, const entries_t& entries, int* numRounds)
  {
    lno_t numVerts = rowmap.extent(0) - 1;
    char_view rowStatus("is membership", numVerts);
    char_view colStatus(Kokkos::ViewAllocateWithoutInitializing("tuple state"), numVerts);
    hash_view_t colHash(Kokkos::ViewAllocateWithoutInitializing("tuple rand"), numVerts);
    lno_view_t colIdx(Kokkos::ViewAllocateWithoutInitializing("tuple index"), numVerts);
    char_view tempStatus(Kokkos::ViewAllocateWithoutInitializing("tuple state"), numVerts);
    hash_view_t tempHash(Kokkos::ViewAllocateWithoutInitializing("tuple rand"), numVerts);
    lno_view_t tempIdx(Kokkos::ViewAllocateWithoutInitializing("tuple index"), numVerts);
    all_worklists_t allWorklists(Kokkos::ViewAllocateWithoutInitializing("All worklists"), numVerts, 3);
    lno_view_t rowWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 0);
    lno_view_t colWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 1);
    lno_view_t tempWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 2);
    KokkosKernels::Impl::sequential_fill(rowWorklist);
    KokkosKernels::Impl::sequential_fill(colWorklist);
    lno_t rowWorkLen = numVerts;
    lno_t colWorkLen = numVerts;
    hash_t round = 0;
    while (rowWorkLen)
    {
      hash_t hashedRound = KokkosKernels::Impl::xorshiftHash(round);
      //Step 1: gather max priority to each column
      Kokkos::parallel_for(range_pol(0, colWorkLen),
      KOKKOS_LAMBDA(lno_t w)
      {
        lno_t i = colWorklist(w);
        int max_state = rowStatus(i);
        hash_t max_rand = KokkosKernels::Impl::xorshiftHash((hash_t) i + hashedRound);
        lno_t max_idx = i;
        for (size_type j = rowmap(i); j < rowmap(i + 1); j++) {
            lno_t v = entries(j);
            hash_t vhash = KokkosKernels::Impl::xorshiftHash((hash_t) v + hashedRound);
            bool is_max = false;
            if (rowStatus(v) > max_state) {
                is_max = true;
            }
            else if (rowStatus(v) == max_state) {
              if (vhash > max_rand) {
                is_max = true;
              }
              else if (vhash == max_rand) {
                if (v > max_idx) {
                    is_max = true;
                }
              }
            }
            if (is_max) {
              max_state = rowStatus(v);
              max_rand = vhash;
              max_idx = v;
            }
        }
        colStatus(i) = max_state;
        colHash(i) = max_rand;
        colIdx(i) = max_idx;
      });
      //now tuple_*_update has the column max (distance-1 neighborhoods)
      Kokkos::parallel_for(range_pol(0, rowWorkLen),
      KOKKOS_LAMBDA(lno_t w)
      {
        lno_t i = rowWorklist(w);
        int max_state = colStatus(i);
        uint32_t max_rand = colHash(i);
        lno_t max_idx = colIdx(i);
        for (size_type j = rowmap(i); j < rowmap(i + 1); j++) {
            lno_t v = entries(j);
            bool is_max = false;
            if (colStatus(v) > max_state) {
                is_max = true;
            }
            else if (colStatus(v) == max_state) {
              if (colHash(v) > max_rand) {
                is_max = true;
              }
              else if (colHash(v) == max_rand) {
                if (colIdx(v) > max_idx) {
                    is_max = true;
                }
              }
            }
            if (is_max) {
              max_state = colStatus(v);
              max_rand = colHash(v);
              max_idx = colIdx(v);
            }
        }
        tempStatus(i) = max_state;
        tempHash(i) = max_rand;
        tempIdx(i) = max_idx;
      });
      //Select vertices which have the max status in their neighborhood, by comparing their current status views with temp*
      Kokkos::parallel_for(range_pol(0, rowWorkLen),
      KOKKOS_LAMBDA(const lno_t w) {
        lno_t i = rowWorklist(w);
        if (tempIdx(i) == i) {
          //vertex i has max status in neighborhood so is in set
          rowStatus(i) = 1;
        }
        else if(tempStatus(i) == 1) {
          //vertex i is out of set (within neighborhood of another that was selected to be in set)
          rowStatus(i) = -1;
        }
      });
      //now that row statuses have been updated, compact both worklists (tuple_state_update has the col statuses, and rowStatus has the rows)
      Kokkos::parallel_scan(range_pol(0, rowWorkLen),
      KOKKOS_LAMBDA(const lno_t w, lno_t& lindex, bool finalPass) {
        lno_t i = rowWorklist(w);
        if(rowStatus(i) == 0)
        {
          if(finalPass)
            tempWorklist(lindex) = i;
          lindex++;
        }
      }, rowWorkLen);
      std::swap(rowWorklist, tempWorklist);
      Kokkos::parallel_scan(range_pol(0, colWorkLen),
      KOKKOS_LAMBDA(const lno_t w, lno_t& lindex, bool finalPass) {
        lno_t i = colWorklist(w);
        if(colStatus(i) == 0)
        {
          if(finalPass)
            tempWorklist(lindex) = i;
          lindex++;
        }
      }, colWorkLen);
      std::swap(colWorklist, tempWorklist);
      round++;
    }
    if(numRounds)
      *numRounds = round;
    lno_t numInSet = 0;
    Kokkos::parallel_reduce(range_pol(0, numVerts),
    KOKKOS_LAMBDA(lno_t i, lno_t& lcount)
    {
      if(rowStatus(i) == 1)
        lcount++;
    }, numInSet);
    labels_t setList(Kokkos::ViewAllocateWithoutInitializing("D2MIS"), numInSet);
    Kokkos::parallel_scan(range_pol(0, numVerts),
    KOKKOS_LAMBDA(lno_t i, lno_t& lcount, bool finalPass)
    {
      if(rowStatus(i) == 1)
      {
        if(finalPass)
          setList(lcount) = i;
        lcount++;
      }
    });
    return setList;
  }
};

}}}

#endif
