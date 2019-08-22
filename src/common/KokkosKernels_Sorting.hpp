/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#ifndef _KOKKOSKERNELS_SORTING_HPP
#define _KOKKOSKERNELS_SORTING_HPP
#include "Kokkos_Core.hpp"
#include "Kokkos_Atomic.hpp"
#include "Kokkos_ArithTraits.hpp"
#include "impl/Kokkos_Timer.hpp"
#include <type_traits>

namespace KokkosKernels {
namespace Impl {

//Radix sort for nonnegative integers, on a single thread within a team.
//Pros: few diverging branches, so good for sorting on a single GPU thread/warp.
//Con: requires auxiliary storage, this version only works for integers (although float/double is possible)
template<typename Ordinal, typename ValueType, typename Thread>
KOKKOS_INLINE_FUNCTION static void
radixSortThread(ValueType* values, ValueType* valuesAux, Ordinal n, const Thread& thread)
{
  static_assert(std::is_integral<ValueType>::value, "radixSort can only be run on integers.");
  if(n <= 1)
    return;
  //sort 4 bits at a time, into 16 buckets
  ValueType mask = 0xF;
  //Is the data currently held in values (false) or valuesAux (true)?
  bool inAux = false;
  //maskPos counts the low bit index of mask (0, 4, 8, ...)
  Ordinal maskPos = 0;
  Ordinal sortBits = 0;
  ValueType minVal = Kokkos::ArithTraits<ValueType>::max();
  ValueType maxVal = 0;
  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(thread, n),
  [=](Ordinal i, ValueType& lminval)
  {
    if(values[i] < lminval)
      lminval = values[i];
  }, Kokkos::Min<ValueType>(minVal));
  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(thread, n),
  [=](Ordinal i, ValueType& lmaxval)
  {
    if(values[i] > lmaxkey)
      lmaxkey = values[i];
  }, Kokkos::Max<ValueType>(maxVal));
  //apply a bias so that key range always starts at 0
  //also invert key values here for a descending sort
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, n),
  [=](Ordinal i)
  {
    values[i] -= minVal;
  });
  Kokkos::single(Kokkos::PerThread(thread),
  [=]()
  {
    ValueType upperBound = maxVal - minVal;
    while(upperBound)
    {
      upperBound >>= 1;
      sortBits++;
    }
    for(Ordinal s = 0; s < (sortBits + 3) / 4; s++)
    {
      //Count the number of elements in each bucket
      Ordinal count[16] = {0};
      Ordinal offset[17];
      if(!inAux)
      {
        for(Ordinal i = 0; i < n; i++)
        {
          count[(values[i] & mask) >> maskPos]++;
        }
      }
      else
      {
        for(Ordinal i = 0; i < n; i++)
        {
          count[(valuesAux[i] & mask) >> maskPos]++;
        }
      }
      offset[0] = 0;
      //get offset as the prefix sum for count
      for(Ordinal i = 0; i < 16; i++)
      {
        offset[i + 1] = offset[i] + count[i];
      }
      //now for each element in [lo, hi), move it to its offset in the other buffer
      //this branch should be ok because whichBuf is the same on all threads
      if(!inAux)
      {
        //copy from *Over to *Aux
        for(Ordinal i = 0; i < n; i++)
        {
          Ordinal bucket = (values[i] & mask) >> maskPos;
          valuesAux[offset[bucket + 1] - count[bucket]] = values[i];
          count[bucket]--;
        }
      }
      else
      {
        //copy from *Aux to *Over
        for(Ordinal i = 0; i < n; i++)
        {
          Ordinal bucket = (valuesAux[i] & mask) >> maskPos;
          values[offset[bucket + 1] - count[bucket]] = valuesAux[i];
          count[bucket]--;
        }
      }
      inAux = !inAux;
      mask = mask << 4;
      maskPos += 4;
    }
  });
  //move values back into main array if they are currently in aux
  if(inAux)
  {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, n),
    [=](Ordinal i)
    {
      values[i] = valuesAux[i];
    });
  }
}

//Bitonic merge sort (requires only comparison operators and trivially-copyable)
//In-place, plenty of parallelism for GPUs, and memory references are coalesced
//Good diagram of the algorithm at https://en.wikipedia.org/wiki/Bitonic_sorter
template<typename Ordinal, typename ValueType, typename TeamMember>
KOKKOS_INLINE_FUNCTION static void
bitonicSortTeam(ValueType* values, Ordinal n, const TeamMember& mem)
{
  //Algorithm only works on power-of-two input size only.
  //If n is not a power-of-two, will implicitly pretend
  //that values[i] for i >= n is just the max for ValueType, so it never gets swapped
  Ordinal npot = 1;
  Ordinal levels = 0;
  while(npot < n)
  {
    levels++;
    npot <<= 1;
  }
  for(Ordinal i = 0; i < levels; i++)
  {
    for(Ordinal j = 0; j <= i; j++)
    {
      // n/2 pairs of items are compared in parallel
      Kokkos::parallel_for(Kokkos::ThreadTeamRange(mem, npot / 2),
        [=](const Ordinal t)
        {
          //How big are the brown/pink boxes?
          Ordinal boxSize = Ordinal(2) << (i - j);
          //Which box contains this thread?
          Ordinal boxID = t >> (1 + i - j);         //t * 2 / boxSize;
          Ordinal boxStart = boxID << (1 + i - j);  //boxID * boxSize
          Ordinal boxOffset = t - (boxStart << 1);  //t - boxID * boxSize / 2;
          Ordinal elem1 = boxStart + boxOffset;
          if(j == 0)
          {
            //first phase (brown box): within a block, compare with the opposite value in the box
            Ordinal elem2 = boxStart + boxSize - 1 - boxOffset;
            if(elem2 < n)
            {
              //both elements in bounds, so compare them and swap if out of order
              if(values[elem1] >= values[elem2])
              {
                ValueType temp = values[elem1];
                values[elem1] = values[elem2];
                values[elem2] = temp;
              }
            }
          }
          else
          {
            //later phases (pink box): within a block, compare with fixed distance (boxSize / 2) apart
            Ordinal elem2 = elem1 + boxSize / 2;
            if(elem2 < n)
            {
              if(values[elem1] >= values[elem2])
              {
                ValueType temp = values[elem1];
                values[elem1] = values[elem2];
                values[elem2] = temp;
              }
            }
          }
        });
      mem.team_barrier();
    }
  }
}

#endif

