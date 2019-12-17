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

/// \file Test_Common_Sorting.hpp
/// \brief Tests for radixSort and bitonicSort in KokkoKernels_Sorting.hpp

#ifndef KOKKOSKERNELS_SORTINGTEST_HPP
#define KOKKOSKERNELS_SORTINGTEST_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <KokkosKernels_SimpleUtils.hpp>
#include <KokkosKernels_Utils.hpp>
#include <KokkosKernels_Sorting.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <cstdlib>

//Generate n randomized counts with mean <avg>.
//Then prefix-sum into randomOffsets.
//This simulates a CRS rowmap or other batched sorting scenario
template<typename OrdView, typename ExecSpace>
size_t generateRandomOffsets(OrdView& randomCounts, OrdView& randomOffsets, size_t n, size_t avg)
{
  randomCounts = OrdView("Counts", n);
  randomOffsets = OrdView("Offsets", n);
  auto countsHost = Kokkos::create_mirror_view(randomCounts);
  size_t total = 0;
  for(size_t i = 0; i < n; i++)
  {
    countsHost(i) = 0.5 + rand() % (avg * 2);
    total += countsHost(i);
  }
  Kokkos::deep_copy(randomCounts, countsHost);
  Kokkos::deep_copy(randomOffsets, randomCounts);
  KokkosKernels::Impl::kk_exclusive_parallel_prefix_sum<OrdView, ExecSpace>(n, randomOffsets);
  return total;
}

struct Coordinates
{
  Coordinates()
  {
     x = 0;
     y = 0;
     z = 0;
  }
  Coordinates(double x_, double y_, double z_)
  {
     x = x_;
     y = y_;
     z = z_;
  }
  double x;
  double y;
  double z;
};

//Generate random integer, up to RAND_MAX
template<typename T>
T getRandom()
{
  return rand() % Kokkos::ArithTraits<T>::max();
}

//Generate a uniform double between (-5, 5)
template<>
double getRandom<double>()
{
  return -5 + (10.0 * rand()) / RAND_MAX;
}

template<>
Coordinates getRandom<Coordinates>()
{
  return Coordinates(getRandom<double>(), getRandom<double>(), getRandom<double>());
}

template<typename View>
void fillRandom(View v)
{
  srand(12345);
  typedef typename View::value_type Value;
  auto vhost = Kokkos::create_mirror_view(v);
  for(size_t i = 0; i < v.extent(0); i++)
    vhost(i) = getRandom<Value>();
  Kokkos::deep_copy(v, vhost);
}

template<typename ValView, typename OrdView>
struct TestSerialRadixFunctor
{
  typedef typename ValView::value_type Value;

  TestSerialRadixFunctor(ValView& values_, ValView& valuesAux_, OrdView& counts_, OrdView& offsets_)
    : values(values_), valuesAux(valuesAux_), counts(counts_), offsets(offsets_)
  {}
  template<typename TeamMem>
  KOKKOS_INLINE_FUNCTION void operator()(const TeamMem t) const
  {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, counts.extent(0)),
      [=](const int i)
      {
        KokkosKernels::Impl::SerialRadixSort<int, Value>(&values(offsets(i)), &valuesAux(offsets(i)), counts(i));
      });
  }
  ValView values;
  ValView valuesAux;
  OrdView counts;
  OrdView offsets;
};

template<typename ExecSpace, typename Scalar>
void testSerialRadixSort()
{
  //Create a view of randomized data
  typedef typename ExecSpace::memory_space mem_space;
  typedef Kokkos::View<int*, mem_space> OrdView;
  //typedef Kokkos::View<int*, Kokkos::HostSpace> OrdViewHost;
  typedef Kokkos::View<Scalar*, mem_space> ValView;
  OrdView counts;
  OrdView offsets;
  //Generate k sub-array sizes, each with size about 20
  size_t k = 100;
  size_t subSize = 20;
  size_t n = generateRandomOffsets<OrdView, ExecSpace>(counts, offsets, k, subSize);
  auto countsHost = Kokkos::create_mirror_view(counts);
  auto offsetsHost = Kokkos::create_mirror_view(offsets);
  Kokkos::deep_copy(countsHost, counts);
  Kokkos::deep_copy(offsetsHost, offsets);
  ValView data("Radix sort testing data", n);
  fillRandom(data);
  ValView dataAux("Radix sort aux data", n);
  //Run the sorting on device in all sub-arrays in parallel, just using vector loops
  typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
  Kokkos::parallel_for(team_policy(1, Kokkos::AUTO(), 32),
        TestSerialRadixFunctor<ValView, OrdView>(data, dataAux, counts, offsets));
  //Sort using std::sort on host to do correctness test
  Kokkos::View<Scalar*, Kokkos::HostSpace> gold("Host sorted", n);
  Kokkos::deep_copy(gold, data);
  for(size_t i = 0; i < k; i++)
  {
    Scalar* begin = &gold(offsetsHost(i));
    Scalar* end = begin + countsHost(i);
    std::sort(begin, end);
  }
  //Copy result to host
  auto dataHost = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(dataHost, data);
  for(size_t i = 0; i < n; i++)
  {
    ASSERT_EQ(dataHost(i), gold(i));
  }
}

template<typename ValView, typename OrdView>
struct TestTeamBitonicFunctor
{
  typedef typename ValView::value_type Value;

  TestTeamBitonicFunctor(ValView& values_, OrdView& counts_, OrdView& offsets_)
    : values(values_), counts(counts_), offsets(offsets_)
  {}

  template<typename TeamMem>
  KOKKOS_INLINE_FUNCTION void operator()(const TeamMem t) const
  {
    int i = t.league_rank();
    KokkosKernels::Impl::TeamBitonicSort<int, Value, TeamMem>(&values(offsets(i)), counts(i), t);
  }

  ValView values;
  OrdView counts;
  OrdView offsets;
};

template<typename ExecSpace, typename Scalar>
void testTeamBitonicSort()
{
  //Create a view of randomized data
  typedef typename ExecSpace::memory_space mem_space;
  typedef Kokkos::View<int*, mem_space> OrdView;
  //typedef Kokkos::View<int*, Kokkos::HostSpace> OrdViewHost;
  typedef Kokkos::View<Scalar*, mem_space> ValView;
  OrdView counts;
  OrdView offsets;
  //Generate k sub-array sizes, each with size about 20
  size_t k = 100;
  size_t subSize = 100;
  size_t n = generateRandomOffsets<OrdView, ExecSpace>(counts, offsets, k, subSize);
  auto countsHost = Kokkos::create_mirror_view(counts);
  auto offsetsHost = Kokkos::create_mirror_view(offsets);
  Kokkos::deep_copy(countsHost, counts);
  Kokkos::deep_copy(offsetsHost, offsets);
  ValView data("Bitonic sort testing data", n);
  fillRandom(data);
  //Run the sorting on device in all sub-arrays in parallel
  Kokkos::parallel_for(Kokkos::TeamPolicy<ExecSpace>(k, Kokkos::AUTO()),
      TestTeamBitonicFunctor<ValView, OrdView>(data, counts, offsets));
  //Copy result to host
  auto dataHost = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(dataHost, data);
  //Sort using std::sort on host to do correctness test
  Kokkos::View<Scalar*, Kokkos::HostSpace> gold("Host sorted", n);
  Kokkos::deep_copy(gold, data);
  for(size_t i = 0; i < k; i++)
  {
    Scalar* begin = &gold(offsetsHost(i));
    Scalar* end = begin + countsHost(i);
    std::sort(begin, end);
  }
  for(size_t i = 0; i < n; i++)
  {
    ASSERT_EQ(dataHost(i), gold(i));
  }
}

template<typename View>
struct CheckSortedFunctor
{
  CheckSortedFunctor(View& v_)
    : v(v_) {}
  KOKKOS_INLINE_FUNCTION void operator()(int i, int& lval) const
  {
    if(v(i) > v(i + 1))
      lval = 0;
  }
  View v;
};

template<typename ExecSpace, typename Scalar>
void testBitonicSort(size_t n)
{
  //Create a view of randomized data
  typedef typename ExecSpace::memory_space mem_space;
  typedef Kokkos::View<Scalar*, mem_space> ValView;
  ValView data("Bitonic sort testing data", n);
  fillRandom(data);
  KokkosKernels::Impl::bitonicSort<ValView, ExecSpace, int>(data);
  int ordered = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecSpace>(0, n - 1),
      CheckSortedFunctor<ValView>(data), Kokkos::Min<int>(ordered));
  ASSERT_TRUE(ordered);
}

//Check that the view is weakly ordered according to Comparator:
//Comparator never says that element i+1 belongs before element i.
template<typename View, typename Comparator>
struct CheckOrderedFunctor
{
  CheckOrderedFunctor(View& v_)
    : v(v_) {}
  KOKKOS_INLINE_FUNCTION void operator()(int i, int& lval) const
  {
    Comparator comp;
    if(comp(v(i + 1), v(i)))
      lval = 0;
  }
  View v;
};

template<typename Scalar>
struct CompareDescending
{
  KOKKOS_INLINE_FUNCTION bool operator()(const Scalar lhs, const Scalar rhs) const
  {
    return lhs > rhs;
  }
};

template<typename ExecSpace>
void testBitonicSortDescending()
{
  typedef char Scalar;
  typedef CompareDescending<Scalar> Comp;
  //Create a view of randomized data
  typedef typename ExecSpace::memory_space mem_space;
  typedef Kokkos::View<Scalar*, mem_space> ValView;
  size_t n = 12521;
  ValView data("Bitonic sort testing data", n);
  fillRandom(data);
  KokkosKernels::Impl::bitonicSort<ValView, ExecSpace, int, Comp>(data);
  int ordered = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecSpace>(0, n - 1),
      CheckOrderedFunctor<ValView, Comp>(data), Kokkos::Min<int>(ordered));
  ASSERT_TRUE(ordered);
}

struct LexCompare
{
  KOKKOS_INLINE_FUNCTION bool operator()(const Coordinates lhs, const Coordinates rhs) const
  {
    if(lhs.x < rhs.x)
      return true;
    else if(lhs.x > rhs.x)
      return false;
    else if(lhs.y < rhs.y)
      return true;
    else if(lhs.y > rhs.y)
      return false;
    else if(lhs.z < rhs.z)
      return true;
    return false;
  }
};

template<typename ExecSpace>
void testBitonicSortLexicographic()
{
  typedef Coordinates Scalar;
  //Create a view of randomized data
  typedef typename ExecSpace::memory_space mem_space;
  typedef Kokkos::View<Scalar*, mem_space> ValView;
  size_t n = 9521;
  ValView data("Bitonic sort testing data", n);
  fillRandom(data);
  KokkosKernels::Impl::bitonicSort<ValView, ExecSpace, int, LexCompare>(data);
  int ordered = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecSpace>(0, n - 1),
      CheckOrderedFunctor<ValView, LexCompare>(data), Kokkos::Min<int>(ordered));
  ASSERT_TRUE(ordered);
}

TEST_F( TestCategory, common_Sorting) {
  testSerialRadixSort<TestExecSpace, char>();
  testSerialRadixSort<TestExecSpace, int>();
  testTeamBitonicSort<TestExecSpace, int>();
  testTeamBitonicSort<TestExecSpace, double>();
  testBitonicSort<TestExecSpace, char>(215673);
  testBitonicSort<TestExecSpace, int>(92314);
  testBitonicSort<TestExecSpace, double>(60234);
  testBitonicSort<TestExecSpace, char>(424);
  testBitonicSort<TestExecSpace, int>(123);
  testBitonicSort<TestExecSpace, double>(53);
  testBitonicSortDescending<TestExecSpace>();
  testBitonicSortLexicographic<TestExecSpace>();
}

#endif

