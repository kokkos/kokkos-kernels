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

#include "KokkosKernels_Utils.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <Kokkos_Bitset.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include "KokkosGraph_Distance1Color.hpp"
#include "KokkosKernels_Uniform_Initialized_MemoryPool.hpp"
#include "KokkosKernels_BitUtils.hpp"
#include "KokkosKernels_SimpleUtils.hpp"
#include "KokkosSparse_partitioning_impl.hpp"

#ifndef _KOKKOSGSIMP_HPP
#define _KOKKOSGSIMP_HPP

namespace KokkosSparse{


  namespace Impl{


    template <typename HandleType, typename lno_row_view_t_, typename lno_nnz_view_t_, typename scalar_nnz_view_t_>
    class ClusterGaussSeidel{

    public:

      typedef lno_row_view_t_ in_lno_row_view_t;
      typedef lno_nnz_view_t_ in_lno_nnz_view_t;
      typedef scalar_nnz_view_t_ in_scalar_nnz_view_t;

      typedef typename HandleType::HandleExecSpace MyExecSpace;
      typedef typename HandleType::HandleTempMemorySpace MyTempMemorySpace;
      typedef typename HandleType::HandlePersistentMemorySpace MyPersistentMemorySpace;


      typedef typename in_lno_row_view_t::non_const_value_type row_lno_t;

      typedef typename HandleType::size_type size_type;
      typedef typename HandleType::nnz_lno_t nnz_lno_t;
      typedef typename HandleType::nnz_scalar_t nnz_scalar_t;


      typedef typename in_lno_row_view_t::const_type const_lno_row_view_t;
      typedef typename in_lno_row_view_t::non_const_type non_const_lno_row_view_t;

      typedef typename lno_nnz_view_t_::const_type const_lno_nnz_view_t;
      typedef typename lno_nnz_view_t_::non_const_type non_const_lno_nnz_view_t;

      typedef typename scalar_nnz_view_t_::const_type const_scalar_nnz_view_t;
      typedef typename scalar_nnz_view_t_::non_const_type non_const_scalar_nnz_view_t;




      typedef typename HandleType::row_lno_temp_work_view_t row_lno_temp_work_view_t;
      typedef typename HandleType::row_lno_persistent_work_view_t row_lno_persistent_work_view_t;
      typedef typename HandleType::row_lno_persistent_work_host_view_t row_lno_persistent_work_host_view_t; //Host view type



      typedef typename HandleType::nnz_lno_temp_work_view_t nnz_lno_temp_work_view_t;
      typedef typename HandleType::nnz_lno_persistent_work_view_t nnz_lno_persistent_work_view_t;
      typedef typename HandleType::nnz_lno_persistent_work_host_view_t nnz_lno_persistent_work_host_view_t; //Host view type


      typedef typename HandleType::scalar_temp_work_view_t scalar_temp_work_view_t;
      typedef typename HandleType::scalar_persistent_work_view_t scalar_persistent_work_view_t;

      typedef Kokkos::RangePolicy<MyExecSpace> my_exec_space;
      typedef nnz_lno_t color_t;
      typedef Kokkos::View<color_t *, MyTempMemorySpace> color_view_t;
      typedef Kokkos::Bitset<MyExecSpace> bitset_t;
      typedef Kokkos::ConstBitset<MyExecSpace> const_bitset_t;

      typedef Kokkos::TeamPolicy<MyExecSpace> team_policy_t ;
      typedef typename team_policy_t::member_type team_member_t ;

    private:
      HandleType *handle;
      nnz_lno_t num_rows, num_cols;

      const_lno_row_view_t row_map;
      const_lno_nnz_view_t entries;
      const_scalar_nnz_view_t values;

      const_scalar_nnz_view_t given_inverse_diagonal;

      bool have_diagonal_given;
      bool is_symmetric;

    public:
      struct PSGS
      {
        // CSR storage of the matrix.
        row_lno_persistent_work_view_t _xadj;
        nnz_lno_persistent_work_view_t _adj;     
        scalar_persistent_work_view_t _adj_vals;

        //Input/output vectors, as in Ax = y
        scalar_persistent_work_view_t _Xvector;
        scalar_persistent_work_view_t _Yvector;

        scalar_persistent_work_view_t _inverse_diagonal;

        nnz_lno_persistent_work_view_t _clusterOffsets;
        nnz_lno_persistent_work_view_t _clusterVerts;
        nnz_lno_persistent_work_view_t _cluster_color_adj;

        nnz_scalar_t omega;

        PSGS(row_lno_persistent_work_view_t xadj_, nnz_lno_persistent_work_view_t adj_, scalar_persistent_work_view_t adj_vals_,
             scalar_persistent_work_view_t Xvector_, scalar_persistent_work_view_t Yvector_, nnz_lno_persistent_work_view_t cluster_color_adj_,
             nnz_lno_persistent_work_view_t clusterOffsets_, nnz_lno_persistent_work_view_t clusterVerts_, 
             nnz_scalar_t omega_,
             scalar_persistent_work_view_t inverse_diagonal_):
          _xadj             (xadj_),
          _adj              (adj_),
          _adj_vals         (adj_vals_),
          _Xvector          (Xvector_),
          _Yvector          (Yvector_),
          _cluster_color_adj(cluster_color_adj_),
          _clusterOffsets   (clusterOffsets_),
          _clusterVerts     (clusterVerts_),
          _inverse_diagonal (inverse_diagonal_),
          omega             (omega_)
        {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const nnz_lno_t &ii) const {
          //ii is the index of a cluster within the color mapping
          nnz_lno_t cluster = _cluster_color_adj(ii);
          for(nnz_lno_t j = _clusterOffsets(cluster); j < _clusterOffsets(cluster + 1); j++)
          {
            nnz_lno_t row = _clusterVerts(j);
            size_type row_begin = _xadj[row];
            size_type row_end = _xadj[row + 1];
            nnz_scalar_t sum = _Yvector[row];
            for (size_type adjind = row_begin; adjind < row_end; ++adjind)
            {
              nnz_lno_t col = _adj[adjind];
              nnz_scalar_t val = _adj_vals[adjind];
              sum -= val * _Xvector[col];
            }
            _Xvector[row] += omega * sum * _inverse_diagonal[row];
          }
        }
      };

      struct Team_PSGS
      {
        //CSR storage of the matrix
        row_lno_persistent_work_view_t _xadj;
        nnz_lno_persistent_work_view_t _adj;
        scalar_persistent_work_view_t _adj_vals;

        //X,Y vectors, as in Ax = y
        scalar_persistent_work_view_t _Xvector;
        scalar_persistent_work_view_t _Yvector;
        nnz_lno_t _color_set_begin;
        nnz_lno_t _color_set_end;

        scalar_persistent_work_view_t _inverse_diagonal;
        nnz_lno_t _team_work_size; //number of clusters a team will handle (should be multiple of team size)

        int _suggested_team_size;
        bool _is_backward;

        nnz_scalar_t _omega;

        Team_PSGS(row_lno_persistent_work_view_t xadj_, nnz_lno_persistent_work_view_t adj_, scalar_persistent_work_view_t adj_vals_,
                  scalar_persistent_work_view_t Xvector_, scalar_persistent_work_view_t Yvector_,
                  nnz_lno_t color_set_begin, nnz_lno_t color_set_end,
                  nnz_lno_persistent_work_view_t color_adj_,
                  scalar_persistent_work_view_t inverse_diagonal_,
                  nnz_scalar_t omega_ = Kokkos::Details::ArithTraits<nnz_scalar_t>::one(),
                  nnz_lno_t team_work_size_ = 1,
                  int suggested_team_size_ = 1,
                  int vector_size_ = 1):
          _xadj( xadj_),
          _adj( adj_),
          _adj_vals( adj_vals_),
          _Xvector( Xvector_),
          _Yvector( Yvector_),
          _color_set_begin(color_set_begin),
          _color_set_end(color_set_end), _inverse_diagonal(inverse_diagonal_),
          _team_work_size(team_work_size_),
          _suggested_team_size(suggested_team_size_),
          _vector_size(vector_size_),
          _is_backward(false),
          _omega(omega_)
        {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const team_member_t& teamMember) const
        {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, team_work_size),
            [&] (size_type threadWorkItem)
            {
              //this thread sequentially processes workItem contiguous clusters in the color set.
              nnz_lno_t ii = _color_set_begin + team_work_size * teamMember.league_rank() + threadWorkItem;
              if (ii >= _color_set_end)
                return;
              size_type row_begin = _xadj[ii];
              size_type row_end = _xadj[ii + 1];
              nnz_scalar_t product = 0;
              Kokkos::parallel_reduce(
                                      Kokkos::ThreadVectorRange(teamMember, row_end - row_begin),
                                      [&] (size_type i, nnz_scalar_t & valueToUpdate) {
                                        size_type adjind = i + row_begin;
                                        nnz_lno_t colIndex = _adj[adjind];
                                        nnz_scalar_t val = _adj_vals[adjind];
                                        valueToUpdate += val * _Xvector[colIndex];
                                      },
                                      product);
              Kokkos::single(Kokkos::PerThread(teamMember),[=] () {
                  _Xvector[ii] += _omega * (_Yvector[ii] - product) * _inverse_diagonal[ii]
                });
            });
        }
      };

      /**
       * \brief constructor
       */

      GaussSeidel(HandleType *handle_,
                  nnz_lno_t num_rows_,
                  nnz_lno_t num_cols_,
                  const_lno_row_view_t row_map_,
                  const_lno_nnz_view_t entries_,
                  const_scalar_nnz_view_t values_):
        handle(handle_), num_rows(num_rows_), num_cols(num_cols_),
        row_map(row_map_), entries(entries_), values(values_),
        have_diagonal_given(false),
        is_symmetric(true)
      {}

      GaussSeidel(HandleType *handle_,
                  nnz_lno_t num_rows_,
                  nnz_lno_t num_cols_,
                  const_lno_row_view_t row_map_,
                  const_lno_nnz_view_t entries_,
                  bool is_symmetric_ = true):
        handle(handle_),
        num_rows(num_rows_), num_cols(num_cols_),
        row_map(row_map_),
        entries(entries_),
        values(),
        have_diagonal_given(false),
        is_symmetric(is_symmetric_)
      {}

      /**
       * \brief constructor
       */
      GaussSeidel(HandleType *handle_,
                  nnz_lno_t num_rows_,
                  nnz_lno_t num_cols_,
                  const_lno_row_view_t row_map_,
                  const_lno_nnz_view_t entries_,
                  const_scalar_nnz_view_t values_,
                  bool is_symmetric_):
        handle(handle_),
        num_rows(num_rows_), num_cols(num_cols_),
        row_map(row_map_), entries(entries_), values(values_),
        have_diagonal_given(false),
        is_symmetric(is_symmetric_)
      {}

      GaussSeidel(HandleType *handle_,
                  nnz_lno_t num_rows_,
                  nnz_lno_t num_cols_,
                  const_lno_row_view_t row_map_,
                  const_lno_nnz_view_t entries_,
                  const_scalar_nnz_view_t values_,
                  const_scalar_nnz_view_t given_inverse_diagonal_,
                  bool is_symmetric_):
        handle(handle_),
        num_rows(num_rows_), num_cols(num_cols_),
        row_map(row_map_), entries(entries_), values(values_),
        given_inverse_diagonal(given_inverse_diagonal_),
        have_diagonal_given(true),
        is_symmetric(is_symmetric_)
      {}

      //Functors used for symbolic
      struct OrderToPermFunctor
      {
        OrderToPermFunctor(nnz_lno_persistent_work_view_t& order_, nnz_lno_persistent_work_view_t& perm_)
          : order(order_), perm(perm_)
        {}
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
        {
          perm[order[i]] = i;
        }
        nnz_lno_persistent_work_view_t order;
        nnz_lno_persistent_work_view_t perm;
      };

      //Functor to swap the numbers of two colors,
      //so that the last cluster has the last color.
      //Except, doesn't touch the color of the last cluster,
      //since that value is needed the entire time this is running.
      struct ClusterColorRelabelFunctor
      {
        typedef typename HandleType::GraphColoringHandleType GCHandle;
        typedef typename GCHandle::color_view_t ColorView;
        typedef Kokkos::View<row_lno_t*, MyTempMemorySpace> RowmapView;
        typedef Kokkos::View<nnz_lno_t*, MyTempMemorySpace> EntriesView;
        ClusterColorRelabelFunctor(ColorView& colors_, color_t numClusterColors_, nnz_lno_t numClusters_)
          : colors(colors_), numClusterColors(numClusterColors_), numClusters(numClusters_)
        {}

        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
        {
          if(colors(i) == numClusterColors)
            colors(i) = colors(numClusters - 1);
          else if(colors(i) == colors(numClusters - 1))
            colors(i) = numClusterColors;
        }

        ColorView colors;
        color_t numClusterColors;
        nnz_lno_t numClusters;
      };

      //Relabel the last cluster, after running ClusterColorRelabelFunctor.
      //Call with a one-element range policy.
      struct RelabelLastColorFunctor
      {
        typedef typename HandleType::GraphColoringHandleType GCHandle;
        typedef typename GCHandle::color_view_t ColorView;

        RelabelLastColorFunctor(ColorView& colors_, color_t numClusterColors_, nnz_lno_t numClusters_)
          : colors(colors_), numClusterColors(numClusterColors_), numClusters(numClusters_)
        {}

        KOKKOS_INLINE_FUNCTION void operator()(const size_type) const
        {
          colors(numClusters - 1) = numClusterColors;
        }
        
        ColorView colors;
        color_t numClusterColors;
        nnz_lno_t numClusters;
      };

      struct ClusterToVertexColoring
      {
        typedef typename HandleType::GraphColoringHandleType GCHandle;
        typedef typename GCHandle::color_view_t ColorView;

        ClusterToVertexColoring(ColorView& clusterColors_, ColorView& vertexColors_, nnz_lno_t numRows_, nnz_lno_t numClusters_, nnz_lno_t clusterSize_)
          : clusterColors(clusterColors_), vertexColors(vertexColors_), numRows(numRows_), numClusters(numClusters_), clusterSize(clusterSize_)
        {}

        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
        {
          size_type cluster = i / clusterSize;
          size_type clusterOffset = i - cluster * clusterSize;
          vertexColors(i) = ((clusterColors(cluster) - 1) * clusterSize) + clusterOffset + 1;
        }

        ColorView clusterColors;
        ColorView vertexColors;
        nnz_lno_t numRows;
        nnz_lno_t numClusters;
        nnz_lno_t clusterSize;
      };

      template<typename nnz_view_t>
      struct ClusterSizeFunctor
      {
        ClusterSizeFunctor(nnz_view_t& counts_, nnz_view_t& vertClusters_)
          : counts(counts_), vertClusters(vertClusters_)
        {}
        KOKKOS_INLINE_FUNCTION void operator()(const nnz_lno_t i) const
        {
          Kokkos::atomic_increment(&counts(vertClusters(i)));
        }
        nnz_view_t counts;
        nnz_view_t vertClusters;
      };

      template<typename nnz_view_t>
      struct FillClusterVertsFunctor
      {
        FillClusterVertsFunctor(nnz_view_t& clusterOffsets_, nnz_view_t& clusterVerts_, nnz_view_t& vertClusters_, nnz_view_t& insertCounts_)
          : clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_), vertClusters(vertClusters_), insertCounts(insertCounts_)
        {}
        KOKKOS_INLINE_FUNCTION void operator()(const nnz_lno_t i) const
        {
          nnz_lno_t cluster = vertClusters(i);
          nnz_lno_t offset = clusterOffsets(cluster) + Kokkos::atomic_fetch_add(&insertCounts(cluster), 1);
          clusterVerts(offset) = i;
        }
        nnz_view_t clusterOffsets;
        nnz_view_t clusterVerts;
        nnz_view_t vertClusters;
        nnz_view_t insertCounts;
      };

      template<typename Rowmap, typename Colinds, typename nnz_view_t>
      struct BuildCrossClusterMaskFunctor
      {
        BuildCrossClusterMaskFunctor(Rowmap& rowmap_, Colinds& colinds_, nnz_view_t& clusterOffsets_, nnz_view_t& clusterVerts_, nnz_view_t& vertClusters_, bitset_t& mask_)
          : rowmap(rowmap_), colinds(colinds_), clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_), vertClusters(vertClusters_), mask(mask_)
        {}

        //Used a fixed-size hash set in shared memory
        KOKKOS_INLINE_FUNCTION constexpr int tableSize() const
        {
          //Should always be a power-of-two, so that X % tableSize() reduces to a bitwise and.
          return 512;
        }

        //Given a cluster index, get the hash table index.
        //This is the 32-bit xorshift RNG, but it works as a hash function.
        KOKKOS_INLINE_FUNCTION unsigned xorshiftHash(nnz_lno_t cluster) const
        {
          unsigned x = cluster;
          x ^= x << 13;
          x ^= x >> 17;
          x ^= x << 5;
          return x;
        }

        KOKKOS_INLINE_FUNCTION bool lookup(nnz_lno_t cluster, int* table) const
        {
          unsigned h = xorshiftHash(cluster);
          for(unsigned i = h; i < h + 2; i++)
          {
            if(table[i % tableSize()] == cluster)
              return true;
          }
          return false;
        }

        //Try to insert the edge between cluster (team's cluster) and neighbor (neighboring cluster)
        //by inserting nei into the table.
        KOKKOS_INLINE_FUNCTION bool insert(nnz_lno_t cluster, nnz_lno_t nei, int* table) const
        {
          unsigned h = xorshiftHash(nei);
          for(unsigned i = h; i < h + 2; i++)
          {
            if(Kokkos::atomic_compare_exchange_strong(&table[i % tableSize()], cluster, nei))
              return true;
          }
          return false;
        }

        KOKKOS_INLINE_FUNCTION void operator()(const team_member_t t) const
        {
          nnz_lno_t cluster = t.league_rank();
          nnz_lno_t clusterSize = clusterOffsets(cluster + 1) - clusterOffsets(cluster);
          //Use a fixed-size hash table per thread to accumulate neighbor of the cluster.
          //If it fills up (very unlikely) then just count every remaining edge going to another cluster
          //not already in the table; this provides a reasonable upper bound for overallocating the cluster graph.
          //each thread handles a cluster
          int* table = (int*) t.team_shmem().get_shmem(tableSize() * sizeof(int));
          //mark every entry as cluster (self-loop) to represent free/empty
          Kokkos::parallel_for(Kokkos::TeamVectorRange(t, tableSize()),
            [&](const nnz_lno_t i)
            {
              table[i] = cluster;
            });
          t.team_barrier();
          //now, for each row belonging to the cluster, iterate through the neighbors
          Kokkos::parallel_for(Kokkos::TeamThreadRange(t, clusterSize),
            [&] (const nnz_lno_t i)
            {
              nnz_lno_t row = clusterVerts(clusterOffsets(cluster) + i);
              nnz_lno_t rowDeg = rowmap(row + 1) - rowmap(row);
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, rowDeg),
                [&] (const nnz_lno_t j)
                {
                  nnz_lno_t nei = colinds(rowmap(row) + j);
                  nnz_lno_t neiCluster = vertClusters(nei);
                  if(neiCluster != cluster)
                  {
                    //Have a neighbor. Try to find it in the table.
                    if(!lookup(neiCluster, table))
                    {
                      //Not in the table. Try to insert it.
                      insert(cluster, neiCluster, table);
                      //Whether or not insertion succeeded,
                      //this is a cross-cluster edge possibly not seen before
                      mask.set(rowmap(row) + j);
                    }
                  }
                });
            });
        }

        size_t team_shmem_size(int teamSize) const
        {
          return tableSize() * sizeof(int);
        }

        Rowmap rowmap;
        Colinds colinds;
        nnz_view_t clusterOffsets;
        nnz_view_t clusterVerts;
        nnz_view_t vertClusters;
        bitset_t mask;
      };

      template<typename Rowmap, typename Colinds, typename nnz_view_t>
      struct FillClusterEntriesFunctor
      {
        FillClusterEntriesFunctor(
            Rowmap& rowmap_, Colinds& colinds_, nnz_view_t& clusterRowmap_, nnz_view_t& clusterEntries_, nnz_view_t& clusterOffsets_, nnz_view_t& clusterVerts_, nnz_view_t& vertClusters_, bitset_t& edgeMask_)
          : rowmap(rowmap_), colinds(colinds_), clusterRowmap(clusterRowmap_), clusterEntries(clusterEntries_), clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_), vertClusters(vertClusters_), edgeMask(edgeMask_)
        {}
        //Run this scan over entries in clusterVerts (reordered point rows)
        KOKKOS_INLINE_FUNCTION void operator()(const nnz_lno_t i, nnz_lno_t& lcount, const bool& finalPass) const
        {
          nnz_lno_t numRows = rowmap.extent(0) - 1;
          nnz_lno_t row = clusterVerts(i);
          size_type rowStart = rowmap(row);
          size_type rowEnd = rowmap(row + 1);
          nnz_lno_t cluster = vertClusters(row);
          nnz_lno_t clusterStart = clusterOffsets(cluster);
          //Count the number of entries in this row.
          //This is how much lcount will be increased by,
          //yielding the offset corresponding to
          //these point entries in the cluster entries.
          nnz_lno_t rowEntries = 0;
          for(size_type j = rowStart; j < rowEnd; j++)
          {
            if(edgeMask.test(j))
              rowEntries++;
          }
          if(finalPass)
          {
            //if this is the last row in the cluster, update the upper bound in clusterRowmap
            if(i == clusterStart)
            {
              clusterRowmap(cluster) = lcount;
            }
            nnz_lno_t clusterEdge = lcount;
            //populate clusterEntries for these edges
            for(size_type j = rowStart; j < rowEnd; j++)
            {
              if(edgeMask.test(j))
              {
                clusterEntries(clusterEdge++) = vertClusters(colinds(j));
              }
            }
          }
          //update the scan result at the end (exclusive)
          lcount += rowEntries;
          if(i == numRows - 1 && finalPass)
          {
            //on the very last row, set the last entry of the cluster rowmap
            clusterRowmap(clusterRowmap.extent(0) - 1) = lcount;
          }
        }
        Rowmap rowmap;
        Colinds colinds;
        nnz_view_t clusterRowmap;
        nnz_view_t clusterEntries;
        nnz_view_t clusterOffsets;
        nnz_view_t clusterVerts;
        nnz_view_t vertClusters;
        const_bitset_t edgeMask;
      };

      //Assign cluster labels to vertices, given that the vertices are naturally
      //ordered so that contiguous groups of vertices form decent clusters.
      template<typename View>
      struct NopVertClusteringFunctor
      {
        NopVertClusteringFunctor(View& vertClusters_, nnz_lno_t clusterSize_) :
            vertClusters(vertClusters_),
            numRows(vertClusters.extent(0)),
            clusterSize(clusterSize_)
        {}
        KOKKOS_INLINE_FUNCTION void operator()(const nnz_lno_t i) const
        {
          vertClusters(i) = i / clusterSize;
        }
        View vertClusters;
        nnz_lno_t numRows;
        nnz_lno_t clusterSize;
      };

      template<typename View>
      struct ReorderedClusteringFunctor
      {
        ReorderedClusteringFunctor(View& vertClusters_, View& ordering_, nnz_lno_t clusterSize_) :
            vertClusters(vertClusters_),
            ordering(ordering_),
            numRows(vertClusters.extent(0)),
            clusterSize(clusterSize_)
        {}
        KOKKOS_INLINE_FUNCTION void operator()(const nnz_lno_t i) const
        {
          vertClusters(i) = ordering(i) / clusterSize;
        }
        View vertClusters;
        View ordering;
        nnz_lno_t numRows;
        nnz_lno_t clusterSize;
      };


      void initialize_symbolic()
      {
        typename HandleType::GraphColoringHandleType *gchandle = this->handle->get_graph_coloring_handle();

        if (gchandle == NULL)
        {
            this->handle->create_graph_coloring_handle();
            //this->handle->create_gs_handle();
            this->handle->get_gs_handle()->set_owner_of_coloring();
            gchandle = this->handle->get_graph_coloring_handle();
        }

        const_lno_row_view_t xadj = this->row_map;
        const_lno_nnz_view_t adj = this->entries;
        size_type nnz = adj.extent(0);

#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        Kokkos::Impl::Timer timer;
#endif
        auto gsHandler = this->handle->get_gs_handle();
        typename HandleType::GraphColoringHandleType::color_view_t colors;
        color_t numColors;
        typedef Kokkos::View<row_lno_t*, MyTempMemorySpace> rowmap_t;
        typedef Kokkos::View<nnz_lno_t*, MyTempMemorySpace> colind_t;
        typedef Kokkos::View<const row_lno_t*, MyTempMemorySpace> const_rowmap_t;
        typedef Kokkos::View<const nnz_lno_t*, MyTempMemorySpace> const_colind_t;
        rowmap_t tmp_xadj;
        colind_t tmp_adj;
        if(is_symmetric)
        {
          tmp_xadj = xadj;
          tmp_adj = adj;
        }
        else
        {
          KokkosKernels::Impl::symmetrize_graph_symbolic_hashmap
            <const_rowmap_t, const_colind_t, rowmap_t, colind_t, MyExecSpace>
            (num_rows, xadj, adj, tmp_xadj, tmp_adj);
        }
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "SYMMETRIZING TIME: " << timer.seconds() << std::endl;
        timer.reset();
#endif
        typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();
        typedef nnz_lno_persistent_work_view_t nnz_view_t;
        nnz_lno_t clusterSize = gsHandler->get_cluster_size();
        nnz_lno_t numClusters = (num_rows + clusterSize - 1) / clusterSize;
        bool onCuda = false;
#ifdef KOKKOS_ENABLE_CUDA
        onCuda = std::is_same<MyExecSpace, Kokkos::Cuda>::value;
#endif
        auto clusterAlgo = gsHandler->get_clustering_algo();
        if(clusterAlgo == CLUSTER_DEFAULT)
        {
          //Use CM if > 50 entries per row, otherwise balloon clustering.
          //CM is quite fast on CPUs if the level sets fan out quickly, otherwise slow and non-scalable.
          if(!onCuda && (adj.extent(0) / num_rows > 50))
            clusterAlgo = CLUSTER_CUTHILL_MCKEE;
          else
            clusterAlgo = CLUSTER_BALLOON;
        }
        switch(clusterAlgo)
        {
          case CLUSTER_CUTHILL_MCKEE:
          {
            RCM<HandleType, rowmap_t, colinds_t> rcm(num_rows, xadj, adj);
            nnz_view_t cmOrder = rcm.cuthill_mckee();
            vertClusters = nnz_view_t("Cluster labels", num_rows);
            Kokkos::parallel_for(my_exec_space(0, num_rows), ReorderedClusteringFunctor<nnz_view_t>(vertClusters, cmOrder, clusterSize));
            break;
          }
          case CLUSTER_BALLOON:
          {
            BalloonClustering<HandleType, rowmap_t, colinds_t> balloon(num_rows, xadj, adj);
            vertClusters = sssp.run(clusterSize);
            break;
          }
          case CLUSTER_DO_NOTHING:
          {
            vertClusters = nnz_view_t("Cluster labels", num_rows);
            Kokkos::parallel_for(my_exec_space(0, num_rows), NopVertClusteringFunctor<nnz_view_t>(vertClusters, clusterSize));
            break;
          }
          default:
            throw std::runtime_error("Clustering algo " + std::to_string((int) clusterAlgo) + " is not implemented");
        }
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "Graph clustering: " << timer.seconds() << '\n';
        timer.reset();
#endif
        //Construct the cluster offset and vertex array. These allow fast iteration over all vertices in a given cluster.
        nnz_view_t clusterOffsets("Cluster offsets", numClusters + 1);
        nnz_view_t clusterVerts("Cluster -> vertices", num_rows);
        Kokkos::parallel_for(my_exec_space(0, num_rows), ClusterSizeFunctor<nnz_view_t>(clusterOffsets, vertClusters));
        KokkosKernels::Impl::exclusive_parallel_prefix_sum<nnz_view_t, MyExecSpace>(numClusters + 1, clusterOffsets);
        {
          nnz_view_t tempInsertCounts("Temporary cluster insert counts", numClusters);
          Kokkos::parallel_for(my_exec_space(0, num_rows), FillClusterVertsFunctor<nnz_view_t>(clusterOffsets, clusterVerts, vertClusters, tempInsertCounts));
        }
#if KOKKOSSPARSE_IMPL_PRINTDEBUG
        {
          auto clusterOffsetsHost = Kokkos::create_mirror_view(clusterOffsets);
          auto clusterVertsHost = Kokkos::create_mirror_view(clusterVerts);
          puts("Clusters (cluster #, and vertex #s):");
          for(nnz_lno_t i = 0; i < numClusters; i++)
          {
            printf("%d: ", (int) i);
            for(nnz_lno_t j = clusterOffsetsHost(i); j < clusterOffsetsHost(i + 1); j++)
            {
              printf("%d ", (int) clusterVerts(j));
            }
            putchar('\n');
          }
          printf("\n\n\n");
        }
#endif
        //Determine the set of edges (in the point graph) that cross between two distinct clusters
        int vectorSize = this->handle->get_suggested_vector_size(num_rows, adj.extent(0));
        bitset_t crossClusterEdgeMask(adj.extent(0));
        size_type numClusterEdges;
        {
          BuildCrossClusterMaskFunctor<rowmap_t, colinds_t, nnz_view_t>
            buildEdgeMask(xadj, adj, clusterOffsets, clusterVerts, vertClusters, crossClusterEdgeMask);
          int sharedPerTeam = buildEdgeMask.team_shmem_size(0);
          int teamSize = KokkosKernels::Impl::get_suggested_team_size<team_policy_t>(buildEdgeMask, vectorSize, sharedPerTeam, 0);
          Kokkos::parallel_for(team_policy_t(numClusters, teamSize, vectorSize).set_scratch_size(0, Kokkos::PerTeam(sharedPerTeam)), buildEdgeMask);
          numClusterEdges = crossClusterEdgeMask.count();
        }
        nnz_view_t clusterRowmap("Cluster graph rowmap", numClusters + 1);
        nnz_view_t clusterEntries("Cluster graph colinds", numClusterEdges);
        Kokkos::parallel_scan(my_exec_space(0, num_rows), FillClusterEntriesFunctor<rowmap_t, colinds_t, nnz_view_t>(xadj, adj, clusterRowmap, clusterEntries, clusterOffsets, clusterVerts, vertClusters, crossClusterEdgeMask));
#if KOKKOSSPARSE_IMPL_PRINTDEBUG
        {
          auto clusterRowmapHost = Kokkos::create_mirror_view(clusterRowmap);
          auto clusterEntriesHost = Kokkos::create_mirror_view(clusterEntries);
          puts("Cluster graph (cluster #, and neighbors):");
          for(nnz_lno_t i = 0; i < numClusters; i++)
          {
            printf("%d: ", (int) i);
            for(nnz_lno_t j = clusterRowmapHost(i); j < clusterRowmapHost(i + 1); j++)
            {
              printf("%d ", (int) clusterEntries(j));
            }
            putchar('\n');
          }
          printf("\n\n\n");
        }
#endif
        //Create a handle that uses nnz_lno_t as the size_type, since the cluster graph should never be larger than 2^31 entries.
        KokkosKernels::Experimental::KokkosKernelsHandle<nnz_lno_t, nnz_lno_t, double, MyExecSpace, MyPersistentMemorySpace, MyPersistentMemorySpace> kh;
        kh.create_graph_coloring_handle(KokkosGraph::COLORING_DEFAULT);
        KokkosGraph::Experimental::graph_color_symbolic(&kh, numClusters, numClusters, clusterRowmap, clusterEntries);
        //retrieve colors
        auto coloringHandle = kh.get_graph_coloring_handle();
        colors = coloringHandle->get_vertex_colors();
        numColors = coloringHandle->get_num_colors();
        kh.destroy_graph_coloring_handle();
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "Coloring: " << timer.seconds() << '\n';
        timer.reset();
#endif

#if KOKKOSSPARSE_IMPL_RUNSEQUENTIAL
        numColors = numClusters;
        KokkosKernels::Impl::print_1Dview(colors);
        std::cout << "numCol:" << numClusters << " numClusters:" << numClusters << '\n';
        typename HandleType::GraphColoringHandleType::color_view_t::HostMirror  h_colors = Kokkos::create_mirror_view (colors);
        for(int i = 0; i < num_rows; ++i){
          h_colors(i) = i + 1;
        }
        Kokkos::deep_copy(colors, h_colors);
#endif
        gsHandler->set_num_colors(numColors);
        //gsHandler->set_new_adj_val(newvals_);
        if (this->handle->get_gs_handle()->is_owner_of_coloring()){
          this->handle->destroy_graph_coloring_handle();
          this->handle->get_gs_handle()->set_owner_of_coloring(false);
        }
        this->handle->get_gs_handle()->set_call_symbolic(true);


        this->handle->get_gs_handle()->allocate_x_y_vectors(this->num_rows, this->num_cols);
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "ALLOC:" << timer.seconds() << std::endl;
#endif
      }

      void initialize_numeric()
      {
        if (this->handle->get_gs_handle()->is_symbolic_called() == false){
          this->initialize_symbolic();
        }
        //Timer for whole numeric
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        Kokkos::Impl::Timer timer;
#endif
        typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();
        if(gsHandler->
        {

          const_lno_row_view_t xadj = this->row_map;
          const_lno_nnz_view_t adj = this->entries;
          const_scalar_nnz_view_t adj_vals = this->values;

          size_type nnz = adj_vals.extent(0);


          int suggested_vector_size = this->handle->get_suggested_vector_size(num_rows, nnz);
          int suggested_team_size = this->handle->get_suggested_team_size(suggested_vector_size);
          nnz_lno_t rows_per_team = this->handle->get_team_work_size(suggested_team_size, MyExecSpace::concurrency(), num_rows);

            Get_Matrix_Diagonals gmd(_xadj, _adj, _adj_vals, permuted_inverse_diagonal,
                                     this->num_rows,
                                     rows_per_team, 1, 1);

            if (this->handle->get_handle_exec_space() == KokkosKernels::Impl::Exec_CUDA || block_size > 1){
              Kokkos::parallel_for("KokkosSparse::GaussSeidel::team_get_matrix_diagonals",
                                   team_policy_t(num_rows / rows_per_team + 1 , suggested_team_size, suggested_vector_size),
                                   gmd );
            }
            else {
              Kokkos::parallel_for("KokkosSparse::GaussSeidel::get_matrix_diagonals",
                                   my_exec_space(0,num_rows),
                                   gmd );
            }

          } else {

            if (block_size > 1)
              KokkosKernels::Impl::permute_block_vector
                <const_scalar_nnz_view_t,
                 scalar_persistent_work_view_t,
                 nnz_lno_persistent_work_view_t, MyExecSpace>(
                                                              num_rows, block_size,
                                                              old_to_new_map,
                                                              given_inverse_diagonal,
                                                              permuted_inverse_diagonal
                                                              );
            else
              KokkosKernels::Impl::permute_vector
                <const_scalar_nnz_view_t,
                 scalar_persistent_work_view_t,
                 nnz_lno_persistent_work_view_t, MyExecSpace>(
                                                              num_rows,
                                                              old_to_new_map,
                                                              given_inverse_diagonal,
                                                              permuted_inverse_diagonal
                                                              );

          }

          MyExecSpace().fence();
          this->handle->get_gs_handle()->set_permuted_inverse_diagonal(permuted_inverse_diagonal);

          this->handle->get_gs_handle()->set_call_numeric(true);

        }
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "NUMERIC:" << timer.seconds() << std::endl;
#endif
      }

      struct Get_Matrix_Diagonals{
        row_lno_persistent_work_view_t _xadj;
        nnz_lno_persistent_work_view_t _adj; // CSR storage of the graph.
        scalar_persistent_work_view_t _adj_vals; // CSR storage of the graph.
        scalar_persistent_work_view_t _diagonals;

        nnz_lno_t num_total_rows;
        nnz_lno_t rows_per_team;
        nnz_lno_t block_size;
        nnz_lno_t block_matrix_size;

        nnz_scalar_t one;


        Get_Matrix_Diagonals(
                             row_lno_persistent_work_view_t xadj_,
                             nnz_lno_persistent_work_view_t adj_,
                             scalar_persistent_work_view_t adj_vals_,
                             scalar_persistent_work_view_t diagonals_,
                             nnz_lno_t num_total_rows_,
                             nnz_lno_t rows_per_team_) :
          _xadj( xadj_),
          _adj( adj_),
          _adj_vals( adj_vals_), _diagonals(diagonals_),
          num_total_rows(num_total_rows_), rows_per_team(rows_per_team_),
          one(Kokkos::Details::ArithTraits<nnz_scalar_t>::one())
        {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const nnz_lno_t & row_id) const {
          size_type row_begin = _xadj[row_id];
          size_type row_end = _xadj[row_id + 1] ;
          nnz_lno_t row_size = row_end - row_begin;
          for (nnz_lno_t col_ind = 0; col_ind < row_size; ++col_ind){
            size_type nnz_ind = col_ind + row_begin;
            nnz_lno_t column_id = _adj[nnz_ind];
            if (column_id == row_id) {
              nnz_scalar_t val = _adj_vals[nnz_ind];
              _diagonals[row_id] = one / val;
              break;
            }
          }
        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const team_member_t &team) const{

          const nnz_lno_t i_begin = team.league_rank() * rows_per_team;
          const nnz_lno_t i_end = i_begin + rows_per_team <= num_total_rows ? i_begin + rows_per_team : num_total_rows;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team,i_begin,i_end), [&] (const nnz_lno_t& row_id) {
              size_type row_begin = _xadj[row_id];
              size_type row_end = _xadj[row_id + 1] ;
              nnz_lno_t row_size = row_end - row_begin;

              Kokkos::parallel_for (Kokkos::ThreadVectorRange(team,row_size), [&] (const nnz_lno_t& col_ind) {
                  size_type val_index = col_ind + row_begin;
                  nnz_lno_t column_id = _adj[val_index];
                  if (column_id == row_id){
                    size_type _val_index = row_begin * block_matrix_size + col_ind * block_size;
                    for (nnz_lno_t r = 0; r < block_size; ++r){
                      nnz_scalar_t val = _adj_vals[_val_index];
                      _diagonals[row_id * block_size + r] = one / val;
                      _val_index += row_size * block_size + 1;
                    }
                  }
                });
            });
        }
      };

      template <typename x_value_array_type, typename y_value_array_type>
      void apply(
                       x_value_array_type x_lhs_output_vec,
                       y_value_array_type y_rhs_input_vec,
                       bool init_zero_x_vector = false,
                       int numIter = 1,
                       nnz_scalar_t omega = Kokkos::Details::ArithTraits<nnz_scalar_t>::one(),
                       bool apply_forward = true,
                       bool apply_backward = true,
                       bool update_y_vector = true)
      {
        typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();
        scalar_persistent_work_view_t Permuted_Yvector = gsHandler->get_permuted_y_vector();
        scalar_persistent_work_view_t Permuted_Xvector = gsHandler->get_permuted_x_vector();


        row_lno_persistent_work_view_t xadj_ = gsHandler->get_new_xadj();
        nnz_lno_persistent_work_view_t adj_ = gsHandler->get_new_adj();
        nnz_lno_persistent_work_view_t color_adj = gsHandler->get_color_adj();
        nnz_lno_persistent_work_host_view_t h_color_xadj = gsHandler->get_color_xadj();

        color_t numColors = gsHandler->get_num_colors();



        if (update_y_vector){
          KokkosKernels::Impl::permute_vector
            <y_value_array_type,
             scalar_persistent_work_view_t,
             nnz_lno_persistent_work_view_t, MyExecSpace>(
                                                          num_rows,
                                                          old_to_new_map,
                                                          y_rhs_input_vec,
                                                          Permuted_Yvector
                                                          );
        }
        MyExecSpace().fence();
        if(init_zero_x_vector){
          KokkosKernels::Impl::zero_vector<scalar_persistent_work_view_t, MyExecSpace>(num_cols, Permuted_Xvector);
        }
        else{
          KokkosKernels::Impl::permute_vector
            <x_value_array_type, scalar_persistent_work_view_t, nnz_lno_persistent_work_view_t, MyExecSpace>(
                                                                                                             num_cols,
                                                                                                             old_to_new_map,
                                                                                                             x_lhs_output_vec,
                                                                                                             Permuted_Xvector
                                                                                                             );
        }
        MyExecSpace().fence();

        row_lno_persistent_work_view_t permuted_xadj = gsHandler->get_new_xadj();
        nnz_lno_persistent_work_view_t permuted_adj = gsHandler->get_new_adj();
        scalar_persistent_work_view_t permuted_adj_vals = gsHandler->get_new_adj_val();
        scalar_persistent_work_view_t permuted_inverse_diagonal = gsHandler->get_permuted_inverse_diagonal();



#if KOKKOSSPARSE_IMPL_PRINTDEBUG
        std::cout << "--point Before X:";
        KokkosKernels::Impl::print_1Dview(Permuted_Xvector,true);
        std::cout << "--point Before Y:";
        KokkosKernels::Impl::print_1Dview(Permuted_Yvector,true);
#endif

        if (gsHandler->get_algorithm_type()== GS_PERMUTED){
          PSGS gs(permuted_xadj, permuted_adj, permuted_adj_vals,
                  Permuted_Xvector, Permuted_Yvector, color_adj, omega, permuted_inverse_diagonal);

          this->IterativePSGS(
                              gs,
                              numColors,
                              h_color_xadj,
                              numIter,
                              apply_forward,
                              apply_backward);
        }
        else{

          pool_memory_space m_space(0, 0, 0,  KokkosKernels::Impl::ManyThread2OneChunk, false);

          Team_PSGS gs(permuted_xadj, permuted_adj, permuted_adj_vals,
                       Permuted_Xvector, Permuted_Yvector,0,0, permuted_inverse_diagonal, m_space,0,0,omega);

          this->IterativePSGS(
                              gs,
                              numColors,
                              h_color_xadj,
                              numIter,
                              apply_forward,
                              apply_backward);
        }

        //Kokkos::parallel_for( my_exec_space(0,nr), PermuteVector(x_lhs_output_vec, Permuted_Xvector, color_adj));


        KokkosKernels::Impl::permute_vector
          <scalar_persistent_work_view_t,x_value_array_type,  nnz_lno_persistent_work_view_t, MyExecSpace>(
                                                                                                           num_cols,
                                                                                                           color_adj,
                                                                                                           Permuted_Xvector,
                                                                                                           x_lhs_output_vec
                                                                                                           );
        MyExecSpace().fence();
      }

      void IterativePSGS(
                         Team_PSGS &gs,
                         color_t numColors,
                         nnz_lno_persistent_work_host_view_t h_color_xadj,
                         int num_iteration,
                         bool apply_forward,
                         bool apply_backward){

        for (int i = 0; i < num_iteration; ++i){
          this->DoPSGS(gs, numColors, h_color_xadj, apply_forward, apply_backward);
        }
      }

      void DoPSGS(Team_PSGS &gs, color_t numColors, nnz_lno_persistent_work_host_view_t h_color_xadj,
                  bool apply_forward,
                  bool apply_backward){

        nnz_lno_t suggested_team_size = gs.suggested_team_size;
        nnz_lno_t team_row_chunk_size = gs.team_work_size;
        int vector_size = gs.vector_size;
        nnz_lno_t block_size = this->handle->get_gs_handle()->get_block_size();

        if (apply_forward)
        {
          gs.is_backward = false;
          for (color_t i = 0; i < numColors; ++i){
            nnz_lno_t color_index_begin = h_color_xadj(i);
            nnz_lno_t color_index_end = h_color_xadj(i + 1);
            int overall_work = color_index_end - color_index_begin;// /256 + 1;
            gs._color_set_begin = color_index_begin;
            gs._color_set_end = color_index_end;

            if (block_size == 1){
              Kokkos::parallel_for("KokkosSparse::GaussSeidel::Team_PSGS::forward",
                                   team_policy_t(overall_work / team_row_chunk_size + 1 , suggested_team_size, vector_size),
                                   gs );
            } else if (gs.num_max_vals_in_l2 == 0){
              //if (i == 0)std::cout << "block_team" << std::endl;
              Kokkos::parallel_for("KokkosSparse::GaussSeidel::BLOCK_Team_PSGS::forward",
                                   block_team_fill_policy_t(overall_work / team_row_chunk_size + 1 , suggested_team_size, vector_size),
                                   gs );
            }
            else {
              //if (i == 0)    std::cout << "big block_team" << std::endl;

              Kokkos::parallel_for("KokkosSparse::GaussSeidel::BIGBLOCK_Team_PSGS::forward",
                                   bigblock_team_fill_policy_t(overall_work / team_row_chunk_size + 1 , suggested_team_size, vector_size),
                                   gs );
            }

            MyExecSpace().fence();
          }
        }
        if (apply_backward){
          gs.is_backward = true;
          if (numColors > 0)
            for (color_t i = numColors - 1;  ; --i){
              nnz_lno_t color_index_begin = h_color_xadj(i);
              nnz_lno_t color_index_end = h_color_xadj(i + 1);
              nnz_lno_t numberOfTeams = color_index_end - color_index_begin;// /256 + 1;
              gs._color_set_begin = color_index_begin;
              gs._color_set_end = color_index_end;
              if (block_size == 1){

                Kokkos::parallel_for("KokkosSparse::GaussSeidel::Team_PSGS::backward",
                                     team_policy_t(numberOfTeams / team_row_chunk_size + 1 , suggested_team_size, vector_size),
                                     gs );
              }
              else if ( gs.num_max_vals_in_l2 == 0){
                //if (i == 0) std::cout << "block_team backward" << std::endl;

                Kokkos::parallel_for("KokkosSparse::GaussSeidel::BLOCK_Team_PSGS::backward",
                                     block_team_fill_policy_t(numberOfTeams / team_row_chunk_size + 1 , suggested_team_size, vector_size),
                                     gs );
              }
              else {
                //if (i == 0)               std::cout << "big block_team backward" << std::endl;

                Kokkos::parallel_for("KokkosSparse::GaussSeidel::BIGBLOCK_Team_PSGS::backward",
                                     bigblock_team_fill_policy_t(numberOfTeams / team_row_chunk_size + 1 , suggested_team_size, vector_size),
                                     gs );
              }
              MyExecSpace().fence();
              if (i == 0){
                break;
              }
            }
        }
      }

      void IterativePSGS(
                         PSGS &gs,
                         color_t numColors,
                         nnz_lno_persistent_work_host_view_t h_color_xadj,
                         int num_iteration,
                         bool apply_forward,
                         bool apply_backward){

        for (int i = 0; i < num_iteration; ++i){
          //std::cout << "ier:" << i << std::endl;
          this->DoPSGS(gs, numColors, h_color_xadj, apply_forward, apply_backward);
        }
      }

      void DoPSGS(PSGS &gs, color_t numColors, nnz_lno_persistent_work_host_view_t h_color_xadj,
                  bool apply_forward,
                  bool apply_backward){
        if (apply_forward){
          //std::cout <<  "numColors:" << numColors << std::endl;
          for (color_t i = 0; i < numColors; ++i){
            nnz_lno_t color_index_begin = h_color_xadj(i);
            nnz_lno_t color_index_end = h_color_xadj(i + 1);
            //std::cout <<  "i:" << i << " color_index_begin:" << color_index_begin << " color_index_end:" << color_index_end << std::endl;
            Kokkos::parallel_for ("KokkosSparse::GaussSeidel::PSGS::forward",
                                  my_exec_space (color_index_begin, color_index_end) , gs);
            MyExecSpace().fence();
          }
        }
        if (apply_backward && numColors){
          for (size_type i = numColors - 1; ; --i){
            nnz_lno_t color_index_begin = h_color_xadj(i);
            nnz_lno_t color_index_end = h_color_xadj(i + 1);
            Kokkos::parallel_for ("KokkosSparse::GaussSeidel::PSGS::backward",
                                  my_exec_space (color_index_begin, color_index_end) , gs);
            MyExecSpace().fence();
            if (i == 0){
              break;
            }
          }
        }
      }
    };

  }
}

#endif

