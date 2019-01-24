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

#ifndef KOKKOSSPARSE_IMPL_SPMV_STRUCT_DEF_HPP_
#define KOKKOSSPARSE_IMPL_SPMV_STRUCT_DEF_HPP_

#include "Kokkos_InnerProductSpaceTraits.hpp"
#include "KokkosBlas1_scal.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

namespace KokkosSparse {
namespace Impl {

enum {FD, FE};

// This TransposeFunctor is functional, but not necessarily performant.
template<class AMatrix,
         class XVector,
         class YVector,
         int dobeta,
         bool conjugate>
struct SPMV_Struct_Transpose_Functor {
  typedef typename AMatrix::execution_space            execution_space;
  typedef typename AMatrix::non_const_ordinal_type     ordinal_type;
  typedef typename AMatrix::non_const_value_type       value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type            team_member;
  typedef Kokkos::Details::ArithTraits<value_type>     ATV;
  typedef typename YVector::non_const_value_type       coefficient_type;
  typedef typename YVector::non_const_value_type       y_value_type;

  const coefficient_type alpha;
  AMatrix m_A;
  XVector m_x;
  const coefficient_type beta;
  YVector m_y;
  const ordinal_type rows_per_thread;

  SPMV_Struct_Transpose_Functor (const coefficient_type& alpha_,
                                 const AMatrix& m_A_,
                                 const XVector& m_x_,
                                 const coefficient_type& beta_,
                                 const YVector& m_y_,
                                 const ordinal_type rows_per_thread_) :
    alpha (alpha_), m_A (m_A_), m_x (m_x_),
    beta (beta_), m_y (m_y_),
    rows_per_thread (rows_per_thread_)
  {}

  KOKKOS_INLINE_FUNCTION void
  operator() (const team_member& dev) const
  {
    // This should be a thread loop as soon as we can use C++11
    for (ordinal_type loop = 0; loop < rows_per_thread; ++loop) {
      // iRow represents a row of the matrix, so its correct type is
      // ordinal_type.
      const ordinal_type iRow = (static_cast<ordinal_type> (dev.league_rank() * dev.team_size() + dev.team_rank()))
                                * rows_per_thread + loop;
      if (iRow >= m_A.numRows ()) {
        return;
      }

      const auto row = m_A.rowConst (iRow);
      const ordinal_type row_length = row.length;

#ifdef __CUDA_ARCH__
      for (ordinal_type iEntry = static_cast<ordinal_type> (threadIdx.x);
           iEntry < row_length;
           iEntry += static_cast<ordinal_type> (blockDim.x))
#else
      for (ordinal_type iEntry = 0;
           iEntry < row_length;
           iEntry ++)
#endif
      {
        const value_type val = conjugate ?
          ATV::conj (row.value(iEntry)) :
          row.value(iEntry);
        const ordinal_type ind = row.colidx(iEntry);

        Kokkos::atomic_add (&m_y(ind), static_cast<y_value_type> (alpha * val * m_x(iRow)));
      }
    }
  }
};

template<class AMatrix,
         class XVector,
         class YVector,
         int dobeta,
         bool conjugate>
struct SPMV_Struct_Functor {
  typedef typename AMatrix::non_const_size_type              size_type;
  typedef typename AMatrix::non_const_ordinal_type           ordinal_type;
  typedef typename AMatrix::non_const_value_type             value_type;
  typedef typename AMatrix::execution_space                  execution_space;
  typedef typename execution_space::scratch_memory_space     scratch_space;
  typedef typename KokkosSparse::SparseRowViewConst<AMatrix> row_view_const;
  typedef typename Kokkos::TeamPolicy<execution_space>       team_policy;
  typedef typename team_policy::member_type                  team_member;
  typedef Kokkos::Details::ArithTraits<value_type>           ATV;
  typedef Kokkos::View<ordinal_type*, scratch_space,
		       Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_ordinal_1d;

  // Tags to perform SPMV on interior and boundaries
  struct interior3ptTag{};    // 1D FD and FE discretization
  struct interior5ptTag{};    // 2D FD discretization
  struct interior9ptTag{};    // 2D FE discretization
  struct interior7ptTag{};    // 3D FD discretization
  struct interior27ptTag{};   // 3D FE discretization
  struct exterior1DTag{};
  struct exterior2DTag{};
  struct exterior3DTag{};

  // Classic spmv params
  const value_type alpha;
  AMatrix  m_A;
  XVector m_x;
  const value_type beta;
  YVector m_y;

  // Additional structured spmv params
  int numDimensions = 0;
  ordinal_type ni  = 0, nj = 0, nk = 0;
  const int stencil_type;
  ordinal_type numInterior, numExterior;
  const int64_t rows_per_team;

  SPMV_Struct_Functor (const Kokkos::View<int*, Kokkos::HostSpace>structure_,
		       const int stencil_type_,
                       const value_type alpha_,
                       const AMatrix m_A_,
                       const XVector m_x_,
                       const value_type beta_,
                       const YVector m_y_,
                       const int64_t rows_per_team_) :
    alpha (alpha_), m_A (m_A_), m_x (m_x_),
    beta (beta_), m_y (m_y_),
    stencil_type(stencil_type_),
    rows_per_team (rows_per_team_)
  {
    static_assert (static_cast<int> (XVector::rank) == 1,
                   "XVector must be a rank 1 View.");
    static_assert (static_cast<int> (YVector::rank) == 1,
                   "YVector must be a rank 1 View.");

    numDimensions = structure_.extent(0);
    if(numDimensions == 1) {
      ni = static_cast<ordinal_type>(structure_(0));
    } else if(numDimensions == 2) {
      ni = static_cast<ordinal_type>(structure_(0));
      nj = static_cast<ordinal_type>(structure_(1));
    } else if(numDimensions == 3) {
      ni = static_cast<ordinal_type>(structure_(0));
      nj = static_cast<ordinal_type>(structure_(1));
      nk = static_cast<ordinal_type>(structure_(2));
    }
  }

  void compute(const int64_t worksets, const int team_size, const int vector_length) {

    if(numDimensions == 1) {
      // Treat interior points using structured algorithm
      numInterior = ni - 2;
      if(numInterior > 0) {
	  size_t shared_size = shared_ordinal_1d::shmem_size(3);
        Kokkos::TeamPolicy<interior3ptTag,
                           execution_space,
                           Kokkos::Schedule<Kokkos::Static> > policy(1,1);
        if(team_size < 0) {
          policy = Kokkos::TeamPolicy<interior3ptTag, execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,Kokkos::AUTO,vector_length).
	      set_scratch_size(0, Kokkos::PerTeam( shared_size ));
        } else {
          policy = Kokkos::TeamPolicy<interior3ptTag, execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,team_size,vector_length).
	      set_scratch_size(0, Kokkos::PerTeam( shared_size ));
        }
        Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: interior", policy, *this);
      }

      // Treat exterior points using unstructured algorithm
      numExterior = 2;
      if(numExterior > 0) {
        Kokkos::RangePolicy<exterior1DTag,
                            execution_space,
                            Kokkos::Schedule<Kokkos::Static> > policy(0, numExterior);
        Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: exterior", policy, *this);
      }

    } else if(numDimensions == 2) {
      // Treat interior points using structured algorithm
      numInterior = (ni - 2)*(nj - 2);
      if(numInterior > 0) {
        if(stencil_type == 1) {
	  size_t shared_size = shared_ordinal_1d::shmem_size(5);
          Kokkos::TeamPolicy<interior5ptTag,
                             execution_space,
                             Kokkos::Schedule<Kokkos::Static> > policy(1,1);
          if(team_size < 0) {
            policy = Kokkos::TeamPolicy<interior5ptTag, execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,Kokkos::AUTO,vector_length).
	      set_scratch_size(0, Kokkos::PerTeam( shared_size ));
          } else {
            policy = Kokkos::TeamPolicy<interior5ptTag, execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,team_size,vector_length).
	      set_scratch_size(0, Kokkos::PerTeam( shared_size ));
          }
          Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: interior", policy, *this);
        } else if(stencil_type == 2) {
	  size_t shared_size = shared_ordinal_1d::shmem_size(9);
          Kokkos::TeamPolicy<interior9ptTag,
                             execution_space,
                             Kokkos::Schedule<Kokkos::Dynamic> > policy(1,1);
          if(team_size < 0) {
            policy = Kokkos::TeamPolicy<interior9ptTag, execution_space, Kokkos::Schedule<Kokkos::Dynamic> >(worksets,Kokkos::AUTO,vector_length).
	      set_scratch_size(0, Kokkos::PerTeam( shared_size ));
          } else {
            policy = Kokkos::TeamPolicy<interior9ptTag, execution_space, Kokkos::Schedule<Kokkos::Dynamic> >(worksets,team_size,vector_length).
	      set_scratch_size(0, Kokkos::PerTeam( shared_size ));
          }
          Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: interior", policy, *this);
        }
      }

      // Treat exterior points using unstructured algorithm
      numExterior = ni*nj - numInterior;
      if(numExterior > 0) {
        Kokkos::RangePolicy<exterior2DTag,
                            execution_space,
                            Kokkos::Schedule<Kokkos::Static> > policy(0, numExterior);
        Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: exterior", policy, *this);
      }
    } else if(numDimensions == 3) {
      // Treat interior points using structured algorithm
      numInterior = (ni - 2)*(nj - 2)*(nk - 2);
      if(numInterior > 0) {
        if(stencil_type == 1) {
	  size_t shared_size = shared_ordinal_1d::shmem_size(7);
          Kokkos::TeamPolicy<interior7ptTag,
                             execution_space,
                             Kokkos::Schedule<Kokkos::Static> > policy(1,1);
          if(team_size < 0) {
            policy = Kokkos::TeamPolicy<interior7ptTag, execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,Kokkos::AUTO,vector_length).
	      set_scratch_size(0, Kokkos::PerTeam( shared_size ));
          } else {
            policy = Kokkos::TeamPolicy<interior7ptTag, execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,team_size,vector_length).
	      set_scratch_size(0, Kokkos::PerTeam( shared_size ));
          }
          Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: interior", policy, *this);
        } else if(stencil_type == 2) {
	  size_t shared_size = shared_ordinal_1d::shmem_size(27);
          Kokkos::TeamPolicy<interior27ptTag,
                             execution_space,
                             Kokkos::Schedule<Kokkos::Dynamic> > policy(1,1);
          if(team_size < 0) {
            policy = Kokkos::TeamPolicy<interior27ptTag, execution_space, Kokkos::Schedule<Kokkos::Dynamic> >(worksets,Kokkos::AUTO,vector_length).
	      set_scratch_size(0, Kokkos::PerTeam(shared_size));
          } else {
            policy = Kokkos::TeamPolicy<interior27ptTag, execution_space, Kokkos::Schedule<Kokkos::Dynamic> >(worksets,team_size,vector_length).
	      set_scratch_size(0, Kokkos::PerTeam(shared_size));
          }
          Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: interior", policy, *this);
        }
      }

      // Treat exterior points using unstructured algorithm
      numExterior = ni*nj*nk - numInterior;
      if(numExterior > 0) {
        Kokkos::RangePolicy<exterior3DTag,
                            execution_space,
                            Kokkos::Schedule<Kokkos::Static> > policy(0, numExterior);
        Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: exterior", policy, *this);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const interior3ptTag&, const team_member& dev) const
  {
    // Allocate and initialize columnOffsets array for the team
    shared_ordinal_1d columnOffsets(dev.team_scratch(0), 3);
    Kokkos::single(Kokkos::PerTeam(dev), [&] () {
    	columnOffsets(0) = -1;
    	columnOffsets(1) = 0;
    	columnOffsets(2) = 1;
      });
    dev.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev, 0, rows_per_team), [&] (const ordinal_type& loop) {
        const ordinal_type interiorIdx = static_cast<ordinal_type> ( dev.league_rank() ) * rows_per_team + loop;
        if(interiorIdx >= numInterior) { return; }

        ordinal_type rowIdx;
        rowIdx = interiorIdx + 1;

        const size_type rowOffset = m_A.graph.row_map(rowIdx);
        const value_type* value_ptr = m_A.values.data() + rowOffset;
	value_type sum = 0.0;
	Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev, 3), [&] (const ordinal_type& idx, value_type& lclSum) {
	    lclSum += *(value_ptr + idx)*m_x(rowIdx + columnOffsets(idx));
	  }, sum);

	Kokkos::single(Kokkos::PerThread(dev), [&] () {
	    m_y(rowIdx) = beta*m_y(rowIdx) + alpha*sum;
	  });
      });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const interior5ptTag&, const team_member& dev) const
  {
    // Allocate and initialize columnOffsets array for the team
    shared_ordinal_1d columnOffsets(dev.team_scratch(0), 5);
    Kokkos::single(Kokkos::PerTeam(dev), [&] () {
    	columnOffsets(0) = -ni;
    	columnOffsets(1) = -1;
    	columnOffsets(2) = 0;
    	columnOffsets(3) = 1;
    	columnOffsets(4) = ni;
      });
    dev.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev, 0, rows_per_team), [&] (const ordinal_type& loop) {
        const ordinal_type interiorIdx = static_cast<ordinal_type> ( dev.league_rank() ) * rows_per_team + loop;
        if(interiorIdx >= numInterior) { return; }

        ordinal_type i, j, rowIdx;
        j = interiorIdx / (ni - 2);
        i = interiorIdx % (ni - 2);
        rowIdx = (j + 1)*ni + i + 1;

        const size_type rowOffset = m_A.graph.row_map(rowIdx);
        const value_type* value_ptr = m_A.values.data() + rowOffset;
	value_type sum = 0.0;
	Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev, 5), [&] (const ordinal_type& idx, value_type& lclSum) {
	    lclSum += *(value_ptr + idx)*m_x(rowIdx + columnOffsets(idx));
	  }, sum);

	Kokkos::single(Kokkos::PerThread(dev), [&] () {
	    m_y(rowIdx) = beta*m_y(rowIdx) + alpha*sum;
	  });
      });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const interior9ptTag&, const team_member& dev) const
  {
    // Allocate and initialize columnOffsets array for the team
    shared_ordinal_1d columnOffsets(dev.team_scratch(0), 9);
    Kokkos::single(Kokkos::PerTeam(dev), [&] () {
    	columnOffsets(0) = -ni - 1;
    	columnOffsets(1) = -ni;
    	columnOffsets(2) = -ni + 1;
    	columnOffsets(3) = -1;
    	columnOffsets(4) = 0;
    	columnOffsets(5) = 1;
	columnOffsets(6) = ni - 1;
	columnOffsets(7) = ni;
	columnOffsets(8) = ni + 1;
      });
    dev.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev, 0, rows_per_team), [&] (const ordinal_type& loop) {
        const ordinal_type interiorIdx = static_cast<ordinal_type> ( dev.league_rank() ) * rows_per_team + loop;
        if(interiorIdx >= numInterior) { return; }

        ordinal_type i, j, rowIdx;
        j = interiorIdx / (ni - 2);
        i = interiorIdx % (ni - 2);
        rowIdx = (j + 1)*ni + i + 1;

        const size_type rowOffset = m_A.graph.row_map(rowIdx);
        const value_type* value_ptr = &(m_A.values(rowOffset));
	value_type sum = 0.0;
	Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev, 9), [&] (const ordinal_type& idx, value_type& lclSum) {
	    lclSum += *(value_ptr + idx)*m_x(rowIdx + columnOffsets(idx));
	  }, sum);

	Kokkos::single(Kokkos::PerThread(dev), [&] () {
	    m_y(rowIdx) = beta*m_y(rowIdx) + alpha*sum;
	  });
      });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const interior7ptTag&, const team_member& dev) const
  {
    // Allocate and initialize columnOffsets array for the team
    shared_ordinal_1d columnOffsets(dev.team_scratch(0), 7);
    Kokkos::single(Kokkos::PerTeam(dev), [&] () {
    	columnOffsets(0) = -ni*nj;
    	columnOffsets(1) = -ni;
    	columnOffsets(2) = -1;
    	columnOffsets(3) = 0;
    	columnOffsets(4) = 1;
    	columnOffsets(5) = ni;
	columnOffsets(6) = ni*nj;
      });
    dev.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev, 0, rows_per_team), [&] (const ordinal_type& loop) {
        const ordinal_type interiorIdx = static_cast<ordinal_type> ( dev.league_rank() ) * rows_per_team + loop;
        if(interiorIdx >= numInterior) { return; }

        ordinal_type i, j, k, rowIdx, rem;
        k = interiorIdx / ((ni - 2)*(nj - 2));
        rem  = interiorIdx % ((ni - 2)*(nj - 2));
        j = rem / (ni - 2);
        i = rem % (ni - 2);
        rowIdx = (k + 1)*nj*ni + (j + 1)*ni + (i + 1);

        const size_type rowOffset = m_A.graph.row_map(rowIdx);
        const value_type* value_ptr = &(m_A.values(rowOffset));
	value_type sum = 0.0;
	Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev, 7), [&] (const ordinal_type& idx, value_type& lclSum) {
	    lclSum += *(value_ptr + idx)*m_x(rowIdx + columnOffsets(idx));
	  }, sum);

	Kokkos::single(Kokkos::PerThread(dev), [&] () {
	    m_y(rowIdx) = beta*m_y(rowIdx) + alpha*sum;
	  });
      });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const interior27ptTag&, const team_member& dev) const
  {
    // Allocate and initialize columnOffsets array for the team
    shared_ordinal_1d columnOffsets(dev.team_scratch(0), 27);
    Kokkos::single(Kokkos::PerTeam(dev), [&] () {
    	columnOffsets(0)  = -ni*nj - ni - 1;
    	columnOffsets(1)  = -ni*nj - ni;
    	columnOffsets(2)  = -ni*nj - ni + 1;
    	columnOffsets(3)  = -ni*nj - 1;
    	columnOffsets(4)  = -ni*nj;
    	columnOffsets(5)  = -ni*nj + 1;
    	columnOffsets(6)  = -ni*nj + ni - 1;
    	columnOffsets(7)  = -ni*nj + ni;
    	columnOffsets(8)  = -ni*nj + ni + 1;
    	columnOffsets(9)  = -ni - 1;
    	columnOffsets(10) = -ni;
    	columnOffsets(11) = -ni + 1;
    	columnOffsets(12) = -1;
    	columnOffsets(13) = 0;
    	columnOffsets(14) = 1;
    	columnOffsets(15) = ni - 1;
    	columnOffsets(16) = ni;
    	columnOffsets(17) = ni + 1;
    	columnOffsets(18) = ni*nj - ni - 1;
    	columnOffsets(19) = ni*nj - ni;
    	columnOffsets(20) = ni*nj - ni + 1;
    	columnOffsets(21) = ni*nj - 1;
    	columnOffsets(22) = ni*nj;
    	columnOffsets(23) = ni*nj + 1;
    	columnOffsets(24) = ni*nj + ni - 1;
    	columnOffsets(25) = ni*nj + ni;
    	columnOffsets(26) = ni*nj + ni + 1;
      });
    dev.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev, 0, rows_per_team), [&] (const ordinal_type& loop) {
        const ordinal_type interiorIdx = static_cast<ordinal_type> ( dev.league_rank() ) * rows_per_team + loop;
        if(interiorIdx >= numInterior) { return; }

        ordinal_type i, j, k, rowIdx, rem;
        k = interiorIdx / ((ni - 2)*(nj - 2));
        rem  = interiorIdx % ((ni - 2)*(nj - 2));
        j = rem / (ni - 2);
        i = rem % (ni - 2);
        rowIdx = (k + 1)*nj*ni + (j + 1)*ni + (i + 1);

        const size_type rowOffset = m_A.graph.row_map(rowIdx);
        const value_type* value_ptr = &(m_A.values(rowOffset));
	value_type sum = 0.0;
	Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev, 27), [&] (const ordinal_type& idx, value_type& lclSum) {
	    lclSum += *(value_ptr + idx)*m_x(rowIdx + columnOffsets(idx));
	  }, sum);
	Kokkos::single(Kokkos::PerThread(dev), [&] () {
	    m_y(rowIdx) = beta*m_y(rowIdx) + alpha*sum;
	  });
      });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const exterior1DTag&, const ordinal_type& exteriorIdx) const
  {
    typedef typename YVector::non_const_value_type y_value_type;

    ordinal_type rowIdx = exteriorIdx*(ni - 1);

    const size_type rowOffset = m_A.graph.row_map(rowIdx);
    const ordinal_type row_length = static_cast<ordinal_type> (m_A.graph.row_map(rowIdx + 1) - rowOffset);
    const value_type* value_ptr = &(m_A.values(rowOffset));
    const ordinal_type* column_ptr = &(m_A.graph.entries(rowOffset));
    y_value_type sum = 0;
    for(ordinal_type entryIdx = 0; entryIdx < row_length; ++entryIdx) {
      sum += (*(value_ptr + entryIdx))*m_x(*(column_ptr + entryIdx));
    }
    m_y(rowIdx) = beta*m_y(rowIdx) + alpha*sum;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const exterior2DTag&, const ordinal_type& exteriorIdx) const
  {
    typedef typename YVector::non_const_value_type y_value_type;
    const ordinal_type topFlag = exteriorIdx / (ni + 2*nj - 4);
    const ordinal_type bottomFlag = static_cast<ordinal_type>((exteriorIdx / ni) == 0);

    ordinal_type rowIdx = 0;
    if(bottomFlag == 1) {
      rowIdx = exteriorIdx;
    } else if(topFlag == 1) {
      rowIdx = exteriorIdx - (ni + 2*nj - 4)
        + ni*(nj - 1);
    } else {
      ordinal_type edgeIdx = (exteriorIdx - ni) / 2;
      ordinal_type edgeFlg = (exteriorIdx - ni) % 2;
      rowIdx = (edgeIdx + 1)*ni + edgeFlg*(ni - 1);
    }

    const size_type rowOffset = m_A.graph.row_map(rowIdx);
    const ordinal_type row_length = static_cast<ordinal_type> (m_A.graph.row_map(rowIdx + 1) - rowOffset);
    const value_type* value_ptr = &(m_A.values(rowOffset));
    const ordinal_type* column_ptr = &(m_A.graph.entries(rowOffset));
    y_value_type sum = 0;
    for(ordinal_type entryIdx = 0; entryIdx < row_length; ++entryIdx) {
      sum += (*(value_ptr + entryIdx))*m_x(*(column_ptr + entryIdx));
    }
    m_y(rowIdx) = beta*m_y(rowIdx) + alpha*sum;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const exterior3DTag&, const ordinal_type& exteriorIdx) const
  {
    typedef typename YVector::non_const_value_type y_value_type;
    const ordinal_type topFlag = static_cast<ordinal_type>(numExterior - exteriorIdx - 1 < ni*nj);
    const ordinal_type bottomFlag = static_cast<ordinal_type>(exteriorIdx / (ni*nj) == 0);

    ordinal_type rowIdx = 0;
    if(bottomFlag == 1) {
      rowIdx = exteriorIdx;
    } else if(topFlag == 1) {
      rowIdx = exteriorIdx - ni*nj - 2*(nk - 2)*(nj + ni - 2) + (nk - 1)*ni*nj;
    } else {
      ordinal_type k, rem;
      k = (exteriorIdx - ni*nj) / (2*(ni - 1 + nj - 1));
      rem = (exteriorIdx - ni*nj) % (2*(ni - 1 + nj - 1));
      // ordinal_type frontFlg = static_cast<ordinal_type>(rem < ni);
      // ordinal_type backFlg = static_cast<ordinal_type>(rem - ni - 2*(nj - 1) - 1 > 0);
      if(rem < ni) {
        rowIdx = (k + 1)*ni*nj + rem;
      } else if(rem < ni + 2*(nj - 2)) {
        ordinal_type edgeIdx = (rem - ni) / 2;
        ordinal_type edgeFlg = (rem - ni) % 2;
        if(edgeFlg == 0) {
          rowIdx = (k + 1)*ni*nj + (edgeIdx + 1)*ni;
        } else if(edgeFlg == 1) {
          rowIdx = (k + 1)*ni*nj + (edgeIdx + 2)*ni - 1;
        }
      } else {
        rowIdx = (k + 1)*ni*nj + rem - ni - 2*(nj - 2) + (nj - 1)*ni;
      }
    }

    const size_type rowOffset = m_A.graph.row_map(rowIdx);
    const ordinal_type row_length = static_cast<ordinal_type> (m_A.graph.row_map(rowIdx + 1) - rowOffset);
    const value_type* value_ptr = &(m_A.values(rowOffset));
    const ordinal_type* column_ptr = &(m_A.graph.entries(rowOffset));
    y_value_type sum = 0;
    for(ordinal_type entryIdx = 0; entryIdx < row_length; ++entryIdx) {
      sum += (*(value_ptr + entryIdx))*m_x(*(column_ptr + entryIdx));
    }
    m_y(rowIdx) = beta*m_y(rowIdx) + alpha*sum;
  }
};

template<class execution_space>
int64_t spmv_struct_launch_parameters(int64_t numInterior, int64_t nnz, int nnz_per_row,
                                      int64_t rows_per_thread, int& team_size, int& vector_length) {
  int64_t rows_per_team;

  if(nnz_per_row < 1) nnz_per_row = 1;

  // Determine rows per thread
  if(rows_per_thread < 1) {
    #ifdef KOKKOS_ENABLE_CUDA
    if(std::is_same<Kokkos::Cuda,execution_space>::value)
      rows_per_thread = 1;
    else
    #endif
    {
      if(nnz_per_row < 20 && numInterior*nnz_per_row > 5000000 ) {
        rows_per_thread = 256;
      } else
        rows_per_thread = 64;
    }
  }

  #ifdef KOKKOS_ENABLE_CUDA
  if(team_size < 1) {
    if(std::is_same<Kokkos::Cuda,execution_space>::value)
    { team_size = 256/vector_length; }
    else
    { team_size = 1; }
  }
  #endif

  rows_per_team = rows_per_thread * team_size;

  if(rows_per_team < 0) {
    int64_t nnz_per_team = 4096;
    int64_t conc = execution_space::concurrency();
    while((conc * nnz_per_team * 4 > nnz)&&(nnz_per_team > 256)) nnz_per_team/=2;
    rows_per_team = (nnz_per_team + nnz_per_row - 1)/nnz_per_row;
  }


  return rows_per_team;
}

template<class AMatrix,
         class XVector,
         class YVector,
         int dobeta,
         bool conjugate>
static void
spmv_struct_beta_no_transpose (const int stencil_type,
                               const Kokkos::View<int*, Kokkos::HostSpace>& structure,
                               typename YVector::const_value_type& alpha,
                               const AMatrix& A,
                               const XVector& x,
                               typename YVector::const_value_type& beta,
                               const YVector& y)
{
  typedef typename AMatrix::ordinal_type ordinal_type;
  typedef typename AMatrix::execution_space execution_space;
  if (A.numRows () <= static_cast<ordinal_type> (0)) {
    return;
  }

  int team_size = -1;
  int vector_length = -1;
  int nnzPerRow = -1;
  int64_t rows_per_thread = -1;
  int64_t numInteriorPts = 0;

  if(structure.extent(0) == 1) {
    numInteriorPts = structure(0) - 2;
    vector_length = 1;
  } else if(structure.extent(0) == 2) {
    numInteriorPts = (structure(1) - 2)*(structure(0) - 2);
    if(stencil_type == 1) {
      vector_length = 2;
    } else if(stencil_type == 2) {
      vector_length = 4;
    }
  } else if(structure.extent(0) == 3) {
    numInteriorPts = (structure(2) - 2)*(structure(1) - 2)*(structure(0) - 2);
    if(stencil_type == 1) {
      vector_length = 2;
    } else if(stencil_type == 2) {
      vector_length = 8;
    }
  }

  int64_t rows_per_team = spmv_struct_launch_parameters<execution_space>(numInteriorPts,
                                                                         A.nnz(),
                                                                         nnzPerRow,
                                                                         rows_per_thread,
                                                                         team_size,
                                                                         vector_length);
  int64_t worksets = (numInteriorPts + rows_per_team - 1) / rows_per_team;

  // std::cout << "worksets=" << worksets
  //           << ", rows_per_team=" << rows_per_team
  //           << ", team_size=" << team_size
  //           << ",  vector_length=" << vector_length << std::endl;

  SPMV_Struct_Functor<AMatrix,XVector,YVector,dobeta,conjugate> func(structure,
								     stencil_type,
								     alpha,A,x,beta,y,
								     rows_per_team);

  func.compute(worksets, team_size, vector_length);
}

template<class AMatrix,
         class XVector,
         class YVector,
         int dobeta,
         bool conjugate>
static void
spmv_struct_beta_transpose (const int stencil_type,
                            const Kokkos::View<int*, Kokkos::HostSpace>& structure,
                            typename YVector::const_value_type& alpha,
                            const AMatrix& A,
                            const XVector& x,
                            typename YVector::const_value_type& beta,
                            const YVector& y)
{
  typedef typename AMatrix::ordinal_type ordinal_type;

  if (A.numRows () <= static_cast<ordinal_type> (0)) {
    return;
  }

  // We need to scale y first ("scaling" by zero just means filling
  // with zeros), since the functor works by atomic-adding into y.
  if (dobeta != 1) {
    KokkosBlas::scal (y, beta, y);
  }

  typedef typename AMatrix::size_type size_type;

  // Assuming that no row contains duplicate entries, NNZPerRow
  // cannot be more than the number of columns of the matrix.  Thus,
  // the appropriate type is ordinal_type.
  const ordinal_type NNZPerRow = static_cast<ordinal_type> (A.nnz () / A.numRows ());

  int vector_length = 1;
  while( (static_cast<ordinal_type> (vector_length*2*3) <= NNZPerRow) && (vector_length<32) ) vector_length*=2;

  typedef SPMV_Struct_Transpose_Functor<AMatrix, XVector, YVector, dobeta, conjugate> OpType;

  typename AMatrix::const_ordinal_type nrow = A.numRows();

  OpType op (alpha, A, x, beta, y, RowsPerThread<typename AMatrix::execution_space> (NNZPerRow));

  const int rows_per_thread = RowsPerThread<typename AMatrix::execution_space > (NNZPerRow);
  const int team_size = Kokkos::TeamPolicy<typename AMatrix::execution_space>(rows_per_thread, Kokkos::AUTO, vector_length).team_size_recommended(op, Kokkos::ParallelForTag());
  const int rows_per_team = rows_per_thread * team_size;
  const size_type nteams = (nrow+rows_per_team-1)/rows_per_team;
  Kokkos::parallel_for("KokkosSparse::spmv_struct<Transpose>", Kokkos::TeamPolicy< typename AMatrix::execution_space >
     ( nteams , team_size , vector_length ) , op );

}

template<class AMatrix,
         class XVector,
         class YVector,
         int dobeta>
static void
spmv_struct_beta (const char mode[],
                  const int stencil_type,
                  const Kokkos::View<int*, Kokkos::HostSpace>& structure,
                  typename YVector::const_value_type& alpha,
                  const AMatrix& A,
                  const XVector& x,
                  typename YVector::const_value_type& beta,
                  const YVector& y)
{
  if (mode[0] == NoTranspose[0]) {
    spmv_struct_beta_no_transpose<AMatrix,XVector,YVector,dobeta,false>
      (stencil_type, structure, alpha, A, x, beta, y);
  }
  else if (mode[0] == Conjugate[0]) {
    spmv_struct_beta_no_transpose<AMatrix,XVector,YVector,dobeta,true>
      (stencil_type, structure, alpha, A, x, beta, y);
  }
  else if (mode[0]==Transpose[0]) {
    spmv_struct_beta_transpose<AMatrix,XVector,YVector,dobeta,false>
      (stencil_type, structure, alpha, A, x, beta, y);
  }
  else if(mode[0]==ConjugateTranspose[0]) {
    spmv_struct_beta_transpose<AMatrix,XVector,YVector,dobeta,true>
      (stencil_type, structure, alpha, A, x, beta, y);
  }
  else {
    Kokkos::Impl::throw_runtime_exception("Invalid Transpose Mode for KokkosSparse::spmv_struct()");
  }
}

}
}

#endif // KOKKOSSPARSE_IMPL_SPMV_STRUCT_DEF_HPP_
