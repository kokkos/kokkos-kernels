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

#ifndef KOKKOSSPARSE_IMPL_SPMV_DEF_HPP_
#define KOKKOSSPARSE_IMPL_SPMV_DEF_HPP_

#include "Kokkos_InnerProductSpaceTraits.hpp"
#include "KokkosBlas1_scal.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv_impl_omp.hpp"

namespace KokkosSparse {
namespace Impl {

template<class InputType, class DeviceType>
struct GetCoeffView {
  typedef Kokkos::View<InputType*,Kokkos::LayoutLeft,DeviceType> view_type;
  typedef Kokkos::View<typename view_type::non_const_value_type*,
                       Kokkos::LayoutLeft,DeviceType> non_const_view_type;
  static non_const_view_type get_view(const InputType in, const int size) {
    non_const_view_type aview("CoeffView",size);
    if(size>0)
      Kokkos::deep_copy(aview,in);
    return aview;
  }
};

template<class IT, class IL, class ID, class IM, class IS, class DeviceType>
struct GetCoeffView<Kokkos::View<IT*,IL,ID,IM,IS>,DeviceType> {
  typedef Kokkos::View<IT*,IL,ID,IM,IS> view_type;
  static Kokkos::View<IT*,IL,ID,IM,IS> get_view(const Kokkos::View<IT*,IL,ID,IM,IS>& in, int size) {
    return in;
  }
};


// This TransposeFunctor is functional, but not necessarily performant.
template<class AMatrix,
         class XVector,
         class YVector,
         int dobeta,
         bool conjugate>
struct SPMV_Transpose_Functor {
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

  SPMV_Transpose_Functor (const coefficient_type& alpha_,
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
struct SPMV_Functor {
  typedef typename AMatrix::execution_space            execution_space;
  typedef typename AMatrix::non_const_ordinal_type     ordinal_type;
  typedef typename AMatrix::non_const_value_type       value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type            team_member;
  typedef Kokkos::Details::ArithTraits<value_type>     ATV;

  const value_type alpha;
  AMatrix  m_A;
  XVector m_x;
  const value_type beta;
  YVector m_y;

  const ordinal_type rows_per_team;

  SPMV_Functor (const value_type alpha_,
                const AMatrix m_A_,
                const XVector m_x_,
                const value_type beta_,
                const YVector m_y_,
                const int rows_per_team_) :
     alpha (alpha_), m_A (m_A_), m_x (m_x_),
     beta (beta_), m_y (m_y_),
     rows_per_team (rows_per_team_)
  {
    static_assert (static_cast<int> (XVector::rank) == 1,
                   "XVector must be a rank 1 View.");
    static_assert (static_cast<int> (YVector::rank) == 1,
                   "YVector must be a rank 1 View.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member& dev) const
  {
    typedef typename YVector::non_const_value_type y_value_type;

    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev,0,rows_per_team), [&] (const ordinal_type& loop) {

      const ordinal_type iRow = static_cast<ordinal_type> ( dev.league_rank() ) * rows_per_team + loop;
      if (iRow >= m_A.numRows ()) {
        return;
      }
      const KokkosSparse::SparseRowViewConst<AMatrix> row = m_A.rowConst(iRow);
      const ordinal_type row_length = static_cast<ordinal_type> (row.length);
      y_value_type sum = 0;

      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev,row_length), [&] (const ordinal_type& iEntry, y_value_type& lsum) {
        const value_type val = conjugate ?
                ATV::conj (row.value(iEntry)) :
                row.value(iEntry);
        lsum += val * m_x(row.colidx(iEntry));
      },sum);

      Kokkos::single(Kokkos::PerThread(dev), [&] () {
        sum *= alpha;

        if (dobeta == 0) {
          m_y(iRow) = sum ;
        } else {
          m_y(iRow) = beta * m_y(iRow) + sum;
        }
      });
    });
  }
};

template<class execution_space>
int64_t spmv_launch_parameters(int64_t numRows, int64_t nnz, int64_t& rows_per_thread, int& team_size, int& vector_length) {
  int64_t rows_per_team;
  int64_t nnz_per_row = nnz/numRows;

  if(nnz_per_row < 1) nnz_per_row = 1;

  if(vector_length < 1) {
    vector_length = 1;
#ifdef KOKKOS_ENABLE_CUDA
    if(std::is_same<Kokkos::Cuda,execution_space>::value)
      while(vector_length<32 && vector_length*6 < nnz_per_row) {vector_length*=2;}
#endif
  }

  // Determine rows per thread
  if(rows_per_thread < 1) {
    #ifdef KOKKOS_ENABLE_CUDA
    if(std::is_same<Kokkos::Cuda,execution_space>::value)
      rows_per_thread = 1;
    else
    #endif
    {
      if(nnz_per_row < 20 && nnz > 5000000 ) {
        rows_per_thread = 256;
      } else
        rows_per_thread = 64;
    }
  }

  if(team_size < 1) {
  #ifdef KOKKOS_ENABLE_CUDA
    if(std::is_same<Kokkos::Cuda,execution_space>::value)
    {
      team_size = 256/vector_length;
    }
    else
  #endif
    {
      team_size = 1;
    }
  }

  rows_per_team = rows_per_thread * team_size;

  if(rows_per_team < 0) {
    int64_t nnz_per_team = 4096;
    int64_t conc = execution_space::concurrency();
    while((conc * nnz_per_team * 4> nnz)&&(nnz_per_team>256)) nnz_per_team/=2;
    rows_per_team = (nnz_per_team+nnz_per_row - 1)/nnz_per_row;
  }


  return rows_per_team;
}

template<class AMatrix,
         class XVector,
         class YVector,
         int dobeta,
         bool conjugate>
static void
spmv_beta_no_transpose (typename YVector::const_value_type& alpha,
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

  #ifdef KOKKOS_ENABLE_OPENMP
  if((std::is_same<execution_space,Kokkos::OpenMP>::value) &&
     (std::is_same<typename std::remove_cv<typename AMatrix::value_type>::type,double>::value) &&
     (std::is_same<typename XVector::non_const_value_type,double>::value) &&
     (std::is_same<typename YVector::non_const_value_type,double>::value) &&
     ((int) A.graph.row_block_offsets.extent(0) == (int) omp_get_max_threads()+1) &&
     (((uintptr_t)(const void*)(x.data())%64)==0) && (((uintptr_t)(const void*)(y.data())%64)==0)
     ) {
    spmv_raw_openmp_no_transpose<AMatrix,XVector,YVector>(alpha,A,x,beta,y);
    return;
  }
  #endif
  int team_size = -1;
  int vector_length = -1;
  int64_t rows_per_thread = -1;

  int64_t rows_per_team = spmv_launch_parameters<execution_space>(A.numRows(),A.nnz(),rows_per_thread,team_size,vector_length);
  int64_t worksets = (y.extent(0)+rows_per_team-1)/rows_per_team;

  SPMV_Functor<AMatrix,XVector,YVector,dobeta,conjugate> func (alpha,A,x,beta,y,rows_per_team);

  if(A.nnz()>10000000) {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic> > policy(1,1);
    if(team_size<0)
      policy = Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic> >(worksets,Kokkos::AUTO,vector_length);
    else
      policy = Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic> >(worksets,team_size,vector_length);
    Kokkos::parallel_for("KokkosSparse::spmv<NoTranspose,Dynamic>",policy,func);
  } else {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static> > policy(1,1);
    if(team_size<0)
      policy = Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,Kokkos::AUTO,vector_length);
    else
      policy = Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,team_size,vector_length);
    Kokkos::parallel_for("KokkosSparse::spmv<NoTranspose,Static>",policy,func);
  }
}

template<class AMatrix,
         class XVector,
         class YVector,
         int dobeta,
         bool conjugate>
static void
spmv_beta_transpose (typename YVector::const_value_type& alpha,
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

  typedef SPMV_Transpose_Functor<AMatrix, XVector, YVector, dobeta, conjugate> OpType;

  typename AMatrix::const_ordinal_type nrow = A.numRows();

  OpType op (alpha, A, x, beta, y, RowsPerThread<typename AMatrix::execution_space> (NNZPerRow));

  const int rows_per_thread = RowsPerThread<typename AMatrix::execution_space > (NNZPerRow);
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
  const int team_size = Kokkos::TeamPolicy<typename AMatrix::execution_space>::team_size_recommended (op, vector_length);
#else
  const int team_size = Kokkos::TeamPolicy<typename AMatrix::execution_space>(rows_per_thread, Kokkos::AUTO, vector_length).team_size_recommended(op, Kokkos::ParallelForTag());
#endif
  const int rows_per_team = rows_per_thread * team_size;
  const size_type nteams = (nrow+rows_per_team-1)/rows_per_team;
  Kokkos::parallel_for("KokkosSparse::spmv<Transpose>", Kokkos::TeamPolicy< typename AMatrix::execution_space >
     ( nteams , team_size , vector_length ) , op );

}

template<class AMatrix,
         class XVector,
         class YVector,
         int dobeta>
static void
spmv_beta (const char mode[],
                 typename YVector::const_value_type& alpha,
                 const AMatrix& A,
                 const XVector& x,
                 typename YVector::const_value_type& beta,
                 const YVector& y)
{
  if (mode[0] == NoTranspose[0]) {
    spmv_beta_no_transpose<AMatrix,XVector,YVector,dobeta,false>
      (alpha,A,x,beta,y);
  }
  else if (mode[0] == Conjugate[0]) {
    spmv_beta_no_transpose<AMatrix,XVector,YVector,dobeta,true>
      (alpha,A,x,beta,y);
  }
  else if (mode[0]==Transpose[0]) {
    spmv_beta_transpose<AMatrix,XVector,YVector,dobeta,false>
      (alpha,A,x,beta,y);
  }
  else if(mode[0]==ConjugateTranspose[0]) {
    spmv_beta_transpose<AMatrix,XVector,YVector,dobeta,true>
      (alpha,A,x,beta,y);
  }
  else {
    Kokkos::Impl::throw_runtime_exception("Invalid Transpose Mode for KokkosSparse::spmv()");
  }
}


// Functor for implementing transpose and conjugate transpose sparse
// matrix-vector multiply with multivector (2-D View) input and
// output.  This functor works, but is not necessarily performant.
template<class AMatrix,
         class XVector,
         class YVector,
         int doalpha,
         int dobeta,
         bool conjugate>
struct SPMV_MV_Transpose_Functor {
  typedef typename AMatrix::execution_space            execution_space;
  typedef typename AMatrix::non_const_ordinal_type     ordinal_type;
  typedef typename AMatrix::non_const_value_type       A_value_type;
  typedef typename YVector::non_const_value_type       y_value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type            team_member;
  typedef typename YVector::non_const_value_type       coefficient_type;

  const coefficient_type alpha;
  AMatrix m_A;
  XVector m_x;
  const coefficient_type beta;
  YVector m_y;

  const ordinal_type n;
  const ordinal_type rows_per_thread;

  SPMV_MV_Transpose_Functor (const coefficient_type& alpha_,
                             const AMatrix& m_A_,
                             const XVector& m_x_,
                             const coefficient_type& beta_,
                             const YVector& m_y_,
                             const ordinal_type rows_per_thread_) :
    alpha (alpha_),
    m_A (m_A_), m_x (m_x_), beta (beta_), m_y (m_y_), n (m_x_.extent(1)),
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
           iEntry < static_cast<ordinal_type> (row_length);
           iEntry += static_cast<ordinal_type> (blockDim.x))
#else
      for (ordinal_type iEntry = 0;
           iEntry < row_length;
           iEntry ++)
#endif
      {
        const A_value_type val = conjugate ?
          Kokkos::Details::ArithTraits<A_value_type>::conj (row.value(iEntry)) :
          row.value(iEntry);
        const ordinal_type ind = row.colidx(iEntry);

        if (doalpha != 1) {
          #ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
          #pragma unroll
          #endif
          for (ordinal_type k = 0; k < n; ++k) {
            Kokkos::atomic_add (&m_y(ind,k),
                                static_cast<y_value_type> (alpha * val * m_x(iRow, k)));
          }
        } else {
          #ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
          #pragma unroll
          #endif
          for (ordinal_type k = 0; k < n; ++k) {
            Kokkos::atomic_add (&m_y(ind,k),
                                static_cast<y_value_type> (val * m_x(iRow, k)));
          }
        }
      }
    }
  }
};




template<class AMatrix,
         class XVector,
         class YVector,
         int doalpha,
         int dobeta,
         bool conjugate>
struct SPMV_MV_Functor {
  typedef typename AMatrix::execution_space              execution_space;
  typedef typename AMatrix::non_const_ordinal_type       ordinal_type;
  typedef typename AMatrix::non_const_value_type         value_type;
  typedef typename Kokkos::TeamPolicy<execution_space>   team_policy;
  typedef typename team_policy::member_type              team_member;
  typedef Kokkos::Details::ArithTraits<value_type>       ATV;

  const value_type alpha;
  AMatrix m_A;
  XVector m_x;
  const value_type beta;
  YVector m_y;

  const ordinal_type rows_per_team;
  const ordinal_type numVecs;
  const ordinal_type vecRemainder;

  const value_type value_type_zero = ATV::zero();

  struct value16 {
    value_type x0, x1, x2, x3, x4, x5, x6, x7;
    value_type x8, x9, x10, x11, x12, x13, x14, x15;

    KOKKOS_INLINE_FUNCTION
    void operator += (const value16 input) {
      x0  += input.x0;
      x1  += input.x1;
      x2  += input.x2;
      x3  += input.x3;
      x4  += input.x4;
      x5  += input.x5;
      x6  += input.x6;
      x7  += input.x7;
      x8  += input.x8;
      x9  += input.x9;
      x10 += input.x10;
      x11 += input.x11;
      x12 += input.x12;
      x13 += input.x13;
      x14 += input.x14;
      x15 += input.x15;
    }

    KOKKOS_INLINE_FUNCTION
    void operator += (volatile const value16 input) volatile {
      x0  += input.x0;
      x1  += input.x1;
      x2  += input.x2;
      x3  += input.x3;
      x4  += input.x4;
      x5  += input.x5;
      x6  += input.x6;
      x7  += input.x7;
      x8  += input.x8;
      x9  += input.x9;
      x10 += input.x10;
      x11 += input.x11;
      x12 += input.x12;
      x13 += input.x13;
      x14 += input.x14;
      x15 += input.x15;
    }

    KOKKOS_INLINE_FUNCTION
    value_type get(const ordinal_type vecIdx) {
      switch (vecIdx) {
      case 0:
        return x0;
      case 1:
        return x1;
      case 2:
        return x2;
      case 3:
        return x3;
      case 4:
        return x4;
      case 5:
        return x5;
      case 6:
        return x6;
      case 7:
        return x7;
      case 8:
        return x8;
      case 9:
        return x9;
      case 10:
        return x10;
      case 11:
        return x11;
      case 12:
        return x12;
      case 13:
        return x13;
      case 14:
        return x14;
      case 15:
        return x15;
      }

      return 0;
    }
  };

  struct value8 {
    value_type x0, x1, x2, x3, x4, x5, x6, x7;

    KOKKOS_INLINE_FUNCTION
    void operator += (const value8 input) {
      x0 += input.x0;
      x1 += input.x1;
      x2 += input.x2;
      x3 += input.x3;
      x4 += input.x4;
      x5 += input.x5;
      x6 += input.x6;
      x7 += input.x7;
    }

    KOKKOS_INLINE_FUNCTION
    void operator += (volatile const value8 input) volatile {
      x0 += input.x0;
      x1 += input.x1;
      x2 += input.x2;
      x3 += input.x3;
      x4 += input.x4;
      x5 += input.x5;
      x6 += input.x6;
      x7 += input.x7;
    }

    KOKKOS_INLINE_FUNCTION
    value_type get(const ordinal_type vecIdx) {
      switch (vecIdx) {
      case 0:
        return x0;
      case 1:
        return x1;
      case 2:
        return x2;
      case 3:
        return x3;
      case 4:
        return x4;
      case 5:
        return x5;
      case 6:
        return x6;
      case 7:
        return x7;
      }

      return 0;
    }
  };

  struct value4 {
    value_type x0, x1, x2, x3;

    KOKKOS_INLINE_FUNCTION
    void operator += (const value4 input) {
      x0 += input.x0;
      x1 += input.x1;
      x2 += input.x2;
      x3 += input.x3;
    }

    KOKKOS_INLINE_FUNCTION
    void operator += (volatile const value4 input) volatile {
      x0 += input.x0;
      x1 += input.x1;
      x2 += input.x2;
      x3 += input.x3;
    }

    KOKKOS_INLINE_FUNCTION
    value_type get(const ordinal_type vecIdx) {
      switch (vecIdx) {
      case 0:
        return x0;
      case 1:
        return x1;
      case 2:
        return x2;
      case 3:
        return x3;
      }

      return 0;
    }
  };

  struct value3 {
    value_type x0, x1, x2;

    KOKKOS_INLINE_FUNCTION
    void operator += (const value3 input) {
      x0 += input.x0;
      x1 += input.x1;
      x2 += input.x2;
    }

    KOKKOS_INLINE_FUNCTION
    void operator += (volatile const value3 input) volatile {
      x0 += input.x0;
      x1 += input.x1;
      x2 += input.x2;
    }

    KOKKOS_INLINE_FUNCTION
    value_type get(const ordinal_type vecIdx) {
      switch (vecIdx) {
      case 0:
        return x0;
      case 1:
        return x1;
      case 2:
        return x2;
      }

      return 0;
    }
  };

  struct value2 {
    value_type x0, x1;

    KOKKOS_INLINE_FUNCTION
    void operator += (const value2 input) {
      x0 += input.x0;
      x1 += input.x1;
    }

    KOKKOS_INLINE_FUNCTION
    void operator += (volatile const value2 input) volatile {
      x0 += input.x0;
      x1 += input.x1;
    }

    KOKKOS_INLINE_FUNCTION
    value_type get(const ordinal_type vecIdx) {
      switch (vecIdx) {
      case 0:
        return x0;
      case 1:
        return x1;
      }

      return 0;
    }
  };

  SPMV_MV_Functor (const value_type alpha_,
                   const AMatrix m_A_,
                   const XVector m_x_,
                   const value_type beta_,
                   const YVector m_y_,
                   const ordinal_type rows_per_team_) :
    alpha (alpha_), m_A (m_A_), m_x (m_x_),
    beta (beta_), m_y (m_y_),
    rows_per_team (rows_per_team_),
    numVecs (m_x_.extent(1)),
    vecRemainder (m_x_.extent(1) % 16)
  {
    static_assert (static_cast<int> (XVector::rank) == 2,
                   "XVector must be a rank 2 View.");
    static_assert (static_cast<int> (YVector::rank) == 2,
                   "YVector must be a rank 2 View.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member& dev) const
  {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev,0,rows_per_team), [&] (const ordinal_type& loop) {
        const ordinal_type iRow = static_cast<ordinal_type>(dev.league_rank()) * rows_per_team + loop;
        if (iRow >= m_A.numRows ()) {
          return;
        }
        const KokkosSparse::SparseRowViewConst<AMatrix> row = m_A.rowConst(iRow);
        const ordinal_type row_length = static_cast<ordinal_type> (row.length);
        ordinal_type remainder = vecRemainder;

	for(ordinal_type vecOffset = 0; vecOffset < numVecs - 15; vecOffset += 16) {
	  value16 sum = {};
	  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev, row_length),
				  [&] (const ordinal_type& entryIdx, value16& lsum) {

				    // Extract data from current row
				    const value_type val = alpha*(conjugate ? ATV::conj(row.value(entryIdx)) : row.value(entryIdx));
				    const ordinal_type colIdx = row.colidx(entryIdx);

				    // Perform dot product accros lhs columns
				    lsum.x0  += m_x(colIdx, vecOffset +  0)*val;
				    lsum.x1  += m_x(colIdx, vecOffset +  1)*val;
				    lsum.x2  += m_x(colIdx, vecOffset +  2)*val;
				    lsum.x3  += m_x(colIdx, vecOffset +  3)*val;
				    lsum.x4  += m_x(colIdx, vecOffset +  4)*val;
				    lsum.x5  += m_x(colIdx, vecOffset +  5)*val;
				    lsum.x6  += m_x(colIdx, vecOffset +  6)*val;
				    lsum.x7  += m_x(colIdx, vecOffset +  7)*val;
				    lsum.x8  += m_x(colIdx, vecOffset +  8)*val;
				    lsum.x9  += m_x(colIdx, vecOffset +  9)*val;
				    lsum.x10 += m_x(colIdx, vecOffset + 10)*val;
				    lsum.x11 += m_x(colIdx, vecOffset + 11)*val;
				    lsum.x12 += m_x(colIdx, vecOffset + 12)*val;
				    lsum.x13 += m_x(colIdx, vecOffset + 13)*val;
				    lsum.x14 += m_x(colIdx, vecOffset + 14)*val;
				    lsum.x15 += m_x(colIdx, vecOffset + 15)*val;
				  }, sum);

          Kokkos::parallel_for(Kokkos::ThreadVectorRange(dev, 16), [&] (const ordinal_type& vecIdx) {
              m_y(iRow, vecOffset + vecIdx) = beta*m_y(iRow, vecOffset + vecIdx) + sum.get(vecIdx);
	  });
	} // Loop over numVecs

        if(remainder > 7) {
          value8 sum = {};
          Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev, row_length),
                                  [&] (const ordinal_type& entryIdx, value8& lsum) {

                                    // Extract data from current row
                                    const value_type val = alpha*(conjugate ? ATV::conj(row.value(entryIdx)) : row.value(entryIdx));
                                    const ordinal_type colIdx = row.colidx(entryIdx);

                                    // Perform dot product accros lhs columns
                                    lsum.x0 += m_x(colIdx, numVecs - remainder)*val;
                                    lsum.x1 += m_x(colIdx, numVecs - remainder + 1)*val;
                                    lsum.x2 += m_x(colIdx, numVecs - remainder + 2)*val;
                                    lsum.x3 += m_x(colIdx, numVecs - remainder + 3)*val;
                                    lsum.x4 += m_x(colIdx, numVecs - remainder + 4)*val;
                                    lsum.x5 += m_x(colIdx, numVecs - remainder + 5)*val;
                                    lsum.x6 += m_x(colIdx, numVecs - remainder + 6)*val;
                                    lsum.x7 += m_x(colIdx, numVecs - remainder + 7)*val;
                                  }, sum);

          Kokkos::parallel_for(Kokkos::ThreadVectorRange(dev, 8), [&] (const ordinal_type& vecIdx) {
              m_y(iRow, numVecs - remainder + vecIdx) =
                beta*m_y(iRow, numVecs - remainder + vecIdx) + sum.get(vecIdx);
            });
          remainder -= 8;
        }

        if(remainder > 3) {
          value4 sum = {};
          Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev, row_length),
                                  [&] (const ordinal_type& entryIdx, value4& lsum) {

                                    // Extract data from current row
                                    const value_type val = alpha*(conjugate ? ATV::conj(row.value(entryIdx)) : row.value(entryIdx));
                                    const ordinal_type colIdx = row.colidx(entryIdx);

                                    // Perform dot product accros lhs columns
                                    lsum.x0 += m_x(colIdx, numVecs - remainder)*val;
                                    lsum.x1 += m_x(colIdx, numVecs - remainder + 1)*val;
                                    lsum.x2 += m_x(colIdx, numVecs - remainder + 2)*val;
                                    lsum.x3 += m_x(colIdx, numVecs - remainder + 3)*val;
                                  }, sum);

          Kokkos::parallel_for(Kokkos::ThreadVectorRange(dev, 4), [&] (const ordinal_type& vecIdx) {
              m_y(iRow, numVecs - remainder + vecIdx) =
                beta*m_y(iRow, numVecs - remainder + vecIdx) + sum.get(vecIdx);
            });
          remainder -= 4;
        }

        switch(remainder > 3 ? remainder - 4 : remainder) {
        case 3:
          {
            value3 sum = {};
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev, row_length),
                                    [&] (const ordinal_type& entryIdx, value3& lsum) {

                                      // Extract data from current row
                                      const value_type val = alpha*(conjugate ? ATV::conj(row.value(entryIdx)) : row.value(entryIdx));
                                      const ordinal_type colIdx = row.colidx(entryIdx);

                                      // Perform dot product accros lhs columns
                                      lsum.x0 += m_x(colIdx, numVecs - 3)*val;
                                      lsum.x1 += m_x(colIdx, numVecs - 2)*val;
                                      lsum.x2 += m_x(colIdx, numVecs - 1)*val;
                                    }, sum);

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(dev, 3), [&] (const ordinal_type& vecIdx) {
                m_y(iRow, numVecs - 3 + vecIdx) =
                  beta*m_y(iRow, numVecs - 3 + vecIdx) + sum.get(vecIdx);
              });
          }
          break;

        case 2:
          {
            value2 sum = {};
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev, row_length),
                                    [&] (const ordinal_type& entryIdx, value2& lsum) {

                                      // Extract data from current row
                                      const value_type val = alpha*(conjugate ? ATV::conj(row.value(entryIdx)) : row.value(entryIdx));
                                      const ordinal_type colIdx = row.colidx(entryIdx);

                                      // Perform dot product accros lhs columns
                                      lsum.x0 += m_x(colIdx, numVecs - 2)*val;
                                      lsum.x1 += m_x(colIdx, numVecs - 1)*val;
                                    }, sum);

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(dev, 2), [&] (const ordinal_type& vecIdx) {
                m_y(iRow, numVecs - 2 + vecIdx) =
                  beta*m_y(iRow, numVecs - 2 + vecIdx) + sum.get(vecIdx);
              });
          }
          break;

        case 1:
          {
            value_type sum = value_type_zero;
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev, row_length),
                                    [&] (const ordinal_type& entryIdx, value_type& lsum) {

                                      // Extract data from current row
                                      const value_type val = alpha*(conjugate ? ATV::conj(row.value(entryIdx)) : row.value(entryIdx));
                                      const ordinal_type colIdx = row.colidx(entryIdx);

                                      // Perform dot product accros lhs columns
                                      lsum += m_x(colIdx, numVecs - 1)*val;
                                    }, sum);
            Kokkos::single(Kokkos::PerThread(dev), [&] () {
                m_y(iRow, numVecs - 1) = beta*m_y(iRow, numVecs - 1) + sum;
              });
          }
          break;
        } // switch

      }); // TeamThreadRange
  }
};


template<class AMatrix,
         class XVector,
         class YVector,
         int doalpha,
         int dobeta,
         bool conjugate>
struct SPMV_MV_OpenMP_Functor {
  typedef typename AMatrix::execution_space              execution_space;
  typedef typename AMatrix::non_const_ordinal_type       ordinal_type;
  typedef typename AMatrix::non_const_value_type         value_type;
  typedef typename Kokkos::TeamPolicy<execution_space>   team_policy;
  typedef typename team_policy::member_type              team_member;
  typedef Kokkos::Details::ArithTraits<value_type>       ATV;

  const value_type alpha;
  AMatrix  m_A;
  XVector m_x;
  const value_type beta;
  YVector m_y;

  const ordinal_type rows_per_thread;
  const ordinal_type numVecs;
  const ordinal_type vecRemainder;
  ordinal_type rows_per_team;

  const value_type value_type_zero = ATV::zero();

  SPMV_MV_OpenMP_Functor (const value_type alpha_,
			  const AMatrix m_A_,
			  const XVector m_x_,
			  const value_type beta_,
			  const YVector m_y_,
			  const int rows_per_thread_) :
    alpha (alpha_), m_A (m_A_), m_x (m_x_),
    beta (beta_), m_y (m_y_),
    rows_per_thread (static_cast<ordinal_type>(rows_per_thread_)),
    numVecs (m_x_.extent(1)),
    vecRemainder (m_x_.extent(1) % 16)
  {
    static_assert (static_cast<int> (XVector::rank) == 2,
                   "XVector must be a rank 2 View.");
    static_assert (static_cast<int> (YVector::rank) == 2,
                   "YVector must be a rank 2 View.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member& dev) const
  {
    for(ordinal_type loop = 0; loop < rows_per_thread; ++loop) {
      const ordinal_type iRow = static_cast<ordinal_type>(dev.league_rank()*dev.team_size() + dev.team_rank())*rows_per_thread + loop;
      if (iRow >= m_A.numRows ()) {
	return;
      }
      ordinal_type remainder = vecRemainder;

      // Split multivector into chunks of 16 vectors
      for(ordinal_type vecOffset = 0; vecOffset < numVecs - 15; vecOffset += 16) {
	strip_mine<16>(iRow, vecOffset);
      } // Loop over numVecs

      switch(remainder) {
      case 15:
	strip_mine<15>(iRow, numVecs - 15);
	break;

      case 14:
	strip_mine<14>(iRow, numVecs - 14);
	break;

      case 13:
	strip_mine<13>(iRow, numVecs - 13);
	break;

      case 12:
	strip_mine<12>(iRow, numVecs - 12);
	break;

      case 11:
	strip_mine<11>(iRow, numVecs - 11);
	break;

      case 10:
	strip_mine<10>(iRow, numVecs - 10);
	break;

      case 9:
	strip_mine<9>(iRow, numVecs - 9);
	break;

      case 8:
	strip_mine<8>(iRow, numVecs - 8);
	break;

      case 7:
	strip_mine<7>(iRow, numVecs - 7);
	break;

      case 6:
	strip_mine<6>(iRow, numVecs - 6);
	break;

      case 5:
	strip_mine<5>(iRow, numVecs - 5);
	break;

      case 4:
	strip_mine<4>(iRow, numVecs - 4);
	break;

      case 3:
	strip_mine<3>(iRow, numVecs - 3);
	break;

      case 2:
	strip_mine<2>(iRow, numVecs - 2);
	break;

      case 1:
	{
	  const KokkosSparse::SparseRowViewConst<AMatrix> row = m_A.rowConst(iRow);
	  const ordinal_type row_length = static_cast<ordinal_type> (row.length);
	  value_type sum = value_type_zero;
	  for(int entryIdx = 0; entryIdx < row_length; ++entryIdx) {
	    // Extract data from current row
	    const value_type val = alpha*(conjugate ? ATV::conj(row.value(entryIdx)) : row.value(entryIdx));
	    const ordinal_type colIdx = row.colidx(entryIdx);
	    sum += m_x(colIdx, numVecs - 1)*val;
	  }

	  m_y(iRow, numVecs - remainder) = beta*m_y(iRow, numVecs - 1) + sum;
	}
	break;
      } // switch
    } // Loop over rows in chunk
  } // operator()

  template <int unroll>
  KOKKOS_INLINE_FUNCTION
  void strip_mine(const ordinal_type iRow,
		  const ordinal_type vecOffset) const {

    value_type sum[unroll];

#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
    for(int vecIdx = 0; vecIdx < unroll; ++vecIdx) {
      sum[vecIdx] = value_type_zero;
    }

    const KokkosSparse::SparseRowViewConst<AMatrix> row = m_A.rowConst(iRow);

#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
#ifdef KOKKOS_ENABLE_PRAGMA_LOOPCOUNT
#pragma loop count(15)
#endif
    for(int entryIdx = 0; entryIdx < row.length; ++entryIdx) {
      // Extract data from current row
      const value_type val = (conjugate ? ATV::conj(row.value(entryIdx)) : row.value(entryIdx));
      const ordinal_type colIdx = row.colidx(entryIdx);

#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for(int vecIdx = 0; vecIdx < unroll; ++vecIdx) {
	// Perform dot product accros lhs columns
	sum[vecIdx] += m_x(colIdx, vecOffset + vecIdx)*val;
      }
    }

    if(doalpha == -1) {
      for(int vecIdx = 0; vecIdx < unroll; ++vecIdx) {
	value_type sumt = sum[vecIdx];
	sum[vecIdx] = -sumt;
      }
    } else if(doalpha*doalpha != 1) {
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for(int vecIdx = 0; vecIdx < unroll; ++vecIdx) {
	sum[vecIdx] *= alpha;
      }
    }

    if(dobeta == 0) {
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (ordinal_type vecIdx = 0; vecIdx < unroll; ++vecIdx) {
	m_y(iRow, vecOffset + vecIdx) = sum[vecIdx];
      }
    } else if(dobeta == 1) {
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (ordinal_type vecIdx = 0; vecIdx < unroll; ++vecIdx) {
	m_y(iRow, vecOffset + vecIdx) += sum[vecIdx];
      }
    } else if(dobeta == -1) {
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (ordinal_type vecIdx = 0; vecIdx < unroll; ++vecIdx) {
	m_y(iRow, vecOffset + vecIdx) -= sum[vecIdx];
      }
    } else {
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (ordinal_type vecIdx = 0; vecIdx < unroll; ++vecIdx) {
	m_y(iRow, vecOffset + vecIdx) = beta*m_y(iRow, vecOffset + vecIdx) + sum[vecIdx];
      }
    }
  } // strip_mine()

};  // SPMV_MV_OpenMP_Functor


template<class AMatrix,
         class XVector,
         class YVector,
         int doalpha,
         int dobeta,
         bool conjugate>
static void
spmv_alpha_beta_mv_no_transpose (const typename YVector::non_const_value_type& alpha,
                                 const AMatrix& A,
                                 const XVector& x,
                                 const typename YVector::non_const_value_type& beta,
                                 const YVector& y)
{
  typedef typename AMatrix::ordinal_type ordinal_type;

  if (A.numRows () <= static_cast<ordinal_type> (0)) {
    return;
  }
  if (doalpha == 0) {
    if (dobeta != 1) {
      KokkosBlas::scal (y, beta, y);
    }
    return;
  }
  else {
    typedef typename AMatrix::size_type size_type;

    // Assuming that no row contains duplicate entries, NNZPerRow
    // cannot be more than the number of columns of the matrix.  Thus,
    // the appropriate type is ordinal_type.
    const ordinal_type numRows = A.numRows();

#ifndef KOKKOS_FAST_COMPILE // This uses templated functions on doalpha and dobeta and will produce 16 kernels

#ifdef KOKKOS_ENABLE_OPENMP
    if(std::is_same<Kokkos::OpenMP, typename AMatrix::execution_space>::value) {
      int vector_length = 1;
      const ordinal_type NNZPerRow = static_cast<ordinal_type> (A.nnz () / numRows);
      while( (static_cast<ordinal_type> (vector_length*2*3) <= NNZPerRow) && (vector_length<8) ) vector_length*=2;
      const int rows_per_thread = RowsPerThread<typename AMatrix::execution_space>(NNZPerRow);

      SPMV_MV_OpenMP_Functor<AMatrix, XVector, YVector, doalpha, dobeta, conjugate>
	op (alpha, A, x, beta, y, rows_per_thread);

      const int team_size = Kokkos::TeamPolicy<typename AMatrix::execution_space >::team_size_recommended(op, vector_length);
      const int rows_per_team = rows_per_thread*team_size;
      const size_type worksets = (numRows + rows_per_team - 1) / rows_per_team;

      Kokkos::parallel_for("KokkosSparse::spmv_mv<MV,NoTranspose>", Kokkos::TeamPolicy<typename AMatrix::execution_space >
			   ( worksets , team_size , vector_length ) , op );
    } else
#endif
    {
      int team_size = -1;
      int vector_length = -1;
      int64_t rows_per_thread = -1;
      const int64_t rows_per_team = spmv_launch_parameters<typename AMatrix::execution_space>(numRows,
											      A.nnz(),
											      rows_per_thread,
											      team_size,
											      vector_length);
      const size_type worksets = (numRows + rows_per_team - 1)/rows_per_team;

      SPMV_MV_Functor<AMatrix, XVector, YVector, doalpha, dobeta, conjugate>
	op (alpha, A, x, beta, y, rows_per_team);
      if(team_size < 0) {
	Kokkos::parallel_for("KokkosSparse::spmv_mv<MV,NoTranspose>", Kokkos::TeamPolicy< typename AMatrix::execution_space >
			     ( worksets , Kokkos::AUTO , vector_length ) , op );
      } else {
	Kokkos::parallel_for("KokkosSparse::spmv_mv<MV,NoTranspose>", Kokkos::TeamPolicy< typename AMatrix::execution_space >
			     ( worksets , team_size , vector_length ) , op );
      }
    }

#else // KOKKOS_FAST_COMPILE this will only instantiate one Kernel for alpha/beta
    const size_type worksets = (numRows + rows_per_team - 1)/rows_per_team;

    typedef SPMV_MV_Functor<AMatrix, XVector, YVector, 2, 2, conjugate> OpType;
    OpType op (alpha, A, x, beta, y, rows_per_team);
    if(team_size < 0) {
      Kokkos::parallel_for("KokkosSparse::spmv_mv<MV,NoTranspose>",  Kokkos::TeamPolicy< typename AMatrix::execution_space >
                           ( worksets , Kokkos::AUTO , vector_length ) , op );
    } else {
      Kokkos::parallel_for("KokkosSparse::spmv_mv<MV,NoTranspose>",  Kokkos::TeamPolicy< typename AMatrix::execution_space >
                           ( worksets , team_size , vector_length ) , op );
    }

#endif // KOKKOS_FAST_COMPILE
  }
}

template<class AMatrix,
         class XVector,
         class YVector,
         int doalpha,
         int dobeta,
         bool conjugate>
static void
spmv_alpha_beta_mv_transpose (const typename YVector::non_const_value_type& alpha,
                              const AMatrix& A,
                              const XVector& x,
                              const typename YVector::non_const_value_type& beta,
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

  if (doalpha != 0) {
    typedef typename AMatrix::size_type size_type;

    // Assuming that no row contains duplicate entries, NNZPerRow
    // cannot be more than the number of columns of the matrix.  Thus,
    // the appropriate type is ordinal_type.
    const ordinal_type NNZPerRow = static_cast<ordinal_type> (A.nnz () / A.numRows ());

    int vector_length = 1;
    while( (static_cast<ordinal_type> (vector_length*2*3) <= NNZPerRow) && (vector_length<8) ) vector_length*=2;

#ifndef KOKKOS_FAST_COMPILE // This uses templated functions on doalpha and dobeta and will produce 16 kernels

    typedef SPMV_MV_Transpose_Functor<AMatrix, XVector, YVector,
      doalpha, dobeta, conjugate> OpType;
    OpType op (alpha, A, x, beta, y, RowsPerThread<typename AMatrix::execution_space> (NNZPerRow));

    typename AMatrix::const_ordinal_type nrow = A.numRows();

    // FIXME (mfh 07 Jun 2016) Shouldn't we use ordinal_type here
    // instead of int?  For example, if the number of threads is 1,
    // then this is just the number of rows.  Ditto for rows_per_team.
    // team_size is a hardware resource thing so it might legitimately
    // be int.
    const int rows_per_thread = RowsPerThread<typename AMatrix::execution_space >(NNZPerRow);
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
    const int team_size = Kokkos::TeamPolicy< typename AMatrix::execution_space >::team_size_recommended(op,vector_length);
#else
  const int team_size = Kokkos::TeamPolicy<typename AMatrix::execution_space>(rows_per_thread, Kokkos::AUTO, vector_length).team_size_recommended(op, Kokkos::ParallelForTag());
#endif
    const int rows_per_team = rows_per_thread * team_size;
    const size_type nteams = (nrow+rows_per_team-1)/rows_per_team;
    Kokkos::parallel_for ("KokkosSparse::spmv<MV,Transpose>",  Kokkos::TeamPolicy< typename AMatrix::execution_space >
       ( nteams , team_size , vector_length ) , op );

#else // KOKKOS_FAST_COMPILE this will only instantiate one Kernel for alpha/beta

    typedef SPMV_MV_Transpose_Functor<AMatrix, XVector, YVector,
      2, 2, conjugate, SizeType> OpType;

    typename AMatrix::const_ordinal_type nrow = A.numRows();

    OpType op (alpha, A, x, beta, y, RowsPerThread<typename AMatrix::execution_space> (NNZPerRow));

    // FIXME (mfh 07 Jun 2016) Shouldn't we use ordinal_type here
    // instead of int?  For example, if the number of threads is 1,
    // then this is just the number of rows.  Ditto for rows_per_team.
    // team_size is a hardware resource thing so it might legitimately
    // be int.
    const int rows_per_thread = RowsPerThread<typename AMatrix::execution_space >(NNZPerRow);
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
    const int team_size = Kokkos::TeamPolicy< typename AMatrix::execution_space >::team_size_recommended(op,vector_length);
#else
  const int team_size = Kokkos::TeamPolicy<typename AMatrix::execution_space>(rows_per_thread, Kokkos::AUTO, vector_length).team_size_recommended(op, Kokkos::ParallelForTag());
#endif
    const int rows_per_team = rows_per_thread * team_size;
    const size_type nteams = (nrow+rows_per_team-1)/rows_per_team;
    Kokkos::parallel_for("KokkosSparse::spmv<MV,Transpose>",  Kokkos::TeamPolicy< typename AMatrix::execution_space >
       ( nteams , team_size , vector_length ) , op );

#endif // KOKKOS_FAST_COMPILE
  }
}

template<class AMatrix,
         class XVector,
         class YVector,
         int doalpha,
         int dobeta>
static void
spmv_alpha_beta_mv (const char mode[],
                    const typename YVector::non_const_value_type& alpha,
                    const AMatrix& A,
                    const XVector& x,
                    const typename YVector::non_const_value_type& beta,
                    const YVector& y)
{
  if (mode[0] == NoTranspose[0]) {
    spmv_alpha_beta_mv_no_transpose<AMatrix, XVector, YVector, doalpha, dobeta, false> (alpha, A, x, beta, y);
  }
  else if (mode[0] == Conjugate[0]) {
    spmv_alpha_beta_mv_no_transpose<AMatrix, XVector, YVector, doalpha, dobeta, true> (alpha, A, x, beta, y);
  }
  else if (mode[0] == Transpose[0]) {
    spmv_alpha_beta_mv_transpose<AMatrix, XVector, YVector, doalpha, dobeta, false> (alpha, A, x, beta, y);
  }
  else if (mode[0] == ConjugateTranspose[0]) {
    spmv_alpha_beta_mv_transpose<AMatrix, XVector, YVector, doalpha, dobeta, true> (alpha, A, x, beta, y);
  }
  else {
    Kokkos::Impl::throw_runtime_exception ("Invalid Transpose Mode for KokkosSparse::spmv()");
  }
}

template<class AMatrix,
         class XVector,
         class YVector,
         int doalpha>
void
spmv_alpha_mv (const char mode[],
               const typename YVector::non_const_value_type& alpha,
               const AMatrix& A,
               const XVector& x,
               const typename YVector::non_const_value_type& beta,
               const YVector& y)
{
  typedef typename YVector::non_const_value_type coefficient_type;
  typedef Kokkos::Details::ArithTraits<coefficient_type> KAT;

  if (beta == KAT::zero ()) {
    spmv_alpha_beta_mv<AMatrix, XVector, YVector, doalpha, 0> (mode, alpha, A, x, beta, y);
  }
  else if (beta == KAT::one ()) {
    spmv_alpha_beta_mv<AMatrix, XVector, YVector, doalpha, 1> (mode, alpha, A, x, beta, y);
  }
  else if (beta == -KAT::one ()) {
    spmv_alpha_beta_mv<AMatrix, XVector, YVector, doalpha, -1> (mode, alpha, A, x, beta, y);
  }
  else {
    spmv_alpha_beta_mv<AMatrix, XVector, YVector, doalpha, 2> (mode, alpha, A, x, beta, y);
  }
}

}
}

#endif // KOKKOSSPARSE_IMPL_SPMV_DEF_HPP_
