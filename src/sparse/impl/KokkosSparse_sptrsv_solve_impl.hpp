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

#ifndef KOKKOSSPARSE_IMPL_SPTRSV_SOLVE_HPP_
#define KOKKOSSPARSE_IMPL_SPTRSV_SOLVE_HPP_

/// \file KokkosSparse_impl_sptrsv.hpp
/// \brief Implementation(s) of sparse triangular solve.

#include <KokkosKernels_config.h>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_sptrsv_handle.hpp>

#if defined(KOKKOSKERNELS_ENABLE_SUPERNODAL)
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas2_team_gemv.hpp>

#include "KokkosBatched_Util.hpp"

#include "KokkosBatched_Trsv_Decl.hpp"
#include "KokkosBatched_Trsv_Serial_Impl.hpp"

#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Gemv_Team_Impl.hpp"
#include "KokkosBatched_Gemv_Serial_Impl.hpp"
#endif


//#define LVL_OUTPUT_INFO

namespace KokkosSparse {
namespace Impl {
namespace Experimental {


struct UnsortedTag {};

template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct LowerTriLvlSchedRPSolverFunctor
{
  typedef typename EntriesType::non_const_value_type lno_t;
  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  LowerTriLvlSchedRPSolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, NGBLType nodes_grouped_by_level_ ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const lno_t i) const {
    auto rowid = nodes_grouped_by_level(i);
    // Assuming indices are sorted per row, diag entry is final index in the list

    auto soffset = row_map(rowid);
    auto eoffset = row_map(rowid+1);
    auto rhs_rowid = rhs(rowid);

    for ( auto ptr = soffset; ptr < eoffset; ++ptr ) {
      auto colid = entries(ptr);
      auto val   = values(ptr);
      if ( colid != rowid ) {
        rhs_rowid = rhs_rowid - val*lhs(colid);
      }
      else {
        lhs(rowid) = rhs_rowid/val;
      }
    } // end for ptr
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const lno_t i) const {
    auto rowid = nodes_grouped_by_level(i);
    auto soffset = row_map(rowid);
    auto eoffset = row_map(rowid+1);
    auto rhs_rowid = rhs(rowid);
    auto diag = -1;

    for ( auto ptr = soffset; ptr < eoffset; ++ptr ) {
      auto colid = entries(ptr);
      auto val   = values(ptr);
      if ( colid != rowid ) {
        rhs_rowid = rhs_rowid - val*lhs(colid);
      }
      else {
        diag = ptr;
      }
    } // end for ptr
    lhs(rowid) = rhs_rowid/values(diag);
  }
};


template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct LowerTriLvlSchedTP1SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  long node_count;
  long node_groups;


  LowerTriLvlSchedTP1SolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, long node_count_, long node_groups_ = 0) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()( const member_type & team ) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_team = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

      Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
      }, diff );

        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this; can also use Kokkos::single
        if ( my_team == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at eoffset - 1
          lhs(rowid) = (rhs_rowid+diff)/values(eoffset-1);
        }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_team = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

        auto diag = -1;

        Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
          else {
            diag = ptr;
          }
        }, diff );

        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this; can also use Kokkos::single
        if ( my_team == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at eoffset - 1
          lhs(rowid) = (rhs_rowid+diff)/values(diag);
        }
  }
};

// FIXME CUDA: This algorithm not working with all integral type combos
// In any case, this serves as a skeleton for 3-level hierarchical parallelism for alg dev
template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct LowerTriLvlSchedTP2SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  long node_count;
  long node_groups;


  LowerTriLvlSchedTP2SolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, long node_count_, long node_groups_ = 0) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {

            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
            }, diff );

            // ASSUMPTION: sorted diagonal value located at eoffset - 1
            lhs(rowid) = (rhs_rowid+diff)/values(eoffset-1);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {
            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            auto diag = -1;
            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
              else {
                diag = ptr;
              }
            }, diff );

            // ASSUMPTION: sorted diagonal value located at eoffset - 1
            lhs(rowid) = (rhs_rowid+diff)/values(diag);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }
};

#if defined(KOKKOSKERNELS_ENABLE_SUPERNODAL)
template <class ColptrView, class RowindType, class ValuesType, class LHSType, class NGBLType>
struct LowerTriSupernodalFunctor
{
  typedef typename LHSType::execution_space execution_space;
  typedef typename execution_space::memory_space memory_space;

  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename ValuesType::non_const_value_type scalar_t;
  typedef Kokkos::Details::ArithTraits<scalar_t> STS;

  typedef typename Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space, Kokkos::MemoryUnmanaged> supernode_view_t;
  typedef typename Kokkos::View<scalar_t*,                      memory_space, Kokkos::MemoryUnmanaged> vector_view_t;

  typedef Kokkos::View<int*, memory_space>  supercols_t;
  typedef typename Kokkos::View<scalar_t*,  memory_space> WorkspaceType;

  typedef Kokkos::pair<int,int> range_type;

  const int *supercols;
  ColptrView colptr;
  RowindType rowind;
  ValuesType values;

  int level;
  supercols_t kernel_type;
  supercols_t diag_kernel_type;

  LHSType X;

  WorkspaceType work; // needed with gemv for update&scatter
  supercols_t work_offset;

  NGBLType nodes_grouped_by_level;

  long node_count;
  long node_groups;

  // constructor
  LowerTriSupernodalFunctor (// supernode info
                             const int *supercols_,
                             // L in CSR
                             const ColptrView  &colptr_,
                             const RowindType &rowind_,
                             const ValuesType &values_,
                             // options to pick kernel type
                             int level_,
                             supercols_t &kernel_type_,
                             supercols_t &diag_kernel_type_,
                             // right-hand-side (input), solution (output)
                             LHSType &X_,
                             // workspace
                             WorkspaceType work_,
                             supercols_t &work_offset_,
                             //
                             const NGBLType &nodes_grouped_by_level_,
                             long  node_count_,
                             long  node_groups_ = 0) :
    supercols(supercols_),
    colptr(colptr_), rowind(rowind_), values(values_),
    level(level_), kernel_type(kernel_type_), diag_kernel_type(diag_kernel_type_),
    X(X_), work(work_), work_offset(work_offset_),
    nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & team) const {

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */
    const int league_rank = team.league_rank(); // batch id
    const int team_size = team.team_size ();
    const int team_rank = team.team_rank ();
    scalar_t zero = STS::zero ();
    scalar_t one  = STS::one ();

    auto s = nodes_grouped_by_level (node_count + league_rank);

    // supernodal column size
    int j1 = supercols[s];
    int j2 = supercols[s+1];
    int nscol = j2 - j1 ;       // number of columns in the s-th supernode column

    int i1 = colptr (j1);
    int i2 = colptr (j1+1);
    int nsrow  = i2 - i1;       // "total" number of rows in all the supernodes (diagonal+off-diagonal)
    int nsrow2 = nsrow - nscol; // "total" number of rows in all the off-diagonal supernodes

    // create a view for the s-th supernocal column
    scalar_t *dataL = const_cast<scalar_t*> (values.data ());
    Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space, Kokkos::MemoryUnmanaged> viewL (&dataL[i1], nsrow, nscol);

    // extract part of the solution, corresponding to the diagonal block
    auto Xj = subview (X, range_type(j1, j2));

    // workspace
    int workoffset = work_offset (s);

    //if (team_rank == 0) {
    //  printf( " >> LowerTriSupernodalFunctor (league=%d, team=%d/%d <<\n",league_rank,team_rank,team_size );
    //  printf( " s = %d, nscol=%d, nsrows=%d (j1=%d, j2=%d), (i1=%d, i2=%d), work_offset=%d\n",s, nscol,nsrow, j1,j2, i1,i2, workoffset );
    //  printf( " kernel types(%d): %d %d\n",level,diag_kernel_type(level), kernel_type(level) );
    //}

    if (diag_kernel_type (level) != 3) { // not a device-level TRSM-solve
      #define SUPERLU_INVERT_OFFDIAG
      #if defined(SUPERLU_INVERT_OFFDIAG)
      // combined TRSM solve with diagonal + GEMV update with off-diagonal
      auto Y = subview (work, range_type(workoffset, workoffset+nsrow));  // needed for gemv instead of trmv/trsv
      auto Ljj = Kokkos::subview (viewL, range_type (0, nsrow), Kokkos::ALL ());
      KokkosBatched::TeamGemv<member_type,
                              KokkosBatched::Trans::NoTranspose,
                              KokkosBatched::Algo::Gemv::Unblocked>
        ::invoke(team, one, Ljj, Xj, zero, Y);
      team.team_barrier ();
      for (int ii = team_rank; ii < nscol; ii += team_size) {
        Xj(ii) = Y(ii);
      }
      team.team_barrier ();
      #else
      /* TRSM with diagonal block */
      // extract diagonal and off-diagonal blocks of L
      auto Ljj = Kokkos::subview (viewL, range_type (0, nscol), Kokkos::ALL ());
      // workspace
      auto Y = subview (work, range_type(workoffset, workoffset+nscol));  // needed for gemv instead of trmv/trsv
      /*if (nscol == 1) {
        if (team_rank == 0) {
          Xj(0) *= Ljj(0, 0);
        }
      } else */{
        for (int ii = team_rank; ii < nscol; ii += team_size) {
          Y(ii) = Xj(ii);
        }
        team.team_barrier ();
        //if (diag_kernel_type (level) == 0) {
          // calling team-level "Unblocked" gemv on small-size diagonal in KokkosBatched
          KokkosBatched::TeamGemv<member_type,
                                  KokkosBatched::Trans::NoTranspose,
                                  KokkosBatched::Algo::Gemv::Unblocked>
            ::invoke(team, one, Ljj, Y, zero, Xj);
        /*} else if (diag_kernel_type (level) == 1) {
          // calling team-level "Blocked" gemv on small-size diagonal in KokkosBatched
          KokkosBatched::TeamGemv<member_type,
                                  KokkosBatched::Trans::NoTranspose,
                                  KokkosBatched::Algo::Gemv::Blocked>
            ::invoke(team, one, Ljj, Y, zero, Xj);
        } else {
          // calling team-based gemv on medium-size diagonal in KokkosBlas
          KokkosBlas::Experimental::
          gemv (team, 'N',
                one,  Ljj,
                      Y,
                zero, Xj);
        }*/
      }
      team.team_barrier ();
      #endif
    }
    //for (int ii=0; ii < nscol; ii++) printf( "%d %e\n",j1+ii,X(j1+ii));
    //printf( "\n" );

    if (nsrow2 > 0) {
      /* GEMM to update with off diagonal blocks */
      auto Z = subview (work, range_type(workoffset+nscol, workoffset+nsrow)); 
      #if !defined(SUPERLU_INVERT_OFFDIAG)
      if (kernel_type (level) != 3) { // not device-level GEMV update
        auto Lij = Kokkos::subview (viewL, range_type (nscol, nsrow), Kokkos::ALL ());
        //if (kernel_type (level) == 0) {
          KokkosBatched::TeamGemv<member_type,
                                  KokkosBatched::Trans::NoTranspose,
                                  KokkosBatched::Algo::Gemv::Unblocked>
            ::invoke(team, one, Lij, Xj, zero, Z);
        /*} else {
          KokkosBlas::Experimental::
          gemv(team, 'N',
               -one, Lij,
                     Xj,
               zero, Z);
        }*/
        team.team_barrier();
      }
      #endif
      /* scatter vectors back into X */
      int ps2 = i1 + nscol ;     // offset into rowind 
      Kokkos::View<scalar_t*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic> > Xatomic(X.data(), X.extent(0));
      for (int ii = team_rank; ii < nsrow2; ii += team_size) {
        int i = rowind (ps2 + ii);
        //printf( " X(%d) = %e - %e = %e\n",i,X(i),Z(ii),X(i)-Z(ii) );
        Xatomic (i) -= Z (ii);
      }
    }
    team.team_barrier();
  }
};

template <class ColptrType, class RowindType, class ValuesType, class LHSType, class NGBLType>
struct UpperTriSupernodalFunctor
{
  typedef typename LHSType::execution_space execution_space;
  typedef typename execution_space::memory_space memory_space;

  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename ValuesType::non_const_value_type scalar_t;
  typedef Kokkos::Details::ArithTraits<scalar_t> STS;

  typedef typename Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space, Kokkos::MemoryUnmanaged> supernode_view_t;
  typedef typename Kokkos::View<scalar_t*,                      memory_space, Kokkos::MemoryUnmanaged> vector_view_t;

  typedef Kokkos::View<int*, memory_space>  supercols_t;
  typedef typename Kokkos::View<scalar_t*, memory_space> WorkspaceType;

  typedef Kokkos::pair<int,int> range_type;

  const int *supercols;
  ColptrType colptr;
  RowindType rowind;
  ValuesType values;

  int level;
  supercols_t kernel_type;
  supercols_t diag_kernel_type;

  LHSType X;

  WorkspaceType work; // needed with gemv for update&scatter
  supercols_t work_offset;

  NGBLType nodes_grouped_by_level;

  long node_count;
  long node_groups;

  // constructor
  UpperTriSupernodalFunctor (// supernode info
                             const int *supercols_,
                             // L in CSR
                             const ColptrType &colptr_,
                             const RowindType &rowind_,
                             const ValuesType &values_,
                             // options to pick kernel type
                             int level_,
                             supercols_t &kernel_type_,
                             supercols_t &diag_kernel_type_,
                             // right-hand-side (input), solution (output)
                             LHSType &X_,
                             // workspace
                             WorkspaceType &work_,
                             supercols_t &work_offset_,
                             //
                             const NGBLType &nodes_grouped_by_level_,
                             long  node_count_,
                             long  node_groups_ = 0) :
    supercols(supercols_),
    colptr(colptr_), rowind(rowind_), values(values_),
    level(level_), kernel_type(kernel_type_), diag_kernel_type(diag_kernel_type_),
    X(X_), work(work_), work_offset(work_offset_),
    nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & team) const {

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */
    const int league_rank = team.league_rank(); // batch id
    const int team_size = team.team_size ();
    const int team_rank = team.team_rank ();

    scalar_t zero = STS::zero ();
    scalar_t one = STS::one ();

    auto s = nodes_grouped_by_level (node_count + league_rank);

    //printf( " Upper: node_count=%ld, s=%d\n",node_count,s );
    int j1 = supercols[s];
    int j2 = supercols[s+1];
    int nscol = j2 - j1;         // number of columns in the s-th supernode column

    int i1 = colptr (j1);
    int i2 = colptr (j1+1);
    int nsrow = i2 - i1 ;        // "total" number of rows in all the supernodes (diagonal+off-diagonal)
    int nsrow2 = nsrow - nscol;  // "total" number of rows in all the off-diagonal supernodes

    // create a view of the s-th supernocal column of L
    scalar_t *dataL = const_cast<scalar_t*> (values.data ());
    Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space, Kokkos::MemoryUnmanaged> viewL (&dataL[i1], nsrow, nscol);

    // extract part of solution, corresponding to the diagonal block L(s, s)
    auto Xj = subview (X, range_type(j1, j2));

    // workspaces
    int workoffset = work_offset (s);

    //if (team_rank == 0) {
    //  printf( " >> UpperTriSupernodalFunctor (league=%d, team=%d/%d <<\n",league_rank,team_rank,team_size );
    //  printf( " s = %d, nscol=%d, nsrows=%d (j1=%d, j2=%d), (i1=%d, i2=%d), work_offset=%d\n",s, nscol,nsrow, j1,j2, i1,i2, workoffset );
    //}

    if (nsrow2 > 0) {
      /* gather vector into Z */
      int ps2 = i1 + nscol;     // offset into rowind 
      auto Z = subview(work, range_type(workoffset+nscol, workoffset+nsrow));  // needed with gemv for update&scatter
      for (int ii = team_rank; ii < nsrow2 ; ii += team_size) {
        int i = rowind (ps2 + ii);
        Z (ii) = X (i);
      }
      team.team_barrier();
      //if (diag_kernel_type (level) != 3)
      { // not device-level GEMV-udpate
        /* GEMM to update with off diagonal blocks, Xj = -Lij^T * Z */
        auto Lij = Kokkos::subview (viewL, range_type (nscol, nsrow), Kokkos::ALL ());
        //if (diag_kernel_type (level) == 0) {
          KokkosBatched::TeamGemv<member_type,
                                  KokkosBatched::Trans::Transpose,
                                  KokkosBatched::Algo::Gemv::Unblocked>
            ::invoke(team, -one, Lij, Z, one, Xj);
        /*} else {
          KokkosBlas::Experimental::
          gemv(team, 'T',
              -one, Lij,
                    Z,
               one, Xj);
        }*/
        team.team_barrier();
      }
    }

    //if (diag_kernel_type (level) != 3) // TODO: there seems to be a bug in KokkosBlas::gemv with "T" and CUDA
    { // not device-level TRSM-solve
      /* TRSM with diagonal block */
      // extract diagonal and off-diagonal blocks of L
      auto Ljj = Kokkos::subview (viewL, range_type (0, nscol), Kokkos::ALL ());

      // workspace
      auto Y = subview (work, range_type(workoffset, workoffset+nscol));  // needed for gemv instead of trmv/trsv
      if (nscol == 1) {
        Xj (0) *= Ljj(0, 0);
      } else {
        for (int ii = team_rank; ii < nscol; ii += team_size) {
          Y (ii) = Xj (ii);
        }
        team.team_barrier();
        //if (diag_kernel_type (level) == 0) {
          // caling team-level kernel in KokkosBatched on a small-size diagonal
          KokkosBatched::TeamGemv<member_type,
                                  KokkosBatched::Trans::Transpose,
                                  KokkosBatched::Algo::Gemv::Unblocked>
            ::invoke(team, one, Ljj, Y, zero, Xj);
        /*} else {
          // caling team-level kernel in KokkosBlas on a small-size diagonal
          KokkosBlas::Experimental::
          gemv(team, 'T',
               one,  Ljj,
                     Y,
               zero, Xj);
        }*/
      }
      team.team_barrier();
    }
    //for (int ii=0; ii < nscol; ii++) printf( "%d %e\n",j1+ii,X(j1+ii));
    //printf( "\n" );
  }
};
#endif

template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct UpperTriLvlSchedRPSolverFunctor
{
  typedef typename EntriesType::non_const_value_type lno_t;
  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;


  UpperTriLvlSchedRPSolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_ ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const lno_t i) const {
    auto rowid = nodes_grouped_by_level(i);
    // Assuming indices are sorted per row, diag entry is final index in the list
    long soffset = row_map(rowid);
    long eoffset = row_map(rowid+1);
    auto rhs_rowid = rhs(rowid);
    for ( long ptr = eoffset-1; ptr >= soffset; --ptr ) {
      auto colid = entries(ptr);
      auto val   = values(ptr);
      if ( colid != rowid ) {
        rhs_rowid = rhs_rowid - val*lhs(colid);
      }
      else {
        lhs(rowid) = rhs_rowid/val;
      }
    } // end for ptr
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const lno_t i) const {
    auto rowid = nodes_grouped_by_level(i);
    long soffset = row_map(rowid);
    long eoffset = row_map(rowid+1);
    auto rhs_rowid = rhs(rowid);
    auto diag = -1;
    for ( long ptr = eoffset-1; ptr >= soffset; --ptr ) {
      auto colid = entries(ptr);
      auto val   = values(ptr);
      if ( colid != rowid ) {
        rhs_rowid = rhs_rowid - val*lhs(colid);
      }
      else {
        diag = ptr;
      }
    } // end for ptr
    lhs(rowid) = rhs_rowid/values(diag);
  }

};

template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct UpperTriLvlSchedTP1SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  long node_count;
  long node_groups;


  UpperTriLvlSchedTP1SolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, long node_count_, long node_groups_ = 0 ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_team = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

        Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
        }, diff );

        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this, also can use Kokkos::single
        if ( my_team == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at start offset
          lhs(rowid) = (rhs_rowid+diff)/values(soffset);
        }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_team = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

        auto diag = -1;

        Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
          else {
            diag = ptr;
          }
        }, diff );
        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this, also can use Kokkos::single
        if ( my_team == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at start offset
          lhs(rowid) = (rhs_rowid+diff)/values(diag);
        }
  }

};


// FIXME CUDA: This algorithm not working with all integral type combos
// In any case, this serves as a skeleton for 3-level hierarchical parallelism for alg dev
template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct UpperTriLvlSchedTP2SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  long node_count;
  long node_groups;


  UpperTriLvlSchedTP2SolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, long node_count_, long node_groups_ = 0 ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()( const member_type & team ) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {

            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
            }, diff );

            // ASSUMPTION: sorted diagonal value located at start offset
            lhs(rowid) = (rhs_rowid+diff)/values(soffset);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team ) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {
            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            auto diag = -1;
            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
              else {
                diag = ptr;
              }
            }, diff );

            // ASSUMPTION: sorted diagonal value located at start offset
            lhs(rowid) = (rhs_rowid+diff)/values(diag);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }

};


template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType, class RHSType, class LHSType >
void lower_tri_solve( TriSolveHandle & thandle, const RowMapType row_map, const EntriesType entries, const ValuesType values, const RHSType & rhs, LHSType &lhs) {

  typedef typename TriSolveHandle::execution_space execution_space;
  typedef typename execution_space::memory_space memory_space;

  typedef typename TriSolveHandle::size_type size_type;
  typedef typename TriSolveHandle::nnz_lno_view_t NGBLType;

  typedef typename ValuesType::non_const_value_type scalar_t;
  typedef Kokkos::Details::ArithTraits<scalar_t> STS;

  typedef typename TriSolveHandle::supercols_t supercols_t;
  typedef typename TriSolveHandle::supercols_host_t supercols_host_t;

  typedef Kokkos::pair<int,int> range_type;

  auto nlevels = thandle.get_num_levels();
  // Keep this a host View, create device version and copy to back to host during scheduling
  auto nodes_per_level = thandle.get_nodes_per_level();
  auto hnodes_per_level = Kokkos::create_mirror_view(nodes_per_level);
  Kokkos::deep_copy(hnodes_per_level, nodes_per_level);  

  auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

#if defined(KOKKOSKERNELS_ENABLE_SUPERNODAL)
  auto nodes_grouped_by_level_host = Kokkos::create_mirror_view (nodes_grouped_by_level);
  Kokkos::deep_copy (nodes_grouped_by_level_host, nodes_grouped_by_level);

  auto row_map_host = Kokkos::create_mirror_view (row_map);
  Kokkos::deep_copy (row_map_host, row_map);

  Kokkos::View<scalar_t*, memory_space> work = thandle.get_workspace ();

  supercols_t kernel_type = thandle.get_kernel_type ();
  supercols_t diag_kernel_type = thandle.get_diag_kernel_type ();
  supercols_t work_offset = thandle.get_work_offset ();
#endif

  size_type node_count = 0;

  // This must stay serial; would be nice to try out Cuda's graph stuff to reduce kernel launch overhead
  for ( size_type lvl = 0; lvl < nlevels; ++lvl ) {
    size_type lvl_nodes = hnodes_per_level(lvl);

    if ( lvl_nodes != 0 ) {
      if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP ) {
        Kokkos::parallel_for( "parfor_fixed_lvl", Kokkos::RangePolicy<execution_space>( node_count, node_count+lvl_nodes ), LowerTriLvlSchedRPSolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> (row_map, entries, values, lhs, rhs, nodes_grouped_by_level) );
      }
      else if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1 ) {
        typedef Kokkos::TeamPolicy<execution_space> policy_type;
        int team_size = thandle.get_team_size();

        LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
        if ( team_size == -1 )
          Kokkos::parallel_for("parfor_l_team", policy_type( lvl_nodes , Kokkos::AUTO ), tstf);
        else
          Kokkos::parallel_for("parfor_l_team", policy_type( lvl_nodes , team_size ), tstf);
      }
      /*
      // TP2 algorithm has issues with some offset-ordinal combo to be addressed
      else if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHED_TP2 ) {
        typedef Kokkos::TeamPolicy<execution_space> tvt_policy_type;

        int team_size = thandle.get_team_size();
        if ( team_size == -1 ) {
          team_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 128;
        }
        int vector_size = thandle.get_team_size();
        if ( vector_size == -1 ) {
          vector_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 4;
        }

        // This impl: "chunk" lvl_nodes into node_groups; a league_rank is responsible for processing that many nodes
        //       TeamThreadRange over number of node_groups
        //       To avoid masking threads, 1 thread (team) per node in node_group
        //       ThreadVectorRange responsible for the actual solve computation
        const int node_groups = team_size;

        LowerTriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count, node_groups);
        Kokkos::parallel_for("parfor_u_team_vector", tvt_policy_type( (int)std::ceil((float)lvl_nodes/(float)node_groups) , team_size, vector_size ), tstf);
      } // end elseif
      */
#if defined(KOKKOSKERNELS_ENABLE_SUPERNODAL)
      else if (thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_NAIVE ||
               thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_ETREE ||
               thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_DAG) {

        //#define profile_supernodal_etree
        #ifdef profile_supernodal_etree
        Kokkos::Timer timer;
        timer.reset();
        #endif
        supercols_host_t kernel_type_host = thandle.get_kernel_type_host ();
        supercols_host_t diag_kernel_type_host = thandle.get_diag_kernel_type_host ();

        if (kernel_type_host (lvl) == 3) {
          // using device-level kernels (functor is called to scatter the results)
          scalar_t zero = STS::zero ();
          scalar_t one = STS::one ();

          supercols_host_t work_offset_host = thandle.get_work_offset_host ();
          scalar_t *dataL = const_cast<scalar_t*> (values.data ());

          for (int league_rank = 0; league_rank < lvl_nodes; league_rank++) {

            auto s = nodes_grouped_by_level_host (node_count + league_rank);

            // supernodal column size
            const int* supercols_host = thandle.get_supercols_host ();
            int j1 = supercols_host[s];
            int j2 = supercols_host[s+1];
            int nscol = j2 - j1 ;        // number of columns in the s-th supernode column

            int i1 = row_map_host (j1);
            int i2 = row_map_host (j1+1);
            int nsrow = i2 - i1;         // "total" number of rows in all the supernodes (diagonal+off-diagonal)

            // workspace  (needed for gemv instead of trmv/trsv)
            int workoffset = work_offset_host (s);

            // create a view for the s-th supernocal block column
            Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space, Kokkos::MemoryUnmanaged> viewL (&dataL[i1], nsrow, nscol);

            // "triangular-solve" to compute Xj
            #if defined(SUPERLU_INVERT_OFFDIAG)
            auto Y = Kokkos::subview (work, range_type(workoffset, workoffset+nsrow));
            auto Xj = Kokkos::subview (lhs, range_type (j1, j2));                      // part of the solution, corresponding to the diagonal block
            auto Ljj = Kokkos::subview (viewL, range_type (0, nsrow), Kokkos::ALL ()); // s-th supernocal column of L
            KokkosBlas::
            gemv("N", one,  Ljj,
                            Xj,
                      zero, Y);
            Kokkos::deep_copy(Xj, Y);
            #else
            auto Y = Kokkos::subview (work, range_type(workoffset, workoffset+nscol));
            auto Xj = Kokkos::subview (lhs, range_type (j1, j2));                      // part of the solution, corresponding to the diagonal block
            auto Ljj = Kokkos::subview (viewL, range_type (0, nscol), Kokkos::ALL ()); // diagonal block of s-th supernocal column of L
            Kokkos::deep_copy(Y, Xj);
            KokkosBlas::
            gemv("N", one,  Ljj,
                            Y,
                      zero, Xj);

            // update with off-diagonal blocks
            int nsrow2 = nsrow - nscol;  // "total" number of rows in all the off-diagonal supernodes
            if (nsrow2 > 0) {
              auto Z = Kokkos::subview (work, range_type(workoffset+nscol, workoffset+nsrow));  // workspace, needed with gemv for update&scatter
              auto Lij = Kokkos::subview (viewL, range_type (nscol, nsrow), Kokkos::ALL ()); // off-diagonal blocks of s-th supernodal column of L
              KokkosBlas::
              gemv("N", one,  Lij,
                              Xj,
                        zero, Z);
            }
            #endif
          }
        }

        // launching sparse-triangular solve functor
        const int* supercols = thandle.get_supercols ();
        LowerTriSupernodalFunctor<RowMapType, EntriesType, ValuesType, LHSType, NGBLType> 
          sptrsv_functor (supercols, row_map, entries, values, lvl, kernel_type, diag_kernel_type, lhs,
                          work, work_offset, nodes_grouped_by_level, node_count);

        typedef Kokkos::TeamPolicy<execution_space> team_policy_type;
        Kokkos::parallel_for ("parfor_lsolve_supernode", team_policy_type(lvl_nodes , Kokkos::AUTO), sptrsv_functor);
        #ifdef profile_supernodal_etree
        Kokkos::fence();
        std::cout << " > SUPERNODAL LowerTri: " << lvl << " " << timer.seconds() << std::endl;
        #endif
      }
#endif
      node_count += lvl_nodes;

    } // end if
  } // end for lvl

} // end lower_tri_solve


template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType, class RHSType, class LHSType >
void upper_tri_solve( TriSolveHandle & thandle, const RowMapType row_map, const EntriesType entries, const ValuesType values, const RHSType & rhs, LHSType &lhs) {

  typedef typename TriSolveHandle::execution_space execution_space;
  typedef typename execution_space::memory_space memory_space;

  typedef typename TriSolveHandle::size_type size_type;
  typedef typename TriSolveHandle::nnz_lno_view_t NGBLType;

  typedef typename ValuesType::non_const_value_type scalar_t;
  typedef Kokkos::Details::ArithTraits<scalar_t> STS;

  typedef typename TriSolveHandle::supercols_t supercols_t;

#if defined(SUPERNODAL_SPTRSV_USOLVE_WITH_KOKKOS_BLAS)
  typedef Kokkos::pair<int,int> range_type;
  typedef typename TriSolveHandle::supercols_host_t supercols_host_t;
#endif

  auto nlevels = thandle.get_num_levels();
  // Keep this a host View, create device version and copy to back to host during scheduling
  auto nodes_per_level = thandle.get_nodes_per_level();
  auto hnodes_per_level = Kokkos::create_mirror_view(nodes_per_level);
  Kokkos::deep_copy(hnodes_per_level, nodes_per_level);

  auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

#if defined(KOKKOSKERNELS_ENABLE_SUPERNODAL)
  //auto nodes_grouped_by_level_host = Kokkos::create_mirror_view (nodes_grouped_by_level);
  //Kokkos::deep_copy (nodes_grouped_by_level_host, nodes_grouped_by_level);

  auto row_map_host = Kokkos::create_mirror_view (row_map);
  Kokkos::deep_copy (row_map_host, row_map);

  // workspace
  Kokkos::View<scalar_t*, memory_space> work = thandle.get_workspace ();

  supercols_t kernel_type = thandle.get_kernel_type ();
  supercols_t diag_kernel_type = thandle.get_diag_kernel_type ();
  supercols_t work_offset = thandle.get_work_offset ();
#endif

  size_type node_count = 0;

  // This must stay serial; would be nice to try out Cuda's graph stuff to reduce kernel launch overhead
  for ( size_type lvl = 0; lvl < nlevels; ++lvl ) {
    size_type lvl_nodes = hnodes_per_level(lvl);

    if ( lvl_nodes != 0 ) {

      if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP ) {
        Kokkos::parallel_for( "parfor_fixed_lvl", Kokkos::RangePolicy<execution_space>( node_count, node_count+lvl_nodes ), UpperTriLvlSchedRPSolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> (row_map, entries, values, lhs, rhs, nodes_grouped_by_level) );
      }
      else if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1 ) {
        typedef Kokkos::TeamPolicy<execution_space> policy_type;

        int team_size = thandle.get_team_size();

        UpperTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
        if ( team_size == -1 )
          Kokkos::parallel_for("parfor_u_team", policy_type( lvl_nodes , Kokkos::AUTO ), tstf);
        else
          Kokkos::parallel_for("parfor_u_team", policy_type( lvl_nodes , team_size ), tstf);
      }
      /*
      // TP2 algorithm has issues with some offset-ordinal combo to be addressed
      else if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHED_TP2 ) {
        typedef Kokkos::TeamPolicy<execution_space> tvt_policy_type;

        int team_size = thandle.get_team_size();
        if ( team_size == -1 ) {
          team_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 128;
        }
        int vector_size = thandle.get_team_size();
        if ( vector_size == -1 ) {
          vector_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 4;
        }

        // This impl: "chunk" lvl_nodes into node_groups; a league_rank is responsible for processing that many nodes
        //       TeamThreadRange over number of node_groups
        //       To avoid masking threads, 1 thread (team) per node in node_group
        //       ThreadVectorRange responsible for the actual solve computation
        const int node_groups = team_size;

        UpperTriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count, node_groups);
        Kokkos::parallel_for("parfor_u_team_vector", tvt_policy_type( (int)std::ceil((float)lvl_nodes/(float)node_groups) , team_size, vector_size ), tstf);
      } // end elseif
      */
#if defined(KOKKOSKERNELS_ENABLE_SUPERNODAL)
      else if (thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_NAIVE ||
               thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_ETREE ||
               thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_DAG) {

        #ifdef profile_supernodal_etree
        Kokkos::Timer timer;
        timer.reset();
        #endif
        // launching sparse-triangular solve functor
        const int* supercols = thandle.get_supercols ();
        UpperTriSupernodalFunctor<RowMapType, EntriesType, ValuesType, LHSType, NGBLType> 
          sptrsv_functor (supercols, row_map, entries, values,lvl, kernel_type, diag_kernel_type, lhs,
                          work, work_offset, nodes_grouped_by_level, node_count);

        typedef Kokkos::TeamPolicy<execution_space> policy_type;
        Kokkos::parallel_for ("parfor_usolve_supernode", policy_type (lvl_nodes , Kokkos::AUTO), sptrsv_functor);

        #if defined(SUPERNODAL_SPTRSV_USOLVE_WITH_KOKKOS_BLAS)
        supercols_host_t kernel_type_host = thandle.get_kernel_type_host ();
        supercols_host_t diag_kernel_type_host = thandle.get_diag_kernel_type_host ();

        if (kernel_type_host (lvl) == 3) {
          // using device-level kernels (functor is called to gather the input into workspace)
          scalar_t zero = STS::zero ();
          scalar_t one = STS::one ();

          supercols_host_t work_offset_host = thandle.get_work_offset_host ();
          scalar_t *dataL = const_cast<scalar_t*> (values.data ());

          for (int league_rank = 0; league_rank < lvl_nodes; league_rank++) {

            auto s = nodes_grouped_by_level_host (node_count + league_rank);

            // supernodal column size
            const int* supercols_host = thandle.get_supercols_host ();
            int j1 = supercols_host[s];
            int j2 = supercols_host[s+1];
            int nscol = j2 - j1 ;        // number of columns in the s-th supernode column

            int i1 = row_map_host (j1);
            int i2 = row_map_host (j1+1);
            int nsrow = i2 - i1;         // "total" number of rows in all the supernodes (diagonal+off-diagonal)
            int nsrow2 = nsrow - nscol;  // "total" number of rows in all the off-diagonal supernodes

            // workspace
            int workoffset = work_offset_host (s);
 
            // create a view for the s-th supernocal block column
            Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space, Kokkos::MemoryUnmanaged> viewL (&dataL[i1], nsrow, nscol);

            // extract part of the solution, corresponding to the diagonal block
            auto Xj = Kokkos::subview (lhs, range_type(j1, j2));
            auto Y = Kokkos::subview (work, range_type(workoffset, workoffset+nscol));  // needed for gemv instead of trmv/trsv

            // update with off-diagonal blocks
            if (nsrow2 > 0) {
              // extract the off-diagonal blocks of s-th supernodal column of L
              auto Lij = Kokkos::subview (viewL, range_type (nscol, nsrow), Kokkos::ALL ());
              auto Z = Kokkos::subview (work, range_type(workoffset+nscol, workoffset+nscol+nsrow2));  // needed with gemv for update&scatter
              KokkosBlas::
              gemv("N", -one, Lij,
                              Z,
                         one, Xj);
            }

            // "triangular-solve" to compute Xj
            // extract the diagonal block of s-th supernocal column of L
            auto Ljj = Kokkos::subview (viewL, range_type (0, nscol), Kokkos::ALL ());
            Kokkos::deep_copy(Y, Xj);
            KokkosBlas::
            gemv("T", one,  Ljj,
                            Y,
                      zero, Xj);
          }
        }
        #endif
        #ifdef profile_supernodal_etree
        Kokkos::fence();
        std::cout << " > SUPERNODAL UpperTri: " << lvl << " " << timer.seconds() << std::endl;
        #endif
      }
#endif
      node_count += lvl_nodes;

    } // end if
  } // end for lvl

} // end upper_tri_solve


} // namespace Experimental
} // namespace Impl
} // namespace KokkosSparse

#endif
