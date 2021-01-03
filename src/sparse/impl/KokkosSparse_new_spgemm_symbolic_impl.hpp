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
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/


#include "KokkosKernels_Utils.hpp"

namespace KokkosSparse{

  namespace Impl{


    struct tAnalyzeDenseSparseB{};
    struct tAnalyzeRowLimitsBdense{};
    struct tAnalyzeRowLimitsBsparse{};
    struct tAnalyzeDenseSparseA{};
    struct tAnalyzeRowLimitsAdense{};
    struct tAnalyzeRowLimitsAsparse{};

    template <typename HandleType, typename LayoutType>
    struct SPGEMM <HandleType, LayoutType>::SymbolicFunctor
    {
 
      ordinal_t numRowsA;
      ordinal_t numRowsB;
      ordinal_t numColsB;
 
      const_row_map_t row_mapA;
      const_entries_t entriesA;
      const_values_t valuesA;

      const_row_map_t row_mapB;
      const_entries_t entriesB;
      const_values_t valuesB;

      size_view_t rowFlopsA;
      minmax_view_t rowLimitsA;
      minmax_view_t rowLimitsB;

      ord_view_t denseRowsA;
      ord_view_t denseRowsB;

      ord_view_t ptrDenseSparse; // length: 5, covers the following 5 variables on device
      ordinal_t ptrDenseRowsB; 
      ordinal_t ptrSparseRowsB;
      ordinal_t ptrDenseRowsA;
      ordinal_t ptrSparseRowsA;
      ordinal_t ptrSingleEntryRowsA; 

      ordinal_t minCol;
      ordinal_t maxCol;

      int vectorSize;
      int teamSize;

      SymbolicFunctor(ordinal_t numRowsA_, ordinal_t numRowsB_, ordinal_t numColsB_,
		      const_row_map_t row_mapA_, const_entries_t entriesA_, const_values_t valuesA_,
		      const_row_map_t row_mapB_, const_entries_t entriesB_, const_values_t valuesB_,
		      size_view_t rowFlopsA_, minmax_view_t rowLimitsA_, minmax_view_t rowLimitsB_, 
		      ord_view_t denseRowsA_, ord_view_t denseRowsB_,
		      int vectorSize_):
	numRowsA(numRowsA_), numRowsB(numRowsB_), numColsB(numColsB_), 
	row_mapA(row_mapA_), entriesA(entriesA_), valuesA(valuesA_), 
	row_mapB(row_mapB_), entriesB(entriesB_), valuesB(valuesB_),
	rowFlopsA(rowFlopsA_), rowLimitsA(rowLimitsA_), rowLimitsB(rowLimitsB_), 
	denseRowsA(denseRowsA_), denseRowsB(denseRowsB_),
	vectorSize(vectorSize_) // not currently used
	
      {
	typedef Kokkos::Details::ArithTraits<ordinal_t> AT;
      	minCol = AT::max();
      	maxCol = AT::min(); 

	// These might be useless
	teamSize = 1024 / vectorSize;
	int numTeams = numRowsB/teamSize + 1;

      }

      void analyze() {

	///////////////////////////////////////////////////////////////
	// Set the pointers for sparse and dense rows in A and B as follows
	// First two entries are about B and last three entries are about A
	///////////////////////////////////////////////////////////////
  
	ptrDenseSparse = ord_view_t(Kokkos::ViewAllocateWithoutInitializing("ptrDenseSparse"), 5);
	auto h_ptrDenseSparse = Kokkos::create_mirror(ptrDenseSparse);
	h_ptrDenseSparse[0] = -1;         //points to where to write dense B rows after incrementing
	h_ptrDenseSparse[1] = numRowsB;   //points to where to write sparse B rows after decrementing
	h_ptrDenseSparse[2] = -1;         //points to where to write dense A rows after incrementing
	h_ptrDenseSparse[3] = numRowsA;   //points to where to write dense A rows after decrementing
	h_ptrDenseSparse[4] = numRowsA-1; //points to where to write single-entry A rows after incrementing
	Kokkos::deep_copy(ptrDenseSparse, h_ptrDenseSparse);


	///////////////////////////////////////////////////////////////
	// 1. Find the max # nonzeros in B rows
	// 2. Find the # of dense B rows: numDenseRowsB 
	// 3. Write indices of dense/sparse B rows in denseRowsB so that
	//  -rows with >256 entries are in positions 0 .. numDenseRowsB-1
	//  -rows with <=256 entries are in numDenseRowsB ... numRowsB-1
	///////////////////////////////////////////////////////////////

	ordinal_t maxNnzInBrow = 0;
	Kokkos::RangePolicy<tAnalyzeDenseSparseB, ExecSpace> pAnalyzeDenseSparseB(0, numRowsB);
	Kokkos::parallel_reduce ("AnalyzeDenseSparseB", pAnalyzeDenseSparseB, *this, Kokkos::Max<ordinal_t>(maxNnzInBrow));


	///////////////////////////////////////////////////////////////
	// 1. Find the max # nonzeros in A rows
	// 2. Find the # of dense A rows: numDenseRowsA 
	// 3. Find the # of single entry A rows: ptrSingleEntryRowsA + 1 - numRowsA 
	// 4. Write indices of dense/sparse/single-entry A rows in denseRowsA so that
	//  -rows with >256 entries are in positions 0 .. numDenseRowsA-1
	//  -rows with <=256 entries are in ptrSparseRowsA ... numRowsA-1
	//  -rows with 1 entry are in numRowsA .. ptrSingleEntryRowsA
	///////////////////////////////////////////////////////////////

	ordinal_t maxNnzInArow = 0;
	Kokkos::RangePolicy<tAnalyzeDenseSparseA, ExecSpace> pAnalyzeDenseSparseA(0, numRowsA);
	Kokkos::parallel_reduce ("AnalyzeDenseSparseA", pAnalyzeDenseSparseA, *this, Kokkos::Max<ordinal_t>(maxNnzInArow));


	///////////////////////////////////////////////////////////////
	// Transfer dense-sparse analysis results
	///////////////////////////////////////////////////////////////
	ExecSpace().fence();
	Kokkos::deep_copy(h_ptrDenseSparse, ptrDenseSparse);
	ptrDenseRowsB = h_ptrDenseSparse[0];
	ptrSparseRowsB = h_ptrDenseSparse[1];
	ptrDenseRowsA = h_ptrDenseSparse[2];
	ptrSparseRowsA = h_ptrDenseSparse[3];
	ptrSingleEntryRowsA = h_ptrDenseSparse[4];
	
	/*//print dense rows
	auto h_denseRowsB = Kokkos::create_mirror_view(denseRowsB);
	Kokkos::deep_copy(h_denseRowsB, denseRowsB);
	for(ordinal_t i = 0; i < ptrDenseRowsB; i++)
	std::cout << i << ": " << h_denseRowsB(i) << std::endl;*/


	///////////////////////////////////////////////////////////////
	// Determine row limits (min and max column indices in each row) in B
	///////////////////////////////////////////////////////////////
	
	Kokkos::Timer timer;
	Kokkos::TeamPolicy<tAnalyzeRowLimitsBdense, ExecSpace> pAnalyzeRowLimitsBdense(ptrDenseRowsB+1, Kokkos::AUTO);
	Kokkos::RangePolicy<tAnalyzeRowLimitsBsparse, ExecSpace> pAnalyzeRowLimitsBsparse(ptrDenseRowsB+1, numRowsB);

	if(ptrDenseRowsB >= 0)
	  Kokkos::parallel_for("AnalyzeRowLimitsBdense", pAnalyzeRowLimitsBdense, *this);
	Kokkos::parallel_for("AnalyzeRowLimitsBsparse", pAnalyzeRowLimitsBsparse, *this);
	ExecSpace().fence();
	double rowLimitsTime = timer.seconds();
	
	/*//print row limits
	auto h_rowLimitsB = Kokkos::create_mirror_view(rowLimitsB);
	Kokkos::deep_copy(h_rowLimitsB, rowLimitsB);
	for(ordinal_t i = 0; i < numRowsB; i++)
	std::cout << i << ": " << h_rowLimitsB(i,0) << " " << h_rowLimitsB(i,1) << std::endl;*/ 
	
	
	std::cout << "DetermineRowLimits: " << rowLimitsTime << " MaxNnzInBrow: " << maxNnzInBrow << " #DenseRows: " <<ptrDenseRowsB+1 << std::endl;


	///////////////////////////////////////////////////////////////
	// Determine row limits (min and max column indices in each row) in A
	// Determine flop count for each row of A
	///////////////////////////////////////////////////////////////
	
	timer.reset();
	Kokkos::TeamPolicy<tAnalyzeRowLimitsAdense, ExecSpace> pAnalyzeRowLimitsAdense(ptrDenseRowsA+1, Kokkos::AUTO);
	Kokkos::RangePolicy<tAnalyzeRowLimitsAsparse, ExecSpace> pAnalyzeRowLimitsAsparse(ptrDenseRowsA+1, numRowsA);

	ExecSpace().fence();
	if(ptrDenseRowsA >= 0)
	  Kokkos::parallel_for("AnalyzeRowLimitsAdense", pAnalyzeRowLimitsAdense, *this);
	Kokkos::parallel_for("AnalyzeRowLimitsAsparse", pAnalyzeRowLimitsAsparse, *this);
	ExecSpace().fence();
	rowLimitsTime = timer.seconds();
	
	std::cout << "DetermineRowLimits: " << rowLimitsTime << " MaxNnzInArow: " << maxNnzInArow << " #DenseRowsA: " << ptrDenseRowsA+1
		  << " numRowsA: "<< numRowsA
		  << " ptrSingleEntryRowsA: "<< ptrSingleEntryRowsA+1 << std::endl;
       

      }


      KOKKOS_INLINE_FUNCTION
      void operator()(const tAnalyzeDenseSparseB&, const ordinal_t &row_index, ordinal_t &update) const {

	ordinal_t myNnz = row_mapB[row_index+1]-row_mapB[row_index]; 
	if(myNnz > update) update = myNnz;

	if(myNnz > 256) {
	  int index = Kokkos::atomic_fetch_add(&ptrDenseSparse[0], 1);
	  denseRowsB[index] = row_index;
	}
	else {
	  int index = Kokkos::atomic_fetch_add(&ptrDenseSparse[1], -1);
	  denseRowsB[index] = row_index;
	}
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(const tAnalyzeDenseSparseA&, const ordinal_t &row_index, ordinal_t &update) const {

	ordinal_t myNnz = row_mapA[row_index+1]-row_mapA[row_index]; 
	if(myNnz > update) update = myNnz;

	if(myNnz > 256) {
	  int index = Kokkos::atomic_fetch_add(&ptrDenseSparse[2], 1);
	  denseRowsA[index] = row_index;
	}
	else if(myNnz > 1){
	  int index = Kokkos::atomic_fetch_add(&ptrDenseSparse[3], -1);
	  denseRowsB[index] = row_index;
	}
	else{
	  int index = Kokkos::atomic_fetch_add(&ptrDenseSparse[4], 1);
	  denseRowsA[index] = row_index;
	}
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(const tAnalyzeRowLimitsBdense&, const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember ) const {

	ordinal_t row_index = denseRowsB[teamMember.league_rank()] ;
	Kokkos::MinMaxScalar<ordinal_t> result;
	Kokkos::MinMax<ordinal_t> reducer(result);
	Kokkos::parallel_reduce(Kokkos::TeamVectorRange(teamMember, row_mapB[row_index], row_mapB[row_index+1]), [&] (const offset_t& i, Kokkos::MinMaxScalar<ordinal_t> &update) {
			  if(entriesB[i] < update.min_val) update.min_val = entriesB[i];
			  if(entriesB[i] > update.max_val) update.max_val = entriesB[i];
			}, reducer);
	Kokkos::single(Kokkos::PerTeam(teamMember),[&] () {
	    rowLimitsB(row_index, 0) = result.min_val;
	    rowLimitsB(row_index, 1) = result.max_val;	    
	  });
	
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(const tAnalyzeRowLimitsBsparse&, const ordinal_t &i) const {

      	ordinal_t myMinCol = minCol, myMaxCol = maxCol;
	ordinal_t row_index = denseRowsB[i];
      	for(offset_t j = row_mapB[row_index]; j < row_mapB[row_index+1]; j++){
      	  myMinCol = KOKKOSKERNELS_MACRO_MIN(myMinCol, entriesB[j]);
      	  myMaxCol = KOKKOSKERNELS_MACRO_MAX(myMaxCol, entriesB[j]);
      	}

      	rowLimitsB(row_index,0) = myMinCol;
      	rowLimitsB(row_index,1) = myMaxCol;
      }
      


      KOKKOS_INLINE_FUNCTION
      void operator()(const tAnalyzeRowLimitsAdense&, const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember ) const {

	ordinal_t row_index = denseRowsA[teamMember.league_rank()] ;

	size_t rowFlop = 0;
	Kokkos::parallel_reduce(Kokkos::TeamVectorRange(teamMember, row_mapA[row_index], row_mapA[row_index+1]), [&] (const offset_t& i, size_t &update) {
	    ordinal_t rowB = entriesA[i];
	    update += (row_mapB[rowB+1] - row_mapB[rowB]);
	  }, Kokkos::Sum<size_t, ExecSpace>(rowFlop));

	Kokkos::single(Kokkos::PerTeam(teamMember),[&] () {
	    rowFlopsA(row_index) = rowFlop;
	  });


	Kokkos::MinMaxScalar<ordinal_t> result;
	Kokkos::MinMax<ordinal_t, ExecSpace> reducer(result);
	Kokkos::parallel_reduce(Kokkos::TeamVectorRange(teamMember, row_mapA[row_index], row_mapA[row_index+1]), [&] (const offset_t& i, Kokkos::MinMaxScalar<ordinal_t> &update) {
	    ordinal_t rowB = entriesA[i];
	    if(rowLimitsB(rowB, 0) < update.min_val) update.min_val = rowLimitsB(rowB, 0);
	    if(rowLimitsB(rowB, 1) > update.max_val) update.max_val = rowLimitsB(rowB, 1);
	  }, reducer);

	Kokkos::single(Kokkos::PerTeam(teamMember),[&] () {
	    rowLimitsA(row_index, 0) = result.min_val;
	    rowLimitsA(row_index, 1) = result.max_val;	    
	  });
	
      }


      KOKKOS_INLINE_FUNCTION
      void operator()(const tAnalyzeRowLimitsAsparse&, const ordinal_t &i) const {

      	ordinal_t myMinCol = minCol, myMaxCol = maxCol;
	ordinal_t row_index = denseRowsA[i];
	size_t rowFlop = 0;
      	for(offset_t j = row_mapA[row_index]; j < row_mapA[row_index+1]; j++){
	  ordinal_t rowB = entriesA[j];
      	  myMinCol = KOKKOSKERNELS_MACRO_MIN(myMinCol, rowLimitsB(rowB, 0));
      	  myMaxCol = KOKKOSKERNELS_MACRO_MAX(myMaxCol, rowLimitsB(rowB, 1));
	  rowFlop += (row_mapB[rowB+1] - row_mapB[rowB]);
      	}

      	rowLimitsB(row_index,0) = myMinCol;
      	rowLimitsB(row_index,1) = myMaxCol;
	rowFlopsA(row_index) = rowFlop;
      }

    };

    template <typename HandleType, typename LayoutType>
    void
    SPGEMM<HandleType,LayoutType>::symbolic_impl(row_map_t row_mapC)
    {

      size_view_t rowFlopsA(Kokkos::ViewAllocateWithoutInitializing("rowFlopsA"), a_row_cnt);
      minmax_view_t rowLimitsA(Kokkos::ViewAllocateWithoutInitializing("rowLimitsA"), a_row_cnt);
      minmax_view_t rowLimitsB(Kokkos::ViewAllocateWithoutInitializing("rowLimitsB"), b_row_cnt);

      ord_view_t denseRowsA(Kokkos::ViewAllocateWithoutInitializing("denseRowsA"), a_row_cnt*2);
      ord_view_t denseRowsB(Kokkos::ViewAllocateWithoutInitializing("denseRowsB"), b_row_cnt);

      int vectorSize = this->handle->get_suggested_vector_size(b_row_cnt, entriesB.extent(0));

      SymbolicFunctor sf(a_row_cnt, b_row_cnt, b_col_cnt,
			 row_mapA, entriesA, valuesA,
			 row_mapB, entriesB, valuesB,
			 rowFlopsA, rowLimitsA, rowLimitsB, denseRowsA, denseRowsB,
			 vectorSize); // vectorSize is not currently used

      sf.analyze();

    }

  }
}
