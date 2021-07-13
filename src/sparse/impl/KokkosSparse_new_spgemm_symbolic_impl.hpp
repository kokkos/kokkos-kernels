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

// This implementation consists of two major steps: analyze and symbolic. 
//
// The analyze step
//    - first computes the max FLOPS for each row of A and
//    - then categorizes the rows of A into 6 different categories as follows:
//         1) rows with 0 or 1 entry,
//         2-5) sparse rows whose computations fits into shared memory, and
//         6) dense rows whose computations may not fit into the shared memory.
// The analyze step writes the number of rows in these categories into an array of size 6.
// It also writes those rows in each different category into a 2D array of size numRowsA x 6.
// 
// The symbolic step, for each category c,
//    - checks the numbers of rows in category c and launches the parallel_for specialized for 
//      category c if that number is greater than zero.
  
#include "KokkosKernels_Utils.hpp"

namespace KokkosSparse{

  namespace Impl{

    // Tags to be used by different parallel_fors
    struct tAnalyze{};
    struct tSymbolicSparse{};
    struct tSymbolicDense{};
    struct tSymbolicSingleEntry{};
    
    // The functor
    template <typename HandleType, typename LayoutType>
    struct SPGEMM <HandleType, LayoutType>::SymbolicFunctor
    {
 
      // User-provided data members
      ordinal_t numRowsA;
      ordinal_t numRowsB;
      ordinal_t numColsB;
 
      const_row_map_t row_mapA;
      const_entries_t entriesA;
      const_values_t valuesA;

      const_row_map_t row_mapB;
      const_entries_t entriesB;
      const_values_t valuesB;

      row_map_t row_mapC;

      // Algorithm-internal members
      ord_view_t rowCountsA;         // #rows in each category 
      host_view_t h_rowCountsA; 
      two_view_t rowOrderA;          // actual row ids in different categories 
      
      pool_t pool;                   // memory pool which will be used by the dense rows
      size_t maxFlop;                // max flop needed to compute a row
      size_t maxBeginsSize;          // smallest power of 2 greater than maxFlops 

      int vectorSize;                // used by the dense rows (suggested vector size)
      size_t shmemSize;              // this is set by user, currently it is not used 

      
      bool verbose;
      
      SymbolicFunctor(ordinal_t numRowsA_, ordinal_t numRowsB_, ordinal_t numColsB_,
		      const_row_map_t row_mapA_, const_entries_t entriesA_, const_values_t valuesA_,
		      const_row_map_t row_mapB_, const_entries_t entriesB_, const_values_t valuesB_,
		      row_map_t row_mapC_,
		      two_view_t rowOrderA_,
		      int shmemSize_, int vectorSize_,
		      bool verbose_):
	numRowsA(numRowsA_), numRowsB(numRowsB_), numColsB(numColsB_), 
	row_mapA(row_mapA_), entriesA(entriesA_), valuesA(valuesA_), 
	row_mapB(row_mapB_), entriesB(entriesB_), valuesB(valuesB_),
	row_mapC(row_mapC_),
	rowOrderA(rowOrderA_),
	shmemSize(shmemSize_), vectorSize(vectorSize_),
	verbose(verbose_)
      {
      }

      void analyze() {
	rowCountsA = ord_view_t("rowCountsA", 6);

 	Kokkos::RangePolicy<tAnalyze, ExecSpace> pAnalyze(0, numRowsA);
	Kokkos::parallel_reduce("Analyze", pAnalyze, *this, Kokkos::Max<size_t>(maxFlop));
	ExecSpace().fence();

	h_rowCountsA = Kokkos::create_mirror_view(rowCountsA);
	Kokkos::deep_copy(h_rowCountsA, rowCountsA);

	if(verbose)
	std::cout << "Numrows: " << numRowsA 
		  << "\n Single: " << h_rowCountsA(0) 
		  << "\n   <=25: " << h_rowCountsA(1) 
		  << "\n   <=50: " << h_rowCountsA(2) 
		  << "\n  <=100: " << h_rowCountsA(3) 
		  << "\n  <=200: " << h_rowCountsA(4) 
		  << "\n   >200: " << h_rowCountsA(5)
		  << "\n MAX FLOPS: " << maxFlop
		  << std::endl;
      }

      void symbolic(){
	
	// Single-entry rows (rows with 0 or 1 nonzero)
	if(h_rowCountsA(0) > 0) {
	  Kokkos::RangePolicy<tSymbolicSingleEntry, ExecSpace> pSymbolicSingleEntry(0, h_rowCountsA(0));
	  Kokkos::parallel_for("SymbolicSingleEntry", pSymbolicSingleEntry, *this);
	}

	// Sparse rows
	size_t threadSharedSize = 21248;
	for(int i = 1; i < 5; i++) {
	  if(h_rowCountsA(i) > 0) {
	    int curTeamSize = 64 >> (i-1);	    
	    int curNumTeams = h_rowCountsA(i)/curTeamSize+1;
	    if(verbose)
	      std::cout << "TEAMSIZE: " << curTeamSize << " for " << h_rowCountsA(i) << std::endl;
	    Kokkos::TeamPolicy<tSymbolicSparse, ExecSpace> pSymbolicSparse(curNumTeams, curTeamSize);
	    pSymbolicSparse = pSymbolicSparse.set_scratch_size(0, Kokkos::PerTeam(threadSharedSize));
	    Kokkos::parallel_for("SymbolicSparse", pSymbolicSparse, *this);
	  }
	}

	// Dense rows
	if(h_rowCountsA(5) > 0) {

	  Kokkos::Timer timer;

	  // Compute the chunk size for the memory pool
	  maxBeginsSize = 1;
	  while(maxBeginsSize < maxFlop) maxBeginsSize = maxBeginsSize << 1;	    
	  size_t chunkSize = (maxBeginsSize + maxFlop) * 2;
	  if(verbose)
	    std::cout << "ChunkSize: " << chunkSize << " maxBeginsSize: " << maxBeginsSize << std::endl;

	  // Create the memory pool. I am not sure if we want to use 1024 or 4096 chunks
	  KokkosKernels::Impl::PoolType myPoolType = KokkosKernels::Impl::ManyThread2OneChunk; // this may change for CPU execution
	  pool = pool_t(1024, chunkSize, -1,  myPoolType);

	  // Launch the parallel_for for the dense rows
	  int curTeamSize = 8;
	  int curNumTeams = h_rowCountsA(5)/curTeamSize+1;
	  if(verbose)
	    std::cout << "TEAMSIZE: " << curTeamSize << " for " << h_rowCountsA(5) << std::endl;
	  Kokkos::TeamPolicy<tSymbolicDense, ExecSpace> pSymbolicDense(curNumTeams, curTeamSize, vectorSize);
	  Kokkos::parallel_for("SymbolicDense", pSymbolicDense.set_scratch_size(0, Kokkos::PerTeam(21760)), *this);
      
	  if(verbose) {
	    double denseTime = timer.seconds();
	    std::cout << "Dense: " << denseTime <<  std::endl;
	  }
	}
	ExecSpace().fence();

	// Prefix sum the entries in row_mapC 
	KokkosKernels::Impl::kk_inclusive_parallel_prefix_sum<row_map_t, ExecSpace>(numRowsA+1, row_mapC);
      }

      // The following analyzes all rows of A in terms of the FLOPs that they will require in A*B.
      // It writes the id of the each row into the category they fall in.
      // It also computes the maxFlops.
      KOKKOS_INLINE_FUNCTION
      void operator()(const tAnalyze&, const ordinal_t &row_index, size_t &update) const {

	// Single-entry rows (rows with 1 or 0 nonzeros)
	if(row_mapA[row_index+1] < row_mapA[row_index] + 2) {
	  int index = Kokkos::atomic_fetch_add(&rowCountsA[0], 1);
	  rowOrderA(index,0) = row_index;
	  return;
	}
	
	// Compute the FLOPs required by the current row
	size_t rowFlop = 0;
	for(offset_t j = row_mapA[row_index]; j < row_mapA[row_index+1]; j++){
	  ordinal_t rowB = entriesA[j];
	  rowFlop += (row_mapB[rowB+1] - row_mapB[rowB]);
	}

	// If the FLOP exceeds numColsB, set it to numColsB.
	// We will use the FLOP information to estimate the maximm usage of the hashmap accumulators.
	rowFlop = rowFlop > size_t(numColsB) ? size_t(numColsB) : rowFlop;

	// Keep track of the max FLOP as well
	if(rowFlop > update) update = rowFlop;

	// Sparse rows with FLOPs <= 25
	if(rowFlop <= 25){
	  int index = Kokkos::atomic_fetch_add(&rowCountsA[1], 1);
	  rowOrderA(index, 1) = row_index;
	}
	// Sparse rows with FLOPs <= 50 and > 25
	else if (rowFlop <= 50){
	  int index = Kokkos::atomic_fetch_add(&rowCountsA[2], 1);
	  rowOrderA(index, 2) = row_index;	  
	}
	// Sparse rows with FLOPs <= 100 and > 50
	else if (rowFlop <= 100){
	  int index = Kokkos::atomic_fetch_add(&rowCountsA[3], 1);
	  rowOrderA(index, 3) = row_index;	  
	}
	// Sparse rows with FLOPs <= 200 and > 100
	else if (rowFlop <= 200){
	  int index = Kokkos::atomic_fetch_add(&rowCountsA[4], 1);
	  rowOrderA(index, 4) = row_index;	  
	}
	// Dense rows (rows with FLOPs > 200)
	else{
	  int index = Kokkos::atomic_fetch_add(&rowCountsA[5], 1);
	  rowOrderA(index, 5) = row_index;	  
	}
      }

      // The following performs the symbolic phase for the single-entry rows
      KOKKOS_INLINE_FUNCTION
      void operator()(const tSymbolicSingleEntry&, const ordinal_t &i) const {

	ordinal_t row_index = rowOrderA(i,0);
	if(row_mapA[row_index] == row_mapA[row_index+1])
	  row_mapC(row_index+1) = 0;
	else {
	  ordinal_t row = entriesA[row_mapA[row_index]];
	  row_mapC(row_index+1) = row_mapB[row+1]-row_mapB[row];
	}
      }

      // The following performs the symbolic phase for the sparse rows.
      // It only uses the level-1 hashmap.
      // The max FLOPs required by those rows guarantee that the level-1 hashmap will not overflow.
      // The number of threads per team varies (8, 16, 32,anf 64).
      // The size of the hashmaps (which exist in each thread) is adjusted according to the team size.
      // Shared memory size used by each team is fixed (21248 bytes). 
      KOKKOS_INLINE_FUNCTION
      void operator()(const tSymbolicSparse&, const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember ) const {

	int curTeamSize = teamMember.team_size();
	ordinal_t rank = teamMember.league_rank() * curTeamSize + teamMember.team_rank();

	int kernel = 1;   // teamSize = 64
	if(curTeamSize == 32) 
	  kernel = 2;
	else if(curTeamSize == 16)
	  kernel = 3;
	else if(curTeamSize == 8)
	  kernel = 4;

	if(rank >= rowCountsA(kernel))
	  return;

	// These values are correct when teamSize=64
	ordinal_t threadMem = 83*sizeof(ordinal_t);
	ordinal_t beginsSize = 32;
	ordinal_t keysSize = 25;

	// Increase the hashmap size if there are fewer threads than 64 
	threadMem = threadMem * (64/curTeamSize);
	beginsSize = beginsSize * (64/curTeamSize);
	keysSize = keysSize * (64/curTeamSize);
	
	// Go to the thread-private region of the shared memory
	ordinal_t threadRank = teamMember.team_rank();
	char *shared_memory = (char *) (teamMember.team_shmem().get_shmem(21248));
	shared_memory += threadMem*threadRank;;

	// Set the pointers to be used by the hashmap
	ordinal_t *begins = (ordinal_t *)shared_memory;
	shared_memory += beginsSize*sizeof(ordinal_t);
	ordinal_t *nexts = (ordinal_t *)shared_memory;
	shared_memory += keysSize*sizeof(ordinal_t);
	ordinal_t *keys = (ordinal_t *)shared_memory;
	shared_memory += keysSize*sizeof(ordinal_t);
	ordinal_t *used_hash_sizes = (ordinal_t *)shared_memory;

	// Initialize begins.
	Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, beginsSize), [&](const ordinal_t &i){
	    begins[i] = -1;
	  });

	// Initialize the number of keys to zero
	Kokkos::single(Kokkos::PerThread(teamMember),[&] () {
	    used_hash_sizes[0] = 0;
	  });

	// Create the hashmap
	hmap_t hmfast(keysSize, beginsSize-1, begins, nexts, keys, nullptr);

	// Insert the entries into the hashmap
	size_t failed = 0;
	ordinal_t row_index = rowOrderA(rank, kernel);
	Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember, row_mapA[row_index], row_mapA[row_index+1]), [&] (const offset_t &i, size_t &update) {
	    ordinal_t rowB = entriesA[i];
	    for(offset_t j = row_mapB[rowB]; j < row_mapB[rowB+1]; j++) {
	      ordinal_t colB = entriesB[j];
	      int err = hmfast.vector_atomic_insert_into_hash(colB, used_hash_sizes);
	      update += err;
	    }
	  }, Kokkos::Sum<size_t>(failed));

	// Set the size of the row
	Kokkos::single(Kokkos::PerThread(teamMember),[&] () {
	    row_mapC(row_index+1) = used_hash_sizes[0];
	  });
      }

      // The following performs the symbolic phase for the dense rows.
      // It uses both level-1 and level-2 hashmaps.
      // A chunk from the memory pool is requested when level-1 hashmap overflows.
      // teemSize=8 is picked based on experiments on a limited dataset.
      // Shared memory size used by each team (21760 bytes) is slightly larger than that of the sparse one (21248 bytes).
      KOKKOS_INLINE_FUNCTION
      void operator()(const tSymbolicDense&, const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember ) const {

	int curTeamSize = teamMember.team_size();
	ordinal_t rank = teamMember.league_rank() * curTeamSize + teamMember.team_rank();
	
	if(rank >= rowCountsA(5))
	  return;

	// Using slightly more shared memory than the sparse one
	ordinal_t threadMem = 85*sizeof(ordinal_t);
	ordinal_t beginsSize = 32;
	ordinal_t keysSize = 25;
	threadMem = threadMem * (64/curTeamSize);
	beginsSize = beginsSize * (64/curTeamSize);
	keysSize = keysSize * (64/curTeamSize);

	// Go to the thread-private region of the shared memory
	ordinal_t threadRank = teamMember.team_rank();
	char *shared_memory = (char *) (teamMember.team_shmem().get_shmem(21248));
	shared_memory += threadMem*threadRank;;

	// Set the pointers for the level-1 hashmap
	ordinal_t *begins = (ordinal_t *)shared_memory;
	shared_memory += beginsSize*sizeof(ordinal_t);
	ordinal_t *nexts = (ordinal_t *)shared_memory;
	shared_memory += keysSize*sizeof(ordinal_t);
	ordinal_t *keys = (ordinal_t *)shared_memory;
	shared_memory += keysSize*sizeof(ordinal_t);
	ordinal_t *used_hash_sizes = (ordinal_t *)shared_memory;
	shared_memory += 2*sizeof(ordinal_t);
	ordinal_t *globally_used_hash_count = (ordinal_t *)shared_memory;

	// Initialize begins
	Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, beginsSize),[&](const ordinal_t &i){
	    begins[i] = -1;
	  });

	// Initialize the number of used keys and hashes 
	Kokkos::single(Kokkos::PerThread(teamMember),[&] () {
	    used_hash_sizes[0] = 0;   // level-1
	    used_hash_sizes[1] = 0;   // level-2
	    globally_used_hash_count[0] = 0;
	  });

	// Create both hashmaps. The arrays of the level-2 hashmap are not allocated yet.
	hmap_t hmfast(keysSize, beginsSize-1, begins, nexts, keys, nullptr);
	hmap_t hmslow(maxFlop, maxBeginsSize-1, NULL, NULL, NULL, NULL);

	// Try to insert the entries in the level-1 hashmap
	size_t failed = 0;
	bool slowAllocated = false;
	ordinal_t *globally_used_hash_indices = NULL;
	ordinal_t row_index = rowOrderA(rank,5);
	for(offset_t i = row_mapA[row_index]; i < row_mapA[row_index+1]; i++) {
	  size_t iterfailed = 0;
	  ordinal_t rowB = entriesA[i];
	  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember, row_mapB[rowB], row_mapB[rowB+1]), [&] (const offset_t &j, size_t &update) {
	      ordinal_t colB = entriesB[j];
	      int err = hmfast.vector_atomic_insert_into_hash(colB, used_hash_sizes);
	      update += err;
	    }, Kokkos::Sum<size_t>(iterfailed));
	  failed += iterfailed;
	  if(failed > 0)
	    break;
	}

	// Use the level-2 hashmap if the level-1 hashmap overflowed
	volatile ordinal_t * tmp = NULL;
	if(failed) {
	  if(!slowAllocated) {
	    
	    // Request a chunk from the memory pool
            while (tmp == NULL){
	      Kokkos::single(Kokkos::PerThread(teamMember),[&] (volatile ordinal_t * &memptr) {
		  memptr = (volatile ordinal_t * )( pool.allocate_chunk(rank));
		}, tmp);
	    }

	    // Set the pointers of the level-2 hashmap in the allocated chunk
	    slowAllocated = true;
	    globally_used_hash_indices = (ordinal_t *)tmp;
	    tmp += maxBeginsSize;
	    hmslow.hash_begins = (ordinal_t *)tmp;
	    tmp += maxBeginsSize;
	    hmslow.hash_nexts = (ordinal_t *)tmp;
	    tmp += maxFlop;
	    hmslow.keys = (ordinal_t *)tmp;
	  }

	  // Insert the entries which don't exist in the level-1 hashmap into the level-2 hashmap
	  for(offset_t i = row_mapA[row_index]; i < row_mapA[row_index+1]; i++) {
	    ordinal_t rowB = entriesA[i]; 
	    Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, row_mapB[rowB], row_mapB[rowB+1]), [&] (const offset_t &j) {
		ordinal_t colB = entriesB[j];
		int err = hmfast.vector_atomic_insert_into_hash(colB, used_hash_sizes);	      
		if(err) hmslow.vector_atomic_insert_into_hash_TrackHashes(colB, used_hash_sizes+1,
									  globally_used_hash_count,
									  globally_used_hash_indices
									  );
	      });
	  }
	}

	// Clean the hashes used in the memory pool chunk and release it
	if(slowAllocated) {
	  
	  ordinal_t dirty_hashes = globally_used_hash_count[0];
	  Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, dirty_hashes),[&] (ordinal_t i) {
	      ordinal_t dirty_hash = globally_used_hash_indices[i];
	      hmslow.hash_begins[dirty_hash] = -1;
	    });

	  Kokkos::single(Kokkos::PerThread(teamMember),[&] () {
	      pool.release_chunk((const ordinal_t *)globally_used_hash_indices);
	    });
	}
	
	// Set the size of the row
	Kokkos::single(Kokkos::PerThread(teamMember),[&] () {

	    if(used_hash_sizes[0] > keysSize)
	      used_hash_sizes[0] = keysSize;
	    row_mapC(row_index+1) = used_hash_sizes[0] + used_hash_sizes[1];
	  });	
      }
    };

    template <typename HandleType, typename LayoutType>
    void
    SPGEMM<HandleType,LayoutType>::symbolic_impl(row_map_t row_mapC)
    {

      Kokkos::Timer timer;

      // This 2D array will hold the indices of the rows in each category
      two_view_t rowOrderA(Kokkos::ViewAllocateWithoutInitializing("rowOrderA"), a_row_cnt);
      double allocTime = timer.seconds();

      // Suggested vector size is only used for the dense rows
      int vectorSize = handle->get_suggested_vector_size(b_row_cnt, entriesB.extent(0));
      int shmemSize = handle->get_shmem_size();  // curently not used 

      // Create the functor
      timer.reset();
      SymbolicFunctor sf(a_row_cnt, b_row_cnt, b_col_cnt,
			 row_mapA, entriesA, valuesA,
			 row_mapB, entriesB, valuesB,
			 row_mapC,
			 rowOrderA,
			 shmemSize, vectorSize,
			 handle->get_verbose());
      double structTime = timer.seconds();

      // Perform the analysis
      // NOTE: The arrays/scalars created/used by this function can also be useful in the numeric phase.
      // SpGEMM handle can be used to transfer those arrays into the numeric phase.
      // Candidates to be transferred: rowOrderA, rowCountsA, maxFlops, etc.
      timer.reset();
      sf.analyze();
      double analyzeTime = timer.seconds();
      
      // Perform the symbolic phase
      timer.reset();
      sf.symbolic();
      double symbolicTime = timer.seconds();

      if(handle->get_verbose())
	std::cout << "Alloc: " << allocTime << " Struct: " << structTime << " Analyze: " << analyzeTime << " Symbolic: " << symbolicTime << std::endl;

    }

  }
}
