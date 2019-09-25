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

#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

#ifndef _SPTRSVHANDLE_HPP
#define _SPTRSVHANDLE_HPP

namespace KokkosSparse {
namespace Experimental {

// TP2 algorithm has issues with some offset-ordinal combo to be addressed
enum class SPTRSVAlgorithm { SEQLVLSCHD_RP, SEQLVLSCHD_TP1/*, SEQLVLSCHED_TP2*/, SUPERNODAL_NAIVE, SUPERNODAL_ETREE, SUPERNODAL_DAG };

template <class size_type_, class lno_t_, class scalar_t_,
          class ExecutionSpace,
          class TemporaryMemorySpace,
          class PersistentMemorySpace>
class SPTRSVHandle {
public:

  typedef ExecutionSpace HandleExecSpace;
  typedef TemporaryMemorySpace HandleTempMemorySpace;
  typedef PersistentMemorySpace HandlePersistentMemorySpace;

  typedef ExecutionSpace execution_space;
  typedef HandlePersistentMemorySpace memory_space;


  typedef typename std::remove_const<size_type_>::type  size_type;
  typedef const size_type const_size_type;

  typedef typename std::remove_const<lno_t_>::type  nnz_lno_t;
  typedef const nnz_lno_t const_nnz_lno_t;

  typedef typename std::remove_const<scalar_t_>::type  nnz_scalar_t;
  typedef const nnz_scalar_t const_nnz_scalar_t;


  typedef typename Kokkos::View<size_type *, HandleTempMemorySpace> nnz_row_view_temp_t;
  typedef typename Kokkos::View<size_type *, HandlePersistentMemorySpace> nnz_row_view_t;
 // typedef typename row_lno_persistent_work_view_t::HostMirror row_lno_persistent_work_host_view_t; //Host view type

  typedef typename Kokkos::View<nnz_scalar_t *, HandleTempMemorySpace> nnz_scalar_view_temp_t;
  typedef typename Kokkos::View<nnz_scalar_t *, HandlePersistentMemorySpace> nnz_scalar_view_t;


  typedef typename Kokkos::View<nnz_lno_t *, HandleTempMemorySpace> nnz_lno_view_temp_t;
  typedef typename Kokkos::View<nnz_lno_t *, HandlePersistentMemorySpace> nnz_lno_view_t;
 // typedef typename nnz_lno_persistent_work_view_t::HostMirror nnz_lno_persistent_work_host_view_t; //Host view type


  typedef typename std::make_signed<typename nnz_row_view_t::non_const_value_type>::type signed_integral_t;
  typedef Kokkos::View< signed_integral_t*, typename nnz_row_view_t::array_layout, typename nnz_row_view_t::device_type, typename nnz_row_view_t::memory_traits > signed_nnz_lno_view_t;

#ifdef KOKKOSKERNELS_ENABLE_SUPERNODAL
  typedef typename execution_space::memory_space  supercols_memory_space;

  typedef Kokkos::DefaultHostExecutionSpace                      supercols_host_execution_space;
  typedef typename supercols_host_execution_space::memory_space  supercols_host_memory_space;

  typedef Kokkos::View<int*, supercols_memory_space>       supercols_t;
  typedef Kokkos::View<int*, supercols_host_memory_space>  supercols_host_t;

  typedef typename Kokkos::View<nnz_scalar_t*, memory_space> WorkspaceType;
#endif

private:

  signed_nnz_lno_view_t level_list;
  nnz_lno_view_t nodes_per_level;
  nnz_lno_view_t nodes_grouped_by_level;

  size_type nrows;
  size_type nlevel;

  bool lower_tri;

  bool symbolic_complete;

  SPTRSVAlgorithm algm;

  int team_size;
  int vector_size;

#ifdef KOKKOSKERNELS_ENABLE_SUPERNODAL
  // number of supernodal columns
  signed_integral_t nsuper;

  // etree, parent's id for each supernodal column
  supercols_host_t etree_host;

  // dag
  int **dag_host;

  // map from supernode to column id, i.e., superdols[s] = the first column id of s-th supernode
  supercols_host_t supercols_host; // on the host
  supercols_t      supercols;      // on the default host/device

  // workspace size
  signed_integral_t lwork;
  WorkspaceType work;
  // offset to workspace for each supernodal column
  supercols_host_t work_offset_host;
  supercols_t      work_offset;

  // type of kernels used at each level
  int sup_size_unblocked;
  int sup_size_blocked;
  supercols_host_t diag_kernel_type_host;
  supercols_t      diag_kernel_type;
  supercols_host_t kernel_type_host;
  supercols_t      kernel_type;

  int num_streams;
  #if defined(KOKKOS_ENABLE_CUDA)
  cudaStream_t *cuda_streams;
  #endif
#endif

public:

  SPTRSVHandle ( SPTRSVAlgorithm choice, const size_type nrows_, bool lower_tri_, bool symbolic_complete_ = false ) :
    level_list(),
    nodes_per_level(),
    nodes_grouped_by_level(),
    nrows(nrows_),
    nlevel(0),
    lower_tri( lower_tri_ ),
    symbolic_complete( symbolic_complete_ ),
    algm(choice),
    team_size(-1),
    vector_size(-1)
  {}

#if 0
  SPTRSVHandle ( SPTRSVAlgorithm choice, const size_type nrows_, bool lower_tri_, bool symbolic_complete_ = false ) :
    level_list( Kokkos::ViewAllocateWithoutInitializing("level_list"), nrows),
    nodes_per_level("nodes_per_level", nrows),
    nodes_grouped_by_level("nodes_grouped_by_level", nrows),
    nrows(nrows_),
    nlevel(0),
    lower_tri( lower_tri_ ),
    symbolic_complete( symbolic_complete_ ),
    algm(choice)
  {
    // WithoutInitializing
    Kokkos::deep_copy( level_list, signed_integral_t(-1) );
  }

/*
  template <class rhslno_row_view_t_,
          class rhslno_nnz_view_t_,
          class rhsscalar_nnz_view_t_,
          class rhsExecutionSpace,
          class rhsMemorySpace>
  SPTRSVHandle ( SPTRSVHandle< rhslno_row_view_t_, rhslno_nnz_view_t_, rhsscalar_nnz_view_t_, rhsExecutionSpace, rhsMemorySpace > & rhs ) {

    this->level_list = rhs.level_list;
    this->nodes_per_level = rhs.nodes_per_level;
    this->nodes_grouped_by_level = rhs.nodes_grouped_by_level;
    this->nrows = rhs.nrows;
    this->nlevel = rhs.nlevel;
    this->lower_tri = rhs.lower_tri;
    this->symbolic_complete = rhs.symbolic_complete;
    this->algm = rhs.algm;
  }

  template <class rhslno_row_view_t_,
          class rhslno_nnz_view_t_,
          class rhsscalar_nnz_view_t_,
          class rhsExecutionSpace,
          class rhsMemorySpace>
  SPTRSVHandle & operator= ( SPTRSVHandle< rhslno_row_view_t_, rhslno_nnz_view_t_, rhsscalar_nnz_view_t_, rhsExecutionSpace, rhsMemorySpace > & rhs ) {

    this->level_list = rhs.level_list;
    this->nodes_per_level = rhs.nodes_per_level;
    this->nodes_grouped_by_level = rhs.nodes_grouped_by_level;
    this->nrows = rhs.nrows;
    this->nlevel = rhs.nlevel;
    this->lower_tri = rhs.lower_tri;
    this->symbolic_complete = rhs.symbolic_complete;
    this->algm = rhs.algm;
    return *this;
  }
*/

#endif

  void reset_handle( const size_type nrows_ ) {
    set_nrows(nrows_);
    set_num_levels(0);
    level_list = signed_nnz_lno_view_t( Kokkos::ViewAllocateWithoutInitializing("level_list"), nrows_),
    Kokkos::deep_copy( level_list, signed_integral_t(-1) );
    nodes_per_level =  nnz_lno_view_t("nodes_per_level", nrows_),
    nodes_grouped_by_level = nnz_lno_view_t("nodes_grouped_by_level", nrows_),
    reset_symbolic_complete();
  }

  virtual ~SPTRSVHandle() {};


  void set_algorithm(SPTRSVAlgorithm choice) { algm = choice; }

  SPTRSVAlgorithm get_algorithm() { return algm; }

  KOKKOS_INLINE_FUNCTION
  signed_nnz_lno_view_t get_level_list() const { return level_list; }

  KOKKOS_INLINE_FUNCTION
  nnz_lno_view_t get_nodes_per_level() const { return nodes_per_level; }

  KOKKOS_INLINE_FUNCTION
  nnz_lno_view_t get_nodes_grouped_by_level() const { return nodes_grouped_by_level; }

  KOKKOS_INLINE_FUNCTION
  size_type get_nrows() const { return nrows; }

  KOKKOS_INLINE_FUNCTION
  void set_nrows(const size_type nrows_) { this->nrows = nrows_; }

  bool is_lower_tri() const { return lower_tri; }
  bool is_upper_tri() const { return !lower_tri; }
  void set_is_lower_tri(bool lower_tri_) { lower_tri = lower_tri_; }

  bool is_symbolic_complete() const { return symbolic_complete; }

  size_type get_num_levels() const { return nlevel; }
  void set_num_levels(size_type nlevels_) { this->nlevel = nlevels_; }

  void set_symbolic_complete() { this->symbolic_complete = true; }
  void reset_symbolic_complete() { this->symbolic_complete = false; }

  void set_team_size(const int ts) {this->team_size = ts;}
  int get_team_size() const {return this->team_size;}

  void set_vector_size(const int vs) {this->vector_size = vs;}
  int get_vector_size() const {return this->vector_size;}

#ifdef KOKKOSKERNELS_ENABLE_SUPERNODAL
  // set nsuper and supercols (# of supernodes, and map from supernode to column id
  void set_supernodes (signed_integral_t nsuper_, int* supercols_, int *etree_) {
    int default_sup_size_unblocked_ = 100;
    int default_sup_size_blocked_ = 200;
    set_supernodes(nsuper_, supercols_, etree_, default_sup_size_unblocked_, default_sup_size_blocked_);
  }

  void set_supernodes (signed_integral_t nsuper_, int* supercols_, int *etree_, int sup_size_unblocked_, int sup_size_blocked_) {
    this->nsuper = nsuper_;

    // etree
    this->etree_host = supercols_host_t (etree_, nsuper_);

    // supercols
    this->supercols_host = supercols_host_t (supercols_, 1+nsuper_);
    this->supercols = supercols_t ("supercols", 1+nsuper_);
    Kokkos::deep_copy (this->supercols, this->supercols_host);

    // workspace offset
    this->work_offset_host = supercols_host_t ("workoffset_host", nsuper_);
    this->work_offset = supercols_t ("workoffset", nsuper_);

    // kernel type 
    this->sup_size_unblocked = sup_size_unblocked_;
    this->sup_size_blocked = sup_size_blocked_;
    this->diag_kernel_type_host = supercols_host_t ("diag_kernel_type_host", nsuper_);
    this->diag_kernel_type = supercols_t ("diag_kernel_type", nsuper_);
    this->kernel_type_host = supercols_host_t ("kernel_type_host", nsuper_);
    this->kernel_type = supercols_t ("kernel_type", nsuper_);

    // dag, set to be null
    this->dag_host = NULL;

    // number of streams
    this->num_streams = 0;
  }

  // set supernodal dag
  void set_supernodal_dag (int** dag_) {
    this->dag_host = dag_;
  }

  // return number of supernodes
  signed_integral_t get_num_supernodes () {
    return this->nsuper;
  }

  // return map to supernode to column id
  const int* get_supercols () {
    return this->supercols.data ();
  }

  const int* get_supercols_host () {
    return this->supercols_host.data ();
  }

  // return parents info in etree of supernodes
  const int* get_etree_parents () {
    return this->etree_host.data ();
  }

  // return parents info in etree of supernodes
  int** get_supernodal_dag () {
    return this->dag_host;
  }

  // workspace size
  void set_workspace_size (signed_integral_t lwork_) {
    this->lwork = lwork_;
    this->work = WorkspaceType("work", lwork);
  }
  signed_integral_t get_workspace_size () {
    return this->lwork;
  }

  // workspace
  KOKKOS_INLINE_FUNCTION
  WorkspaceType get_workspace() const {
    return this->work;
  }

  // workspace
  KOKKOS_INLINE_FUNCTION
  supercols_t get_work_offset() const { 
    return this->work_offset;
  }

  supercols_host_t get_work_offset_host() const { 
    return this->work_offset_host;
  }

  // supernode size tolerance to pick right kernel type
  int get_supernode_size_unblocked() {
    return this->sup_size_unblocked;
  }

  int get_supernode_size_blocked() {
    return this->sup_size_blocked;
  }


  void set_supernode_size_unblocked(int size_unblocked) {
    this->sup_size_unblocked = size_unblocked;
  }

  void set_supernode_size_blocked(int size_blocked) {
    this->sup_size_blocked = size_blocked;
  }

  // kernel type
  supercols_host_t get_kernel_type_host () {
    return this->kernel_type_host;
  }

  supercols_host_t get_diag_kernel_type_host () {
    return this->diag_kernel_type_host;
  }


  KOKKOS_INLINE_FUNCTION
  supercols_t get_kernel_type () {
    return this->kernel_type;
  }

  KOKKOS_INLINE_FUNCTION
  supercols_t get_diag_kernel_type () {
    return this->diag_kernel_type;
  }

  #if defined(KOKKOS_ENABLE_CUDA)
  // streams
  void setNumStreams(int num_streams_) {
    this->num_streams = num_streams_;
    if (num_streams_ > 0) {
      this->cuda_streams = (cudaStream_t*)malloc(num_streams_ * sizeof(cudaStream_t));
      for (int i = 0 ; i < num_streams_; i++) {
        cudaStreamCreate(&(this->cuda_streams[i]));
      }
    }
  }

  cudaStream_t* getStream(int id) {
    return &(this->cuda_streams[id]);
  }
  #endif
#endif

  void print_algorithm() { 
    if ( algm == SPTRSVAlgorithm::SEQLVLSCHD_RP )
      std::cout << "SEQLVLSCHD_RP" << std::endl;;

    if ( algm == SPTRSVAlgorithm::SEQLVLSCHD_TP1 )
      std::cout << "SEQLVLSCHD_TP1" << std::endl;;

    if ( algm == SPTRSVAlgorithm::SUPERNODAL_NAIVE )
      std::cout << "SUPERNODAL_NAIVE" << std::endl;;

    if ( algm == SPTRSVAlgorithm::SUPERNODAL_ETREE )
      std::cout << "SUPERNODAL_ETREE" << std::endl;;

    if ( algm == SPTRSVAlgorithm::SUPERNODAL_DAG )
      std::cout << "SUPERNODAL_DAG" << std::endl;;

    /*
    if ( algm == SPTRSVAlgorithm::SEQLVLSCHED_TP2 ) {
      std::cout << "SEQLVLSCHED_TP2" << std::endl;;
      std::cout << "WARNING: With CUDA this is currently only reliable with int-int ordinal-offset pair" << std::endl;
    }
    */
  }

  inline SPTRSVAlgorithm StringToSPTRSVAlgorithm(std::string & name) {
    if(name=="SPTRSV_DEFAULT")             return SPTRSVAlgorithm::SEQLVLSCHD_RP;
    else if(name=="SPTRSV_RANGEPOLICY")    return SPTRSVAlgorithm::SEQLVLSCHD_RP;
    else if(name=="SPTRSV_TEAMPOLICY1")    return SPTRSVAlgorithm::SEQLVLSCHD_TP1;
    /*else if(name=="SPTRSV_TEAMPOLICY2")    return SPTRSVAlgorithm::SEQLVLSCHED_TP2;*/
    else
      throw std::runtime_error("Invalid SPTRSVAlgorithm name");
  }

};

} // namespace Experimental
} // namespace Kokkos

#endif
