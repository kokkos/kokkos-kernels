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

#ifndef _KOKKOSSPGEMMIMPL_HPP
#define _KOKKOSSPGEMMIMPL_HPP

//#define KOKKOSKERNELS_ANALYZE_COMPRESSION
#define KOKKOSKERNELS_ANALYZE_MEMORYACCESS
//#define HASHTRACK

//#define TRACK_INSERTS
//#define GPU_EXPERIMENTAL
//#define NUMERIC_USE_STATICMEM
//#define twostep
#include <KokkosKernels_Utils.hpp>
#include <KokkosKernels_SimpleUtils.hpp>
#include <KokkosKernels_SparseUtils.hpp>
#include <KokkosKernels_VectorUtils.hpp>
#include <KokkosKernels_HashmapAccumulator.hpp>
#include "KokkosKernels_Uniform_Initialized_MemoryPool.hpp"
#include "KokkosKernels_GraphColor.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace KokkosKernels{

namespace Experimental{

namespace Graph{
namespace Impl{


template <typename HandleType,
  typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
  typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
class KokkosSPGEMM{
public:

  typedef a_row_view_t_ a_row_view_t;
  typedef a_lno_nnz_view_t_ a_in_lno_nnz_view_t;
  typedef a_scalar_nnz_view_t_ a_in_scalar_nnz_view_t;

  typedef b_lno_row_view_t_ b_in_lno_row_view_t;
  typedef b_lno_nnz_view_t_ b_in_lno_nnz_view_t;
  typedef b_scalar_nnz_view_t_ b_in_scalar_nnz_view_t;



  typedef typename a_row_view_t::non_const_value_type size_type;
  typedef typename a_row_view_t::const_value_type const_size_type;


  typedef typename a_in_lno_nnz_view_t::non_const_value_type nnz_lno_t;
  typedef typename a_in_lno_nnz_view_t::const_value_type const_nnz_lno_t;

  typedef typename a_in_scalar_nnz_view_t::non_const_value_type scalar_t;
  typedef typename a_in_scalar_nnz_view_t::const_value_type const_scalar_t;


  typedef typename a_row_view_t::const_type const_a_lno_row_view_t;
  typedef typename a_row_view_t::non_const_type non_const_a_lno_row_view_t;

  typedef typename a_in_lno_nnz_view_t::const_type const_a_lno_nnz_view_t;
  typedef typename a_in_lno_nnz_view_t::non_const_type non_const_a_lno_nnz_view_t;

  typedef typename a_in_scalar_nnz_view_t::const_type const_a_scalar_nnz_view_t;
  typedef typename a_in_scalar_nnz_view_t::non_const_type non_const_a_scalar_nnz_view_t;


  typedef typename b_in_lno_row_view_t::const_type const_b_lno_row_view_t;
  typedef typename b_in_lno_row_view_t::non_const_type non_const_b_lno_row_view_t;

  typedef typename b_in_lno_nnz_view_t::const_type const_b_lno_nnz_view_t;
  typedef typename b_in_lno_nnz_view_t::non_const_type non_const_b_lno_nnz_view_t;

  typedef typename b_in_scalar_nnz_view_t::const_type const_b_scalar_nnz_view_t;
  typedef typename b_in_scalar_nnz_view_t::non_const_type non_const_b_scalar_nnz_view_t;

  typedef typename HandleType::HandleExecSpace MyExecSpace;
  typedef typename HandleType::HandleTempMemorySpace MyTempMemorySpace;
  typedef typename HandleType::HandlePersistentMemorySpace MyPersistentMemorySpace;


  typedef typename HandleType::row_lno_temp_work_view_t row_lno_temp_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_view_t row_lno_persistent_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_host_view_t row_lno_persistent_work_host_view_t; //Host view type


  typedef typename HandleType::nnz_lno_temp_work_view_t nnz_lno_temp_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_view_t nnz_lno_persistent_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_host_view_t nnz_lno_persistent_work_host_view_t; //Host view type


  typedef typename HandleType::scalar_temp_work_view_t scalar_temp_work_view_t;
  typedef typename HandleType::scalar_persistent_work_view_t scalar_persistent_work_view_t;


  typedef typename HandleType::bool_persistent_view_t bool_persistent_view_t;
  typedef typename HandleType::bool_temp_view_t bool_temp_view_t;


  typedef Kokkos::RangePolicy<MyExecSpace> my_exec_space;
  typedef Kokkos::TeamPolicy<MyExecSpace> team_policy_t ;
  typedef typename team_policy_t::member_type team_member_t ;

  struct CountTag{};




  struct FillTag{};
  struct MultiCoreDenseAccumulatorTag{};
  struct MultiCoreTag{};
  struct MultiCoreTag2{};
  struct GPUTag{};

  struct Numeric1Tag{};
  struct Numeric2Tag{};
  struct Numeric3Tag{};

  typedef Kokkos::TeamPolicy<MultiCoreDenseAccumulatorTag, MyExecSpace> multicore_dense_team_count_policy_t ;
  typedef Kokkos::TeamPolicy<MultiCoreTag, MyExecSpace> multicore_team_policy_t ;
  typedef Kokkos::TeamPolicy<MultiCoreTag2, MyExecSpace> multicore_team_policy2_t ;

  typedef Kokkos::TeamPolicy<GPUTag, MyExecSpace> gpu_team_policy_t ;
  typedef Kokkos::TeamPolicy<CountTag, MyExecSpace> team_count_policy_t ;
  typedef Kokkos::TeamPolicy<FillTag, MyExecSpace> team_fill_policy_t ;
  typedef Kokkos::TeamPolicy<Numeric1Tag, MyExecSpace> team_numeric1_policy_t ;
  typedef Kokkos::TeamPolicy<Numeric2Tag, MyExecSpace> team_numeric2_policy_t ;
  typedef Kokkos::TeamPolicy<Numeric3Tag, MyExecSpace> team_numeric3_policy_t ;


  typedef Kokkos::TeamPolicy<MultiCoreDenseAccumulatorTag, MyExecSpace, Kokkos::Schedule<Kokkos::Dynamic> > dynamic_multicore_dense_team_count_policy_t ;
  typedef Kokkos::TeamPolicy<MultiCoreTag, MyExecSpace, Kokkos::Schedule<Kokkos::Dynamic> > dynamic_multicore_team_policy_t ;
  typedef Kokkos::TeamPolicy<MultiCoreTag2, MyExecSpace, Kokkos::Schedule<Kokkos::Dynamic> > dynamic_multicore_team_policy2_t ;


  typedef Kokkos::TeamPolicy<CountTag, MyExecSpace, Kokkos::Schedule<Kokkos::Dynamic> > dynamic_team_count_policy_t ;
  typedef Kokkos::TeamPolicy<FillTag, MyExecSpace, Kokkos::Schedule<Kokkos::Dynamic> > dynamic_team_fill_policy_t ;
  typedef Kokkos::TeamPolicy<Numeric1Tag, MyExecSpace, Kokkos::Schedule<Kokkos::Dynamic> > dynamic_team_numeric1_policy_t ;
  typedef Kokkos::TeamPolicy<Numeric2Tag, MyExecSpace, Kokkos::Schedule<Kokkos::Dynamic> > dynamic_team_numeric2_policy_t ;
  typedef Kokkos::TeamPolicy<Numeric3Tag, MyExecSpace, Kokkos::Schedule<Kokkos::Dynamic> > dynamic_team_numeric3_policy_t ;

  typedef Kokkos::TeamPolicy<MyExecSpace, Kokkos::Schedule<Kokkos::Dynamic> > dynamic_team_policy_t ;


private:
  HandleType *handle;
  nnz_lno_t a_row_cnt;
  nnz_lno_t b_row_cnt;
  nnz_lno_t b_col_cnt;


  const_a_lno_row_view_t row_mapA;
  const_a_lno_nnz_view_t entriesA;
  const_a_scalar_nnz_view_t valsA;
  bool transposeA;

  const_b_lno_row_view_t row_mapB;
  const_b_lno_nnz_view_t entriesB;
  const_b_scalar_nnz_view_t valsB;
  bool transposeB;

  const size_t shmem_size;
  const size_t concurrency;
  const bool use_dynamic_schedule;
  const bool KOKKOSKERNELS_VERBOSE;
  //const int KOKKOSKERNELS_VERBOSE = 1;


  //////////////////////////////////////////////////////////////////////////////
  //////Function and Struct for matrix compression.
  //////Declerations are at KokkosKernels_SPGEMM_impl_compression.hpp
  //////////////////////////////////////////////////////////////////////////////

  /**
   * \brief Given a symbolic matrix (a graph), it compresses the graph using bits.
   * \param in_row_map: input row pointers.
   * \param in_entries: input column entries
   * \param out_row_map: output row pointers of the compressed matrix
   * \param out_nnz_indices: output, column set indices of the output matrix.
   * \param out_nnz_sets: output, column sets of the output matrix.
   *
   */
  template <typename in_row_view_t, typename in_nnz_view_t, typename out_rowmap_view_t, typename out_nnz_view_t>
  void compressMatrix(
      nnz_lno_t n, size_type nnz,
      in_row_view_t in_row_map, in_nnz_view_t in_entries,
      out_rowmap_view_t out_row_map,
      out_nnz_view_t out_nnz_indices,
      out_nnz_view_t out_nnz_sets);

  /**
   *\brief Functor to zip the B matrix.
   */
  template <typename row_view_t, typename nnz_view_t, typename new_row_view_t, typename new_nnz_view_t, typename pool_memory_space>
  struct SingleStepZipMatrix;




  //////////////////////////////////////////////////////////////////////////
  /////BELOW CODE IS TO for SPEED SPGEMM
  ////DECL IS AT _speed.hpp
  //////////////////////////////////////////////////////////////////////////
  template <typename a_row_view_t, typename a_nnz_view_t, typename a_scalar_view_t,
            typename b_row_view_t, typename b_nnz_view_t, typename b_scalar_view_t,
            typename c_row_view_t, typename c_nnz_view_t, typename c_scalar_view_t,
            typename mpool_type>
  struct NumericCMEM_CPU;

  template <typename a_row_view_t__, typename a_nnz_view_t__, typename a_scalar_view_t__,
            typename b_row_view_t__, typename b_nnz_view_t__, typename b_scalar_view_t__,
            typename c_row_view_t__, typename c_nnz_view_t__, typename c_scalar_view_t__>
  struct NumericCMEM;
  /**
   * \brief Numeric phase with speed method
   */
  template <typename c_row_view_t, typename c_lno_nnz_view_t, typename c_scalar_nnz_view_t>
  void KokkosSPGEMM_numeric_speed(
      c_row_view_t rowmapC_,
      c_lno_nnz_view_t entriesC_,
      c_scalar_nnz_view_t valuesC_,
      KokkosKernels::Experimental::Util::ExecSpaceType my_exec_space);



  //////////////////////////////////////////////////////////////////////////
  /////BELOW CODE IS TO for colored SPGEMM
  ////DECL IS AT _color.hpp
  //////////////////////////////////////////////////////////////////////////
  template <typename a_row_view_t__, typename a_nnz_view_t__, typename a_scalar_view_t__,
            typename b_row_view_t__, typename b_nnz_view_t__, typename b_scalar_view_t__,
            typename c_row_view_t__, typename c_nnz_view_t__, typename c_scalar_view_t__>
  struct NumericCCOLOR;

  /**
   * \brief Numeric phase with speed method
   */
  template <typename c_row_view_t, typename c_lno_nnz_view_t, typename c_scalar_nnz_view_t>
  void KokkosSPGEMM_numeric_color(
      c_row_view_t rowmapC_,
      c_lno_nnz_view_t entriesC_,
      c_scalar_nnz_view_t valuesC_,
      SPGEMMAlgorithm spgemm_algorithm);


  template <typename c_row_view_t, typename c_nnz_view_t>
  void d2_color_c_matrix(
      c_row_view_t rowmapC,
      c_nnz_view_t entryIndicesC_,

      nnz_lno_t &original_num_colors,
      nnz_lno_persistent_work_host_view_t &h_color_xadj,
      nnz_lno_persistent_work_view_t &color_adj,
      nnz_lno_persistent_work_view_t &vertex_colors_to_store,

      nnz_lno_t &num_colors_in_one_step,
      nnz_lno_t &num_multi_color_steps,
      SPGEMMAlgorithm spgemm_algorithm);

  //////////////////////////////////////////////////////////////////////////
  /////BELOW CODE IS TO for kkmem SPGEMM
  ////DECL IS AT _kkmem.hpp
  //////////////////////////////////////////////////////////////////////////
  template <typename a_row_view_t, typename a_nnz_view_t, typename a_scalar_view_t,
            typename b_row_view_t, typename b_nnz_view_t, typename b_scalar_view_t,
            typename c_row_view_t, typename c_nnz_view_t, typename c_scalar_view_t,
            typename pool_memory_type>
  struct PortableNumericCHASH;

  //KKMEM only difference is work memory does not use output memory for 2nd level accumulator.
  template <typename c_row_view_t, typename c_lno_nnz_view_t, typename c_scalar_nnz_view_t>
  void KokkosSPGEMM_numeric_hash2(
        c_row_view_t rowmapC_,
        c_lno_nnz_view_t entriesC_,
        c_scalar_nnz_view_t valuesC_,
        KokkosKernels::Experimental::Util::ExecSpaceType my_exec_space);

  template <typename c_row_view_t, typename c_lno_nnz_view_t, typename c_scalar_nnz_view_t>
  void KokkosSPGEMM_numeric_hash(
        c_row_view_t rowmapC_,
        c_lno_nnz_view_t entriesC_,
        c_scalar_nnz_view_t valuesC_,
        KokkosKernels::Experimental::Util::ExecSpaceType my_exec_space);
#if defined( KOKKOS_HAVE_OPENMP )
#ifdef KOKKOSKERNELS_HAVE_OUTER
  //OUTER PRODUCT CODES
  struct Triplet;

  template <typename a_col_view_t, typename a_nnz_view_t, typename a_scalar_view_t,
            typename b_row_view_t, typename b_nnz_view_t, typename b_scalar_view_t,
            typename flop_row_view_t>
  struct OuterProduct;

  template <typename triplet_view_t>
  void sort_triplets(triplet_view_t triplets, size_t num_triplets);

  template <typename host_triplet_view_t>
  void merge_triplets_on_slow_memory(
      host_triplet_view_t *triplets,
      size_t num_blocks,
      size_t overall_size,
      host_triplet_view_t output_triplets);

  template <typename triplet_view_t,
            typename c_row_view_t,
            typename c_lno_nnz_view_t, typename c_scalar_nnz_view_t>
  size_t final_collapse_triplets_omp(
      triplet_view_t triplets,
      size_t num_triplets,
      c_row_view_t &rowmapC_,
      c_lno_nnz_view_t &entriesC_,
      c_scalar_nnz_view_t &valuesC_);

  template <typename triplet_view_t>
  size_t collapse_triplets(triplet_view_t triplets, size_t num_triplets);

  template <typename triplet_view_t>
  size_t collapse_triplets_omp(triplet_view_t triplets, size_t num_triplets, triplet_view_t out_triplets);

  template <typename a_row_view_t, typename b_row_view_t, typename flop_row_view_t>
  struct FlopsPerRowOuter;
#endif
#endif

  template <typename c_row_view_t, typename c_lno_nnz_view_t, typename c_scalar_nnz_view_t>
    void KokkosSPGEMM_numeric_outer(
          c_row_view_t &rowmapC_,
          c_lno_nnz_view_t &entriesC_,
          c_scalar_nnz_view_t &valuesC_,
          KokkosKernels::Experimental::Util::ExecSpaceType my_exec_space);
  //////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////


#ifdef KOKKOSKERNELS_ANALYZE_MEMORYACCESS
  //////////////////////////////////////////////////////////////////////////
  /////BELOW CODE IS TO CALCULATE MEMORY ACCESSES WITH HYPERGRAPH MODEL/////
  ////DECL IS AT _memaccess.hpp
  //////////////////////////////////////////////////////////////////////////

  //Functor to calculate how many flops is performed per row of C.
  template <typename a_row_view_t, typename a_nnz_view_t,
            typename b_row_view_t, typename b_nnz_view_t, typename c_row_view_t>
  struct FlopsPerRow;


  void create_read_write_hg(
      size_t &overall_flops,
      row_lno_temp_work_view_t &c_flop_rowmap,
      row_lno_temp_work_view_t &c_comp_a_net_index,
      row_lno_temp_work_view_t &c_comp_b_net_index,
      nnz_lno_temp_work_view_t &c_comp_row_index,
      nnz_lno_temp_work_view_t &c_comp_col_index);

  template <typename c_row_view_t>
  void print_read_write_cost(c_row_view_t rowmapC);


  template <typename c_row_view_t>
  void read_write_cost(
      nnz_lno_t num_colors,
      nnz_lno_t num_multi_colors,
      nnz_lno_t num_parallel_colors,
      bool isGPU,
      int num_cores,

      nnz_lno_t num_hyperthreads_in_core,
      nnz_lno_t hyper_threads_in_team,

      int vectorlane,
      const int cache_line_size,
      const int data_size,
      const int cache_size,

      nnz_lno_persistent_work_host_view_t color_xadj,
      typename nnz_lno_persistent_work_view_t::HostMirror color_adj,
      typename nnz_lno_persistent_work_view_t::HostMirror vertex_colors,

      size_t overall_flops,
      typename row_lno_temp_work_view_t::HostMirror c_flop_rowmap,
      typename row_lno_temp_work_view_t::HostMirror c_comp_a_net_index,
      typename row_lno_temp_work_view_t::HostMirror c_comp_b_net_index,
      typename nnz_lno_temp_work_view_t::HostMirror c_comp_row_index,
      typename nnz_lno_temp_work_view_t::HostMirror c_comp_col_index,
      c_row_view_t rowmapC,
      int write_type //0 -- KKMEM, 1-KKSPEED, 2- KKCOLOR 3-KKMULTICOLOR 4-KKMULTICOLOR2
      );
  struct Cache;
#endif

public:

  //////////////////////////////////////////////////////////////////////////
  /////BELOW CODE IS for public symbolic and numeric functions
  ////DECL IS AT _def.hpp
  //////////////////////////////////////////////////////////////////////////
  template <typename c_row_view_t, typename c_lno_nnz_view_t, typename c_scalar_nnz_view_t>
  void KokkosSPGEMM_numeric(c_row_view_t &rowmapC_, c_lno_nnz_view_t &entriesC_, c_scalar_nnz_view_t &valuesC_);
  //TODO: These are references only for outer product algorithm.
  //If the algorithm is removed, then remove the references.


  /**
   * \brief Symbolic phase of the SPGEMM.
   * \param rowmapC_: row pointers for the result matrix. Allocated before the call with size (n+1),
   * where n is the number of rows of first matrix.
   */
  template <typename c_row_view_t>
  void KokkosSPGEMM_symbolic(c_row_view_t rowmapC_);

  template <typename c_row_view_t, typename c_nnz_view_t>
  void write_matrix_to_plot(
      nnz_lno_t &num_colors,
      nnz_lno_persistent_work_host_view_t &h_color_xadj,
      nnz_lno_persistent_work_view_t &color_adj,
      c_row_view_t &rowmapC, c_nnz_view_t &entryIndicesC_);

  KokkosSPGEMM(
      HandleType *handle_,
      nnz_lno_t m_,
      nnz_lno_t n_,
      nnz_lno_t k_,
      const_a_lno_row_view_t row_mapA_,
      const_a_lno_nnz_view_t entriesA_,
      bool transposeA_,
      const_b_lno_row_view_t row_mapB_,
      const_b_lno_nnz_view_t entriesB_,
      bool transposeB_):handle (handle_), a_row_cnt(m_), b_row_cnt(n_), b_col_cnt(k_),
          row_mapA(row_mapA_), entriesA(entriesA_), valsA(), transposeA(transposeA_),
          row_mapB(row_mapB_), entriesB(entriesB_), valsB(), transposeB(transposeB_),
          shmem_size(handle_->get_shmem_size()), concurrency(MyExecSpace::concurrency()),
          use_dynamic_schedule(handle_->is_dynamic_scheduling()), KOKKOSKERNELS_VERBOSE(handle_->get_verbose())
          //,row_mapC(), entriesC(), valsC()
          {}

  KokkosSPGEMM(
      HandleType *handle_,
      nnz_lno_t m_,
      nnz_lno_t n_,
      nnz_lno_t k_,
        const_a_lno_row_view_t row_mapA_,
        const_a_lno_nnz_view_t entriesA_,
        const_a_scalar_nnz_view_t valsA_,
        bool transposeA_,
        const_b_lno_row_view_t row_mapB_,
        const_b_lno_nnz_view_t entriesB_,
        const_b_scalar_nnz_view_t valsB_,
        bool transposeB_):handle (handle_), a_row_cnt(m_), b_row_cnt(n_), b_col_cnt(k_),
            row_mapA(row_mapA_), entriesA(entriesA_), valsA(valsA_), transposeA(transposeA_),
            row_mapB(row_mapB_), entriesB(entriesB_), valsB(valsB_), transposeB(transposeB_),
            shmem_size(handle_->get_shmem_size()), concurrency(MyExecSpace::concurrency()),
            use_dynamic_schedule(handle_->is_dynamic_scheduling()), KOKKOSKERNELS_VERBOSE(handle_->get_verbose())
            //,row_mapB(), entriesC(), valsC()
            {}







  //////////////////////////////////////////////////////////////////////////
  /////BELOW CODE IS for symbolic phase
  ////DECL IS AT _symbolic.hpp
  //////////////////////////////////////////////////////////////////////////

  /***
   * \brief Functor to calculate the row sizes of C.
   */
  template <typename a_row_view_t, typename a_nnz_view_t,
            typename b_original_row_view_t,
            typename b_compressed_row_view_t, typename b_nnz_view_t,
            typename c_row_view_t, //typename nnz_lno_temp_work_view_t,
            typename pool_memory_space>
  struct StructureC;



  template <typename a_row_view_t, typename a_nnz_view_t,
            typename b_original_row_view_t,
            typename b_compressed_row_view_t, typename b_nnz_view_t,
            typename c_row_view_t, typename nnz_lno_temp_work_view_t,
            typename pool_memory_space>
  struct NonzeroesC;

  /**
   * \brief Functor to calculate the max flops in a row of SPGEMM.
   *
   */
  template <typename a_row_view_t, typename a_nnz_view_t,
            typename b_oldrow_view_t, typename b_row_view_t>
  struct PredicMaxRowNNZ;

  /**
   * \brief function return max flops for a row in the result multiplication.
   * \param m: number of rows in A
   * \param row_mapA: row pointers of A.
   * \param entriesA: column indices of A
   * \param row_pointers_begin_B: beginning of the row indices for B
   * \param row_pointers_end_B: end of the row indices for B
   */
  template <typename a_row_view_t, typename a_nnz_view_t,
            typename b_oldrow_view_t, typename b_row_view_t>
  int getMaxRoughRowNNZ(
      int m,
      a_row_view_t row_mapA_,
      a_nnz_view_t entriesA_,

      b_oldrow_view_t row_pointers_begin_B,
      b_row_view_t row_pointers_end_B)
  {
    //get the execution space type.
    //KokkosKernels::Experimental::Util::ExecSpaceType my_exec_space = this->handle->get_handle_exec_space();
    int suggested_vector_size = this->handle->get_suggested_vector_size(m, entriesA_.dimension_0());
    int suggested_team_size = this->handle->get_suggested_team_size(suggested_vector_size);
    nnz_lno_t team_row_chunk_size = this->handle->get_team_work_size(suggested_team_size, this->concurrency , m);

    PredicMaxRowNNZ<a_row_view_t, a_nnz_view_t, b_oldrow_view_t, b_row_view_t>
      pcnnnz(
        m,
        row_mapA_,
        entriesA_,
        row_pointers_begin_B,
        row_pointers_end_B,
        team_row_chunk_size );


    typename b_oldrow_view_t::non_const_value_type rough_size = 0;
    Kokkos::parallel_reduce( team_policy_t(m / team_row_chunk_size  + 1 , suggested_team_size, suggested_vector_size), pcnnnz, rough_size);
    MyExecSpace::fence();
    return rough_size;
  }


  template <typename a_row_view_t, typename a_nnz_view_t,
              typename b_original_row_view_t,
              typename b_compressed_row_view_t, typename b_nnz_view_t,
              typename c_row_view_t>
  void symbolic_c(
      nnz_lno_t m,
      a_row_view_t row_mapA_,
      a_nnz_view_t entriesA_,

      b_original_row_view_t old_row_mapB,
      b_compressed_row_view_t row_mapB_,
      b_nnz_view_t entriesSetIndex,
      b_nnz_view_t entriesSets,

      c_row_view_t rowmapC,
      nnz_lno_t maxNumRoughNonzeros
  ){
    typedef KokkosKernels::Experimental::Util::UniformMemoryPool< MyTempMemorySpace, nnz_lno_t> pool_memory_space;

    //get the number of rows and nonzeroes of B.
    nnz_lno_t brows = row_mapB_.dimension_0() - 1;
    size_type bnnz =  entriesSetIndex.dimension_0();

    //get the SPGEMMAlgorithm to run.
    SPGEMMAlgorithm spgemm_algorithm = this->handle->get_spgemm_handle()->get_algorithm_type();

    KokkosKernels::Experimental::Util::ExecSpaceType my_exec_space = this->handle->get_handle_exec_space();
    size_type compressed_b_size = bnnz;
#ifdef KOKKOSKERNELS_ANALYZE_COMPRESSION
    //TODO: DELETE BELOW
    {
	std::cout << "\t\t!!!!DELETE THIS PART!!!! PRINTING STATS HERE!!!!!" << std::endl;
        KokkosKernels::Experimental::Util::kk_reduce_diff_view <b_original_row_view_t, b_compressed_row_view_t, MyExecSpace>
		(brows, old_row_mapB, row_mapB_, compressed_b_size);
        std::cout << "\tcompressed_b_size:" << compressed_b_size << " bnnz:" << bnnz << std::endl;
        std::cout << "Given compressed maxNumRoughNonzeros:" << maxNumRoughNonzeros << std::endl;
        nnz_lno_t r_maxNumRoughZeros = this->getMaxRoughRowNNZ(a_row_cnt, row_mapA, entriesA, old_row_mapB,row_mapB_ );
        std::cout << "compressed r_maxNumRoughZeros:" << r_maxNumRoughZeros << std::endl;

        size_t compressed_flops = 0;
        size_t original_flops = 0;
        size_t compressd_max_flops= 0;
        size_t original_max_flops = 0;
        for (int i = 0; i < a_row_cnt; ++i){
	  int arb = row_mapA(i);
          int are = row_mapA(i + 1);
          size_t compressed_row_flops = 0;
	  size_t original_row_flops = 0;
	  for (int j = arb; j < are; ++j){
            int ae = entriesA(j);
            compressed_row_flops += row_mapB_(ae) - old_row_mapB(ae);
	    original_row_flops += old_row_mapB(ae + 1) - old_row_mapB(ae);
          }
          if (compressed_row_flops > compressd_max_flops) compressd_max_flops = compressed_row_flops;
          if (original_row_flops > original_max_flops) original_max_flops = original_row_flops;
          compressed_flops += compressed_row_flops;
          original_flops += original_row_flops;
        }
	std::cout 	<< "original_flops:" << original_flops
			<< " compressed_flops:" << compressed_flops
			<< " FLOP_REDUCTION:" << double(compressed_flops) / original_flops
			<< std::endl;
	std::cout 	<< "original_max_flops:" << original_max_flops
			<< " compressd_max_flops:" << compressd_max_flops
			<< " MEM_REDUCTION:" << double(compressd_max_flops) / original_max_flops * 2
			<< std::endl;
        std::cout 	<< "\tOriginal_B_SIZE:" << bnnz
			<< " Compressed_b_size:" << compressed_b_size
			<< std::endl;
	std::cout << " AR AC ANNZ BR BC BNNZ original_flops compressed_flops FLOP_REDUCTION original_max_flops compressd_max_flops MEM_REDUCTION riginal_B_SIZE Compressed_b_size B_SIZE_REDUCTION" <<  std::endl;
	std::cout << " " << a_row_cnt << " " << b_row_cnt << " " << entriesA.dimension_0() << " " << b_row_cnt << " " << b_col_cnt << " " << entriesB.dimension_0() << " " <<  original_flops << " " << compressed_flops << " " << double(compressed_flops) / original_flops << " " << original_max_flops << " " << compressd_max_flops << " " << double(compressd_max_flops) / original_max_flops * 2 << " " << bnnz << " " << compressed_b_size <<" "<< double(compressed_b_size) / bnnz  << std::endl;
    }
    //TODO DELETE ABOVE
#endif
    if (my_exec_space == KokkosKernels::Experimental::Util::Exec_CUDA){
 	KokkosKernels::Experimental::Util::kk_reduce_diff_view <b_original_row_view_t, b_compressed_row_view_t, MyExecSpace> (brows, old_row_mapB, row_mapB_, compressed_b_size);
        if (KOKKOSKERNELS_VERBOSE){
		std::cout << "\tcompressed_b_size:" << compressed_b_size << " bnnz:" << bnnz << std::endl;
	}
    }
    int suggested_vector_size = this->handle->get_suggested_vector_size(brows, compressed_b_size);

    //this kernel does not really work well if the vector size is less than 4.
    if (suggested_vector_size < 4 && my_exec_space == KokkosKernels::Experimental::Util::Exec_CUDA){
        if (KOKKOSKERNELS_VERBOSE){
          std::cout << "\tsuggested_vector_size:" << suggested_vector_size << " setting it to 4 for Structure kernel" << std::endl;
        }
	suggested_vector_size = 4;
    }
    int suggested_team_size = this->handle->get_suggested_team_size(suggested_vector_size);
    nnz_lno_t team_row_chunk_size = this->handle->get_team_work_size(suggested_team_size,concurrency, a_row_cnt);

    //round up maxNumRoughNonzeros to closest power of 2.
    nnz_lno_t min_hash_size = 1;
    while (maxNumRoughNonzeros > min_hash_size){
      min_hash_size *= 2;
    }

    //set the chunksize.
    size_t chunksize = min_hash_size ; //this is for used hash indices
    chunksize += min_hash_size ; //this is for the hash begins
    chunksize += maxNumRoughNonzeros ; //this is for hash nexts
    chunksize += maxNumRoughNonzeros ; //this is for hash keys
    chunksize += maxNumRoughNonzeros ; //this is for hash values

    //initizalize value for the mem pool
    int pool_init_val = -1;

    //if KKSPEED are used on CPU, or KKMEMSPEED is run with threads less than 32
    //than we use dense accumulators.
    if ((   spgemm_algorithm == KokkosKernels::Experimental::Graph::SPGEMM_KK_MEMSPEED  &&
        concurrency <=  sizeof (nnz_lno_t) * 8 &&
        my_exec_space != KokkosKernels::Experimental::Util::Exec_CUDA)
        ||
        (   spgemm_algorithm == KokkosKernels::Experimental::Graph::SPGEMM_KK_SPEED &&
            my_exec_space != KokkosKernels::Experimental::Util::Exec_CUDA)){

      nnz_lno_t col_size = this->b_col_cnt / (sizeof (nnz_lno_t) * 8)+ 1;

      nnz_lno_t max_row_size = KOKKOSKERNELS_MACRO_MIN(col_size, maxNumRoughNonzeros);
      chunksize = col_size + max_row_size;
      //if speed is set, and exec space is cpu, then  we use dense accumulators.
      //or if memspeed is set, and concurrency is not high, we use dense accumulators.
      maxNumRoughNonzeros = col_size;
      pool_init_val = 0;
      if (KOKKOSKERNELS_VERBOSE){
        std::cout << "\tDense Acc - COLS:" << col_size << " max_row_size:" << max_row_size << std::endl;
      }
    }


    size_t num_chunks = concurrency / suggested_vector_size;

    KokkosKernels::Experimental::Util::PoolType my_pool_type = KokkosKernels::Experimental::Util::OneThread2OneChunk;
    if (my_exec_space == KokkosKernels::Experimental::Util::Exec_CUDA) {
      my_pool_type = KokkosKernels::Experimental::Util::ManyThread2OneChunk;
    }


#if defined( KOKKOS_HAVE_CUDA )
    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;
    size_t required_size = num_chunks * chunksize * sizeof(nnz_lno_t);
    if (KOKKOSKERNELS_VERBOSE)
      std::cout << "\tmempool required size:" << required_size << " free_byte:" << free_byte << " total_byte:" << total_byte << std::endl;
    if (required_size + num_chunks > free_byte){
      num_chunks = ((((free_byte - num_chunks)* 0.5) /8 ) * 8) / sizeof(nnz_lno_t) / chunksize;
    }
    {
    nnz_lno_t min_chunk_size = 1;
    while (min_chunk_size * 2 < num_chunks) {
      min_chunk_size *= 2;
    }
    num_chunks = min_chunk_size;
    }
#endif
    if (KOKKOSKERNELS_VERBOSE){
      std::cout << "\tPool Size (MB):" << (num_chunks * chunksize * sizeof(nnz_lno_t)) / 1024. / 1024. << " num_chunks:" << num_chunks << " chunksize:" << chunksize << std::endl;
    }
    Kokkos::Impl::Timer timer1;
    pool_memory_space m_space(num_chunks, chunksize, pool_init_val,  my_pool_type);
    MyExecSpace::fence();

    if (KOKKOSKERNELS_VERBOSE){
      std::cout << "\tPool Alloc Time:" << timer1.seconds() << std::endl;
    }

    StructureC<a_row_view_t, a_nnz_view_t,
    b_original_row_view_t, b_compressed_row_view_t, b_nnz_view_t,
    c_row_view_t, /* nnz_lno_temp_work_view_t,*/ pool_memory_space>
    sc(
        m,
        row_mapA_,
        entriesA_,
        old_row_mapB,
        row_mapB_,
        entriesSetIndex,
        entriesSets,
        rowmapC,
        min_hash_size,
        maxNumRoughNonzeros,
        shmem_size,
        suggested_team_size,
        team_row_chunk_size,
        suggested_vector_size,
        m_space,
        my_exec_space,KOKKOSKERNELS_VERBOSE);

    if (KOKKOSKERNELS_VERBOSE){
      std::cout << "\tStructureC vector_size:" << suggested_vector_size
          << " team_size:" << suggested_team_size
          << " chunk_size:" << team_row_chunk_size
          << " shmem_size:" << shmem_size << std::endl;
    }

    timer1.reset();

    if (my_exec_space == KokkosKernels::Experimental::Util::Exec_CUDA) {
      Kokkos::parallel_for( gpu_team_policy_t(m / suggested_team_size + 1 , suggested_team_size, suggested_vector_size), sc);
    }
    else {
      if (( spgemm_algorithm == KokkosKernels::Experimental::Graph::SPGEMM_KK_MEMSPEED  &&
          concurrency <=  sizeof (nnz_lno_t) * 8)  ||
          spgemm_algorithm == KokkosKernels::Experimental::Graph::SPGEMM_KK_SPEED){

        if (use_dynamic_schedule){
          Kokkos::parallel_for( dynamic_multicore_dense_team_count_policy_t(m / team_row_chunk_size + 1 , suggested_team_size, suggested_vector_size), sc);
        }
        else {
          Kokkos::parallel_for( multicore_dense_team_count_policy_t(m / team_row_chunk_size + 1 , suggested_team_size, suggested_vector_size), sc);
        }
      }
      else {
        if (use_dynamic_schedule){
          Kokkos::parallel_for( dynamic_multicore_team_policy_t(m / team_row_chunk_size + 1 , suggested_team_size, suggested_vector_size), sc);
        }
        else {
          Kokkos::parallel_for( multicore_team_policy_t(m / team_row_chunk_size + 1 , suggested_team_size, suggested_vector_size), sc);
        }
      }
    }
    MyExecSpace::fence();

    if (KOKKOSKERNELS_VERBOSE){
      std::cout << "\tStructureC Kernel time:" << timer1.seconds() << std::endl<< std::endl;
    }
    //if we need to find the max nnz in a row.
    {
      Kokkos::Impl::Timer timer1_;
      size_type c_max_nnz = 0;
      KokkosKernels::Experimental::Util::view_reduce_max<c_row_view_t, MyExecSpace>(m, rowmapC, c_max_nnz);
      MyExecSpace::fence();
      this->handle->get_spgemm_handle()->set_max_result_nnz(c_max_nnz);

      if (KOKKOSKERNELS_VERBOSE){
        std::cout << "\tReduce Max Row Size Time:" << timer1_.seconds() << std::endl;
      }
    }

    KokkosKernels::Experimental::Util::kk_exclusive_parallel_prefix_sum<c_row_view_t, MyExecSpace>(m+1, rowmapC);
    MyExecSpace::fence();


    auto d_c_nnz_size = Kokkos::subview(rowmapC, m);
    auto h_c_nnz_size = Kokkos::create_mirror_view (d_c_nnz_size);
    Kokkos::deep_copy (h_c_nnz_size, d_c_nnz_size);
    typename c_row_view_t::non_const_value_type c_nnz_size = h_c_nnz_size();
    this->handle->get_spgemm_handle()->set_c_nnz(c_nnz_size);


    if (spgemm_algorithm == KokkosKernels::Experimental::Graph::SPGEMM_KK_COLOR ||
        spgemm_algorithm == KokkosKernels::Experimental::Graph::SPGEMM_KK_MULTICOLOR ||
        spgemm_algorithm == KokkosKernels::Experimental::Graph::SPGEMM_KK_MULTICOLOR2){

      if (KOKKOSKERNELS_VERBOSE){
        std::cout << "\tCOLORING PHASE"<<  std::endl;
      }

      nnz_lno_temp_work_view_t entryIndicesC_; //(Kokkos::ViewAllocateWithoutInitializing("entryIndicesC_"), c_nnz_size);

	  timer1.reset();
	  entryIndicesC_ = nnz_lno_temp_work_view_t (Kokkos::ViewAllocateWithoutInitializing("entryIndicesC_"), c_nnz_size);
	  //calculate the structure.
	  NonzeroesC<
	  a_row_view_t, a_nnz_view_t,
	  b_original_row_view_t, b_compressed_row_view_t, b_nnz_view_t,
	  c_row_view_t, nnz_lno_temp_work_view_t,
	  pool_memory_space>
	  nnzc_( m,
			  row_mapA_,
			  entriesA_,
			  old_row_mapB,
			  row_mapB_,
			  entriesSetIndex,
			  entriesSets,
			  rowmapC,
			  entryIndicesC_,
			  min_hash_size,
			  maxNumRoughNonzeros,
			  shmem_size,suggested_vector_size,m_space,
			  my_exec_space);

	  if (my_exec_space == KokkosKernels::Experimental::Util::Exec_CUDA) {
		  Kokkos::parallel_for( gpu_team_policy_t(m / suggested_team_size + 1 , suggested_team_size, suggested_vector_size), nnzc_);
	  }
	  else {
		  if (use_dynamic_schedule){
			  Kokkos::parallel_for( dynamic_multicore_team_policy_t(m / suggested_team_size + 1 , suggested_team_size, suggested_vector_size), nnzc_);
		  }
		  else {
			  Kokkos::parallel_for( multicore_team_policy_t(m / suggested_team_size + 1 , suggested_team_size, suggested_vector_size), nnzc_);
		  }
	  }

	  MyExecSpace::fence();


      if (KOKKOSKERNELS_VERBOSE){
        std::cout << "\t\tCOLORING-NNZ-FILL-TIME:" << timer1.seconds() <<  std::endl;
      }

      nnz_lno_t original_num_colors, num_colors_in_one_step, num_multi_color_steps;
      nnz_lno_persistent_work_host_view_t h_color_xadj;
      nnz_lno_persistent_work_view_t color_adj, vertex_colors_to_store;

      //distance-2 color
      this->d2_color_c_matrix(
          rowmapC, entryIndicesC_,
          original_num_colors, h_color_xadj, color_adj , vertex_colors_to_store,
          num_colors_in_one_step, num_multi_color_steps, spgemm_algorithm);

      std::cout << "original_num_colors:" << original_num_colors << " num_colors_in_one_step:" << num_colors_in_one_step << " num_multi_color_steps:" << num_multi_color_steps << std::endl;
      timer1.reset();

      //sort the color indices.
      for (nnz_lno_t i = 0; i < num_multi_color_steps; ++i){
        //sort the ones that have more than 32 rows.
        if (h_color_xadj(i+1) - h_color_xadj(i) <= 32) continue;
        auto sv = Kokkos::subview(color_adj,Kokkos::pair<nnz_lno_t, nnz_lno_t> (h_color_xadj(i), h_color_xadj(i+1)));
        //KokkosKernels::Experimental::Util::print_1Dview(sv, i ==47);
        //TODO for some reason kokkos::sort is failing on views with size 56 and 112.
        //for now we use std::sort. Delete below comment, and delete the code upto fence.
        //Kokkos::sort(sv);
        //
        auto h_sv = Kokkos::create_mirror_view (sv);
        Kokkos::deep_copy(h_sv,sv);
        auto* p_sv = h_sv.ptr_on_device();
        std::sort (p_sv, p_sv + h_color_xadj(i+1) - h_color_xadj(i));
        Kokkos::deep_copy(sv,h_sv);
        MyExecSpace::fence();
      }

      if (KOKKOSKERNELS_VERBOSE){
        std::cout << "\t\tCOLOR-SORT-TIME:" << timer1.seconds() <<  std::endl;
      }
      this->handle->get_spgemm_handle()->set_color_xadj(
          original_num_colors,
          h_color_xadj, color_adj, vertex_colors_to_store,
          num_colors_in_one_step, num_multi_color_steps);
      this->handle->get_spgemm_handle()->set_c_column_indices(entryIndicesC_);
    }

  }

};


}
}
}
}
#include "KokkosKernels_SPGEMM_imp_outer.hpp"
#include "KokkosKernels_SPGEMM_impl_memaccess.hpp"
#include "KokkosKernels_SPGEMM_impl_kkmem.hpp"
#include "KokkosKernels_SPGEMM_impl_color.hpp"
#include "KokkosKernels_SPGEMM_impl_speed.hpp"
#include "KokkosKernels_SPGEMM_impl_compression.hpp"
#include "KokkosKernels_SPGEMM_impl_def.hpp"
#include "KokkosKernels_SPGEMM_impl_symbolic.hpp"
#endif
