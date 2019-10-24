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
//BMK: DEBUGGING
#include <set>
#include <unordered_map>
#include <vector>
#ifndef _KOKKOSGSIMP_HPP
#define _KOKKOSGSIMP_HPP

namespace KokkosSparse{


  namespace Impl{


    template <typename HandleType, typename lno_row_view_t_, typename lno_nnz_view_t_, typename scalar_nnz_view_t_>
    class GaussSeidel{

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

      struct BlockTag{};
      struct BigBlockTag{};

      typedef Kokkos::TeamPolicy<BlockTag, MyExecSpace> block_team_fill_policy_t ;
      typedef Kokkos::TeamPolicy<BigBlockTag, MyExecSpace> bigblock_team_fill_policy_t ;
      typedef KokkosKernels::Impl::UniformMemoryPool< MyTempMemorySpace, nnz_scalar_t> pool_memory_space;

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

      struct PSGS{
        row_lno_persistent_work_view_t _xadj;
        nnz_lno_persistent_work_view_t _adj; // CSR storage of the graph.
        scalar_persistent_work_view_t _adj_vals; // CSR storage of the graph.

        scalar_persistent_work_view_t _Xvector /*output*/;
        scalar_persistent_work_view_t _Yvector;

        scalar_persistent_work_view_t _permuted_inverse_diagonal;

        nnz_scalar_t omega;

        PSGS(row_lno_persistent_work_view_t xadj_, nnz_lno_persistent_work_view_t adj_, scalar_persistent_work_view_t adj_vals_,
             scalar_persistent_work_view_t Xvector_, scalar_persistent_work_view_t Yvector_, nnz_lno_persistent_work_view_t color_adj_,
             nnz_scalar_t omega_,
             scalar_persistent_work_view_t permuted_inverse_diagonal_):
          _xadj( xadj_),
          _adj( adj_),
          _adj_vals( adj_vals_),
          _Xvector( Xvector_),
          _Yvector( Yvector_), _permuted_inverse_diagonal(permuted_inverse_diagonal_),
          omega(omega_){}

        KOKKOS_INLINE_FUNCTION
        void operator()(const nnz_lno_t &ii) const {

          size_type row_begin = _xadj[ii];
          size_type row_end = _xadj[ii + 1];

          nnz_scalar_t sum = _Yvector[ii];

          for (size_type adjind = row_begin; adjind < row_end; ++adjind){
            nnz_lno_t colIndex = _adj[adjind];
            nnz_scalar_t val = _adj_vals[adjind];
            sum -= val * _Xvector[colIndex];
          }
          nnz_scalar_t invDiagonalVal = _permuted_inverse_diagonal[ii];
          _Xvector[ii] = _Xvector[ii] + omega * sum * invDiagonalVal;
        }
      };

      struct Team_PSGS{

        row_lno_persistent_work_view_t _xadj;
        nnz_lno_persistent_work_view_t _adj; // CSR storage of the graph.
        scalar_persistent_work_view_t _adj_vals; // CSR storage of the graph.

        scalar_persistent_work_view_t _Xvector /*output*/;
        scalar_persistent_work_view_t _Yvector;
        nnz_lno_t _color_set_begin;
        nnz_lno_t _color_set_end;

        scalar_persistent_work_view_t _permuted_inverse_diagonal;
        nnz_lno_t block_size;
        nnz_lno_t team_work_size;
        const size_t shared_memory_size;

        int suggested_team_size;
        const size_t thread_shared_memory_scalar_size;
        int vector_size;
        const pool_memory_space pool;
        const nnz_lno_t num_max_vals_in_l1, num_max_vals_in_l2;
        bool is_backward;

        nnz_scalar_t omega;


        Team_PSGS(row_lno_persistent_work_view_t xadj_, nnz_lno_persistent_work_view_t adj_, scalar_persistent_work_view_t adj_vals_,
                  scalar_persistent_work_view_t Xvector_, scalar_persistent_work_view_t Yvector_,
                  nnz_lno_t color_set_begin, nnz_lno_t color_set_end,
                  scalar_persistent_work_view_t permuted_inverse_diagonal_,
                  pool_memory_space pms,
                  nnz_lno_t _num_max_vals_in_l1 = 0,
                  nnz_lno_t _num_max_vals_in_l2 = 0,
                  nnz_scalar_t omega_ = Kokkos::Details::ArithTraits<nnz_scalar_t>::one(),

                  nnz_lno_t block_size_ = 1,
                  nnz_lno_t team_work_size_ = 1,
                  size_t shared_memory_size_ = 16,
                  int suggested_team_size_ = 1,
                  int vector_size_ = 1):
          _xadj( xadj_),
          _adj( adj_),
          _adj_vals( adj_vals_),
          _Xvector( Xvector_),
          _Yvector( Yvector_),
          _color_set_begin(color_set_begin),
          _color_set_end(color_set_end), _permuted_inverse_diagonal(permuted_inverse_diagonal_),
          block_size(block_size_),
          team_work_size(team_work_size_),
          shared_memory_size(shared_memory_size_),
          suggested_team_size(suggested_team_size_),
          thread_shared_memory_scalar_size(((shared_memory_size / suggested_team_size / 8) * 8 ) / sizeof(nnz_scalar_t) ),
          vector_size(vector_size_), pool (pms), num_max_vals_in_l1(_num_max_vals_in_l1),
          num_max_vals_in_l2(_num_max_vals_in_l2), is_backward(false),
          omega(omega_){}

        KOKKOS_INLINE_FUNCTION
        void operator()(const team_member_t & teamMember) const {

          nnz_lno_t ii = teamMember.league_rank()  * teamMember.team_size()+ teamMember.team_rank() + _color_set_begin;
          if (ii >= _color_set_end)
            return;



          size_type row_begin = _xadj[ii];
          size_type row_end = _xadj[ii + 1];

          nnz_scalar_t product = 0 ;
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
              nnz_scalar_t invDiagonalVal = _permuted_inverse_diagonal[ii];
              _Xvector[ii] += omega * (_Yvector[ii] - product) * invDiagonalVal;
            });
        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const BigBlockTag&, const team_member_t & teamMember) const {


          const nnz_lno_t team_row_begin = teamMember.league_rank() * team_work_size + _color_set_begin;
          const nnz_lno_t team_row_end = KOKKOSKERNELS_MACRO_MIN(team_row_begin + team_work_size, _color_set_end);
          //get the shared memory and shift it based on the thread index so that each thread has private memory.
          nnz_scalar_t *all_shared_memory = (nnz_scalar_t *) (teamMember.team_shmem().get_shmem(shared_memory_size));

          all_shared_memory += thread_shared_memory_scalar_size * teamMember.team_rank();

          //store the diagonal positions, because we want to update them on shared memory if we update them on global memory.
          nnz_lno_t *diagonal_positions = (nnz_lno_t *)all_shared_memory;
          all_shared_memory =  (nnz_scalar_t *) (((nnz_lno_t *)all_shared_memory) + ((block_size / 8) + 1) * 8);

          nnz_scalar_t *all_global_memory = NULL;


          Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, team_row_begin, team_row_end), [&] (const nnz_lno_t& ii) {


              Kokkos::parallel_for(
                                   Kokkos::ThreadVectorRange(teamMember, block_size),
                                   [&] (nnz_lno_t i) {
                                     diagonal_positions[i] = -1;
                                   });

              size_type row_begin = _xadj[ii];
              size_type row_end = _xadj[ii + 1];
              nnz_lno_t row_size = row_end - row_begin;

              nnz_lno_t l1_val_size = row_size * block_size, l2_val_size = 0;
              //if the current row size is larger than shared memory size,
              //than allocate l2 vector.
              if (row_size > num_max_vals_in_l1){
                volatile nnz_scalar_t * tmp = NULL;
                while (tmp == NULL){
                  Kokkos::single(Kokkos::PerThread(teamMember),[&] (volatile nnz_scalar_t * &memptr) {
                      memptr = (volatile nnz_scalar_t * )( pool.allocate_chunk(ii));
                    }, tmp);
                }
                all_global_memory = (nnz_scalar_t *)tmp;
                l1_val_size = num_max_vals_in_l1 * block_size;
                l2_val_size = (row_size * block_size - l1_val_size);
              }
              //bring values to l1 vector
              Kokkos::parallel_for(
                                   Kokkos::ThreadVectorRange(teamMember, l1_val_size),
                                   [&] (nnz_lno_t i) {
                                     size_type adjind = i / block_size + row_begin;
                                     nnz_lno_t colIndex = _adj[adjind];

                                     if (colIndex == ii){
                                       diagonal_positions[i % block_size] = i;
                                     }
                                     all_shared_memory[i] = _Xvector[colIndex * block_size + i % block_size];
                                   });
              //bring values to l2 vector.
              Kokkos::parallel_for(
                                   Kokkos::ThreadVectorRange(teamMember, l2_val_size),
                                   [&] (nnz_lno_t k) {
                                     nnz_lno_t i = l1_val_size + k;

                                     size_type adjind = i / block_size + row_begin;
                                     nnz_lno_t colIndex = _adj[adjind];

                                     if (colIndex == ii){
                                       diagonal_positions[i % block_size] = i;
                                     }
                                     all_global_memory[k] = _Xvector[colIndex * block_size + i % block_size];
                                   });

              row_begin = row_begin * block_size * block_size;
              //sequentially solve in the block.
              //this respects backward and forward sweeps.
              for (int m = 0; m < block_size; ++m ){
                int i = m;
                if (is_backward) i = block_size - m - 1;
                size_type current_row_begin = row_begin + i * row_size * block_size;
                //first reduce l1 dot product.
                //MD: TODO: if thread dot product is implemented it should be called here.
                nnz_scalar_t product = 0 ;
                Kokkos::parallel_reduce(
                                        Kokkos::ThreadVectorRange(teamMember, l1_val_size),
                                        [&] (nnz_lno_t colind, nnz_scalar_t & valueToUpdate) {

                                          valueToUpdate += all_shared_memory[colind] * _adj_vals(current_row_begin + colind);

                                        },
                                        product);
                //l2 dot product.
                //MD: TODO: if thread dot product is implemented, it should be called here again.
                nnz_scalar_t product2 = 0 ;
                Kokkos::parallel_reduce(
                                        Kokkos::ThreadVectorRange(teamMember, l2_val_size),
                                        [&] (nnz_lno_t colind2, nnz_scalar_t & valueToUpdate) {
                                          nnz_lno_t colind = colind2 + l1_val_size;
                                          valueToUpdate += all_global_memory[colind2] * _adj_vals(current_row_begin + colind);
                                        },
                                        product2);

                product += product2;
                //update the new vector entries.
                Kokkos::single(Kokkos::PerThread(teamMember),[=] () {
                    nnz_lno_t block_row_index = ii * block_size + i;
                    nnz_scalar_t invDiagonalVal = _permuted_inverse_diagonal[block_row_index];
                    _Xvector[block_row_index] += omega * (_Yvector[block_row_index] - product) * invDiagonalVal;

                    //we need to update the values of the vector entries if they are already brought to shared memory to sync with global memory.
                    if (diagonal_positions[i] != -1){
                      if (diagonal_positions[i] < l1_val_size)
                        all_shared_memory[diagonal_positions[i]] = _Xvector[block_row_index];
                      else
                        all_global_memory[diagonal_positions[i] - l1_val_size] = _Xvector[block_row_index];
                    }
                  });



#if KOKKOSSPARSE_IMPL_PRINTDEBUG
                if (/*i == 0 && ii == 1*/ ii == 0 || (block_size == 1 && ii < 2) ){
                  std::cout << "\n\n\nrow:" << ii * block_size + i;
                  std::cout << "\nneighbors:";
                  for (int z = 0; z < int (row_size); ++z){
                    std::cout << _adj[_xadj[ii] + z] << " ";
                  }

                  std::cout <<"\n\nrow-0:X -- all-shared-memory:";
                  for (int z = 0; z < int (row_size * block_size); ++z){
                    std::cout << all_shared_memory[z] << " ";
                  }
                  std::cout << std::endl << "product:" << product << std::endl;
                  std::cout << "diagonal" << _permuted_inverse_diagonal[ii * block_size + i] << std::endl;
                  std::cout << "_Yvector" << _Yvector[ii * block_size + i] << std::endl;

                  std::cout << std::endl << "block_row_index:" << ii * block_size + i <<  " _Xvector[block_row_index]:" << _Xvector[ii * block_size + i] << std::endl;
                }
#endif
              }
              if (row_size > num_max_vals_in_l1)
                Kokkos::single(Kokkos::PerThread(teamMember),[&] () {
                    pool.release_chunk(all_global_memory);
                  });
            });


        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const BlockTag&, const team_member_t & teamMember) const {


          const nnz_lno_t team_row_begin = teamMember.league_rank() * team_work_size + _color_set_begin;
          const nnz_lno_t team_row_end = KOKKOSKERNELS_MACRO_MIN(team_row_begin + team_work_size, _color_set_end);

          nnz_scalar_t *all_shared_memory = (nnz_scalar_t *) (teamMember.team_shmem().get_shmem(shared_memory_size));

          all_shared_memory += thread_shared_memory_scalar_size * teamMember.team_rank();


          nnz_lno_t *diagonal_positions = (nnz_lno_t *)all_shared_memory;
          all_shared_memory =  (nnz_scalar_t *) (((nnz_lno_t *)all_shared_memory) + ((block_size / 8) + 1) * 8);



          Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, team_row_begin, team_row_end), [&] (const nnz_lno_t& ii) {
#if KOKKOSSPARSE_IMPL_PRINTDEBUG
              Kokkos::single(Kokkos::PerThread(teamMember),[=] () {
                  for(nnz_lno_t i = 0; i < block_size; diagonal_positions[i++] = -1);
                });
#endif


              Kokkos::parallel_for(
                                   Kokkos::ThreadVectorRange(teamMember, block_size),
                                   [&] (nnz_lno_t i) {
                                     diagonal_positions[i] = -1;
                                   });

              size_type row_begin = _xadj[ii];
              size_type row_end = _xadj[ii + 1];
              nnz_lno_t row_size = row_end - row_begin;

              Kokkos::parallel_for(
                                   Kokkos::ThreadVectorRange(teamMember, row_size * block_size),
                                   [&] (nnz_lno_t i) {


                                     size_type adjind = i / block_size + row_begin;
                                     nnz_lno_t colIndex = _adj[adjind];

                                     if (colIndex == ii){
                                       diagonal_positions[i % block_size] = i;
                                     }
                                     all_shared_memory[i] = _Xvector[colIndex * block_size + i % block_size];

                                   });

              row_begin = row_begin * block_size * block_size;

              for (int m = 0; m < block_size; ++m ){
                int i = m;
                if (is_backward) i = block_size - m - 1;
                size_type current_row_begin = row_begin + i * row_size * block_size;

                nnz_scalar_t product = 0 ;
                Kokkos::parallel_reduce(
                                        Kokkos::ThreadVectorRange(teamMember, row_size * block_size),
                                        [&] (nnz_lno_t colind, nnz_scalar_t & valueToUpdate) {

                                          valueToUpdate += all_shared_memory[colind] * _adj_vals(current_row_begin + colind);

                                        },
                                        product);


                Kokkos::single(Kokkos::PerThread(teamMember),[=] () {
                    nnz_lno_t block_row_index = ii * block_size + i;
                    nnz_scalar_t invDiagonalVal = _permuted_inverse_diagonal[block_row_index];
                    _Xvector[block_row_index] += omega * (_Yvector[block_row_index] - product) * invDiagonalVal;


                    if (diagonal_positions[i] != -1){
                      all_shared_memory[diagonal_positions[i]] = _Xvector[block_row_index];
                    }

                  });

#if !defined(__CUDA_ARCH__)
#if KOKKOSSPARSE_IMPL_PRINTDEBUG
                if (/*i == 0 && ii == 1*/ ii == 0 || (block_size == 1 && ii < 2) ){
                  std::cout << "\n\n\nrow:" << ii * block_size + i;
                  std::cout << "\nneighbors:";
                  for (int z = 0; z < int (row_size); ++z){
                    std::cout << _adj[_xadj[ii] + z] << " ";
                  }

                  std::cout <<"\n\nrow-0:X -- all-shared-memory:";
                  for (int z = 0; z < int (row_size * block_size); ++z){
                    std::cout << all_shared_memory[z] << " ";
                  }
                  std::cout << std::endl << "product:" << product << std::endl;
                  std::cout << "diagonal" << _permuted_inverse_diagonal[ii * block_size + i] << std::endl;
                  std::cout << "_Yvector" << _Yvector[ii * block_size + i] << std::endl;

                  std::cout << std::endl << "block_row_index:" << ii * block_size + i <<  " _Xvector[block_row_index]:" << _Xvector[ii * block_size + i] << std::endl << std::endl<< std::endl;
                }
#endif
#endif
                //row_begin += row_size * block_size;
              }
            });


        }

        size_t team_shmem_size (int team_size) const {
          return shared_memory_size;
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
        is_symmetric(true){}


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
        is_symmetric(is_symmetric_){}



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
        is_symmetric(is_symmetric_){}


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
        is_symmetric(is_symmetric_){}



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
        if(gsHandler->get_algorithm_type() != GS_CLUSTER)
        {
          if (!is_symmetric){

            if (gchandle->get_coloring_algo_type() == KokkosGraph::COLORING_EB){

              gchandle->symmetrize_and_calculate_lower_diagonal_edge_list(num_rows, xadj, adj);
              KokkosGraph::Experimental::graph_color_symbolic <HandleType, const_lno_row_view_t, const_lno_nnz_view_t>
                (this->handle, num_rows, num_rows, xadj , adj);
            }
            else {
              row_lno_temp_work_view_t tmp_xadj;
              nnz_lno_temp_work_view_t tmp_adj;
              KokkosKernels::Impl::symmetrize_graph_symbolic_hashmap
                < const_lno_row_view_t, const_lno_nnz_view_t,
                  row_lno_temp_work_view_t, nnz_lno_temp_work_view_t,
                  MyExecSpace>
                (num_rows, xadj, adj, tmp_xadj, tmp_adj );
              KokkosGraph::Experimental::graph_color_symbolic <HandleType, row_lno_temp_work_view_t, nnz_lno_temp_work_view_t> (this->handle, num_rows, num_rows, tmp_xadj , tmp_adj);
            }
          }
          else {
            KokkosGraph::Experimental::graph_color_symbolic <HandleType, const_lno_row_view_t, const_lno_nnz_view_t> (this->handle, num_rows, num_rows, xadj , adj);
          }
          colors =  gchandle->get_vertex_colors();
          numColors = gchandle->get_num_colors();
        }
        else
        {
          typedef Kokkos::View<row_lno_t*, MyTempMemorySpace> rowmap_t;
          typedef Kokkos::View<nnz_lno_t*, MyTempMemorySpace> colind_t;
          typedef Kokkos::View<const row_lno_t*, MyTempMemorySpace> const_rowmap_t;
          typedef Kokkos::View<const nnz_lno_t*, MyTempMemorySpace> const_colind_t;
          if(is_symmetric)
          {
            colors = initialize_symbolic_cluster<const_rowmap_t, const_colind_t>(xadj, adj, numColors);
          }
          else
          {
            rowmap_t tmp_xadj;
            colind_t tmp_adj;
            KokkosKernels::Impl::symmetrize_graph_symbolic_hashmap
              <const_rowmap_t, const_colind_t, rowmap_t, colind_t, MyExecSpace>
              (num_rows, xadj, adj, tmp_xadj, tmp_adj);
            colors = initialize_symbolic_cluster<rowmap_t, colind_t>(tmp_xadj, tmp_adj, numColors);
          }
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "initialize_symbolic_cluster(): " << timer.seconds() << '\n';
#endif
        std::cout << "Expected (max) degree of parallelism in GS apply: " << (double) num_rows / numColors << std::endl;
        }
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "COLORING_TIME:" << timer.seconds() << std::endl;
        timer.reset();
#endif

#if KOKKOSSPARSE_IMPL_RUNSEQUENTIAL
        numColors = num_rows;
        KokkosKernels::Impl::print_1Dview(colors);
        std::cout << "numCol:" << numColors << " numRows:" << num_rows << " cols:" << num_cols << " nnz:" << adj.extent(0) <<  std::endl;
        typename HandleType::GraphColoringHandleType::color_view_t::HostMirror  h_colors = Kokkos::create_mirror_view (colors);
        for(int i = 0; i < num_rows; ++i){
          h_colors(i) = i + 1;
        }
        Kokkos::deep_copy(colors, h_colors);
#endif
        nnz_lno_persistent_work_view_t color_xadj;

        nnz_lno_persistent_work_view_t color_adj;


#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        timer.reset();
#endif

        KokkosKernels::Impl::create_reverse_map
          <typename HandleType::GraphColoringHandleType::color_view_t,
           nnz_lno_persistent_work_view_t, MyExecSpace>
          (num_rows, numColors, colors, color_xadj, color_adj);
        MyExecSpace().fence();

#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "CREATE_REVERSE_MAP:" << timer.seconds() << std::endl;
        timer.reset();
#endif

        nnz_lno_persistent_work_host_view_t  h_color_xadj = Kokkos::create_mirror_view (color_xadj);
        Kokkos::deep_copy (h_color_xadj , color_xadj);
        MyExecSpace().fence();


#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "DEEP_COPY:" << timer.seconds() << std::endl;
        timer.reset();
#endif


#if defined( KOKKOS_ENABLE_CUDA )
        if (Kokkos::Impl::is_same<Kokkos::Cuda, MyExecSpace >::value){
          for (nnz_lno_t i = 0; i < numColors; ++i){
            nnz_lno_t color_index_begin = h_color_xadj(i);
            nnz_lno_t color_index_end = h_color_xadj(i + 1);

            if (color_index_begin + 1 >= color_index_end ) continue;
            auto colorsubset =
              subview(color_adj, Kokkos::pair<row_lno_t, row_lno_t> (color_index_begin, color_index_end));
            MyExecSpace().fence();
            Kokkos::sort (colorsubset);
            //TODO: MD 08/2017: If I remove the below fence, code fails on cuda.
            //I do not see any reason yet it to fail.
            MyExecSpace().fence();
          }
        }
#endif



        MyExecSpace().fence();
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "SORT_TIME:" << timer.seconds() << std::endl;
        timer.reset();
        //std::cout << "sort" << std::endl;
#endif

        row_lno_persistent_work_view_t permuted_xadj ("new xadj", num_rows + 1);
        nnz_lno_persistent_work_view_t old_to_new_map ("old_to_new_index_", num_rows );
        nnz_lno_persistent_work_view_t permuted_adj ("newadj_", nnz );

        Kokkos::parallel_for( "KokkosSparse::GaussSeidel::create_permuted_xadj", my_exec_space(0,num_rows),
                              create_permuted_xadj(
                                                   color_adj,
                                                   xadj,
                                                   permuted_xadj,
                                                   old_to_new_map));
        //std::cout << "create_permuted_xadj" << std::endl;
        MyExecSpace().fence();

#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "CREATE_PERMUTED_XADJ:" << timer.seconds() << std::endl;

        timer.reset();
#endif


        KokkosKernels::Impl::inclusive_parallel_prefix_sum
          <row_lno_persistent_work_view_t, MyExecSpace>
          (num_rows + 1, permuted_xadj);
        MyExecSpace().fence();

#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "INCLUSIVE_PPS:" << timer.seconds() << std::endl;
        timer.reset();
#endif


        Kokkos::parallel_for( "KokkosSparse::GaussSeidel::fill_matrix_symbolic",my_exec_space(0,num_rows),
                              fill_matrix_symbolic(
                                                   num_rows,
                                                   color_adj,
                                                   xadj,
                                                   adj,
                                                   //adj_vals,
                                                   permuted_xadj,
                                                   permuted_adj,
                                                   //newvals_,
                                                   old_to_new_map));
        MyExecSpace().fence();

#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "SYMBOLIC_FILL:" << timer.seconds() << std::endl;
        timer.reset();
#endif

        nnz_lno_t block_size = this->handle->get_gs_handle()->get_block_size();

        //MD: if block size is larger than 1;
        //the algorithm copies the vector entries into shared memory and reuses this small shared memory for vector entries.
        if (block_size > 1)
          {
            //first calculate max row size.
            size_type max_row_size = 0;
            KokkosKernels::Impl::kk_view_reduce_max_row_size<size_type, MyExecSpace>(num_rows, permuted_xadj.data(), permuted_xadj.data() + 1, max_row_size);
            gsHandler->set_max_nnz(max_row_size);


            nnz_lno_t brows = permuted_xadj.extent(0) - 1;
            size_type bnnz =  permuted_adj.extent(0) * block_size * block_size;

            int suggested_vector_size = this->handle->get_suggested_vector_size(brows, bnnz);
            int suggested_team_size = this->handle->get_suggested_team_size(suggested_vector_size);
            size_t shmem_size_to_use = this->handle->get_shmem_size();
            //nnz_lno_t team_row_chunk_size = this->handle->get_team_work_size(suggested_team_size,MyExecSpace::concurrency(), brows);

            //MD: now we calculate how much memory is needed for shared memory.
            //we have two-level vectors: as in spgemm hashmaps.
            //we try to fit everything into shared memory.
            //if they fit, we can use BlockTeam function in Team_SGS functor.
            //on CPUs, we make L1 vector big enough so that it will always hold it.
            //on GPUs, we have a upper bound for shared memory: handle->get_shmem_size(): this is set to 32128 bytes.
            //If things do not fit into shared memory, we allocate vectors in global memory and run BigBlockTeam in Team_SGS functor.
            size_t level_1_mem = max_row_size * block_size * sizeof(nnz_scalar_t) + ((block_size / 8 ) + 1) * 8 * sizeof(nnz_lno_t);
            level_1_mem = suggested_team_size * level_1_mem;
            size_t level_2_mem = 0;
            nnz_lno_t num_values_in_l1 = max_row_size;
            nnz_lno_t num_values_in_l2 = 0;
            nnz_lno_t num_big_rows = 0;

            KokkosKernels::Impl::ExecSpaceType ex_sp = this->handle->get_handle_exec_space();
            if (ex_sp != KokkosKernels::Impl::Exec_CUDA){
              //again, if it is on CPUs, we make L1 as big as we need.
              size_t l1mem = 1;
              while(l1mem < level_1_mem){
                l1mem *= 2;
              }
              gsHandler->set_level_1_mem(l1mem);
              level_1_mem = l1mem;
              level_2_mem = 0;
            }
            else {
              //on GPUs set the L1 size to max shmem and calculate how much we need for L2.
              //we try to shift with 8 always because of the errors we experience with memory shifts on GPUs.
              level_1_mem = shmem_size_to_use;
              num_values_in_l1 = (shmem_size_to_use / suggested_team_size - ((block_size / 8 ) + 1) * 8 * sizeof(nnz_lno_t)) / sizeof(nnz_scalar_t) / block_size;
              if (((block_size / 8 ) + 1) * 8 * sizeof(nnz_lno_t) > shmem_size_to_use / suggested_team_size ) throw "Shared memory size is to small for the given block size\n";
              if (num_values_in_l1 >= (nnz_lno_t) (max_row_size) ){
                num_values_in_l2 = 0;
                level_2_mem = 0;
                num_big_rows = 0;
              }
              else {

                num_values_in_l2 = max_row_size - num_values_in_l1;
                level_2_mem = num_values_in_l2 * block_size  * sizeof(nnz_scalar_t);
                //std::cout << "level_2_mem:" << level_2_mem << std::endl;
                size_t l2mem = 1;
                while(l2mem < level_2_mem){
                  l2mem *= 2;
                }
                level_2_mem  = l2mem;
                //std::cout << "level_2_mem:" << level_2_mem << std::endl;

                size_type num_large_rows = 0;
                KokkosKernels::Impl::kk_reduce_numrows_larger_than_threshold<row_lno_persistent_work_view_t, MyExecSpace>(brows, permuted_xadj, num_values_in_l1, num_large_rows);
                num_big_rows = KOKKOSKERNELS_MACRO_MIN(num_large_rows, (size_type)(MyExecSpace::concurrency() / suggested_vector_size));
                //std::cout << "num_big_rows:" << num_big_rows << std::endl;

#if defined( KOKKOS_ENABLE_CUDA )
                if (ex_sp == KokkosKernels::Impl::Exec_CUDA) {
                  //check if we have enough memory for this. lower the concurrency if we do not have enugh memory.
                  size_t free_byte ;
                  size_t total_byte ;
                  cudaMemGetInfo( &free_byte, &total_byte ) ;
                  size_t required_size = size_t (num_big_rows) * level_2_mem;
                  if (required_size + num_big_rows * sizeof(int) > free_byte){
                    num_big_rows = ((((free_byte - num_big_rows * sizeof(int))* 0.8) /8 ) * 8) / level_2_mem;
                  }
                  {
                    nnz_lno_t min_chunk_size = 1;
                    while (min_chunk_size * 2 <= num_big_rows) {
                      min_chunk_size *= 2;
                    }
                    num_big_rows = min_chunk_size;
                  }
                }
#endif
              }

            }

            gsHandler->set_max_nnz(max_row_size);
            gsHandler->set_level_1_mem(level_1_mem);
            gsHandler->set_level_2_mem(level_2_mem);

            gsHandler->set_num_values_in_l1(num_values_in_l1);
            gsHandler->set_num_values_in_l2(num_values_in_l2);
            gsHandler->set_num_big_rows(num_big_rows);

          }


        gsHandler->set_color_set_xadj(h_color_xadj);
        gsHandler->set_color_set_adj(color_adj);
        gsHandler->set_num_colors(numColors);
        gsHandler->set_new_xadj(permuted_xadj);
        gsHandler->set_new_adj(permuted_adj);
        //gsHandler->set_new_adj_val(newvals_);
        gsHandler->set_old_to_new_map(old_to_new_map);
        if (this->handle->get_gs_handle()->is_owner_of_coloring()){
          this->handle->destroy_graph_coloring_handle();
          this->handle->get_gs_handle()->set_owner_of_coloring(false);
        }
        this->handle->get_gs_handle()->set_call_symbolic(true);


        this->handle->get_gs_handle()->allocate_x_y_vectors(this->num_rows * block_size, this->num_cols * block_size);
        //std::cout << "all end" << std::endl;
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "ALLOC:" << timer.seconds() << std::endl;
#endif
      }

      /**********************************************/
      /* Functors for initialize_symbolic_cluster() */
      /**********************************************/

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

      /*
      struct ClusterEntryCountingFunctor
      {
        typedef Kokkos::View<row_lno_t*, MyTempMemorySpace> RowmapView;
        typedef Kokkos::View<nnz_lno_t*, MyTempMemorySpace> EntriesView;
        typedef Kokkos::View<size_type*, MyTempMemorySpace> BitsetView;
        ClusterEntryCountingFunctor(RowmapView& clusterRowmap_, BitsetView& denseClusterRow_, nnz_lno_persistent_work_view_t& clusterOrder_, nnz_lno_persistent_work_view_t& clusterPerm_, nnz_lno_t numClusters_, nnz_lno_t clusterSize_, row_lno_persistent_work_view_t& xadj_, nnz_lno_persistent_work_view_t& adj_, nnz_lno_t nthreads_, nnz_lno_t numRows_)
          : clusterRowmap(clusterRowmap_), denseClusterRow(denseClusterRow_), clusterOrder(clusterOrder_), clusterPerm(clusterPerm_), numClusters(numClusters_), clusterSize(clusterSize_), xadj(xadj_), adj(adj_), nthreads(nthreads_), numRows(numRows_)
        {}

        KOKKOS_INLINE_FUNCTION void operator()(const size_type tid) const
        {
          const size_type bitsPerST = 8 * sizeof(size_type);
          const size_type wordsPerRow = (numClusters + bitsPerST - 1) / bitsPerST;
          for(nnz_lno_t i = 0; i < numClusters; i += nthreads)
          {
            nnz_lno_t c = tid + i;
            if(c < numClusters)
            {
              //zero out all bits in dense row for this thread
              //denseRow is a bitset that holds row c of dense cluster graph
              size_type* denseRow = &denseClusterRow(tid * wordsPerRow);
              for(size_type j = 0; j < wordsPerRow; j++)
                denseRow[j] = 0;
              nnz_lno_t clusterBegin = c * clusterSize;
              nnz_lno_t clusterEnd = (c + 1) * clusterSize;
              if(c == numClusters - 1)
                clusterEnd = numRows - 1;
              for(nnz_lno_t orderRow = clusterBegin; orderRow < clusterEnd; orderRow++)
              {
                nnz_lno_t origRow = clusterPerm(orderRow);
                //map neighbors of origRow to the RCM matrix, then to clusters
                for(size_type neiIndex = xadj(origRow); neiIndex < xadj(origRow + 1); neiIndex++)
                {
                  nnz_lno_t nei = adj(neiIndex);
                  nnz_lno_t orderNei = clusterOrder(nei);
                  nnz_lno_t clusterNei = orderNei / clusterSize;
                  //record the entry in dense row
                  //this should be fast since bitsPerST is a power of 2
                  denseRow[clusterNei / bitsPerST] |= (size_type(1) << (clusterNei % bitsPerST));
                }
              }
              //count the 1 bits in denseRow
              size_type numEntries = 0;
              for(size_type j = 0; j < wordsPerRow; j++)
              {
                //use the KokkosKernels popcount intrinsic wrapper
                numEntries += KokkosKernels::Impl::pop_count(denseRow[j]);
              }
              //finally, record the entry count for this row
              clusterRowmap(c) = numEntries;
            }
          }
        }
        RowmapView clusterRowmap;
        BitsetView denseClusterRow;
        nnz_lno_persistent_work_view_t clusterOrder;
        nnz_lno_persistent_work_view_t clusterPerm;
        nnz_lno_t numClusters;
        nnz_lno_t clusterSize;
        row_lno_persistent_work_view_t xadj;
        nnz_lno_persistent_work_view_t adj;
        nnz_lno_t nthreads;
        nnz_lno_t numRows;
      };

      //Functor that fills in clusterEntries
      struct ClusterEntryFunctor
      {
        typedef Kokkos::View<row_lno_t*, MyTempMemorySpace> RowmapView;
        typedef Kokkos::View<nnz_lno_t*, MyTempMemorySpace> EntriesView;
        typedef Kokkos::View<size_type*, MyTempMemorySpace> BitsetView;
        ClusterEntryFunctor(
            RowmapView&                     clusterRowmap_,
            EntriesView&                    clusterEntries_,
            BitsetView&                     denseClusterRow_,
            row_lno_persistent_work_view_t& xadj_,
            nnz_lno_persistent_work_view_t& adj_,
            nnz_lno_persistent_work_view_t& clusterOrder_,
            nnz_lno_persistent_work_view_t& clusterPerm_,
            nnz_lno_t                       numClusters_,
            nnz_lno_t                       clusterSize_,
            nnz_lno_t                       numRows_,
            nnz_lno_t                       nthreads_)
          :
          clusterRowmap(clusterRowmap_), clusterEntries(clusterEntries_),
          denseClusterRow(denseClusterRow_), xadj(xadj_), adj(adj_),
          clusterOrder(clusterOrder_), clusterPerm(clusterPerm_), numClusters(numClusters_),
          clusterSize(clusterSize_), numRows(numRows_), nthreads(nthreads_)
        {}
        KOKKOS_INLINE_FUNCTION void operator()(const size_type tid) const
        {
          const size_type bitsPerST = 8 * sizeof(size_type);
          const size_type wordsPerRow = (numClusters + bitsPerST - 1) / bitsPerST;
          for(nnz_lno_t i = 0; i < numClusters; i += nthreads)
          {
            nnz_lno_t c = tid + i;
            if(c < numClusters)
            {
              //zero out all bits in dense row for this thread
              //denseRow is a bitset that holds row c of dense cluster graph
              size_type* denseRow = &denseClusterRow(tid * wordsPerRow);
              for(size_type j = 0; j < wordsPerRow; j++)
                denseRow[j] = 0;
              nnz_lno_t clusterBegin = c * clusterSize;
              nnz_lno_t clusterEnd = (c + 1) * clusterSize;
              if(c == numClusters - 1)
                clusterEnd = numRows - 1;
              for(nnz_lno_t orderRow = clusterBegin; orderRow < clusterEnd; orderRow++)
              {
                nnz_lno_t origRow = clusterPerm(orderRow);
                //map neighbors of origRow to the RCM matrix, then to clusters
                for(size_type neiIndex = xadj(origRow); neiIndex < xadj(origRow + 1); neiIndex++)
                {
                  nnz_lno_t nei = adj(neiIndex);
                  nnz_lno_t orderNei = clusterOrder(nei);
                  nnz_lno_t clusterNei = orderNei / clusterSize;
                  //record the entry in dense row
                  //this should be fast since bitsPerST is a power of 2
                  denseRow[clusterNei / bitsPerST] |= (size_type(1) << (clusterNei % bitsPerST));
                }
              }
              //write sparse cluster graph entries
              size_type numEntries = 0;
              for(size_type j = 0; j < wordsPerRow; j++)
              {
                for(size_type bitPos = 0; bitPos < bitsPerST; bitPos++)
                {
                  if(denseRow[j] & (size_type(1) << bitPos))
                  {
                    clusterEntries(clusterRowmap(c) + numEntries) = j * bitsPerST + bitPos;
                    numEntries++;
                  }
                }
              }
            }
          }
        }
        RowmapView clusterRowmap;
        EntriesView clusterEntries;
        BitsetView denseClusterRow;
        row_lno_persistent_work_view_t xadj;
        nnz_lno_persistent_work_view_t adj;
        nnz_lno_persistent_work_view_t clusterOrder;
        nnz_lno_persistent_work_view_t clusterPerm;
        nnz_lno_t numClusters;
        nnz_lno_t clusterSize;
        nnz_lno_t numRows;
        nnz_lno_t nthreads;
      };
      */

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

      template<typename rowmap_t, typename colinds_t>
      typename HandleType::GraphColoringHandleType::color_view_t initialize_symbolic_cluster(rowmap_t xadj, colinds_t adj, color_t& numColors)
      {
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        Kokkos::Impl::Timer timer;
#endif
        typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();
        typedef nnz_lno_persistent_work_view_t nnz_view_t;
        //typedef Kokkos::View<nnz_lno_t, typename nnz_view_t::memory_space> single_view_t;
        //compute the RCM ordering of the graph
        nnz_lno_t clusterSize = gsHandler->get_cluster_size();
        nnz_lno_t numClusters = (num_rows + clusterSize - 1) / clusterSize;
        nnz_view_t vertClusters;
        auto clusterAlgo = gsHandler->get_clustering_algo();
        if(clusterAlgo == CLUSTER_DEFAULT)
        {
          //Use CM if > 50 entries per row, otherwise SSSP.
          //CM is quite fast if the level sets fan out quickly, otherwise slow and non-scalable.
          if(adj.extent(0) / num_rows > 50)
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
            BalloonClustering<HandleType, rowmap_t, colinds_t> sssp(num_rows, xadj, adj);
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
        //#define KOKKOSSPARSE_IMPL_PRINTDEBUG 1
#if KOKKOSSPARSE_IMPL_PRINTDEBUG
        {
          auto vertClustersHost = Kokkos::create_mirror_view(vertClusters);
          Kokkos::deep_copy(vertClustersHost, vertClusters);
          puts("Cluster labels for each vertex (20 per line):");
          for(int i = 0 ; i < num_rows; i++)
          {
            printf("%d ", (int) vertClustersHost(i));
            if(i % 20 == 19)
              putchar('\n');
          }
          printf("\n\n\n");
        }
#endif
        //Construct the cluster offset and vertex array. These allow fast iteration over all vertices in a given cluster.
        nnz_view_t clusterOffsets("Cluster offsets", numClusters + 1);
        nnz_view_t clusterVerts("Cluster -> vertices", num_rows);
        Kokkos::parallel_for(my_exec_space(0, num_rows), ClusterSizeFunctor<nnz_view_t>(clusterOffsets, vertClusters));
        MyExecSpace().fence();
        KokkosKernels::Impl::exclusive_parallel_prefix_sum<nnz_view_t, MyExecSpace>(numClusters + 1, clusterOffsets);
        {
          nnz_view_t tempInsertCounts("Temporary cluster insert counts", numClusters);
          Kokkos::parallel_for(my_exec_space(0, num_rows), FillClusterVertsFunctor<nnz_view_t>(clusterOffsets, clusterVerts, vertClusters, tempInsertCounts));
        }
        MyExecSpace().fence();
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
          MyExecSpace().fence();
          numClusterEdges = crossClusterEdgeMask.count();
        }
        nnz_view_t clusterRowmap("Cluster graph rowmap", numClusters + 1);
        nnz_view_t clusterEntries("Cluster graph colinds", numClusterEdges);
        Kokkos::parallel_scan(my_exec_space(0, num_rows), FillClusterEntriesFunctor<rowmap_t, colinds_t, nnz_view_t>(xadj, adj, clusterRowmap, clusterEntries, clusterOffsets, clusterVerts, vertClusters, crossClusterEdgeMask));
        MyExecSpace().fence();
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
        color_view_t clusterColors = coloringHandle->get_vertex_colors();
        nnz_lno_t numClusterColors = coloringHandle->get_num_colors();
        kh.destroy_graph_coloring_handle();
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "Cluster graph coloring with " << numClusterColors << " colors: " << timer.seconds() << '\n';
        timer.reset();
#endif
        //Need to map cluster colors to row colors with no gaps, so make the last cluster have the last color (numClusterColors).
        //This is easy because any permutation of colors preserves a valid coloring. Simply swap colors c1,c2 where
        //c1 = numClusterColors and c2 = clusterColors(numClusters - 1).
        Kokkos::parallel_for(my_exec_space(0, numClusters - 1),
            ClusterColorRelabelFunctor(clusterColors, numClusterColors, numClusters));
        //finally, change the last cluster's color
        MyExecSpace::fence();
        Kokkos::parallel_for(my_exec_space(0, 1),
            RelabelLastColorFunctor(clusterColors, numClusterColors, numClusters));
        MyExecSpace::fence();
        color_view_t vertexColors("Vertex colors (Cluster GS)", num_rows);
        //Now, have a simple formula to label all the vertices with colors
        Kokkos::parallel_for(my_exec_space(0, num_rows),
            ClusterToVertexColoring(clusterColors, vertexColors, num_rows, numClusters, clusterSize));
        //determine the largest vertex color - not necessarily the last vertex's color, since that
        //cluster might be smaller than clusterSize.
        KokkosKernels::Impl::view_reduce_max<color_view_t, MyExecSpace>(num_rows, vertexColors, numColors);
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "Final vertex labeling: " << timer.seconds() << '\n';
        timer.reset();
#endif
        return vertexColors;
#undef KOKKOSSPARSE_IMPL_PRINTDEBUG
      }

      struct create_permuted_xadj{
        nnz_lno_persistent_work_view_t color_adj;
        const_lno_row_view_t oldxadj;
        row_lno_persistent_work_view_t newxadj;
        nnz_lno_persistent_work_view_t old_to_new_index;
        create_permuted_xadj(
                             nnz_lno_persistent_work_view_t color_adj_,
                             const_lno_row_view_t oldxadj_,
                             row_lno_persistent_work_view_t newxadj_,
                             nnz_lno_persistent_work_view_t old_to_new_index_):
          color_adj(color_adj_), oldxadj(oldxadj_),
          newxadj(newxadj_),old_to_new_index(old_to_new_index_){}

        KOKKOS_INLINE_FUNCTION
        void operator()(const nnz_lno_t &i) const{
          nnz_lno_t index = color_adj(i);
          newxadj(i + 1) = oldxadj[index + 1] - oldxadj[index];
          old_to_new_index[index] = i;
        }
      };

      struct fill_matrix_symbolic{
        nnz_lno_t num_rows;
        nnz_lno_persistent_work_view_t color_adj;
        const_lno_row_view_t oldxadj;
        const_lno_nnz_view_t oldadj;
        //value_array_type oldadjvals;
        row_lno_persistent_work_view_t newxadj;
        nnz_lno_persistent_work_view_t newadj;
        //value_persistent_work_array_type newadjvals;
        nnz_lno_persistent_work_view_t old_to_new_index;
        fill_matrix_symbolic(
                             nnz_lno_t num_rows_,
                             nnz_lno_persistent_work_view_t color_adj_,
                             const_lno_row_view_t oldxadj_,
                             const_lno_nnz_view_t oldadj_,
                             //value_array_type oldadjvals_,
                             row_lno_persistent_work_view_t newxadj_,
                             nnz_lno_persistent_work_view_t newadj_,
                             //value_persistent_work_array_type newadjvals_,
                             nnz_lno_persistent_work_view_t old_to_new_index_):
          num_rows(num_rows_),
          color_adj(color_adj_), oldxadj(oldxadj_), oldadj(oldadj_), //oldadjvals(oldadjvals_),
          newxadj(newxadj_), newadj(newadj_), //newadjvals(newadjvals_),
          old_to_new_index(old_to_new_index_){}

        KOKKOS_INLINE_FUNCTION
        void operator()(const nnz_lno_t &i) const{
          nnz_lno_t index = color_adj(i);
          size_type xadj_begin = newxadj(i);

          size_type old_xadj_end = oldxadj[index + 1];
          for (size_type j = oldxadj[index]; j < old_xadj_end; ++j){
            nnz_lno_t neighbor = oldadj[j];
            if(neighbor < num_rows) neighbor = old_to_new_index[neighbor];
            newadj[xadj_begin++] = neighbor;
            //newadjvals[xadj_begin++] = oldadjvals[j];
          }
        }
      };


      struct fill_matrix_numeric{
        nnz_lno_persistent_work_view_t color_adj;
        const_lno_row_view_t oldxadj;
        const_scalar_nnz_view_t oldadjvals;
        row_lno_persistent_work_view_t newxadj;
        scalar_persistent_work_view_t newadjvals;

        nnz_lno_t num_total_rows;
        nnz_lno_t rows_per_team;
        nnz_lno_t block_matrix_size;
        fill_matrix_numeric(
                            nnz_lno_persistent_work_view_t color_adj_,
                            const_lno_row_view_t oldxadj_,
                            const_scalar_nnz_view_t oldadjvals_,
                            row_lno_persistent_work_view_t newxadj_,
                            scalar_persistent_work_view_t newadjvals_,
                            nnz_lno_t num_total_rows_,
                            nnz_lno_t rows_per_team_ , nnz_lno_t block_matrix_size_):
          color_adj(color_adj_), oldxadj(oldxadj_),  oldadjvals(oldadjvals_),
          newxadj(newxadj_), newadjvals(newadjvals_),
          num_total_rows(num_total_rows_), rows_per_team(rows_per_team_), block_matrix_size(block_matrix_size_){}

        KOKKOS_INLINE_FUNCTION
        void operator()(const nnz_lno_t &i) const{
          nnz_lno_t index = color_adj(i);
          size_type xadj_begin = newxadj(i) * block_matrix_size;
          size_type old_xadj_end = oldxadj[index + 1] * block_matrix_size;

          for (size_type j = oldxadj[index] * block_matrix_size ; j < old_xadj_end; ++j){
            newadjvals[xadj_begin++] = oldadjvals[j];
          }
        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const team_member_t &team) const{

          const nnz_lno_t i_begin = team.league_rank() * rows_per_team;
          const nnz_lno_t i_end = i_begin + rows_per_team <= num_total_rows ? i_begin + rows_per_team : num_total_rows;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team,i_begin,i_end), [&] (const nnz_lno_t& i) {
              nnz_lno_t index = color_adj(i);
              size_type xadj_begin = newxadj(i) * block_matrix_size;

              size_type old_xadj_begin = oldxadj[index] * block_matrix_size;
              size_type old_xadj_end = oldxadj[index + 1] * block_matrix_size;
              Kokkos::parallel_for (Kokkos::ThreadVectorRange(team,old_xadj_end-old_xadj_begin), [&] (const nnz_lno_t& j) {
                  newadjvals[xadj_begin + j] = oldadjvals[old_xadj_begin + j];
                });
            });
        }
      };


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
                             nnz_lno_t rows_per_team_ ,
                             nnz_lno_t block_size_,
                             nnz_lno_t block_matrix_size_):
          _xadj( xadj_),
          _adj( adj_),
          _adj_vals( adj_vals_), _diagonals(diagonals_),
          num_total_rows(num_total_rows_), rows_per_team(rows_per_team_),
          block_size(block_size_),block_matrix_size(block_matrix_size_),one(Kokkos::Details::ArithTraits<nnz_scalar_t>::one()){}

        KOKKOS_INLINE_FUNCTION
        void operator()(const nnz_lno_t & row_id) const {
          size_type row_begin = _xadj[row_id];
          size_type row_end = _xadj[row_id + 1] ;
          nnz_lno_t row_size = row_end - row_begin;
          for (nnz_lno_t col_ind = 0; col_ind < row_size; ++col_ind){
            size_type nnz_ind = col_ind + row_begin;
            nnz_lno_t column_id = _adj[nnz_ind];
            if (column_id == row_id){
              size_type val_index = row_begin * block_matrix_size + col_ind;
              for (nnz_lno_t r = 0; r < block_size; ++r){
                nnz_scalar_t val = _adj_vals[val_index];
                _diagonals[row_id * block_size + r] = one / val;
                val_index += row_size + 1;
              }
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

                      //std::cout << "row_id * block_size + r:" << row_id * block_size + r << " _diagonals[row_id * block_size + r]:" << _diagonals[row_id * block_size + r] << std::endl;
                    }
                  }
                });
            });
        }
      };

      void initialize_numeric(){

        if (this->handle->get_gs_handle()->is_symbolic_called() == false){
          this->initialize_symbolic();
        }
        //else
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        Kokkos::Impl::Timer timer;
#endif
        {


          const_lno_row_view_t xadj = this->row_map;
          const_lno_nnz_view_t adj = this->entries;
          const_scalar_nnz_view_t adj_vals = this->values;

          size_type nnz = adj_vals.extent(0);

          typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();



          row_lno_persistent_work_view_t newxadj_ = gsHandler->get_new_xadj();
          nnz_lno_persistent_work_view_t old_to_new_map = gsHandler->get_old_to_new_map();
          nnz_lno_persistent_work_view_t newadj_ = gsHandler->get_new_adj();

          nnz_lno_persistent_work_view_t color_adj = gsHandler->get_color_adj();
          scalar_persistent_work_view_t permuted_adj_vals (Kokkos::ViewAllocateWithoutInitializing("newvals_"), nnz );


          int suggested_vector_size = this->handle->get_suggested_vector_size(num_rows, nnz);
          int suggested_team_size = this->handle->get_suggested_team_size(suggested_vector_size);
          nnz_lno_t rows_per_team = this->handle->get_team_work_size(suggested_team_size,MyExecSpace::concurrency(), num_rows);


          nnz_lno_t block_size = this->handle->get_gs_handle()->get_block_size();
          nnz_lno_t block_matrix_size = block_size  * block_size ;

          //MD NOTE: 03/27/2018: below fill matrix operations will work fine with block size 1.
          //If the block size is more than 1, below code assumes that the rows are sorted similar to point crs.
          //for example given a block crs with 3 blocks in a column a,b,c where each of them is 3x3 matrix as below.
          // a11 a12 a13   b11 b12 b13    c11 c12 c13
          // a21 a22 a23   b21 b22 b23    c21 c22 c23
          // a31 a32 a33   b31 b32 b33    c31 c32 c33
          // this copy assumes the storage in the following order
          // a11 a12 a13   b11 b12 b13    c11 c12 c13 a21 a22 a23   b21 b22 b23    c21 c22 c23 a31 a32 a33   b31 b32 b33    c31 c32 c33
          // this is the order that is used in the rest of the algorithm.
          // !!!!!!!!!!!!if the input has different format than this!!!!!!!!!!!!!!!!!!
          // change fill_matrix_numeric so that they store the internal matrix as above.
          // the rest will wok fine.

          if (this->handle->get_handle_exec_space() == KokkosKernels::Impl::Exec_CUDA){
            Kokkos::parallel_for( "KokkosSparse::GaussSeidel::Team_fill_matrix_numeric",
                                  team_policy_t(num_rows / rows_per_team + 1 , suggested_team_size, suggested_vector_size),
                                  fill_matrix_numeric(
                                                      color_adj,
                                                      xadj,
                                                      //adj,
                                                      adj_vals,
                                                      newxadj_,
                                                      //newadj_,
                                                      permuted_adj_vals,
                                                      //,old_to_new_map
                                                      this->num_rows,
                                                      rows_per_team,
                                                      block_matrix_size
                                                      ));
          }
          else {
            Kokkos::parallel_for( "KokkosSparse::GaussSeidel::fill_matrix_numeric",my_exec_space(0,num_rows),
                                  fill_matrix_numeric(
                                                      color_adj,
                                                      xadj,
                                                      //adj,
                                                      adj_vals,
                                                      newxadj_,
                                                      //newadj_,
                                                      permuted_adj_vals,
                                                      //,old_to_new_map
                                                      this->num_rows,
                                                      rows_per_team,
                                                      block_matrix_size
                                                      ));
          }
          MyExecSpace().fence();
          gsHandler->set_new_adj_val(permuted_adj_vals);

          scalar_persistent_work_view_t permuted_inverse_diagonal (Kokkos::ViewAllocateWithoutInitializing("permuted_inverse_diagonal"), num_rows * block_size );
          if (!have_diagonal_given) {
            Get_Matrix_Diagonals gmd(newxadj_, newadj_, permuted_adj_vals, permuted_inverse_diagonal,
                                     this->num_rows,
                                     rows_per_team,
                                     block_size,
                                     block_matrix_size);

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

      template <typename x_value_array_type, typename y_value_array_type>
      void block_apply(
                       x_value_array_type x_lhs_output_vec,
                       y_value_array_type y_rhs_input_vec,
                       bool init_zero_x_vector = false,
                       int numIter = 1,
                       nnz_scalar_t omega = Kokkos::Details::ArithTraits<nnz_scalar_t>::one(),
                       bool apply_forward = true,
                       bool apply_backward = true,
                       bool update_y_vector = true){
        if (this->handle->get_gs_handle()->is_numeric_called() == false){
          this->initialize_numeric();
        }

        typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();

        nnz_lno_t block_size = this->handle->get_gs_handle()->get_block_size();
        //nnz_lno_t block_matrix_size = block_size  * block_size ;


        scalar_persistent_work_view_t Permuted_Yvector = gsHandler->get_permuted_y_vector();
        scalar_persistent_work_view_t Permuted_Xvector = gsHandler->get_permuted_x_vector();



        row_lno_persistent_work_view_t newxadj_ = gsHandler->get_new_xadj();
        nnz_lno_persistent_work_view_t old_to_new_map = gsHandler->get_old_to_new_map();
        nnz_lno_persistent_work_view_t newadj_ = gsHandler->get_new_adj();
        nnz_lno_persistent_work_view_t color_adj = gsHandler->get_color_adj();

        color_t numColors = gsHandler->get_num_colors();



        if (update_y_vector){


          KokkosKernels::Impl::permute_block_vector
            <y_value_array_type,
             scalar_persistent_work_view_t,
             nnz_lno_persistent_work_view_t, MyExecSpace>(
                                                          num_rows, block_size,
                                                          old_to_new_map,
                                                          y_rhs_input_vec,
                                                          Permuted_Yvector
                                                          );
        }
        MyExecSpace().fence();
        if(init_zero_x_vector){
          KokkosKernels::Impl::zero_vector<scalar_persistent_work_view_t, MyExecSpace>(num_cols * block_size, Permuted_Xvector);
        }
        else{
          KokkosKernels::Impl::permute_block_vector
            <x_value_array_type, scalar_persistent_work_view_t, nnz_lno_persistent_work_view_t, MyExecSpace>(
                                                                                                             num_cols, block_size,
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
        std::cout << "Y:";
        KokkosKernels::Impl::print_1Dview(Permuted_Yvector);
        std::cout << "Original Y:";
        KokkosKernels::Impl::print_1Dview(y_rhs_input_vec);

        std::cout << "X:";
        KokkosKernels::Impl::print_1Dview(Permuted_Xvector);

        std::cout << "permuted_xadj:"; KokkosKernels::Impl::print_1Dview(permuted_xadj);
        std::cout << "permuted_adj:"; KokkosKernels::Impl::print_1Dview(permuted_adj);
        std::cout << "permuted_adj_vals:"; KokkosKernels::Impl::print_1Dview(permuted_adj_vals);
        std::cout << "permuted_diagonals:"; KokkosKernels::Impl::print_1Dview(permuted_inverse_diagonal);
#endif
        nnz_lno_persistent_work_host_view_t h_color_xadj = gsHandler->get_color_xadj();



        nnz_lno_t brows = permuted_xadj.extent(0) - 1;
        size_type bnnz =  permuted_adj_vals.extent(0);

        int suggested_vector_size = this->handle->get_suggested_vector_size(brows, bnnz);
        int suggested_team_size = this->handle->get_suggested_team_size(suggested_vector_size);
        nnz_lno_t team_row_chunk_size = this->handle->get_team_work_size(suggested_team_size,MyExecSpace::concurrency(), brows);


        //size_t shmem_size_to_use = this->handle->get_shmem_size();
        size_t l1_shmem_size = gsHandler->get_level_1_mem();
        nnz_lno_t num_values_in_l1 = gsHandler->get_num_values_in_l1();

        size_t level_2_mem = gsHandler->get_level_2_mem();
        nnz_lno_t num_values_in_l2 = gsHandler->get_num_values_in_l2();
        nnz_lno_t num_chunks = gsHandler->get_num_big_rows();

        pool_memory_space m_space(num_chunks, level_2_mem / sizeof(nnz_scalar_t), 0,  KokkosKernels::Impl::ManyThread2OneChunk, false);

#if KOKKOSSPARSE_IMPL_PRINTDEBUG
        std::cout   << "l1_shmem_size:" << l1_shmem_size << " num_values_in_l1:" << num_values_in_l1
                    << " level_2_mem:" << level_2_mem << " num_values_in_l2:" << num_values_in_l2
                    << " num_chunks:" << num_chunks << std::endl;
#endif

        Team_PSGS gs(permuted_xadj, permuted_adj, permuted_adj_vals,
                     Permuted_Xvector, Permuted_Yvector,0,0, permuted_inverse_diagonal, m_space,
                     num_values_in_l1, num_values_in_l2,
                     omega,
                     block_size, team_row_chunk_size, l1_shmem_size, suggested_team_size,
                     suggested_vector_size);

        this->IterativePSGS(
                            gs,
                            numColors,
                            h_color_xadj,
                            numIter,
                            apply_forward,
                            apply_backward);


        //Kokkos::parallel_for( my_exec_space(0,nr), PermuteVector(x_lhs_output_vec, Permuted_Xvector, color_adj));


        KokkosKernels::Impl::permute_block_vector
          <scalar_persistent_work_view_t,x_value_array_type,  nnz_lno_persistent_work_view_t, MyExecSpace>(
                                                                                                           num_cols, block_size,
                                                                                                           color_adj,
                                                                                                           Permuted_Xvector,
                                                                                                           x_lhs_output_vec
                                                                                                           );
        MyExecSpace().fence();

#if KOKKOSSPARSE_IMPL_PRINTDEBUG
        std::cout << "After X:";
        KokkosKernels::Impl::print_1Dview(Permuted_Xvector);
        std::cout << "Result X:";
        KokkosKernels::Impl::print_1Dview(x_lhs_output_vec);
        std::cout << "Y:";
        KokkosKernels::Impl::print_1Dview(Permuted_Yvector);

#endif

      }

      template <typename x_value_array_type, typename y_value_array_type>
      void point_apply(
                       x_value_array_type x_lhs_output_vec,
                       y_value_array_type y_rhs_input_vec,
                       bool init_zero_x_vector = false,
                       int numIter = 1,
                       nnz_scalar_t omega = Kokkos::Details::ArithTraits<nnz_scalar_t>::one(),
                       bool apply_forward = true,
                       bool apply_backward = true,
                       bool update_y_vector = true){

        typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();
        scalar_persistent_work_view_t Permuted_Yvector = gsHandler->get_permuted_y_vector();
        scalar_persistent_work_view_t Permuted_Xvector = gsHandler->get_permuted_x_vector();


        row_lno_persistent_work_view_t newxadj_ = gsHandler->get_new_xadj();
        nnz_lno_persistent_work_view_t old_to_new_map = gsHandler->get_old_to_new_map();
        nnz_lno_persistent_work_view_t newadj_ = gsHandler->get_new_adj();
        nnz_lno_persistent_work_view_t color_adj = gsHandler->get_color_adj();

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

        nnz_lno_persistent_work_host_view_t h_color_xadj = gsHandler->get_color_xadj();


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
#if KOKKOSSPARSE_IMPL_PRINTDEBUG
        std::cout << "--point After X:";
        KokkosKernels::Impl::print_1Dview(Permuted_Xvector);
        std::cout << "--point Result X:";
        KokkosKernels::Impl::print_1Dview(x_lhs_output_vec);
#endif

      }

      template <typename x_value_array_type, typename y_value_array_type>
      void apply(
                 x_value_array_type x_lhs_output_vec,
                 y_value_array_type y_rhs_input_vec,
                 bool init_zero_x_vector = false,
                 int numIter = 1,
                 nnz_scalar_t omega = Kokkos::Details::ArithTraits<nnz_scalar_t>::one(),
                 bool apply_forward = true,
                 bool apply_backward = true,
                 bool update_y_vector = true){
        if (this->handle->get_gs_handle()->is_numeric_called() == false){
          this->initialize_numeric();
        }
        nnz_lno_t block_size = this->handle->get_gs_handle()->get_block_size();
        if (block_size == 1){
          this->point_apply(
                            x_lhs_output_vec, y_rhs_input_vec,
                            init_zero_x_vector, numIter ,
                            omega,
                            apply_forward, apply_backward,
                            update_y_vector);
        }
        else {
          this->block_apply(
                            x_lhs_output_vec, y_rhs_input_vec,
                            init_zero_x_vector, numIter, omega,
                            apply_forward, apply_backward,
                            update_y_vector);
        }
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

        /*
          size_type nnz = this->values.extent(0);
          int suggested_vector_size = this->handle->get_suggested_vector_size(num_rows, nnz);
          int suggested_team_size = this->handle->get_suggested_team_size(suggested_vector_size);
          nnz_lno_t team_row_chunk_size = this->handle->get_team_work_size(suggested_team_size,MyExecSpace::concurrency(), brows);
          this->handle->get_gs_handle()->vector_team_size(max_allowed_team_size, vector_size, teamSizeMax, num_rows, nnz);
        */


        if (apply_forward){
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
